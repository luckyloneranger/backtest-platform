"""OU Mean Reversion -- statistical mean-reversion using Ornstein-Uhlenbeck process.

Fits OU model to each stock via OLS regression (statsmodels). Only trades stocks
with statistically significant mean-reversion (halflife < 30, p-value < 0.05).
Entry when price deviates > 2 sigma from OU equilibrium. Exit at OU mean.
Uses CNC for longs (multi-day holds) and MIS for shorts (CNC shorts not allowed).
"""

from collections import deque

import numpy as np
import statsmodels.api as sm

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.position_manager import PositionManager


@register("ou_mean_reversion")
class OUMeanReversion(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config, instruments):
        self.min_history = config.get("min_history", 60)
        self.max_halflife = config.get("max_halflife", 30)
        self.zscore_entry = config.get("zscore_entry", 2.0)
        self.zscore_exit = config.get("zscore_exit", 0.0)
        self.zscore_stop = config.get("zscore_stop", 3.0)
        self.min_pvalue = config.get("min_pvalue", 0.05)
        self.risk_pct = config.get("risk_pct", 0.03)
        self.max_hold_bars = config.get("max_hold_bars", 60)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol state
        self.prices: dict[str, deque] = {}
        self.ou_theta: dict[str, float] = {}
        self.ou_mu: dict[str, float] = {}
        self.ou_sigma: dict[str, float] = {}
        self.ou_halflife: dict[str, float] = {}
        self.ou_pvalue: dict[str, float] = {}
        self.is_tradeable: dict[str, bool] = {}

    def _ensure(self, symbol: str):
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=300)
            self.ou_theta[symbol] = 0.0
            self.ou_mu[symbol] = 0.0
            self.ou_sigma[symbol] = 0.0
            self.ou_halflife[symbol] = float("inf")
            self.ou_pvalue[symbol] = 1.0
            self.is_tradeable[symbol] = False

    def _fit_ou(self, symbol: str):
        """Fit OU model via OLS: diff(P) = alpha + beta * P_lag + epsilon.

        theta = -beta (mean-reversion speed, should be positive)
        mu = alpha / theta (long-term equilibrium)
        sigma = std(residuals) (volatility around equilibrium)
        halflife = ln(2) / theta
        """
        prices = np.array(self.prices[symbol])
        lag = prices[:-1]
        diff = np.diff(prices)
        lag_with_const = sm.add_constant(lag)

        try:
            model = sm.OLS(diff, lag_with_const).fit()
        except Exception:
            self.is_tradeable[symbol] = False
            return

        beta = model.params[1]
        alpha = model.params[0]
        theta = -beta

        if theta <= 0:
            self.is_tradeable[symbol] = False
            self.ou_theta[symbol] = theta
            self.ou_halflife[symbol] = float("inf")
            return

        mu = alpha / theta
        sigma = float(model.resid.std())
        halflife = np.log(2) / theta
        p_value = float(model.pvalues[1])

        self.ou_theta[symbol] = theta
        self.ou_mu[symbol] = mu
        self.ou_sigma[symbol] = sigma
        self.ou_halflife[symbol] = halflife
        self.ou_pvalue[symbol] = p_value

        self.is_tradeable[symbol] = (
            halflife < self.max_halflife and p_value < self.min_pvalue
        )

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        if "day" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["day"].items():
            self._ensure(symbol)
            self.prices[symbol].append(bar.close)

            n = len(self.prices[symbol])
            if n < self.min_history:
                continue

            # Re-fit OU model each bar
            self._fit_ou(symbol)

            mu = self.ou_mu[symbol]
            sigma = self.ou_sigma[symbol]
            state = self.pm.get_state(symbol)

            # Compute z-score relative to OU equilibrium
            z = (bar.close - mu) / sigma if sigma > 0 else 0.0

            # --- Flat: check entries ---
            if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                if not self.is_tradeable[symbol]:
                    continue

                qty = int(self.risk_pct * snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
                # Cap to available cash to prevent leverage
                max_qty = int(snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
                qty = min(qty, max_qty)

                inst = self.instruments.get(symbol)
                if inst and inst.lot_size > 1:
                    qty = (qty // inst.lot_size) * inst.lot_size

                if qty <= 0:
                    continue

                # Long: price well below equilibrium
                if z < -self.zscore_entry:
                    stop_price = mu - self.zscore_stop * sigma
                    signals += self.pm.enter_long(
                        symbol, qty, 0, "CNC", stop_price,
                    )

                # Short: price well above equilibrium (always MIS)
                elif z > self.zscore_entry:
                    stop_price = mu + self.zscore_stop * sigma
                    signals += self.pm.enter_short(
                        symbol, qty, 0, stop_price,
                    )

            # --- Long: check exits ---
            elif self.pm.is_long(symbol):
                should_exit = False

                # Reverted to mean
                if z >= self.zscore_exit:
                    should_exit = True
                # Diverging further (model wrong)
                elif z < -self.zscore_stop:
                    should_exit = True
                # Time stop
                elif state.bars_held > self.max_hold_bars:
                    should_exit = True

                if should_exit:
                    signals += self.pm.exit_position(symbol)

                # Update trailing stop based on OU parameters
                elif sigma > 0:
                    new_stop = mu - self.zscore_stop * sigma
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

            # --- Short: check exits ---
            elif self.pm.is_short(symbol):
                should_exit = False

                # Reverted to mean
                if z <= -self.zscore_exit:
                    should_exit = True
                # Diverging further (model wrong)
                elif z > self.zscore_stop:
                    should_exit = True
                # Time stop
                elif state.bars_held > self.max_hold_bars:
                    should_exit = True

                if should_exit:
                    signals += self.pm.exit_position(symbol)

                # Update trailing stop based on OU parameters
                elif sigma > 0:
                    new_stop = mu + self.zscore_stop * sigma
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

        return signals

    def on_complete(self):
        return {
            "strategy_type": "ou_mean_reversion",
            "max_halflife": self.max_halflife,
            "zscore_entry": self.zscore_entry,
        }
