"""Statistical Pairs Trading -- trade mean-reversion of cointegrated spread.

Finds the most cointegrated pair from the symbol list.
Computes spread = price_A - hedge_ratio * price_B
Trades the z-score of the spread:
  z > +2  -> short A, long B  (spread will converge)
  z < -2  -> long A, short B
  z crosses 0 -> exit (mean-reverted)
  |z| > 3 -> stop-loss (diverging further)
"""

from collections import deque
from itertools import combinations
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_cointegration, compute_zscore, compute_halflife
from strategies.position_manager import PositionManager


@register("pairs_trading")
class PairsTrading(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 60}]

    def initialize(self, config, instruments):
        self.lookback_period = config.get("lookback_period", 60)
        self.zscore_period = config.get("zscore_period", 20)
        self.entry_threshold = config.get("entry_threshold", 2.0)
        self.exit_threshold = config.get("exit_threshold", 0.0)
        self.stop_threshold = config.get("stop_threshold", 3.0)
        self.min_pvalue = config.get("min_pvalue", 0.05)
        self.risk_pct = config.get("risk_pct", 0.02)
        self.max_hold_bars = config.get("max_hold_bars", 50)
        self.instruments = instruments

        # Two position managers -- one per leg
        self.pm_a = PositionManager(max_pending_bars=1)
        self.pm_b = PositionManager(max_pending_bars=1)

        # Per-symbol price history
        self.prices: dict[str, deque] = {}

        # Pair state
        self.pair_selected = False
        self.symbol_a: str = ""
        self.symbol_b: str = ""
        self.hedge_ratio: float = 0.0

        # Spread tracking
        self.spread_history: deque = deque(maxlen=self.lookback_period)
        self.bars_in_trade: int = 0
        self.in_trade = False
        self.trade_direction: str = ""  # "long_spread" or "short_spread"

    def _ensure_prices(self, symbol: str):
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=self.lookback_period + 20)

    def _select_pair(self):
        """Test all symbol combinations and pick the most cointegrated pair."""
        symbols = [s for s in self.prices if len(self.prices[s]) >= self.lookback_period]
        if len(symbols) < 2:
            return

        best_pvalue = 1.0
        best_pair = None
        best_hedge = 0.0

        for sym_a, sym_b in combinations(symbols, 2):
            result = compute_cointegration(
                list(self.prices[sym_a]), list(self.prices[sym_b])
            )
            if result is None:
                continue
            p_value, hedge_ratio = result
            if p_value < best_pvalue:
                best_pvalue = p_value
                best_pair = (sym_a, sym_b)
                best_hedge = hedge_ratio

        if best_pair is not None and best_pvalue < self.min_pvalue:
            self.pair_selected = True
            self.symbol_a, self.symbol_b = best_pair
            self.hedge_ratio = best_hedge
            # Pre-fill spread history
            prices_a = list(self.prices[self.symbol_a])
            prices_b = list(self.prices[self.symbol_b])
            n = min(len(prices_a), len(prices_b))
            self.spread_history.clear()
            for i in range(n):
                self.spread_history.append(prices_a[i] - self.hedge_ratio * prices_b[i])

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        self.pm_a.increment_bars()
        self.pm_b.increment_bars()
        signals = self.pm_a.process_fills(snapshot)
        signals += self.pm_b.process_fills(snapshot)
        signals += self.pm_a.resubmit_expired(snapshot)
        signals += self.pm_b.resubmit_expired(snapshot)
        self.pm_a.reconcile(snapshot)
        self.pm_b.reconcile(snapshot)

        # Only process daily bars
        if "day" not in snapshot.timeframes:
            return signals

        # Accumulate close prices
        for symbol, bar in snapshot.timeframes["day"].items():
            self._ensure_prices(symbol)
            self.prices[symbol].append(bar.close)

        # Attempt pair selection if not yet done
        if not self.pair_selected:
            self._select_pair()
            if not self.pair_selected:
                return signals

        # Need both symbols in this bar
        bar_a = snapshot.timeframes["day"].get(self.symbol_a)
        bar_b = snapshot.timeframes["day"].get(self.symbol_b)
        if bar_a is None or bar_b is None:
            return signals

        # Update spread
        spread = bar_a.close - self.hedge_ratio * bar_b.close
        self.spread_history.append(spread)

        # Compute z-score
        zscore = compute_zscore(list(self.spread_history), self.zscore_period)
        if zscore is None:
            return signals

        # Track hold duration
        if self.in_trade:
            self.bars_in_trade += 1

        # Position sizing: each leg gets half of risk capital
        capital = snapshot.portfolio.equity
        half_capital = capital * self.risk_pct / 2.0
        qty_a = max(1, int(half_capital / bar_a.close)) if bar_a.close > 0 else 0
        qty_b = max(1, int(half_capital / bar_b.close)) if bar_b.close > 0 else 0

        if qty_a == 0 or qty_b == 0:
            return signals

        # --- Exit logic (check before entries) ---
        if self.in_trade:
            should_exit = False

            # Mean reversion exit: z-score crosses exit threshold
            if self.trade_direction == "short_spread" and zscore <= self.exit_threshold:
                should_exit = True
            elif self.trade_direction == "long_spread" and zscore >= self.exit_threshold:
                should_exit = True

            # Stop-loss: spread diverging further
            if abs(zscore) > self.stop_threshold:
                should_exit = True

            # Time stop
            if self.bars_in_trade > self.max_hold_bars:
                should_exit = True

            if should_exit:
                signals += self.pm_a.exit_position(self.symbol_a)
                signals += self.pm_b.exit_position(self.symbol_b)
                self.in_trade = False
                self.trade_direction = ""
                self.bars_in_trade = 0
                return signals

        # --- Entry logic ---
        if not self.in_trade:
            a_flat = self.pm_a.is_flat(self.symbol_a)
            b_flat = self.pm_b.is_flat(self.symbol_b)
            a_no_pending = not self.pm_a.has_pending_entry(self.symbol_a)
            b_no_pending = not self.pm_b.has_pending_entry(self.symbol_b)

            if a_flat and b_flat and a_no_pending and b_no_pending:
                if zscore > self.entry_threshold:
                    # Spread too high: short A, long B
                    # Short A (MIS) with far stop
                    signals += self.pm_a.enter_short(
                        self.symbol_a, qty_a, 0, bar_a.close * 100.0
                    )
                    # Long B (MIS) with near-zero stop
                    signals += self.pm_b.enter_long(
                        self.symbol_b, qty_b, 0, "MIS", 0.01
                    )
                    self.in_trade = True
                    self.trade_direction = "short_spread"
                    self.bars_in_trade = 0

                elif zscore < -self.entry_threshold:
                    # Spread too low: long A, short B
                    # Long A (MIS) with near-zero stop
                    signals += self.pm_a.enter_long(
                        self.symbol_a, qty_a, 0, "MIS", 0.01
                    )
                    # Short B (MIS) with far stop
                    signals += self.pm_b.enter_short(
                        self.symbol_b, qty_b, 0, bar_b.close * 100.0
                    )
                    self.in_trade = True
                    self.trade_direction = "long_spread"
                    self.bars_in_trade = 0

        return signals

    def on_complete(self):
        return {
            "strategy_type": "pairs_trading",
            "pair": f"{self.symbol_a}/{self.symbol_b}" if self.pair_selected else "none",
            "hedge_ratio": round(self.hedge_ratio, 4) if self.pair_selected else 0.0,
        }
