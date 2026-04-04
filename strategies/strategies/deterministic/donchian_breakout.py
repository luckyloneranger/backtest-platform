"""Donchian Breakout + Volume + Trailing Stop strategy.

Multi-timeframe trend-following strategy:
- Daily: Donchian channel (N-day high/low) for breakout detection
- 15-minute: Fine-grained entry timing and trailing stop management

Rules:
- BUY when daily close breaks above the N-day high AND volume is above average
- SELL when price hits trailing stop (ATR-based) OR breaks below N-day low
- Dynamic position sizing: risk_pct of capital per trade
- Only one position per symbol at a time

Why this works:
- Donchian breakout captures the start of major trends
- Volume filter avoids false breakouts on low-liquidity days
- Trailing stop locks in profits on trending moves instead of fixed exits
- ATR-based stops adapt to each stock's volatility

Config params:
- channel_period: Donchian channel lookback (default 20 days)
- atr_period: ATR period for trailing stop (default 14 days)
- atr_multiplier: trailing stop distance in ATRs (default 2.0)
- volume_factor: required volume vs average to confirm breakout (default 1.5)
- risk_pct: fraction of capital per trade (default 0.25 = 25%)
"""

from collections import deque

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal


def compute_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    """Average True Range — measures volatility."""
    if len(highs) < period + 1:
        return None

    true_ranges = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    return sum(true_ranges) / period


@register("donchian_breakout")
class DonchianBreakout(Strategy):
    """Donchian Channel Breakout with volume confirmation and ATR trailing stop."""

    def required_data(self) -> list[dict]:
        return [
            {"interval": "day", "lookback": 60},
            {"interval": "15minute", "lookback": 30},
        ]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.channel_period = config.get("channel_period", 20)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.volume_factor = config.get("volume_factor", 1.5)
        self.risk_pct = config.get("risk_pct", 0.25)
        self.instruments = instruments

        # Per-symbol state
        self.daily_highs: dict[str, deque[float]] = {}
        self.daily_lows: dict[str, deque[float]] = {}
        self.daily_closes: dict[str, deque[float]] = {}
        self.daily_volumes: dict[str, deque[int]] = {}
        self.in_position: dict[str, bool] = {}
        self.trailing_stop: dict[str, float] = {}
        self.highest_since_entry: dict[str, float] = {}
        self.current_atr: dict[str, float] = {}

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals = []

        # Update daily data when new daily bar arrives
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure_state(symbol)
                self.daily_highs[symbol].append(bar.high)
                self.daily_lows[symbol].append(bar.low)
                self.daily_closes[symbol].append(bar.close)
                self.daily_volumes[symbol].append(bar.volume)

                # Compute ATR
                atr = compute_atr(
                    list(self.daily_highs[symbol]),
                    list(self.daily_lows[symbol]),
                    list(self.daily_closes[symbol]),
                    self.atr_period,
                )
                if atr is not None:
                    self.current_atr[symbol] = atr

        # Process 15-minute bars for entries and stop management
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            self._ensure_state(symbol)

            # Reconcile in_position flag with portfolio on order rejection
            if self.in_position[symbol]:
                held = any(p.symbol == symbol and p.quantity > 0 for p in snapshot.portfolio.positions)
                if not held:
                    self.in_position[symbol] = False
                    self.trailing_stop[symbol] = 0.0
                    self.highest_since_entry[symbol] = 0.0

            closes = list(self.daily_closes.get(symbol, []))
            highs = list(self.daily_highs.get(symbol, []))
            lows = list(self.daily_lows.get(symbol, []))
            volumes = list(self.daily_volumes.get(symbol, []))

            if len(closes) < self.channel_period + 1:
                continue

            # Donchian channel: highest high and lowest low over last N days
            # Use [:-1] to exclude the current (incomplete) day
            channel_high = max(highs[-(self.channel_period + 1):-1])
            channel_low = min(lows[-(self.channel_period + 1):-1])

            # Average volume over channel period
            avg_volume = sum(volumes[-(self.channel_period + 1):-1]) / self.channel_period

            atr = self.current_atr.get(symbol, 0.0)

            if not self.in_position[symbol]:
                # --- ENTRY: price breaks above channel high with volume ---
                if (
                    bar.close > channel_high
                    and atr > 0
                    and volumes  # ensure we have volume data
                    and volumes[-1] > avg_volume * self.volume_factor
                ):
                    # Dynamic position sizing
                    qty = int(snapshot.portfolio.cash * self.risk_pct / bar.close)

                    # Round to lot size for F&O
                    inst = self.instruments.get(symbol)
                    if inst and inst.lot_size > 1:
                        qty = (qty // inst.lot_size) * inst.lot_size

                    if qty > 0:
                        signals.append(Signal(
                            action="BUY", symbol=symbol, quantity=qty,
                            product_type="CNC",
                        ))
                        self.in_position[symbol] = True
                        self.highest_since_entry[symbol] = bar.close
                        self.trailing_stop[symbol] = bar.close - (atr * self.atr_multiplier)

            else:
                # --- MANAGE POSITION: update trailing stop ---
                if bar.close > self.highest_since_entry[symbol]:
                    self.highest_since_entry[symbol] = bar.close
                    # Move stop up: highest price - ATR * multiplier
                    new_stop = bar.close - (atr * self.atr_multiplier) if atr > 0 else self.trailing_stop[symbol]
                    self.trailing_stop[symbol] = max(self.trailing_stop[symbol], new_stop)

                # --- EXIT: trailing stop hit OR channel low break ---
                should_exit = False

                if bar.close <= self.trailing_stop[symbol]:
                    should_exit = True

                if bar.close < channel_low:
                    should_exit = True

                if should_exit:
                    # Sell entire position
                    held_qty = 0
                    for pos in snapshot.portfolio.positions:
                        if pos.symbol == symbol:
                            held_qty = pos.quantity
                            break

                    if held_qty > 0:
                        signals.append(Signal(
                            action="SELL", symbol=symbol, quantity=held_qty,
                            product_type="CNC",
                        ))
                    self.in_position[symbol] = False
                    self.trailing_stop[symbol] = 0.0
                    self.highest_since_entry[symbol] = 0.0

        return signals

    def _ensure_state(self, symbol: str) -> None:
        if symbol not in self.daily_highs:
            max_len = self.channel_period + self.atr_period + 10
            self.daily_highs[symbol] = deque(maxlen=max_len)
            self.daily_lows[symbol] = deque(maxlen=max_len)
            self.daily_closes[symbol] = deque(maxlen=max_len)
            self.daily_volumes[symbol] = deque(maxlen=max_len)
            self.in_position[symbol] = False
            self.trailing_stop[symbol] = 0.0
            self.highest_since_entry[symbol] = 0.0

    def on_complete(self) -> dict:
        return {
            "strategy_type": "donchian_breakout",
            "channel_period": self.channel_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "volume_factor": self.volume_factor,
            "risk_pct": self.risk_pct,
        }
