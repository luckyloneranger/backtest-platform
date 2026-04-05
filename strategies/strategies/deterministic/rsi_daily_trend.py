"""RSI Mean Reversion with Trend Filter + Pyramiding strategy.

Multi-timeframe strategy that uses:
- 15-minute RSI for entry/exit signals (mean reversion)
- Daily EMA for trend direction (regime filter)
- Daily ATR for trailing stop calculation

Long entry (scaled pyramid in uptrend):
- Level 1: RSI < 40 AND daily trend up -> buy 1/3 of target position
- Level 2: RSI < 30 AND already at Level 1 -> buy another 1/3
- Level 3: RSI < 20 AND already at Level 1+2 -> buy final 1/3

Short entry (scaled pyramid in downtrend):
- Level 1: RSI > 60 AND daily trend down -> sell 1/3 of target position
- Level 2: RSI > 70 AND already at Level 1 -> sell another 1/3
- Level 3: RSI > 80 AND already at Level 1+2 -> sell final 1/3

Trend filter:
- Trend UP = daily close > EMA AND EMA is rising (current EMA > EMA from 5 bars ago)
- Trend DOWN = daily close < EMA AND EMA is falling (current EMA < EMA from 5 bars ago)

Long exits:
- Partial exit: sell 1/2 when RSI > rsi_partial_exit (default 60)
- Full exit: RSI > rsi_full_exit (70) OR trend reversal OR trailing stop OR max loss
- Trailing stop: avg_entry_price - atr_stop_multiplier * daily_ATR
- Max loss stop: avg_entry_price * (1 - max_loss_pct)
- Time stop: exit after max_hold_bars if gain < 0.5%

Short exits:
- Partial cover: buy 1/2 when RSI < (100 - rsi_partial_exit) i.e. RSI < 40
- Full cover: RSI < (100 - rsi_full_exit) i.e. RSI < 30 OR trend reversal OR trailing stop OR max loss
- Trailing stop: avg_entry_price + atr_stop_multiplier * daily_ATR (price above entry is bad)
- Max loss stop: avg_entry_price * (1 + max_loss_pct)
- Time stop: exit after max_hold_bars if gain < 0.5%

Target position = risk_pct * portfolio.cash / price

Config defaults:
- rsi_period=14, ema_period=20
- rsi_entry_1=35, rsi_entry_2=25, rsi_entry_3=15
- rsi_partial_exit=60, rsi_full_exit=70
- risk_pct=0.3, max_pyramid_levels=2, atr_period=14, atr_stop_multiplier=2.0
- max_loss_pct=0.03, max_hold_bars=40, cooldown_bars=50

Short side thresholds are mirrored: 100 - rsi_entry_X and 100 - rsi_exit_X.
"""

from collections import deque
from dataclasses import dataclass, field

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_rsi, compute_ema, compute_atr


@dataclass
class SymbolState:
    """Per-symbol tracking state for the pyramid strategy."""
    direction: str = "flat"  # "flat", "long", "short"
    pyramid_level: int = 0
    avg_entry_price: float = 0.0
    total_qty: int = 0
    entry_bar: int = 0
    partial_taken: bool = False
    trailing_stop: float = 0.0
    last_exit_bar: int = 0  # cooldown: bar when last position was closed
    prices_15m: deque = field(default_factory=lambda: deque(maxlen=110))
    prices_daily: deque = field(default_factory=lambda: deque(maxlen=60))
    daily_highs: deque = field(default_factory=lambda: deque(maxlen=60))
    daily_lows: deque = field(default_factory=lambda: deque(maxlen=60))
    daily_closes: deque = field(default_factory=lambda: deque(maxlen=60))
    ema_history: deque = field(default_factory=lambda: deque(maxlen=10))
    prev_rsi: float | None = None
    bar_count: int = 0
    cached_trend_up: bool = False
    cached_trend_down: bool = False


@register("rsi_daily_trend")
class RsiDailyTrend(Strategy):
    """Mean Reversion with Trend Filter + Pyramiding.

    Uses 15-minute RSI for timing pyramid entries/exits and daily EMA for
    trend direction. Supports long and short positions with scaled entries,
    partial exits, trailing stops, max-loss stops, and time-based exits.
    """

    def required_data(self) -> list[dict]:
        return [
            {"interval": "15minute", "lookback": 100},
            {"interval": "day", "lookback": 50},
        ]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        # RSI parameters (long side; short side mirrors via 100 - X)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_entry_1 = config.get("rsi_entry_1", 35)
        self.rsi_entry_2 = config.get("rsi_entry_2", 25)
        self.rsi_entry_3 = config.get("rsi_entry_3", 15)
        self.rsi_partial_exit = config.get("rsi_partial_exit", 60)
        self.rsi_full_exit = config.get("rsi_full_exit", 70)

        # EMA / trend parameters
        self.ema_period = config.get("ema_period", 20)

        # Position sizing
        self.risk_pct = config.get("risk_pct", 0.3)
        self.max_pyramid_levels = config.get("max_pyramid_levels", 2)

        # Stop parameters
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_multiplier = config.get("atr_stop_multiplier", 2.0)
        self.max_loss_pct = config.get("max_loss_pct", 0.03)
        self.max_hold_bars = config.get("max_hold_bars", 40)

        # Cooldown: don't re-enter same symbol for N bars after exit
        self.cooldown_bars = config.get("cooldown_bars", 50)

        self.instruments = instruments

        # Per-symbol state
        self.states: dict[str, SymbolState] = {}

    def _get_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        if symbol not in self.states:
            self.states[symbol] = SymbolState()
        return self.states[symbol]

    def _held_qty(self, symbol: str, snapshot: MarketSnapshot) -> int:
        """Get long quantity currently held in portfolio for a symbol."""
        for pos in snapshot.portfolio.positions:
            if pos.symbol == symbol and pos.quantity > 0:
                return pos.quantity
        return 0

    def _held_short_qty(self, symbol: str, snapshot: MarketSnapshot) -> int:
        """Get short quantity currently held in portfolio for a symbol.

        Returns the absolute value of negative position quantity.
        """
        for pos in snapshot.portfolio.positions:
            if pos.symbol == symbol and pos.quantity < 0:
                return abs(pos.quantity)
        return 0

    def _update_trend(self, state: SymbolState) -> None:
        """Recompute trend direction after a new daily bar.

        Updates cached_trend_up and cached_trend_down.
        """
        if len(state.prices_daily) == 0:
            state.cached_trend_up = False
            state.cached_trend_down = False
            return

        ema = compute_ema(list(state.prices_daily), self.ema_period)
        if ema is None:
            state.cached_trend_up = False
            state.cached_trend_down = False
            return

        # Store EMA in history for slope detection (once per daily bar)
        state.ema_history.append(ema)

        last_close = state.prices_daily[-1]

        # --- Uptrend: close > EMA AND EMA rising ---
        if last_close > ema:
            if len(state.ema_history) >= 6:
                state.cached_trend_up = state.ema_history[-1] > state.ema_history[-6]
            else:
                # Not enough EMA history yet -- accept close > EMA alone
                state.cached_trend_up = True
        else:
            state.cached_trend_up = False

        # --- Downtrend: close < EMA AND EMA falling ---
        if last_close < ema:
            if len(state.ema_history) >= 6:
                state.cached_trend_down = state.ema_history[-1] < state.ema_history[-6]
            else:
                # Not enough EMA history yet -- accept close < EMA alone
                state.cached_trend_down = True
        else:
            state.cached_trend_down = False

    def _compute_trailing_stop(self, state: SymbolState) -> float:
        """Compute trailing stop based on ATR (for long positions)."""
        atr = compute_atr(
            list(state.daily_highs),
            list(state.daily_lows),
            list(state.daily_closes),
            self.atr_period,
        )
        if atr is None or state.avg_entry_price <= 0:
            return 0.0

        return state.avg_entry_price - self.atr_stop_multiplier * atr

    def _reset_state(self, state: SymbolState) -> None:
        """Reset position-related state after full exit."""
        state.last_exit_bar = state.bar_count  # cooldown starts now
        state.direction = "flat"
        state.pyramid_level = 0
        state.avg_entry_price = 0.0
        state.total_qty = 0
        state.entry_bar = 0
        state.partial_taken = False
        state.trailing_stop = 0.0

    def _process_long_entries(
        self, state: SymbolState, symbol: str, bar_close: float,
        rsi: float, trend_up: bool, atr: float | None,
        level_qty: int, signals: list[Signal],
    ) -> None:
        """Handle long pyramid entries (levels 1-3)."""
        # Cooldown check: don't re-enter too soon after exiting
        if state.direction == "flat" and state.last_exit_bar > 0:
            if state.bar_count - state.last_exit_bar < self.cooldown_bars:
                return

        if state.direction == "flat" and rsi < self.rsi_entry_1 and trend_up and level_qty > 0:
            signals.append(Signal(
                action="BUY", symbol=symbol, quantity=level_qty,
                product_type="MIS",
            ))
            state.direction = "long"
            state.avg_entry_price = bar_close
            state.total_qty = level_qty
            state.pyramid_level = 1
            state.entry_bar = state.bar_count
            state.partial_taken = False
            if atr is not None:
                state.trailing_stop = bar_close - self.atr_stop_multiplier * atr

        elif state.direction == "long" and state.pyramid_level == 1 and self.max_pyramid_levels >= 2 and rsi < self.rsi_entry_2 and level_qty > 0:
            signals.append(Signal(
                action="BUY", symbol=symbol, quantity=level_qty,
                product_type="MIS",
            ))
            old_cost = state.avg_entry_price * state.total_qty
            new_cost = bar_close * level_qty
            state.total_qty += level_qty
            state.avg_entry_price = (old_cost + new_cost) / state.total_qty
            state.pyramid_level = 2

        elif state.direction == "long" and state.pyramid_level == 2 and self.max_pyramid_levels >= 3 and rsi < self.rsi_entry_3 and level_qty > 0:
            signals.append(Signal(
                action="BUY", symbol=symbol, quantity=level_qty,
                product_type="MIS",
            ))
            old_cost = state.avg_entry_price * state.total_qty
            new_cost = bar_close * level_qty
            state.total_qty += level_qty
            state.avg_entry_price = (old_cost + new_cost) / state.total_qty
            state.pyramid_level = 3

    def _process_long_exits(
        self, state: SymbolState, symbol: str, bar_close: float,
        rsi: float, trend_up: bool, atr: float | None,
        snapshot: MarketSnapshot, signals: list[Signal],
    ) -> None:
        """Handle long position exits (partial + full)."""
        current_qty = self._held_qty(symbol, snapshot)
        if current_qty == 0:
            current_qty = state.total_qty

        # Update trailing stop (ratchet upwards)
        if atr is not None and state.avg_entry_price > 0:
            new_stop = bar_close - self.atr_stop_multiplier * atr
            if new_stop > state.trailing_stop:
                state.trailing_stop = new_stop

        # Check partial exit: RSI > partial threshold, not yet taken
        if (
            not state.partial_taken
            and rsi > self.rsi_partial_exit
            and current_qty > 1
        ):
            sell_qty = current_qty // 2
            if sell_qty > 0:
                signals.append(Signal(
                    action="SELL", symbol=symbol, quantity=sell_qty,
                    product_type="MIS",
                ))
                state.partial_taken = True
                state.total_qty -= sell_qty

        # Check full exit conditions
        else:
            should_full_exit = False

            # RSI overbought full exit
            if rsi > self.rsi_full_exit:
                should_full_exit = True

            # Trend reversal exit (only if we have daily data)
            if len(state.prices_daily) > 0 and not trend_up:
                ema = compute_ema(list(state.prices_daily), self.ema_period)
                if ema is not None:
                    should_full_exit = True

            # Trailing stop exit
            if state.trailing_stop > 0 and bar_close <= state.trailing_stop:
                should_full_exit = True

            # Max loss stop
            if state.avg_entry_price > 0:
                max_loss_price = state.avg_entry_price * (1 - self.max_loss_pct)
                if bar_close <= max_loss_price:
                    should_full_exit = True

            # Time stop: held too long with insufficient gain
            bars_held = state.bar_count - state.entry_bar
            if bars_held > self.max_hold_bars and state.avg_entry_price > 0:
                gain_pct = (bar_close - state.avg_entry_price) / state.avg_entry_price
                if gain_pct < 0.005:
                    should_full_exit = True

            if should_full_exit and current_qty > 0:
                signals.append(Signal(
                    action="SELL", symbol=symbol, quantity=current_qty,
                    product_type="MIS",
                ))
                self._reset_state(state)

    def _process_short_entries(
        self, state: SymbolState, symbol: str, bar_close: float,
        rsi: float, trend_down: bool, atr: float | None,
        level_qty: int, signals: list[Signal],
    ) -> None:
        """Handle short pyramid entries (levels 1-3).

        Short side RSI thresholds are mirrored: 100 - rsi_entry_X.
        """
        short_entry_1 = 100 - self.rsi_entry_1  # default 60
        short_entry_2 = 100 - self.rsi_entry_2  # default 70
        short_entry_3 = 100 - self.rsi_entry_3  # default 85

        # Cooldown check
        if state.direction == "flat" and state.last_exit_bar > 0:
            if state.bar_count - state.last_exit_bar < self.cooldown_bars:
                return

        if state.direction == "flat" and rsi > short_entry_1 and trend_down and level_qty > 0:
            signals.append(Signal(
                action="SELL", symbol=symbol, quantity=level_qty,
                product_type="MIS",
            ))
            state.direction = "short"
            state.avg_entry_price = bar_close
            state.total_qty = level_qty
            state.pyramid_level = 1
            state.entry_bar = state.bar_count
            state.partial_taken = False
            # Trailing stop for shorts is ABOVE entry (price going up = bad)
            if atr is not None:
                state.trailing_stop = bar_close + self.atr_stop_multiplier * atr

        elif state.direction == "short" and state.pyramid_level == 1 and self.max_pyramid_levels >= 2 and rsi > short_entry_2 and level_qty > 0:
            signals.append(Signal(
                action="SELL", symbol=symbol, quantity=level_qty,
                product_type="MIS",
            ))
            old_cost = state.avg_entry_price * state.total_qty
            new_cost = bar_close * level_qty
            state.total_qty += level_qty
            state.avg_entry_price = (old_cost + new_cost) / state.total_qty
            state.pyramid_level = 2

        elif state.direction == "short" and state.pyramid_level == 2 and self.max_pyramid_levels >= 3 and rsi > short_entry_3 and level_qty > 0:
            signals.append(Signal(
                action="SELL", symbol=symbol, quantity=level_qty,
                product_type="MIS",
            ))
            old_cost = state.avg_entry_price * state.total_qty
            new_cost = bar_close * level_qty
            state.total_qty += level_qty
            state.avg_entry_price = (old_cost + new_cost) / state.total_qty
            state.pyramid_level = 3

    def _process_short_exits(
        self, state: SymbolState, symbol: str, bar_close: float,
        rsi: float, trend_down: bool, atr: float | None,
        snapshot: MarketSnapshot, signals: list[Signal],
    ) -> None:
        """Handle short position exits (partial cover + full cover).

        Short exit RSI thresholds are mirrored: 100 - rsi_exit_X.
        """
        short_partial_exit = 100 - self.rsi_partial_exit  # default 40
        short_full_exit = 100 - self.rsi_full_exit  # default 30

        current_qty = self._held_short_qty(symbol, snapshot)
        if current_qty == 0:
            current_qty = state.total_qty

        # Update trailing stop for shorts (ratchet downwards -- lower is better for shorts)
        if atr is not None and state.avg_entry_price > 0:
            new_stop = bar_close + self.atr_stop_multiplier * atr
            if new_stop < state.trailing_stop:
                state.trailing_stop = new_stop

        # Partial cover: RSI drops below partial threshold
        if (
            not state.partial_taken
            and rsi < short_partial_exit
            and current_qty > 1
        ):
            cover_qty = current_qty // 2
            if cover_qty > 0:
                signals.append(Signal(
                    action="BUY", symbol=symbol, quantity=cover_qty,
                    product_type="MIS",
                ))
                state.partial_taken = True
                state.total_qty -= cover_qty

        # Check full cover conditions
        else:
            should_full_exit = False

            # RSI oversold full cover
            if rsi < short_full_exit:
                should_full_exit = True

            # Trend reversal exit: trend is no longer down
            if len(state.prices_daily) > 0 and not trend_down:
                ema = compute_ema(list(state.prices_daily), self.ema_period)
                if ema is not None:
                    should_full_exit = True

            # Trailing stop exit: price rises above trailing stop
            if state.trailing_stop > 0 and bar_close >= state.trailing_stop:
                should_full_exit = True

            # Max loss stop: price rises too much above entry
            if state.avg_entry_price > 0:
                max_loss_price = state.avg_entry_price * (1 + self.max_loss_pct)
                if bar_close >= max_loss_price:
                    should_full_exit = True

            # Time stop: held too long with insufficient gain
            bars_held = state.bar_count - state.entry_bar
            if bars_held > self.max_hold_bars and state.avg_entry_price > 0:
                # For shorts, gain = (entry - current) / entry
                gain_pct = (state.avg_entry_price - bar_close) / state.avg_entry_price
                if gain_pct < 0.005:
                    should_full_exit = True

            if should_full_exit and current_qty > 0:
                signals.append(Signal(
                    action="BUY", symbol=symbol, quantity=current_qty,
                    product_type="MIS",
                ))
                self._reset_state(state)

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals = []

        # 1. Update daily prices when a new daily bar arrives
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                state = self._get_state(symbol)
                state.prices_daily.append(bar.close)
                state.daily_highs.append(bar.high)
                state.daily_lows.append(bar.low)
                state.daily_closes.append(bar.close)
                self._update_trend(state)

        # 2. Process 15-minute bars
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            state = self._get_state(symbol)
            state.bar_count += 1

            # Reconcile position with portfolio
            if state.direction == "long":
                held = self._held_qty(symbol, snapshot)
                if held == 0:
                    self._reset_state(state)
                else:
                    state.total_qty = held
            elif state.direction == "short":
                held = self._held_short_qty(symbol, snapshot)
                if held == 0:
                    self._reset_state(state)
                else:
                    state.total_qty = held

            state.prices_15m.append(bar.close)

            # Compute RSI on 15-minute data
            rsi = compute_rsi(list(state.prices_15m), self.rsi_period)
            if rsi is None:
                state.prev_rsi = rsi
                continue

            # Determine daily trend (cached from last daily bar update)
            trend_up = state.cached_trend_up
            trend_down = state.cached_trend_down

            # Compute ATR for trailing stop
            atr = compute_atr(
                list(state.daily_highs),
                list(state.daily_lows),
                list(state.daily_closes),
                self.atr_period,
            )

            # --- Target position sizing ---
            target_qty = int(self.risk_pct * snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
            inst = self.instruments.get(symbol)
            if inst and inst.lot_size > 1:
                target_qty = (target_qty // inst.lot_size) * inst.lot_size
            level_qty = max(target_qty // 3, 1) if target_qty > 0 else 0

            # --- Entries ---
            if state.direction == "flat":
                # Try long entry first, then short
                self._process_long_entries(
                    state, symbol, bar.close, rsi, trend_up, atr, level_qty, signals,
                )
                if state.direction == "flat":
                    self._process_short_entries(
                        state, symbol, bar.close, rsi, trend_down, atr, level_qty, signals,
                    )
            elif state.direction == "long":
                # Continue pyramiding long
                self._process_long_entries(
                    state, symbol, bar.close, rsi, trend_up, atr, level_qty, signals,
                )
            elif state.direction == "short":
                # Continue pyramiding short
                self._process_short_entries(
                    state, symbol, bar.close, rsi, trend_down, atr, level_qty, signals,
                )

            # --- Exits ---
            if state.direction == "long":
                self._process_long_exits(
                    state, symbol, bar.close, rsi, trend_up, atr, snapshot, signals,
                )
            elif state.direction == "short":
                self._process_short_exits(
                    state, symbol, bar.close, rsi, trend_down, atr, snapshot, signals,
                )

            state.prev_rsi = rsi

        return signals

    def on_complete(self) -> dict:
        return {
            "strategy_type": "rsi_daily_trend",
            "rsi_period": self.rsi_period,
            "ema_period": self.ema_period,
            "risk_pct": self.risk_pct,
            "atr_period": self.atr_period,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "max_loss_pct": self.max_loss_pct,
            "max_hold_bars": self.max_hold_bars,
        }
