from collections import deque
from dataclasses import dataclass, field

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal, Position


@dataclass
class SymbolState:
    """Per-symbol tracking state for the SMA crossover strategy."""

    prices: deque  # close prices for SMA computation
    daily_highs: deque  # high prices for ATR
    daily_lows: deque  # low prices for ATR
    daily_closes: deque  # close prices for ATR
    prev_fast_above: bool | None = None
    entry_price: float = 0.0
    avg_entry: float = 0.0
    entry_bar: int = 0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    trailing_stop: float = 0.0
    position_qty: int = 0
    pyramid_count: int = 0
    original_qty: int = 0
    bars_in_position: int = 0
    pending_entry: bool = False       # limit entry order outstanding
    pending_side: str = ""            # "BUY" or "SELL" — direction of pending entry
    pending_qty: int = 0              # quantity of pending entry order
    has_engine_stop: bool = False     # SL-M stop in engine
    product_type: str = "CNC"        # "CNC" or "MIS" — set at entry time


@register("sma_crossover")
class SmaCrossover(Strategy):
    """Trend-following SMA crossover strategy with ATR-based position sizing,
    trailing stops, time stops, and pyramiding. Supports both long and short positions.

    Long entry: golden cross (fast SMA > slow SMA) with minimum trend spread.
    Long exit: death cross, trailing stop (engine SL-M), or time stop.
    Short entry: death cross (fast SMA < slow SMA) with minimum trend spread.
    Short exit: golden cross, trailing stop (engine SL-M), or time stop.
    Pyramiding: add to winning positions when price moves ATR beyond avg_entry.
    Position tracking: positive position_qty = long, negative = short.

    Entry orders use LIMIT at bar.close. Stop-losses use engine SL-M orders.
    Product type (CNC/MIS) is dynamically chosen based on trend spread.
    """

    def required_data(self) -> list[dict]:
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.fast_period: int = config.get("fast_period", 10)
        self.slow_period: int = config.get("slow_period", 30)
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be less than slow_period ({self.slow_period})"
            )

        self.risk_per_trade: float = config.get("risk_per_trade", 0.02)
        self.atr_period: int = config.get("atr_period", 14)
        self.atr_stop_multiplier: float = config.get("atr_stop_multiplier", 2.0)
        self.min_spread: float = config.get("min_spread", 0.005)
        self.max_hold_bars: int = config.get("max_hold_bars", 50)
        self.pyramid_levels: int = config.get("pyramid_levels", 2)
        self.cnc_spread_threshold: float = config.get("cnc_spread_threshold", 0.01)

        self.states: dict[str, SymbolState] = {}
        self.bar_count: int = 0

    def _get_state(self, symbol: str) -> SymbolState:
        """Get or create per-symbol state."""
        if symbol not in self.states:
            self.states[symbol] = SymbolState(
                prices=deque(maxlen=self.slow_period),
                daily_highs=deque(maxlen=self.atr_period + 10),
                daily_lows=deque(maxlen=self.atr_period + 10),
                daily_closes=deque(maxlen=self.atr_period + 10),
            )
        return self.states[symbol]

    def _reconcile_position(self, state: SymbolState, portfolio_positions: list[Position], symbol: str) -> None:
        """Sync local state with the portfolio's actual position."""
        actual_qty = 0
        actual_avg = 0.0
        for pos in portfolio_positions:
            if pos.symbol == symbol:
                actual_qty = pos.quantity
                actual_avg = pos.avg_price
                break

        if actual_qty == 0 and state.position_qty != 0:
            # Position was closed externally (e.g. by engine stop, auto-squareoff)
            state.position_qty = 0
            state.pyramid_count = 0
            state.original_qty = 0
            state.entry_price = 0.0
            state.avg_entry = 0.0
            state.highest_since_entry = 0.0
            state.lowest_since_entry = 0.0
            state.trailing_stop = 0.0
            state.bars_in_position = 0
            state.has_engine_stop = False

        if actual_qty != 0:
            state.position_qty = actual_qty
            state.avg_entry = actual_avg

    def _compute_qty(self, capital: float, atr: float) -> int:
        """ATR-based position sizing: qty = (capital * risk_per_trade) / (ATR * multiplier)."""
        risk_amount = capital * self.risk_per_trade
        stop_distance = atr * self.atr_stop_multiplier
        if stop_distance <= 0:
            return 0
        return int(risk_amount / stop_distance)

    def _reset_state(self, state: SymbolState) -> None:
        """Reset all position-related state fields."""
        state.position_qty = 0
        state.pyramid_count = 0
        state.original_qty = 0
        state.entry_price = 0.0
        state.avg_entry = 0.0
        state.highest_since_entry = 0.0
        state.lowest_since_entry = 0.0
        state.trailing_stop = 0.0
        state.bars_in_position = 0
        state.has_engine_stop = False
        state.pending_entry = False
        state.pending_side = ""
        state.pending_qty = 0

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals: list[Signal] = []
        self.bar_count += 1

        for interval, bars in snapshot.timeframes.items():
            for symbol, bar in bars.items():
                state = self._get_state(symbol)

                # 1. Update price deques
                state.prices.append(bar.close)
                state.daily_highs.append(bar.high)
                state.daily_lows.append(bar.low)
                state.daily_closes.append(bar.close)

                # 2. Compute indicators
                from strategies.indicators import compute_sma, compute_atr

                prices_list = list(state.prices)
                fast_sma = compute_sma(prices_list, self.fast_period)
                slow_sma = compute_sma(prices_list, self.slow_period)

                highs_list = list(state.daily_highs)
                lows_list = list(state.daily_lows)
                closes_list = list(state.daily_closes)
                atr = compute_atr(highs_list, lows_list, closes_list, self.atr_period)

                # 3. Check warmup
                if fast_sma is None or slow_sma is None or atr is None:
                    state.prev_fast_above = None
                    continue

                fast_above = fast_sma > slow_sma

                # 4. Reconcile position with portfolio
                self._reconcile_position(state, snapshot.portfolio.positions, symbol)

                # 5. Handle pending entry orders
                if state.pending_entry:
                    self._handle_pending_entry(
                        state, symbol, bar, snapshot, signals,
                        fast_above, fast_sma, slow_sma, atr,
                    )
                    state.prev_fast_above = fast_above
                    continue

                # 6. If FLAT: check for long entry (golden cross) or short entry (death cross)
                if state.position_qty == 0:
                    if (
                        state.prev_fast_above is not None
                        and fast_above
                        and not state.prev_fast_above
                    ):
                        # Golden cross detected - check trend strength
                        spread = (fast_sma - slow_sma) / slow_sma
                        if spread > self.min_spread and atr > 0:
                            qty = self._compute_qty(snapshot.context.initial_capital, atr)
                            if qty > 0:
                                product = "CNC" if spread > self.cnc_spread_threshold else "MIS"
                                signals.append(
                                    Signal(
                                        action="BUY",
                                        symbol=symbol,
                                        quantity=qty,
                                        order_type="LIMIT",
                                        limit_price=bar.close,
                                        product_type=product,
                                    )
                                )
                                state.pending_entry = True
                                state.pending_side = "BUY"
                                state.pending_qty = qty
                                state.product_type = product

                    elif (
                        state.prev_fast_above is not None
                        and not fast_above
                        and state.prev_fast_above
                    ):
                        # Death cross detected - check trend strength for short entry
                        spread = abs(fast_sma - slow_sma) / slow_sma
                        if spread > self.min_spread and atr > 0:
                            qty = self._compute_qty(snapshot.context.initial_capital, atr)
                            if qty > 0:
                                product = "CNC" if spread > self.cnc_spread_threshold else "MIS"
                                signals.append(
                                    Signal(
                                        action="SELL",
                                        symbol=symbol,
                                        quantity=qty,
                                        order_type="LIMIT",
                                        limit_price=bar.close,
                                        product_type=product,
                                    )
                                )
                                state.pending_entry = True
                                state.pending_side = "SELL"
                                state.pending_qty = qty
                                state.product_type = product

                # 7. If LONG: check stop hit, exits, then pyramiding
                elif state.position_qty > 0:
                    # Check for engine stop hit first
                    if state.has_engine_stop:
                        stop_hit = any(
                            f.symbol == symbol and f.side == "SELL"
                            for f in snapshot.fills
                        )
                        if stop_hit:
                            self._reset_state(state)
                            state.prev_fast_above = fast_above
                            continue

                    state.bars_in_position += 1

                    # Update trailing stop (only moves up)
                    if bar.close > state.highest_since_entry:
                        state.highest_since_entry = bar.close
                    new_stop = state.highest_since_entry - atr * self.atr_stop_multiplier
                    if new_stop > state.trailing_stop:
                        state.trailing_stop = new_stop
                        # Ratchet engine stop
                        if state.has_engine_stop:
                            signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                            signals.append(
                                Signal(
                                    action="SELL",
                                    symbol=symbol,
                                    quantity=state.position_qty,
                                    order_type="SL_M",
                                    stop_price=state.trailing_stop,
                                    product_type=state.product_type,
                                )
                            )

                    sell_all = False
                    # a. Death cross exit
                    if state.prev_fast_above is not None and not fast_above and state.prev_fast_above:
                        sell_all = True

                    # b. Time stop exit
                    if not sell_all and state.bars_in_position > self.max_hold_bars:
                        gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0.0
                        if gain < 0.005:
                            sell_all = True

                    if sell_all:
                        # Cancel all pending orders (including engine stop) before exit
                        signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                        signals.append(
                            Signal(
                                action="SELL",
                                symbol=symbol,
                                quantity=state.position_qty,
                                product_type=state.product_type,
                            )
                        )
                        self._reset_state(state)
                    else:
                        # Check pyramid opportunity
                        if (
                            state.pyramid_count < self.pyramid_levels
                            and atr > 0
                            and bar.close > state.avg_entry + atr
                        ):
                            add_qty = max(1, int(state.original_qty * 0.5))
                            signals.append(
                                Signal(
                                    action="BUY",
                                    symbol=symbol,
                                    quantity=add_qty,
                                    product_type=state.product_type,
                                )
                            )
                            # Cancel and resubmit engine stop with updated qty
                            total_qty = state.position_qty + add_qty
                            state.avg_entry = (
                                (state.avg_entry * state.position_qty + bar.close * add_qty) / total_qty
                            )
                            state.position_qty = total_qty
                            state.pyramid_count += 1
                            if state.has_engine_stop:
                                signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                                signals.append(
                                    Signal(
                                        action="SELL",
                                        symbol=symbol,
                                        quantity=state.position_qty,
                                        order_type="SL_M",
                                        stop_price=state.trailing_stop,
                                        product_type=state.product_type,
                                    )
                                )

                # 8. If SHORT: check stop hit, exits, then pyramiding (mirror of long logic)
                elif state.position_qty < 0:
                    # Check for engine stop hit first
                    if state.has_engine_stop:
                        stop_hit = any(
                            f.symbol == symbol and f.side == "BUY"
                            for f in snapshot.fills
                        )
                        if stop_hit:
                            self._reset_state(state)
                            state.prev_fast_above = fast_above
                            continue

                    state.bars_in_position += 1
                    abs_qty = abs(state.position_qty)

                    # Update trailing stop for short (only moves down)
                    if bar.close < state.lowest_since_entry:
                        state.lowest_since_entry = bar.close
                    new_stop = state.lowest_since_entry + atr * self.atr_stop_multiplier
                    if new_stop < state.trailing_stop:
                        state.trailing_stop = new_stop
                        # Ratchet engine stop
                        if state.has_engine_stop:
                            signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                            signals.append(
                                Signal(
                                    action="BUY",
                                    symbol=symbol,
                                    quantity=abs_qty,
                                    order_type="SL_M",
                                    stop_price=state.trailing_stop,
                                    product_type=state.product_type,
                                )
                            )

                    cover_all = False
                    # a. Golden cross exit (opposite crossover)
                    if state.prev_fast_above is not None and fast_above and not state.prev_fast_above:
                        cover_all = True

                    # b. Time stop exit
                    if not cover_all and state.bars_in_position > self.max_hold_bars:
                        gain = (state.avg_entry - bar.close) / state.avg_entry if state.avg_entry > 0 else 0.0
                        if gain < 0.005:
                            cover_all = True

                    if cover_all:
                        # Cancel all pending orders (including engine stop) before exit
                        signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                        signals.append(
                            Signal(
                                action="BUY",
                                symbol=symbol,
                                quantity=abs_qty,
                                product_type=state.product_type,
                            )
                        )
                        self._reset_state(state)
                    else:
                        # Check short pyramid opportunity (price drops below avg_entry - ATR)
                        if (
                            state.pyramid_count < self.pyramid_levels
                            and atr > 0
                            and bar.close < state.avg_entry - atr
                        ):
                            add_qty = max(1, int(state.original_qty * 0.5))
                            signals.append(
                                Signal(
                                    action="SELL",
                                    symbol=symbol,
                                    quantity=add_qty,
                                    product_type=state.product_type,
                                )
                            )
                            # Cancel and resubmit engine stop with updated qty
                            total_abs = abs_qty + add_qty
                            state.avg_entry = (
                                (state.avg_entry * abs_qty + bar.close * add_qty) / total_abs
                            )
                            state.position_qty = -(total_abs)
                            state.pyramid_count += 1
                            if state.has_engine_stop:
                                signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                                signals.append(
                                    Signal(
                                        action="BUY",
                                        symbol=symbol,
                                        quantity=abs(state.position_qty),
                                        order_type="SL_M",
                                        stop_price=state.trailing_stop,
                                        product_type=state.product_type,
                                    )
                                )

                # 9. Update crossover state
                state.prev_fast_above = fast_above

        return signals

    def _handle_pending_entry(
        self,
        state: SymbolState,
        symbol: str,
        bar,
        snapshot: MarketSnapshot,
        signals: list[Signal],
        fast_above: bool,
        fast_sma: float,
        slow_sma: float,
        atr: float,
    ) -> None:
        """Handle a pending limit entry order — check fills, cancel, or update."""
        if state.pending_side == "BUY":
            # Check if long entry filled
            filled = any(
                f.symbol == symbol and f.side == "BUY" for f in snapshot.fills
            )
            if filled:
                fill_price = next(
                    f.fill_price for f in snapshot.fills
                    if f.symbol == symbol and f.side == "BUY"
                )
                state.pending_entry = False
                state.pending_side = ""
                state.entry_price = fill_price
                state.avg_entry = fill_price
                state.entry_bar = self.bar_count
                state.highest_since_entry = max(fill_price, bar.close)
                state.lowest_since_entry = 0.0
                state.trailing_stop = fill_price - atr * self.atr_stop_multiplier
                state.position_qty = state.pending_qty
                state.original_qty = state.pending_qty
                state.pyramid_count = 0
                state.bars_in_position = 0
                state.pending_qty = 0
                # Submit engine SL-M stop
                signals.append(
                    Signal(
                        action="SELL",
                        symbol=symbol,
                        quantity=state.position_qty,
                        order_type="SL_M",
                        stop_price=state.trailing_stop,
                        product_type=state.product_type,
                    )
                )
                state.has_engine_stop = True
            elif not fast_above:
                # Crossover reversed — cancel unfilled limit
                signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                state.pending_entry = False
                state.pending_side = ""
                state.pending_qty = 0
            else:
                # Still valid — cancel old limit and resubmit at current close
                signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                signals.append(
                    Signal(
                        action="BUY",
                        symbol=symbol,
                        quantity=state.pending_qty,
                        order_type="LIMIT",
                        limit_price=bar.close,
                        product_type=state.product_type,
                    )
                )

        elif state.pending_side == "SELL":
            # Check if short entry filled
            filled = any(
                f.symbol == symbol and f.side == "SELL" for f in snapshot.fills
            )
            if filled:
                fill_price = next(
                    f.fill_price for f in snapshot.fills
                    if f.symbol == symbol and f.side == "SELL"
                )
                state.pending_entry = False
                state.pending_side = ""
                state.entry_price = fill_price
                state.avg_entry = fill_price
                state.entry_bar = self.bar_count
                state.highest_since_entry = 0.0
                state.lowest_since_entry = min(fill_price, bar.close)
                state.trailing_stop = fill_price + atr * self.atr_stop_multiplier
                state.position_qty = -state.pending_qty
                state.original_qty = state.pending_qty
                state.pyramid_count = 0
                state.bars_in_position = 0
                state.pending_qty = 0
                # Submit engine SL-M stop (BUY to cover short)
                signals.append(
                    Signal(
                        action="BUY",
                        symbol=symbol,
                        quantity=abs(state.position_qty),
                        order_type="SL_M",
                        stop_price=state.trailing_stop,
                        product_type=state.product_type,
                    )
                )
                state.has_engine_stop = True
            elif fast_above:
                # Crossover reversed — cancel unfilled short limit
                signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                state.pending_entry = False
                state.pending_side = ""
                state.pending_qty = 0
            else:
                # Still valid — cancel old limit and resubmit at current close
                signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                signals.append(
                    Signal(
                        action="SELL",
                        symbol=symbol,
                        quantity=state.pending_qty,
                        order_type="LIMIT",
                        limit_price=bar.close,
                        product_type=state.product_type,
                    )
                )
