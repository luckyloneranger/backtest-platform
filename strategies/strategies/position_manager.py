"""Shared position management for deterministic strategies.

Handles order lifecycle: entries (LIMIT/MARKET), engine SL-M stops,
trailing stop ratcheting, profit targets, fill detection, DAY expiry
re-submission, and portfolio reconciliation.

Strategies call entry/exit/update methods which return Signal lists.
Each bar, strategies must call process_fills() and resubmit_expired()
before their own logic.
"""

from dataclasses import dataclass, field
from strategies.base import Signal, MarketSnapshot


@dataclass
class PositionState:
    """Per-symbol position tracking state."""
    direction: str = "flat"       # "flat", "long", "short"
    qty: int = 0
    avg_entry: float = 0.0
    entry_bar: int = 0
    trailing_stop: float = 0.0
    product_type: str = "CNC"
    has_engine_stop: bool = False
    has_profit_target: bool = False
    profit_target_price: float = 0.0
    profit_target_qty: int = 0
    partial_taken: bool = False
    pending_entry: bool = False
    pending_side: str = ""        # "BUY" or "SELL"
    pending_qty: int = 0
    pending_bar: int = 0
    pyramid_count: int = 0
    original_qty: int = 0
    bars_held: int = 0


class PositionManager:
    """Manages positions, engine orders, and fills for a strategy."""

    def __init__(self, max_pending_bars: int = 3):
        self.states: dict[str, PositionState] = {}
        self.bar_count: int = 0
        self.max_pending_bars = max_pending_bars

    def get_state(self, symbol: str) -> PositionState:
        if symbol not in self.states:
            self.states[symbol] = PositionState()
        return self.states[symbol]

    # === Query methods ===

    def is_flat(self, symbol: str) -> bool:
        return self.get_state(symbol).direction == "flat"

    def is_long(self, symbol: str) -> bool:
        return self.get_state(symbol).direction == "long"

    def is_short(self, symbol: str) -> bool:
        return self.get_state(symbol).direction == "short"

    def position_qty(self, symbol: str) -> int:
        return self.get_state(symbol).qty

    def avg_entry_price(self, symbol: str) -> float:
        return self.get_state(symbol).avg_entry

    def has_pending_entry(self, symbol: str) -> bool:
        return self.get_state(symbol).pending_entry

    # === Entry methods ===

    def enter_long(self, symbol: str, qty: int, limit_price: float,
                   product_type: str, stop_price: float) -> list[Signal]:
        """Submit entry for long position. Uses LIMIT if limit_price > 0, else MARKET."""
        state = self.get_state(symbol)
        if state.direction != "flat" or state.pending_entry:
            return []

        order_type = "LIMIT" if limit_price > 0 else "MARKET"
        state.pending_entry = True
        state.pending_side = "BUY"
        state.pending_qty = qty
        state.pending_bar = self.bar_count
        state.product_type = product_type
        state.trailing_stop = stop_price
        state.original_qty = qty

        return [Signal(
            action="BUY", symbol=symbol, quantity=qty,
            order_type=order_type, limit_price=limit_price,
            product_type=product_type,
        )]

    def enter_short(self, symbol: str, qty: int, limit_price: float,
                    stop_price: float) -> list[Signal]:
        """Submit entry for short position. Always MIS (CNC shorts not allowed)."""
        state = self.get_state(symbol)
        if state.direction != "flat" or state.pending_entry:
            return []

        order_type = "LIMIT" if limit_price > 0 else "MARKET"
        state.pending_entry = True
        state.pending_side = "SELL"
        state.pending_qty = qty
        state.pending_bar = self.bar_count
        state.product_type = "MIS"  # shorts always MIS
        state.trailing_stop = stop_price
        state.original_qty = qty

        return [Signal(
            action="SELL", symbol=symbol, quantity=qty,
            order_type=order_type, limit_price=limit_price,
            product_type="MIS",
        )]

    def add_pyramid(self, symbol: str, qty: int, limit_price: float) -> list[Signal]:
        """Add to existing position via LIMIT or MARKET."""
        state = self.get_state(symbol)
        if state.direction == "flat":
            return []

        order_type = "LIMIT" if limit_price > 0 else "MARKET"
        action = "BUY" if state.direction == "long" else "SELL"

        state.pending_entry = True
        state.pending_side = action
        state.pending_qty = qty
        state.pending_bar = self.bar_count

        return [Signal(
            action=action, symbol=symbol, quantity=qty,
            order_type=order_type, limit_price=limit_price,
            product_type=state.product_type,
        )]

    def set_profit_target(self, symbol: str, qty: int, limit_price: float) -> list[Signal]:
        """Set a resting LIMIT order for partial profit taking."""
        state = self.get_state(symbol)
        if state.direction == "flat":
            return []

        action = "SELL" if state.direction == "long" else "BUY"
        state.has_profit_target = True
        state.profit_target_price = limit_price
        state.profit_target_qty = qty

        return [Signal(
            action=action, symbol=symbol, quantity=qty,
            order_type="LIMIT", limit_price=limit_price,
            product_type=state.product_type,
        )]

    # === Exit methods ===

    def exit_position(self, symbol: str, qty: int | None = None) -> list[Signal]:
        """Exit full or partial position. Emits CANCEL + MARKET order."""
        state = self.get_state(symbol)
        if state.direction == "flat":
            return []

        signals: list[Signal] = []
        # Cancel any pending engine orders
        signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))

        exit_qty = qty if qty is not None else state.qty
        action = "SELL" if state.direction == "long" else "BUY"

        signals.append(Signal(
            action=action, symbol=symbol, quantity=exit_qty,
            product_type=state.product_type,
        ))

        if qty is None or qty >= state.qty:
            # Full exit
            self._reset(state)
        else:
            # Partial exit
            state.qty -= exit_qty
            state.partial_taken = True

        return signals

    def update_trailing_stop(self, symbol: str, new_stop: float) -> list[Signal]:
        """Ratchet trailing stop. If moved, emits CANCEL + new SL-M."""
        state = self.get_state(symbol)
        if state.direction == "flat" or not state.has_engine_stop:
            return []

        # For longs: stop only moves UP. For shorts: stop only moves DOWN.
        if state.direction == "long" and new_stop <= state.trailing_stop:
            return []
        if state.direction == "short" and new_stop >= state.trailing_stop:
            return []

        state.trailing_stop = new_stop
        action = "SELL" if state.direction == "long" else "BUY"

        return [
            Signal(action="CANCEL", symbol=symbol, quantity=0),
            Signal(
                action=action, symbol=symbol, quantity=state.qty,
                order_type="SL_M", stop_price=new_stop,
                product_type=state.product_type,
            ),
        ]

    # === Lifecycle methods ===

    def process_fills(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Process fills from snapshot. Detect entries, stops, profit targets."""
        signals: list[Signal] = []

        for fill in snapshot.fills:
            symbol = fill.symbol
            state = self.get_state(symbol)

            # --- Entry fill detection ---
            if state.pending_entry and fill.side == state.pending_side:
                state.pending_entry = False

                if state.direction == "flat":
                    # Initial entry fill
                    state.direction = "long" if fill.side == "BUY" else "short"
                    state.qty = fill.quantity
                    state.avg_entry = fill.fill_price
                    state.entry_bar = self.bar_count
                    state.pyramid_count = 0
                    state.partial_taken = False
                    state.bars_held = 0

                    # Submit engine SL-M stop
                    stop_action = "SELL" if state.direction == "long" else "BUY"
                    signals.append(Signal(
                        action=stop_action, symbol=symbol, quantity=state.qty,
                        order_type="SL_M", stop_price=state.trailing_stop,
                        product_type=state.product_type,
                    ))
                    state.has_engine_stop = True

                else:
                    # Pyramid fill — update avg entry
                    old_cost = state.avg_entry * state.qty
                    new_cost = fill.fill_price * fill.quantity
                    state.qty += fill.quantity
                    state.avg_entry = (old_cost + new_cost) / state.qty
                    state.pyramid_count += 1

                    # Cancel old SL-M + submit new for total qty
                    signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                    stop_action = "SELL" if state.direction == "long" else "BUY"
                    signals.append(Signal(
                        action=stop_action, symbol=symbol, quantity=state.qty,
                        order_type="SL_M", stop_price=state.trailing_stop,
                        product_type=state.product_type,
                    ))

                continue

            # --- Stop hit detection ---
            if state.has_engine_stop and state.direction != "flat":
                expected_side = "SELL" if state.direction == "long" else "BUY"
                if fill.side == expected_side and not state.has_profit_target:
                    # Engine SL-M was triggered
                    self._reset(state)
                    continue

            # --- Profit target fill detection ---
            if state.has_profit_target and state.direction != "flat":
                expected_side = "SELL" if state.direction == "long" else "BUY"
                if fill.side == expected_side:
                    state.has_profit_target = False
                    state.partial_taken = True
                    state.qty -= fill.quantity

                    # Move stop to breakeven
                    state.trailing_stop = state.avg_entry
                    signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                    stop_action = "SELL" if state.direction == "long" else "BUY"
                    signals.append(Signal(
                        action=stop_action, symbol=symbol, quantity=state.qty,
                        order_type="SL_M", stop_price=state.avg_entry,
                        product_type=state.product_type,
                    ))
                    continue

        # --- Stale pending entry cancellation ---
        for symbol, state in self.states.items():
            if state.pending_entry and self.bar_count - state.pending_bar > self.max_pending_bars:
                # Check if entry is still in pending_orders
                has_pending = any(
                    po.symbol == symbol and po.order_type == "LIMIT"
                    for po in snapshot.pending_orders
                )
                if has_pending:
                    signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                state.pending_entry = False
                state.pending_side = ""
                state.pending_qty = 0

        return signals

    def resubmit_expired(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Re-submit engine orders that expired due to DAY order validity."""
        signals: list[Signal] = []

        for symbol, state in self.states.items():
            if state.direction == "flat":
                continue

            # Re-submit expired SL-M stop
            if state.has_engine_stop and state.trailing_stop > 0:
                has_stop = any(
                    po.symbol == symbol and po.order_type == "SL_M"
                    for po in snapshot.pending_orders
                )
                if not has_stop:
                    action = "SELL" if state.direction == "long" else "BUY"
                    signals.append(Signal(
                        action=action, symbol=symbol, quantity=state.qty,
                        order_type="SL_M", stop_price=state.trailing_stop,
                        product_type=state.product_type,
                    ))

            # Re-submit expired profit target
            if state.has_profit_target and not state.partial_taken:
                has_target = any(
                    po.symbol == symbol and po.order_type == "LIMIT"
                    for po in snapshot.pending_orders
                )
                if not has_target and state.profit_target_price > 0:
                    action = "SELL" if state.direction == "long" else "BUY"
                    signals.append(Signal(
                        action=action, symbol=symbol, quantity=state.profit_target_qty,
                        order_type="LIMIT", limit_price=state.profit_target_price,
                        product_type=state.product_type,
                    ))

            # Detect expired pending entry
            if state.pending_entry:
                has_pending = any(
                    po.symbol == symbol and po.order_type == "LIMIT"
                    for po in snapshot.pending_orders
                )
                filled = any(
                    f.symbol == symbol and f.side == state.pending_side
                    for f in snapshot.fills
                )
                if not has_pending and not filled:
                    state.pending_entry = False
                    state.pending_side = ""
                    state.pending_qty = 0

        return signals

    def reconcile(self, snapshot: MarketSnapshot):
        """Sync internal state with portfolio positions."""
        for symbol, state in self.states.items():
            if state.direction == "flat":
                continue

            # Find portfolio position
            held = 0
            for pos in snapshot.portfolio.positions:
                if pos.symbol == symbol:
                    held = pos.quantity
                    break

            if state.direction == "long" and held <= 0:
                self._reset(state)
            elif state.direction == "short" and held >= 0:
                self._reset(state)
            elif held != 0:
                state.qty = abs(held)

    def increment_bars(self):
        """Call once per bar to track bar count and position hold duration."""
        self.bar_count += 1
        for state in self.states.values():
            if state.direction != "flat":
                state.bars_held += 1

    def _reset(self, state: PositionState):
        """Reset all position-related state."""
        state.direction = "flat"
        state.qty = 0
        state.avg_entry = 0.0
        state.entry_bar = 0
        state.trailing_stop = 0.0
        state.product_type = "CNC"
        state.has_engine_stop = False
        state.has_profit_target = False
        state.profit_target_price = 0.0
        state.profit_target_qty = 0
        state.partial_taken = False
        state.pending_entry = False
        state.pending_side = ""
        state.pending_qty = 0
        state.pending_bar = 0
        state.pyramid_count = 0
        state.original_qty = 0
        state.bars_held = 0
