from collections import deque
from dataclasses import dataclass, field

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal, Position
from strategies.indicators import compute_sma, compute_atr


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
    trailing_stop: float = 0.0
    position_qty: int = 0
    pyramid_count: int = 0
    original_qty: int = 0
    bars_in_position: int = 0


@register("sma_crossover")
class SmaCrossover(Strategy):
    """Trend-following SMA crossover strategy with ATR-based position sizing,
    trailing stops, time stops, and pyramiding.

    Entry: golden cross (fast SMA > slow SMA) with minimum trend spread.
    Exit: death cross, trailing stop, or time stop.
    Pyramiding: add to winning positions when price exceeds avg_entry + ATR.
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

        if actual_qty == 0 and state.position_qty > 0:
            # Position was closed externally (e.g. by engine)
            state.position_qty = 0
            state.pyramid_count = 0
            state.original_qty = 0
            state.entry_price = 0.0
            state.avg_entry = 0.0
            state.highest_since_entry = 0.0
            state.trailing_stop = 0.0
            state.bars_in_position = 0

        if actual_qty > 0:
            state.position_qty = actual_qty
            state.avg_entry = actual_avg

    def _compute_qty(self, capital: float, atr: float) -> int:
        """ATR-based position sizing: qty = (capital * risk_per_trade) / (ATR * multiplier)."""
        risk_amount = capital * self.risk_per_trade
        stop_distance = atr * self.atr_stop_multiplier
        if stop_distance <= 0:
            return 0
        return int(risk_amount / stop_distance)

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

                # 5. If NOT in position: check entry
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
                                signals.append(
                                    Signal(
                                        action="BUY",
                                        symbol=symbol,
                                        quantity=qty,
                                        product_type="CNC",
                                    )
                                )
                                state.entry_price = bar.close
                                state.avg_entry = bar.close
                                state.entry_bar = self.bar_count
                                state.highest_since_entry = bar.close
                                state.trailing_stop = bar.close - atr * self.atr_stop_multiplier
                                state.position_qty = qty
                                state.original_qty = qty
                                state.pyramid_count = 0
                                state.bars_in_position = 0

                # 6. If IN position: check exits, then pyramiding
                elif state.position_qty > 0:
                    state.bars_in_position += 1

                    # Update trailing stop (only moves up)
                    if bar.close > state.highest_since_entry:
                        state.highest_since_entry = bar.close
                    new_stop = state.highest_since_entry - atr * self.atr_stop_multiplier
                    if new_stop > state.trailing_stop:
                        state.trailing_stop = new_stop

                    sell_all = False
                    # a. Death cross exit
                    if state.prev_fast_above is not None and not fast_above and state.prev_fast_above:
                        sell_all = True

                    # b. Trailing stop exit
                    if not sell_all and bar.close < state.trailing_stop:
                        sell_all = True

                    # c. Time stop exit
                    if not sell_all and state.bars_in_position > self.max_hold_bars:
                        gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0.0
                        if gain < 0.005:
                            sell_all = True

                    if sell_all:
                        signals.append(
                            Signal(
                                action="SELL",
                                symbol=symbol,
                                quantity=state.position_qty,
                                product_type="CNC",
                            )
                        )
                        state.position_qty = 0
                        state.pyramid_count = 0
                        state.original_qty = 0
                        state.entry_price = 0.0
                        state.avg_entry = 0.0
                        state.highest_since_entry = 0.0
                        state.trailing_stop = 0.0
                        state.bars_in_position = 0
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
                                    product_type="CNC",
                                )
                            )
                            # Update average entry
                            total_qty = state.position_qty + add_qty
                            state.avg_entry = (
                                (state.avg_entry * state.position_qty + bar.close * add_qty) / total_qty
                            )
                            state.position_qty = total_qty
                            state.pyramid_count += 1

                # 7. Update crossover state
                state.prev_fast_above = fast_above

        return signals
