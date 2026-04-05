"""Donchian Breakout + Trend Confirmation + Partial Profit + Pyramiding strategy.

Multi-timeframe trend-following strategy:
- Daily: Donchian channel (N-day high/low) for breakout detection, ATR for volatility
- 15-minute: Fine-grained entry timing, trailing stop management, partial exits

Long Rules:
- BUY when 15-min close breaks above N-day channel high with volume confirmation
- Risk-based position sizing: qty = int((cash * risk_per_trade) / (ATR * atr_multiplier))
- Pyramiding: up to 2 add-on entries when price moves 1*ATR above avg entry
- Engine SL-M stop at trailing stop level, engine LIMIT sell at profit target
- Trailing stop ratchet: cancel + resubmit SL-M when stop moves up
- Partial profit: LIMIT sell 1/3 at profit_target_atr * ATR, move stop to breakeven
- Channel low exit: price < N-day low -> cancel engine orders + full exit
- Max loss stop: price < avg_entry * (1 - max_loss_pct) -> cancel + full exit
- Time stop: exit after max_hold_bars if gain < 0.5%
- Dynamic CNC/MIS: CNC if volume > 1.5x average, MIS otherwise

Short Rules:
- SELL when 15-min close breaks below N-day channel low with volume confirmation
- Risk-based position sizing: same formula as long
- Pyramiding: up to 2 add-on entries when price moves 1*ATR below avg entry
- Engine SL-M BUY at trailing stop above entry, engine LIMIT BUY for partial cover
- Trailing stop ratchet: cancel + resubmit SL-M when stop moves down
- Partial cover: LIMIT BUY 1/3 at avg_entry - profit_target_atr * ATR, stop to breakeven
- Channel high exit: price > N-day high -> cancel + full cover
- Max loss stop: price > avg_entry * (1 + max_loss_pct) -> cancel + full cover
- Time stop: cover after max_hold_bars if gain < 0.5%
- Dynamic CNC/MIS: CNC if volume > 1.5x average, MIS otherwise

Config params:
- channel_period: Donchian channel lookback (default 20 days)
- atr_period: ATR period (default 14 days)
- atr_multiplier: trailing stop distance in ATRs (default 1.5)
- volume_factor: required volume vs average to confirm breakout (default 1.0)
- risk_per_trade: fraction of capital risked per trade (default 0.02)
- profit_target_atr: ATR multiple for partial profit exit (default 2.0)
- max_loss_pct: maximum loss before forced exit (default 0.02)
- max_hold_bars: max bars in position before time stop check (default 30)
- pyramid_levels: maximum pyramid add-ons (default 2)
"""

from collections import deque

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_atr


@register("donchian_breakout")
class DonchianBreakout(Strategy):
    """Donchian Channel Breakout with engine stops, limit profits, dynamic CNC/MIS."""

    def required_data(self) -> list[dict]:
        return [
            {"interval": "day", "lookback": 60},
            {"interval": "15minute", "lookback": 30},
        ]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.channel_period = config.get("channel_period", 20)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 1.5)
        self.volume_factor = config.get("volume_factor", 1.0)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.profit_target_atr = config.get("profit_target_atr", 2.0)
        self.max_loss_pct = config.get("max_loss_pct", 0.02)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.pyramid_levels = config.get("pyramid_levels", 2)
        self.instruments = instruments

        # Per-symbol state
        self.daily_highs: dict[str, deque[float]] = {}
        self.daily_lows: dict[str, deque[float]] = {}
        self.daily_closes: dict[str, deque[float]] = {}
        self.daily_volumes: dict[str, deque[int]] = {}
        self.current_atr: dict[str, float] = {}

        # Position management state
        self.in_position: dict[str, bool] = {}
        self.is_short: dict[str, bool] = {}
        self.avg_entry_price: dict[str, float] = {}
        self.original_qty: dict[str, int] = {}
        self.pyramid_count: dict[str, int] = {}
        self.trailing_stop: dict[str, float] = {}
        self.highest_since_entry: dict[str, float] = {}
        self.lowest_since_entry: dict[str, float] = {}
        self.entry_bar: dict[str, int] = {}
        self.bar_counter: dict[str, int] = {}
        self.partial_taken: dict[str, bool] = {}
        self.breakeven_stop: dict[str, bool] = {}

        # Engine order tracking
        self.has_profit_target: dict[str, bool] = {}
        self.has_engine_stop: dict[str, bool] = {}
        self.product_type: dict[str, str] = {}

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals: list[Signal] = []

        # --- Step 1: Update daily data when new daily bar arrives ---
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure_state(symbol)
                self.daily_highs[symbol].append(bar.high)
                self.daily_lows[symbol].append(bar.low)
                self.daily_closes[symbol].append(bar.close)
                self.daily_volumes[symbol].append(bar.volume)

                atr = compute_atr(
                    list(self.daily_highs[symbol]),
                    list(self.daily_lows[symbol]),
                    list(self.daily_closes[symbol]),
                    self.atr_period,
                )
                if atr is not None:
                    self.current_atr[symbol] = atr

        # --- Step 2: Process 15-minute bars ---
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            self._ensure_state(symbol)
            self.bar_counter[symbol] += 1

            # 2a. Detect engine stop/profit fills BEFORE reconciliation
            #     When the engine fills an SL-M or LIMIT order, the position
            #     may already be gone (held_qty == 0). We must detect the fill
            #     before reconciliation resets state and potentially re-enters.
            held_qty = 0
            for pos in snapshot.portfolio.positions:
                if pos.symbol == symbol and pos.quantity != 0:
                    held_qty = pos.quantity
                    break

            if self.in_position[symbol] and not self.is_short.get(symbol, False):
                # Long position — check for SELL fills from engine orders
                if self.has_engine_stop.get(symbol, False) or self.has_profit_target.get(symbol, False):
                    sell_fill = any(f.symbol == symbol and f.side == "SELL" for f in snapshot.fills)
                    if sell_fill and held_qty == 0:
                        # Stop fill consumed all position — reset and skip
                        self._reset_position(symbol)
                        continue

            elif self.in_position[symbol] and self.is_short.get(symbol, False):
                # Short position — check for BUY fills from engine orders
                if self.has_engine_stop.get(symbol, False) or self.has_profit_target.get(symbol, False):
                    buy_fill = any(f.symbol == symbol and f.side == "BUY" for f in snapshot.fills)
                    if buy_fill and held_qty == 0:
                        # Stop fill consumed all position — reset and skip
                        self._reset_position(symbol)
                        continue

            # 2a-bis. Reconcile position with portfolio
            if self.in_position[symbol] and held_qty == 0:
                self._reset_position(symbol)

            # 2b. Compute Donchian channel
            highs = list(self.daily_highs.get(symbol, []))
            lows = list(self.daily_lows.get(symbol, []))
            closes = list(self.daily_closes.get(symbol, []))
            volumes = list(self.daily_volumes.get(symbol, []))

            if len(closes) < self.channel_period + 1:
                continue

            # Exclude current (incomplete) day
            channel_high = max(highs[-(self.channel_period + 1):-1])
            channel_low = min(lows[-(self.channel_period + 1):-1])

            # 2c. Average volume over channel period
            avg_volume = sum(volumes[-(self.channel_period + 1):-1]) / self.channel_period

            atr = self.current_atr.get(symbol, 0.0)

            # 2d. ENTRY LOGIC
            if not self.in_position[symbol]:
                volume_ok = (
                    atr > 0
                    and volumes
                    and volumes[-1] >= avg_volume * self.volume_factor
                )

                # Determine product type based on volume strength
                volume_strong = volumes[-1] > avg_volume * 1.5 if volumes else False
                product = "CNC" if volume_strong else "MIS"

                if volume_ok and bar.close > channel_high:
                    # --- LONG ENTRY ---
                    risk_amount = atr * self.atr_multiplier
                    qty = int((snapshot.portfolio.cash * self.risk_per_trade) / risk_amount) if risk_amount > 0 else 0

                    inst = self.instruments.get(symbol)
                    if inst and inst.lot_size > 1:
                        qty = (qty // inst.lot_size) * inst.lot_size

                    if qty > 0:
                        self.product_type[symbol] = product
                        signals.append(Signal(
                            action="BUY", symbol=symbol, quantity=qty,
                            product_type=product,
                        ))
                        self.in_position[symbol] = True
                        self.is_short[symbol] = False
                        self.avg_entry_price[symbol] = bar.close
                        self.original_qty[symbol] = qty
                        self.pyramid_count[symbol] = 0
                        self.highest_since_entry[symbol] = bar.close
                        self.lowest_since_entry[symbol] = bar.close
                        self.trailing_stop[symbol] = bar.close - (atr * self.atr_multiplier)
                        self.entry_bar[symbol] = self.bar_counter[symbol]
                        self.partial_taken[symbol] = False
                        self.breakeven_stop[symbol] = False
                        self.has_engine_stop[symbol] = False
                        self.has_profit_target[symbol] = False

                elif volume_ok and bar.close < channel_low:
                    # --- SHORT ENTRY ---
                    # Short entries always use MIS (CNC shorts not allowed in Zerodha)
                    product = "MIS"
                    risk_amount = atr * self.atr_multiplier
                    qty = int((snapshot.portfolio.cash * self.risk_per_trade) / risk_amount) if risk_amount > 0 else 0

                    inst = self.instruments.get(symbol)
                    if inst and inst.lot_size > 1:
                        qty = (qty // inst.lot_size) * inst.lot_size

                    if qty > 0:
                        self.product_type[symbol] = product
                        signals.append(Signal(
                            action="SELL", symbol=symbol, quantity=qty,
                            product_type=product,
                        ))
                        self.in_position[symbol] = True
                        self.is_short[symbol] = True
                        self.avg_entry_price[symbol] = bar.close
                        self.original_qty[symbol] = qty
                        self.pyramid_count[symbol] = 0
                        self.highest_since_entry[symbol] = bar.close
                        self.lowest_since_entry[symbol] = bar.close
                        # Short trailing stop is ABOVE entry: entry + ATR * multiplier
                        self.trailing_stop[symbol] = bar.close + (atr * self.atr_multiplier)
                        self.entry_bar[symbol] = self.bar_counter[symbol]
                        self.partial_taken[symbol] = False
                        self.breakeven_stop[symbol] = False
                        self.has_engine_stop[symbol] = False
                        self.has_profit_target[symbol] = False

            # 2e. POSITION MANAGEMENT
            elif not self.is_short[symbol]:
                # ==================== LONG POSITION MANAGEMENT ====================
                product = self.product_type.get(symbol, "CNC")
                avg_entry = self.avg_entry_price[symbol]

                # --- Stop-hit detection ---
                if self.has_engine_stop.get(symbol, False):
                    stop_hit = any(f.symbol == symbol and f.side == "SELL" for f in snapshot.fills)
                    if stop_hit and not self.has_profit_target.get(symbol, False):
                        self.has_engine_stop[symbol] = False
                        self._reset_position(symbol)
                        continue

                # --- Partial profit fill detection ---
                if self.has_profit_target.get(symbol, False):
                    partial_fill = any(f.symbol == symbol and f.side == "SELL" for f in snapshot.fills)
                    if partial_fill:
                        self.has_profit_target[symbol] = False
                        self.partial_taken[symbol] = True
                        # Cancel existing engine stop, resubmit at breakeven
                        signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                        remaining_qty = held_qty
                        signals.append(Signal(
                            action="SELL", symbol=symbol, quantity=remaining_qty,
                            order_type="SL_M", stop_price=avg_entry,
                            product_type=product,
                        ))
                        self.trailing_stop[symbol] = avg_entry
                        self.breakeven_stop[symbol] = True

                # --- Update highest since entry ---
                if bar.close > self.highest_since_entry[symbol]:
                    self.highest_since_entry[symbol] = bar.close

                # --- Re-submit expired engine orders (DAY order expiry at 15:30 IST) ---
                if held_qty > 0:
                    if self.has_engine_stop.get(symbol, False):
                        has_stop = any(
                            po.symbol == symbol and po.order_type == "SL_M"
                            for po in snapshot.pending_orders
                        )
                        if not has_stop and self.trailing_stop.get(symbol, 0) > 0:
                            signals.append(Signal(
                                action="SELL", symbol=symbol,
                                quantity=held_qty,
                                order_type="SL_M",
                                stop_price=self.trailing_stop[symbol],
                                product_type=product,
                            ))

                    if self.has_profit_target.get(symbol, False) and not self.partial_taken.get(symbol, False):
                        has_target = any(
                            po.symbol == symbol and po.order_type == "LIMIT"
                            for po in snapshot.pending_orders
                        )
                        if not has_target:
                            atr_val = self.current_atr.get(symbol, 0)
                            if atr_val > 0:
                                partial_qty = max(1, held_qty // 3)
                                target_price = avg_entry + self.profit_target_atr * atr_val
                                signals.append(Signal(
                                    action="SELL", symbol=symbol, quantity=partial_qty,
                                    order_type="LIMIT", limit_price=target_price,
                                    product_type=product,
                                ))

                # --- Trailing stop ratchet (cancel + resubmit) ---
                if atr > 0:
                    new_stop = self.highest_since_entry[symbol] - (atr * self.atr_multiplier)
                    if new_stop > self.trailing_stop[symbol]:
                        self.trailing_stop[symbol] = new_stop
                        if self.has_engine_stop.get(symbol, False):
                            signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                            signals.append(Signal(
                                action="SELL", symbol=symbol, quantity=held_qty,
                                order_type="SL_M", stop_price=self.trailing_stop[symbol],
                                product_type=product,
                            ))

                # --- EXIT CHECKS (channel low, max loss, time stop) ---
                should_exit = False

                if bar.close < channel_low:
                    should_exit = True

                if bar.close < avg_entry * (1 - self.max_loss_pct):
                    should_exit = True

                bars_held = self.bar_counter[symbol] - self.entry_bar[symbol]
                if bars_held > self.max_hold_bars:
                    gain_pct = (bar.close - avg_entry) / avg_entry if avg_entry > 0 else 0
                    if gain_pct < 0.005:
                        should_exit = True

                if should_exit and held_qty > 0:
                    # Cancel all pending engine orders first
                    signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                    self.has_engine_stop[symbol] = False
                    self.has_profit_target[symbol] = False
                    signals.append(Signal(
                        action="SELL", symbol=symbol, quantity=held_qty,
                        product_type=product,
                    ))
                    self._reset_position(symbol)
                    continue

                # --- Submit engine stop + profit target (first bar after entry) ---
                if not self.has_engine_stop.get(symbol, False) and held_qty > 0:
                    if atr > 0:
                        signals.append(Signal(
                            action="SELL", symbol=symbol, quantity=held_qty,
                            order_type="SL_M", stop_price=self.trailing_stop[symbol],
                            product_type=product,
                        ))
                        self.has_engine_stop[symbol] = True

                        if not self.partial_taken.get(symbol, False):
                            partial_qty = max(1, held_qty // 3)
                            profit_price = avg_entry + self.profit_target_atr * atr
                            signals.append(Signal(
                                action="SELL", symbol=symbol, quantity=partial_qty,
                                order_type="LIMIT", limit_price=profit_price,
                                product_type=product,
                            ))
                            self.has_profit_target[symbol] = True

                # --- Pyramid entry ---
                if (
                    atr > 0
                    and self.pyramid_count[symbol] < self.pyramid_levels
                    and bar.close > avg_entry + atr
                ):
                    add_qty = max(1, self.original_qty[symbol] // 2)
                    signals.append(Signal(
                        action="BUY", symbol=symbol, quantity=add_qty,
                        product_type=product,
                    ))
                    total_qty = held_qty + add_qty
                    self.avg_entry_price[symbol] = (
                        (avg_entry * held_qty + bar.close * add_qty) / total_qty
                        if total_qty > 0 else avg_entry
                    )
                    self.pyramid_count[symbol] += 1
                    continue

            else:
                # ==================== SHORT POSITION MANAGEMENT ====================
                abs_held = abs(held_qty)
                product = self.product_type.get(symbol, "CNC")
                avg_entry = self.avg_entry_price[symbol]

                # --- Stop-hit detection (short: BUY fill = stop hit) ---
                if self.has_engine_stop.get(symbol, False):
                    stop_hit = any(f.symbol == symbol and f.side == "BUY" for f in snapshot.fills)
                    if stop_hit and not self.has_profit_target.get(symbol, False):
                        self.has_engine_stop[symbol] = False
                        self._reset_position(symbol)
                        continue

                # --- Partial cover fill detection (short: BUY fill = partial cover) ---
                if self.has_profit_target.get(symbol, False):
                    partial_fill = any(f.symbol == symbol and f.side == "BUY" for f in snapshot.fills)
                    if partial_fill:
                        self.has_profit_target[symbol] = False
                        self.partial_taken[symbol] = True
                        # Cancel existing engine stop, resubmit at breakeven
                        signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                        remaining_qty = abs_held
                        signals.append(Signal(
                            action="BUY", symbol=symbol, quantity=remaining_qty,
                            order_type="SL_M", stop_price=avg_entry,
                            product_type=product,
                        ))
                        self.trailing_stop[symbol] = avg_entry
                        self.breakeven_stop[symbol] = True

                # --- Update lowest since entry (tracks in favour of short) ---
                if bar.close < self.lowest_since_entry[symbol]:
                    self.lowest_since_entry[symbol] = bar.close

                # --- Re-submit expired engine orders (DAY order expiry at 15:30 IST) ---
                if abs_held > 0:
                    if self.has_engine_stop.get(symbol, False):
                        has_stop = any(
                            po.symbol == symbol and po.order_type == "SL_M"
                            for po in snapshot.pending_orders
                        )
                        if not has_stop and self.trailing_stop.get(symbol, 0) > 0:
                            signals.append(Signal(
                                action="BUY", symbol=symbol,
                                quantity=abs_held,
                                order_type="SL_M",
                                stop_price=self.trailing_stop[symbol],
                                product_type=product,
                            ))

                    if self.has_profit_target.get(symbol, False) and not self.partial_taken.get(symbol, False):
                        has_target = any(
                            po.symbol == symbol and po.order_type == "LIMIT"
                            for po in snapshot.pending_orders
                        )
                        if not has_target:
                            atr_val = self.current_atr.get(symbol, 0)
                            if atr_val > 0:
                                partial_qty = max(1, abs_held // 3)
                                target_price = avg_entry - self.profit_target_atr * atr_val
                                signals.append(Signal(
                                    action="BUY", symbol=symbol, quantity=partial_qty,
                                    order_type="LIMIT", limit_price=target_price,
                                    product_type=product,
                                ))

                # --- Trailing stop ratchet (cancel + resubmit, only moves DOWN for shorts) ---
                if atr > 0:
                    new_stop = self.lowest_since_entry[symbol] + (atr * self.atr_multiplier)
                    if new_stop < self.trailing_stop[symbol]:
                        self.trailing_stop[symbol] = new_stop
                        if self.has_engine_stop.get(symbol, False):
                            signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                            signals.append(Signal(
                                action="BUY", symbol=symbol, quantity=abs_held,
                                order_type="SL_M", stop_price=self.trailing_stop[symbol],
                                product_type=product,
                            ))

                # --- SHORT EXIT CHECKS (channel high, max loss, time stop) ---
                should_cover = False

                # Channel high breakout (trend reversal)
                if bar.close > channel_high:
                    should_cover = True

                # Max loss stop (price rises too much above entry)
                if bar.close > avg_entry * (1 + self.max_loss_pct):
                    should_cover = True

                # Time stop
                bars_held = self.bar_counter[symbol] - self.entry_bar[symbol]
                if bars_held > self.max_hold_bars:
                    # For shorts, gain = (entry - current) / entry
                    gain_pct = (avg_entry - bar.close) / avg_entry if avg_entry > 0 else 0
                    if gain_pct < 0.005:
                        should_cover = True

                if should_cover and abs_held > 0:
                    # Cancel all pending engine orders first
                    signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
                    self.has_engine_stop[symbol] = False
                    self.has_profit_target[symbol] = False
                    signals.append(Signal(
                        action="BUY", symbol=symbol, quantity=abs_held,
                        product_type=product,
                    ))
                    self._reset_position(symbol)
                    continue

                # --- Submit engine stop + profit target (first bar after entry) ---
                if not self.has_engine_stop.get(symbol, False) and abs_held > 0:
                    if atr > 0:
                        signals.append(Signal(
                            action="BUY", symbol=symbol, quantity=abs_held,
                            order_type="SL_M", stop_price=self.trailing_stop[symbol],
                            product_type=product,
                        ))
                        self.has_engine_stop[symbol] = True

                        if not self.partial_taken.get(symbol, False):
                            partial_qty = max(1, abs_held // 3)
                            profit_price = avg_entry - self.profit_target_atr * atr
                            signals.append(Signal(
                                action="BUY", symbol=symbol, quantity=partial_qty,
                                order_type="LIMIT", limit_price=profit_price,
                                product_type=product,
                            ))
                            self.has_profit_target[symbol] = True

                # --- Pyramid (add to short) ---
                if (
                    atr > 0
                    and self.pyramid_count[symbol] < self.pyramid_levels
                    and bar.close < avg_entry - atr
                ):
                    add_qty = max(1, self.original_qty[symbol] // 2)
                    signals.append(Signal(
                        action="SELL", symbol=symbol, quantity=add_qty,
                        product_type=product,
                    ))
                    total_qty = abs_held + add_qty
                    self.avg_entry_price[symbol] = (
                        (avg_entry * abs_held + bar.close * add_qty) / total_qty
                        if total_qty > 0 else avg_entry
                    )
                    self.pyramid_count[symbol] += 1
                    continue

        return signals

    def _ensure_state(self, symbol: str) -> None:
        if symbol not in self.daily_highs:
            max_len = self.channel_period + self.atr_period + 10
            self.daily_highs[symbol] = deque(maxlen=max_len)
            self.daily_lows[symbol] = deque(maxlen=max_len)
            self.daily_closes[symbol] = deque(maxlen=max_len)
            self.daily_volumes[symbol] = deque(maxlen=max_len)
            self._reset_position(symbol)
            self.bar_counter[symbol] = 0

    def _reset_position(self, symbol: str) -> None:
        self.in_position[symbol] = False
        self.is_short[symbol] = False
        self.avg_entry_price[symbol] = 0.0
        self.original_qty[symbol] = 0
        self.pyramid_count[symbol] = 0
        self.trailing_stop[symbol] = 0.0
        self.highest_since_entry[symbol] = 0.0
        self.lowest_since_entry[symbol] = float('inf')
        self.entry_bar[symbol] = 0
        self.partial_taken[symbol] = False
        self.breakeven_stop[symbol] = False
        self.has_engine_stop[symbol] = False
        self.has_profit_target[symbol] = False
        self.product_type[symbol] = "CNC"

    def on_complete(self) -> dict:
        return {
            "strategy_type": "donchian_breakout",
            "channel_period": self.channel_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "volume_factor": self.volume_factor,
            "risk_per_trade": self.risk_per_trade,
            "profit_target_atr": self.profit_target_atr,
            "max_loss_pct": self.max_loss_pct,
            "max_hold_bars": self.max_hold_bars,
            "pyramid_levels": self.pyramid_levels,
        }
