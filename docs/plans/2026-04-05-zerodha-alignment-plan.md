# Zerodha API Alignment — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the backtesting engine with Zerodha Kite Connect API behavior — DAY order expiry, CNC short restriction, SL two-price model, per-order cancellation, IOC validity.

**Architecture:** Engine-level changes in matching.rs, engine.rs, types.rs + proto/Python updates for new fields. Strict enforcement — operations that would fail live are rejected. Strategies updated to comply.

**Tech Stack:** Rust, Python, gRPC/Proto

**Design doc:** `docs/plans/2026-04-05-zerodha-api-alignment-design.md`

---

### Task A: DAY Order Validity + IOC (Gaps 1, 7)

**Files:**
- Modify: `engine/crates/core/src/matching.rs` — cancel_all_pending, IOC auto-cancel, OrderValidity
- Modify: `engine/crates/core/src/engine.rs` — 15:30 IST expiry in event loop
- Modify: `engine/crates/core/src/types.rs` — OrderValidity enum, validity on Signal
- Modify: `engine/crates/proto/proto/strategy.proto` — OrderValidity enum, validity field
- Modify: `engine/crates/core/src/grpc_client.rs` — map validity
- Modify: `strategies/strategies/base.py` — validity field on Signal
- Modify: `strategies/server/server.py` — map validity
- Regen: `cargo build -p backtest-proto` + `cd strategies && ./generate_proto.sh`

**Changes:**

1. **types.rs** — Add OrderValidity:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
   pub enum OrderValidity { Day, Ioc }
   impl Default for OrderValidity { fn default() -> Self { OrderValidity::Day } }
   ```
   Add `pub validity: OrderValidity` to Signal (with `#[serde(default)]`).

2. **matching.rs** — Add to Order struct: `pub validity: OrderValidity` (default Day).
   - Add `pub fn cancel_all_pending(&mut self) -> Vec<Order>` that drains all pending orders.
   - In `process_bar`, after processing fills, auto-cancel unfilled IOC orders:
     ```rust
     self.pending_orders.retain(|o| o.validity != OrderValidity::Ioc);
     ```
   - Propagate validity from Signal to Order in `Order::from_signal`.

3. **engine.rs** — After MIS auto-squareoff (15:20 IST), add DAY order expiry at 15:30:
   ```rust
   let close_time = chrono::NaiveTime::from_hms_opt(15, 30, 0).unwrap();
   if ist_time >= close_time {
       matcher.cancel_all_pending();
   }
   ```

4. **Proto** — Add `enum OrderValidity { DAY = 0; IOC = 1; }` and `OrderValidity validity = 9` to Signal message.

5. **Python** — Add `validity: str = "DAY"` to Signal dataclass. Map in server.py.

**Tests:**
- `test_day_orders_expire_at_1530` — submit limit order, tick past 15:30 IST timestamp, verify pending_orders is empty
- `test_ioc_unfilled_cancelled` — submit IOC limit that doesn't fill, verify cancelled after process_bar
- `test_ioc_filled_not_cancelled` — submit IOC limit that does fill, verify fill occurs normally

**Verify:** `cd engine && cargo test` + `cd strategies && pytest tests/ -v`

**Commit:** `feat: DAY order expiry at 15:30 IST and IOC validity support`

---

### Task B: CNC Short Restriction (Gap 2)

**Files:**
- Modify: `engine/crates/core/src/engine.rs` — reject CNC shorts
- Modify: `strategies/strategies/deterministic/sma_crossover.py` — force MIS for shorts
- Modify: `strategies/strategies/deterministic/donchian_breakout.py` — force MIS for shorts

**Changes:**

1. **engine.rs** — In signal processing, before submitting a Sell order:
   ```rust
   if signal.action == Action::Sell && signal.product_type == ProductType::Cnc {
       // Check if this is closing an existing long position
       let has_long_position = portfolio.positions_snapshot()
           .iter()
           .any(|(sym, _qty, is_short, _pt)| sym == &signal.symbol && !is_short);
       if !has_long_position {
           current_rejections.push(OrderRejection {
               symbol: signal.symbol.clone(),
               side: Side::Sell,
               quantity: signal.quantity,
               reason: "CNC_SHORT_NOT_ALLOWED".into(),
           });
           continue;
       }
   }
   ```

2. **sma_crossover.py** — In short entry logic (death cross), force MIS:
   ```python
   # Short entries always use MIS (CNC shorts not allowed in Zerodha)
   product = "MIS"
   ```

3. **donchian_breakout.py** — In short entry logic (channel low break), force MIS:
   ```python
   product = "MIS"  # shorts always MIS
   ```

**Tests:**
- `test_cnc_short_rejected` — submit CNC SELL without position, verify rejection with reason "CNC_SHORT_NOT_ALLOWED"
- `test_cnc_sell_existing_position_allowed` — buy CNC, then sell CNC, verify sell goes through
- `test_sma_short_uses_mis` — verify SMA short entries emit product_type="MIS"
- `test_donchian_short_uses_mis` — verify Donchian short entries emit product_type="MIS"

**Verify:** `cd engine && cargo test` + `cd strategies && pytest tests/ -v`

**Commit:** `feat: CNC short restriction — reject CNC shorts, strategies use MIS`

---

### Task C: SL Two-Price Model (Gap 3)

**Files:**
- Modify: `engine/crates/core/src/types.rs` — trigger_price on Signal
- Modify: `engine/crates/core/src/matching.rs` — SL trigger→limit two-step, Order gets trigger_price
- Modify: `engine/crates/proto/proto/strategy.proto` — trigger_price field
- Modify: `engine/crates/core/src/grpc_client.rs` — map trigger_price
- Modify: `strategies/strategies/base.py` — trigger_price on Signal
- Modify: `strategies/server/server.py` — map trigger_price

**Changes:**

1. **types.rs** — Add `pub trigger_price: f64` to Signal (default 0.0, serde default).

2. **matching.rs** — Add `pub trigger_price: f64` to Order. Update `Order::from_signal` to copy trigger_price.

   For SL Sell orders, change fill logic:
   ```rust
   // Old: if bar.low <= order.stop_price → fill at stop_price or bar.open
   // New two-step:
   (OrderType::Sl, Side::Sell) => {
       if bar.low <= order.trigger_price {
           // Triggered! Now it's a limit order at order.limit_price
           if bar.open <= order.limit_price {
               // Gapped through limit — fill at bar.open (price improvement)
               fill_price = bar.open;
           } else if bar.low <= order.limit_price {
               // Price touched limit — fill at limit
               fill_price = order.limit_price;
           } else {
               // Triggered but price didn't reach limit — becomes pending limit order
               // Convert to limit order for next bar
               order.order_type = OrderType::Limit;
               order.trigger_price = 0.0; // already triggered
               // Don't fill this bar — stays pending
               continue;
           }
       }
   }
   ```

   For SL Buy: mirror with `bar.high >= trigger_price`.

   For SL-M: no change — trigger_price activates, fills at market (bar.open ± slippage). Already correct.

   Backward compat: if `trigger_price == 0.0`, fall back to using `stop_price` as trigger_price (existing behavior for strategies that don't set trigger_price).

3. **Proto** — Add `double trigger_price = 9` to Signal message.

4. **Python** — Add `trigger_price: float = 0.0` to Signal dataclass.

**Tests:**
- `test_sl_two_price_fill_at_limit` — SL sell: trigger=100, limit=99. Bar low touches 98 → fill at 99
- `test_sl_two_price_gap_through` — trigger=100, limit=99. Bar opens at 95 → fill at 95 (below limit = price improvement)
- `test_sl_two_price_trigger_but_no_fill` — trigger=100, limit=99. Bar low=99.5 (triggers but doesn't reach limit) → becomes pending limit
- `test_sl_backward_compat` — trigger_price=0, stop_price=100 → uses stop_price as trigger (old behavior preserved)

**Verify:** `cd engine && cargo test -p backtest-core -- matching`

**Commit:** `feat: SL two-price model — trigger_price activates, limit_price fills`

---

### Task D: Per-Order Cancellation (Gap 5)

**Files:**
- Modify: `engine/crates/core/src/matching.rs` — order_id on Order, cancel_order by ID
- Modify: `engine/crates/core/src/engine.rs` — route cancel by order_id
- Modify: `engine/crates/core/src/types.rs` — cancel_order_id on Signal
- Modify: `engine/crates/proto/proto/strategy.proto` — order_id, cancel_order_id
- Modify: `engine/crates/core/src/grpc_client.rs` — map order_id on PendingOrderInfo
- Modify: `strategies/strategies/base.py` — order_id on PendingOrder, cancel_order_id on Signal
- Modify: `strategies/server/server.py` — map fields

**Changes:**

1. **matching.rs** — Add to Order: `pub order_id: u64`. Add to OrderMatcher: `next_order_id: u64` (init 1).
   On `submit()`: `order.order_id = self.next_order_id; self.next_order_id += 1`.
   Add `pub fn cancel_order(&mut self, order_id: u64) -> Option<Order>`.
   Update PendingOrderInfo to include `order_id: u64`.

2. **types.rs** — Add `pub cancel_order_id: Option<u64>` to Signal (serde default None).

3. **engine.rs** — On Cancel action:
   ```rust
   if let Some(oid) = signal.cancel_order_id {
       matcher.cancel_order(oid);
   } else {
       matcher.cancel_orders_for_symbol(&signal.symbol);
   }
   ```

4. **Proto** — Add `uint64 order_id = 7` to PendingOrderInfo. Add `uint64 cancel_order_id = 10` to Signal (0 = cancel all for symbol).

5. **Python** — `order_id: int = 0` on PendingOrder. `cancel_order_id: int = 0` on Signal.

**Tests:**
- `test_cancel_specific_order_by_id` — submit 2 orders for same symbol with different IDs, cancel one by ID, verify only that one removed
- `test_cancel_all_for_symbol_still_works` — cancel without order_id, verify all removed (backward compat)
- `test_order_ids_are_sequential` — submit 3 orders, verify IDs are 1, 2, 3
- `test_pending_order_info_has_id` — verify order_id appears in pending_orders snapshot

**Verify:** `cd engine && cargo test` + `cd strategies && pytest tests/ -v`

**Commit:** `feat: per-order cancellation by ID, sequential order IDs`

---

### Task E: Documentation (Gaps 4, 6)

**Files:**
- Modify: `CLAUDE.md`

**Changes:**
Add to Key Conventions:
```
- Order modification is not supported. Use CANCEL + new order (adds one bar latency vs Zerodha's atomic modify).
- Cover orders (CO) are not supported. Strategies achieve the same effect by manually pairing entry + SL-M orders.
- DAY validity: all pending orders expire at 15:30 IST. Strategies must re-place orders on the next trading day.
- IOC validity: orders with validity="IOC" are cancelled if unfilled on the current bar.
- CNC short restriction: selling without a position via CNC is rejected. Use MIS for intraday shorts.
- SL orders use two prices: trigger_price (activation) + limit_price (fill). If price gaps past limit, order becomes a pending limit.
- Per-order cancellation: cancel_order_id targets a specific order. Without it, all orders for the symbol are cancelled.
```

**Commit:** `docs: document Zerodha API alignment in CLAUDE.md`

---

## Execution Order

Tasks A-D are fully independent. Task E after all.

```
A (DAY + IOC)              ─┐
B (CNC short restriction)  ─┤── All parallel
C (SL two-price)           ─┤
D (Per-order cancellation) ─┘
E (Documentation)          ── After A-D
```

## Proto Regeneration Note

Tasks A, C, D all modify strategy.proto. Since they run in parallel, there may be merge conflicts in the proto file. Solution: after all agents commit, do a single proto regeneration pass to resolve any conflicts.
