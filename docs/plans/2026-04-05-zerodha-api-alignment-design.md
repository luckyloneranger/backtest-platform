# Zerodha API Alignment — Design Document

Date: 2026-04-05

## Goal

Align the backtesting engine with Zerodha Kite Connect API behavior. Strict enforcement — the engine rejects operations that would fail in live trading, ensuring backtests match real-world execution.

## Gaps Identified

| # | Gap | Severity | Action |
|---|-----|----------|--------|
| 1 | DAY order validity — pending orders persist indefinitely | HIGH | Implement |
| 2 | CNC short restriction — CNC shorts allowed in engine | MEDIUM | Implement |
| 3 | SL two-price model — SL orders lack trigger_price + limit_price | MEDIUM | Implement |
| 4 | Order modification — no atomic modify, only cancel + re-place | LOW-MED | Document only |
| 5 | Per-order cancellation — cancel-all-for-symbol is too coarse | LOW-MED | Implement |
| 6 | Cover orders — no atomic entry + SL multi-leg | LOW | Document only |
| 7 | IOC validity — no immediate-or-cancel order type | LOW | Implement |

## Feature Designs

### 1. DAY Order Validity (HIGH)

All pending orders (LIMIT, SL, SL-M) expire at 15:30 IST (NSE market close) each trading day.

**Changes to `engine.rs`:**
After MIS auto-squareoff at 15:20 IST (already implemented), add a second time check at 15:30 IST:
```
if ist_time >= 15:30:
    expired = matcher.cancel_all_pending()
    // expired orders are gone — strategies see empty pending_orders on next bar
```

**Changes to `matching.rs`:**
- Add `pub fn cancel_all_pending(&mut self) -> Vec<Order>` — removes and returns all pending orders
- Existing `cancel_orders_for_symbol` stays for strategy-initiated cancellation

**Behavior:**
- Orders placed during the day expire at 15:30 IST
- Strategies must re-place orders on the next trading day if still wanted
- This prevents stale limit orders filling on gap opens days later

### 2. CNC Short Restriction (MEDIUM)

Zerodha prohibits short selling via CNC product type. Only MIS allows equity shorts (intraday only, covered by auto-squareoff).

**Changes to `engine.rs`:**
In signal processing, before submitting an order:
```rust
if signal.action == Action::Sell && signal.product_type == ProductType::Cnc {
    let has_long = portfolio.has_position(&signal.symbol, false); // false = not short
    if !has_long {
        // Reject: CNC short not allowed
        rejections.push(OrderRejection {
            symbol: signal.symbol,
            side: Side::Sell,
            quantity: signal.quantity,
            reason: "CNC_SHORT_NOT_ALLOWED".into(),
        });
        continue;
    }
}
```

Note: Selling an EXISTING CNC long position is fine (closing a position). Only opening a NEW short via CNC is blocked.

**Changes to strategies:**
- `sma_crossover.py`: Force `product_type="MIS"` for all short entries (death cross). Currently uses dynamic CNC/MIS which could assign CNC to shorts.
- `donchian_breakout.py`: Same — force MIS for channel-low short entries.
- `rsi_daily_trend.py`: Already correct — shorts always use MIS.

### 3. SL Two-Price Model (MEDIUM)

Zerodha SL orders have two prices:
- `trigger_price`: when market hits this price, the order activates
- `price` (limit): the order then becomes a LIMIT order at this price

If price gaps past the limit, the order does NOT fill — it sits as an unfilled limit order. Currently our SL orders treat the single `stop_price` as both trigger and fill price.

**Changes to `types.rs`:**
```rust
pub struct Signal {
    // ... existing fields ...
    pub trigger_price: f64,  // NEW: activation price for SL orders
}
```

**Changes to `matching.rs`:**
For `OrderType::Sl`:
- Current: if `bar.low <= stop_price` (sell) → fill at `stop_price` or `bar.open` on gap
- New: if `bar.low <= order.trigger_price` (sell) → order activates. Then check if `bar.low <= order.limit_price` → fill at `limit_price`. If price gapped past limit, order becomes a pending limit order (doesn't fill this bar).
- For SL buy: mirror logic with `bar.high >= trigger_price`

For `OrderType::SlM`:
- No change — SL-M has only a trigger price, fills at market (bar.open) when triggered.

**Changes to `Order` struct:**
Add `trigger_price: f64` field. For SL orders, `trigger_price` activates, `limit_price` is the fill limit. For SL-M, only `trigger_price` matters.

**Proto changes:**
Add `double trigger_price = 8` to Signal message. Update Python Signal dataclass.

### 4. Order Modification — Document Only

Not implementing atomic order modification. The cancel + re-place pattern strategies use is adequate for bar-based backtesting. The one-bar latency difference vs. atomic modify is negligible at 15-minute bars.

Document in CLAUDE.md: "Order modification is not supported. Use CANCEL + new order. This adds one bar of latency compared to Zerodha's atomic modify API."

### 5. Per-Order Cancellation (LOW-MEDIUM)

Currently `CANCEL` removes ALL pending orders for a symbol. With engine-managed stops + profit targets, a strategy might have multiple pending orders for the same symbol (SL-M stop + LIMIT profit target) and need to cancel only one.

**Changes to `matching.rs`:**
- Add `order_id: u64` field to `Order` struct
- `OrderMatcher` tracks a counter: `next_order_id: u64`
- On `submit()`: assign `order_id = self.next_order_id; self.next_order_id += 1`
- Add `pub fn cancel_order(&mut self, order_id: u64) -> Option<Order>` — cancel by specific ID
- Existing `cancel_orders_for_symbol` remains for cancel-all behavior

**Changes to `types.rs`:**
Add `cancel_order_id: Option<u64>` to Signal. When set, only that specific order is cancelled.

**Changes to `engine.rs`:**
On `Action::Cancel`:
```rust
if let Some(oid) = signal.cancel_order_id {
    matcher.cancel_order(oid);
} else {
    matcher.cancel_orders_for_symbol(&signal.symbol);
}
```

**Changes to `PendingOrderInfo`:**
Add `order_id: u64` so strategies can see order IDs in `snapshot.pending_orders` and target specific orders for cancellation.

**Proto changes:**
Add `uint64 order_id` to PendingOrderInfo and `uint64 cancel_order_id` to Signal.

**Python:**
Add `order_id: int = 0` to PendingOrder and `cancel_order_id: int = 0` to Signal.

### 6. Cover Orders — Document Only

Not implementing — strategies achieve the same effect by manually pairing entry + SL-M orders. Document in CLAUDE.md.

### 7. IOC Validity (LOW)

Add order validity: `DAY` (default) or `IOC` (immediate-or-cancel).

**Changes to `types.rs`:**
```rust
pub enum OrderValidity {
    Day,
    Ioc,
}
```
Add `validity: OrderValidity` to `Signal` and `Order` (default `Day`).

**Changes to `matching.rs`:**
After `process_bar`, for any IOC order that didn't fill on this bar → cancel it automatically:
```rust
// Post-process: cancel unfilled IOC orders
self.pending_orders.retain(|o| o.validity != OrderValidity::Ioc);
```

**Proto changes:**
Add `OrderValidity` enum and `validity` field to Signal.

**Python:**
Add `validity: str = "DAY"` to Signal. Valid values: `"DAY"`, `"IOC"`.

## Files Impacted

| File | Changes |
|------|---------|
| `engine/crates/core/src/engine.rs` | DAY expiry at 15:30, CNC short rejection, per-order cancel routing |
| `engine/crates/core/src/matching.rs` | cancel_all_pending, order_id, cancel_order, SL two-price, IOC auto-cancel |
| `engine/crates/core/src/types.rs` | trigger_price, cancel_order_id, OrderValidity, order_id on Signal |
| `engine/crates/proto/proto/strategy.proto` | trigger_price, cancel_order_id, order_id, OrderValidity |
| `engine/crates/core/src/grpc_client.rs` | Map new fields |
| `strategies/strategies/base.py` | trigger_price, cancel_order_id, order_id, validity on Signal/PendingOrder |
| `strategies/server/server.py` | Map new fields |
| `strategies/strategies/deterministic/sma_crossover.py` | Force MIS for shorts |
| `strategies/strategies/deterministic/donchian_breakout.py` | Force MIS for shorts |
| `CLAUDE.md` | Document non-implemented gaps |

## Implementation Grouping

| Task | Gaps | Independent? |
|------|------|-------------|
| A: DAY validity + IOC | 1, 7 | Yes |
| B: CNC short restriction | 2 | Yes |
| C: SL two-price model | 3 | Yes |
| D: Per-order cancellation | 5 | Yes |
| E: Documentation | 4, 6 | Yes (after A-D) |

All code tasks (A-D) are independent and can run in parallel.

## Verification

```bash
cd engine && cargo test                    # all Rust tests pass
cd strategies && pytest tests/ -v          # all Python tests pass
# Run backtests to verify no regression
```
