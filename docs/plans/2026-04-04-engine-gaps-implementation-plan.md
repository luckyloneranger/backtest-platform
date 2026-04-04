# Production-Grade Engine Gaps — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build 17 features across execution, simulation, risk, reporting, and data to bring the backtesting engine to production quality for equity research.

**Architecture:** Features are grouped into 6 implementation tasks (A-F) based on file dependencies. Groups A-D modify engine core, Group E is reporting (independent), Group F is data pipeline (mostly independent). Each task is a committable unit with tests.

**Tech Stack:** Rust (engine), Python (strategies), gRPC/Proto (interface), chrono (dates/times)

**Design doc:** `docs/plans/2026-04-04-engine-gaps-design.md`

---

### Task A: Short Selling + MIS Auto-Squareoff (Features 1, 2)

**Files:**
- Modify: `engine/crates/core/src/types.rs` — add `Direction` enum
- Modify: `engine/crates/core/src/portfolio.rs` — negative positions, short P&L, product_type tracking
- Modify: `engine/crates/core/src/engine.rs` — auto-squareoff logic at 15:20 IST
- Modify: `engine/crates/core/src/matching.rs` — propagate product_type from Order to Fill

**Changes:**

1. **types.rs**: Add `Direction` enum:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
   pub enum Direction { Long, Short }
   ```

2. **portfolio.rs — Short positions**:
   - Add `product_type: ProductType` to `InternalPosition`
   - Add `direction: Direction` to `ClosedTrade`
   - Rewrite `apply_sell` with no position → create short: `InternalPosition { quantity: fill.quantity, avg_price: fill.fill_price, ..., product_type }`. Cash is credited (short sale proceeds).
   - Rewrite `apply_buy` to handle closing shorts: if position has negative-semantic (is_short flag or separate short tracking), closing a short means buying back. Record `ClosedTrade` with `direction: Short`, P&L = `(entry_price - exit_price) * qty`.
   - **Implementation note**: Rather than negative quantity (which complicates all qty math), use a `is_short: bool` field on `InternalPosition`. This keeps quantity always positive but tracks direction.
   - `equity()`: for shorts, position value = `-(qty * current_price)` since we owe shares. But we also received `qty * avg_price` as cash when shorting. So net equity contribution = `qty * (avg_price - current_price)`.
   - `unrealized_pnl` for shorts: `(avg_price - current_price) * qty`

3. **matching.rs — Product type on Fill**:
   - `Fill` already has `costs: f64`. Add `product_type: ProductType`.
   - Set from `order.product_type` during fill construction in `process_bar`.

4. **engine.rs — MIS Auto-squareoff**:
   - After all normal processing for a timestamp, check time-of-day:
     ```rust
     let ist_offset = chrono::FixedOffset::east_opt(19800).unwrap();
     let dt = chrono::DateTime::from_timestamp_millis(*timestamp)
         .unwrap()
         .with_timezone(&ist_offset);
     let time = dt.time();
     let squareoff_time = chrono::NaiveTime::from_hms_opt(15, 20, 0).unwrap();
     ```
   - If `time >= squareoff_time`, iterate portfolio positions where `product_type == Mis`
   - For each, submit a closing market order (SELL for longs, BUY for shorts)
   - Track `auto_squareoff_count` in BacktestResult for reporting

**Tests to add:**
- `test_short_sell_creates_short_position` — sell without position, verify negative position
- `test_buy_closes_short_position` — sell then buy, verify ClosedTrade with direction Short
- `test_short_pnl_correct` — short at 100, cover at 90, PnL = +10 * qty - costs
- `test_short_equity_calculation` — short position reduces equity when price rises
- `test_mis_auto_squareoff` — MIS position exists at 15:20, verify force-close
- `test_cnc_not_squaredoff` — CNC position at 15:20, verify NOT closed
- `test_product_type_on_fill` — verify fill carries product_type from order

**Verify:** `cd engine && cargo test -p backtest-core`

**Commit:** `feat: short selling support and MIS auto-squareoff at 15:20 IST`

---

### Task B: Order Mechanics — Cancellation, Gaps, Volume (Features 3, 4, 5)

**Files:**
- Modify: `engine/crates/core/src/types.rs` — add `Cancel` to Action enum
- Modify: `engine/crates/core/src/matching.rs` — gap fills, volume constraints, cancel method
- Modify: `engine/crates/core/src/engine.rs` — handle Cancel signals, pass pending orders in snapshot
- Modify: `engine/crates/core/src/config.rs` — add `max_volume_pct`
- Modify: `engine/crates/proto/proto/strategy.proto` — Cancel action, PendingOrderInfo
- Modify: `engine/crates/core/src/grpc_client.rs` — pending orders in snapshot
- Modify: `strategies/strategies/base.py` — PendingOrder dataclass
- Modify: `strategies/server/server.py` — Cancel action mapping, pending orders conversion

**Changes:**

1. **types.rs**: Add `Cancel` to Action: `enum Action { Hold, Buy, Sell, Cancel }`

2. **matching.rs — Gap handling**:
   - Limit buy: if bar gaps below limit (`bar.open < order.limit_price` AND `bar.low <= order.limit_price`), fill at `bar.open` not `limit_price`
   - Limit sell: if bar gaps above limit (`bar.open > order.limit_price` AND `bar.high >= order.limit_price`), fill at `bar.open`
   - SL sell: if `bar.open < order.stop_price` (gapped through), fill at `bar.open * (1 - slippage)` — already handled for SL-M in earlier fix, extend to SL
   - SL buy: if `bar.open > order.stop_price`, fill at `bar.open * (1 + slippage)`

3. **matching.rs — Volume constraints**:
   - Add `max_volume_pct: f64` field to `OrderMatcher` (constructor takes it)
   - During fill: `let max_qty = (bar.volume as f64 * self.max_volume_pct).max(1.0) as i32;`
   - If `order.quantity > max_qty`: fill for `max_qty`, create new pending order for remainder
   - Default `max_volume_pct = 1.0` (no constraint) for backwards compat. Set via config.

4. **matching.rs — Order cancellation**:
   - Add `pub fn cancel_orders_for_symbol(&mut self, symbol: &str) -> Vec<Order>` — removes from pending_orders
   - Add `pub fn pending_orders(&self) -> &[Order]` — read access for snapshot

5. **engine.rs — Cancel signal handling + pending orders in snapshot**:
   - In signal processing: if `signal.action == Action::Cancel`, call `matcher.cancel_orders_for_symbol(&signal.symbol)`
   - Add pending orders to MarketSnapshot (new field `pending_orders: Vec<PendingOrderInfo>`)
   - Build from `matcher.pending_orders()` each tick

6. **config.rs**: Add `max_volume_pct: f64` with `#[serde(default = "default_max_volume_pct")]` (default 1.0)

7. **Proto + Python**: Add `CANCEL = 3` to Action enum. Add `PendingOrderInfo` message. Add `pending_orders` to `BarEvent`. Update Python dataclass and server.py mappings.

**Tests to add:**
- `test_limit_buy_gap_fills_at_open` — limit buy at 100, bar opens at 95, fills at 95
- `test_limit_sell_gap_fills_at_open` — limit sell at 100, bar opens at 105, fills at 105
- `test_sl_gap_fills_at_open` — SL sell at 100, bar opens at 90, fills at 90*(1-slippage)
- `test_volume_constraint_clamps_qty` — order for 1000, bar volume 100, max_pct 0.1 → fills 10, 990 remains pending
- `test_cancel_removes_pending_orders` — submit limit, cancel, verify pending empty
- `test_pending_orders_visible_in_snapshot` — submit limit, verify appears in snapshot

**Verify:** `cd engine && cargo test` + `cd strategies && pytest tests/ -v`

**Commit:** `feat: order cancellation, gap handling, volume constraints`

---

### Task C: Risk Controls (Features 7, 8, 9, 10)

**Files:**
- Modify: `engine/crates/core/src/config.rs` — new risk config fields
- Modify: `engine/crates/core/src/engine.rs` — risk checks in event loop
- Modify: `engine/crates/cli/src/commands/run.rs` — CLI flags

**Changes:**

1. **config.rs**: Add fields (all `Option`, serde default None/sensible):
   ```rust
   pub max_drawdown_pct: Option<f64>,      // e.g., 0.10 = 10%
   pub daily_loss_limit: Option<f64>,       // e.g., 20000.0 = ₹20k
   pub max_position_qty: Option<i32>,       // e.g., 500
   pub max_exposure_pct: Option<f64>,       // e.g., 0.80 = 80%
   pub max_volume_pct: f64,                 // from Task B, default 1.0
   ```

2. **engine.rs — Kill switch (max drawdown)**:
   - Track `peak_equity` and `killed` flag before main loop
   - Each bar: update `peak_equity = peak_equity.max(current_equity)`
   - Compute `drawdown = (peak_equity - current_equity) / peak_equity`
   - If `drawdown > max_drawdown_pct`: force-close all positions via matcher, set `killed = true`
   - When `killed`, skip `on_bar` and signal processing for remaining bars
   - Record `kill_reason: Option<String>` in `BacktestResult`

3. **engine.rs — Daily loss limit**:
   - Track `day_start_equity` and `daily_limit_hit` flag
   - At each bar: detect day change via timestamp (different calendar day)
   - On new day: reset `day_start_equity = current_equity`, clear `daily_limit_hit`
   - If `day_start_equity - current_equity > daily_loss_limit`: set `daily_limit_hit = true`
   - When `daily_limit_hit`: reject all Buy signals (still allow Sells), force-close MIS positions

4. **engine.rs — Per-symbol position limit**:
   - Before submitting order: check `portfolio.position_qty(signal.symbol)`
   - If `current_qty + signal.quantity > max_position_qty`: clamp signal quantity
   - If fully clamped to 0: generate `OrderRejection` with reason `POSITION_LIMIT`

5. **engine.rs — Exposure limit**:
   - Before submitting buy: compute total exposure = `sum(abs(qty * price))` for all positions
   - If `exposure + new_trade_value > initial_capital * max_exposure_pct`: reject with `EXPOSURE_LIMIT`

6. **run.rs — CLI flags**:
   ```rust
   #[arg(long)] pub max_drawdown: Option<f64>,
   #[arg(long)] pub daily_loss_limit: Option<f64>,
   #[arg(long)] pub max_position_qty: Option<i32>,
   #[arg(long)] pub max_exposure: Option<f64>,
   #[arg(long, default_value = "1.0")] pub max_volume_pct: f64,
   ```

**Tests to add:**
- `test_kill_switch_stops_trading` — set max_drawdown 5%, force 6% loss, verify no more on_bar calls
- `test_kill_switch_closes_positions` — verify all positions closed when triggered
- `test_daily_loss_limit_blocks_buys` — exceed daily limit, verify Buy rejected but Sell allowed
- `test_daily_loss_resets_next_day` — exceed limit, next day buys work again
- `test_position_limit_clamps_qty` — max 100, already hold 80, buy 50 → clamped to 20
- `test_exposure_limit_rejects_buy` — exposure at 79%, buy would push to 85%, max is 80% → rejected

**Verify:** `cd engine && cargo test -p backtest-core`

**Commit:** `feat: risk controls — kill switch, daily loss, position/exposure limits`

---

### Task D: Trading Calendar (Feature 6)

**Files:**
- Create: `engine/crates/core/src/calendar.rs`
- Modify: `engine/crates/core/src/lib.rs` — add `pub mod calendar`
- Modify: `engine/crates/cli/src/commands/data.rs` — skip holidays in test data generator

**Changes:**

1. **calendar.rs**:
   ```rust
   pub struct TradingCalendar {
       holidays: HashSet<NaiveDate>,
   }

   impl TradingCalendar {
       pub fn nse() -> Self { /* hardcoded NSE holidays 2020-2027 */ }
       pub fn is_trading_day(&self, date: NaiveDate) -> bool { /* !weekend && !holiday */ }
       pub fn next_trading_day(&self, date: NaiveDate) -> NaiveDate { /* advance until trading day */ }
   }
   ```
   - Hardcode ~120 NSE holidays (Diwali, Republic Day, Independence Day, Holi, Good Friday, Eid, Christmas, etc. for 2020-2027)

2. **data.rs**: In `handle_generate_test_data`, skip days where `!calendar.is_trading_day(date)`

**Tests to add:**
- `test_republic_day_is_holiday` — Jan 26 2024 is not a trading day
- `test_weekend_not_trading_day` — Saturday/Sunday return false
- `test_normal_monday_is_trading_day`
- `test_next_trading_day_skips_weekend` — Friday → next is Monday

**Verify:** `cd engine && cargo test -p backtest-core -- calendar`

**Commit:** `feat: NSE trading calendar with holidays 2020-2027`

---

### Task E: Reporting (Features 11, 12, 13, 14)

**Files:**
- Modify: `engine/crates/core/src/metrics.rs` — per-symbol, monthly, benchmark, duration
- Modify: `engine/crates/core/src/reporter.rs` — save new metrics
- Modify: `engine/crates/cli/src/commands/run.rs` — benchmark flag
- Modify: `engine/crates/cli/src/commands/results.rs` — display new metrics

**Changes:**

1. **metrics.rs — Per-symbol breakdown**:
   ```rust
   pub struct SymbolMetrics {
       pub total_trades: usize,
       pub winning_trades: usize,
       pub losing_trades: usize,
       pub win_rate: f64,
       pub total_pnl: f64,
       pub avg_pnl: f64,
   }
   ```
   - Add `per_symbol: HashMap<String, SymbolMetrics>` to `MetricsReport`
   - Compute by grouping `closed_trades` by symbol

2. **metrics.rs — Monthly returns**:
   ```rust
   pub struct MonthlyReturn {
       pub year: i32,
       pub month: u32,
       pub return_pct: f64,
   }
   ```
   - Add `monthly_returns: Vec<MonthlyReturn>` to `MetricsReport`
   - Group equity curve by YYYY-MM, compute `(month_end - month_start) / month_start`

3. **metrics.rs — Benchmark**:
   - Add `benchmark_return_pct: f64` and `alpha_pct: f64` to `MetricsReport`
   - Buy-and-hold: compute from first/last bar prices per symbol, equal-weight
   - Need bars data passed to `MetricsReport::compute()` — add parameter or compute in `run.rs`

4. **metrics.rs — Trade duration**:
   - Add to `TradeStatistics`: `avg_duration_ms: f64`, `min_duration_ms: i64`, `max_duration_ms: i64`
   - Compute from `exit_timestamp_ms - entry_timestamp_ms` for each `ClosedTrade`

5. **reporter.rs**: Serialize new fields into `metrics.json`

6. **results.rs**: Print per-symbol table, monthly returns, benchmark/alpha in `results show`

7. **run.rs**: Add `--benchmark` flag (default `buy_and_hold`). Compute benchmark return from bars_by_interval.

**Tests to add:**
- `test_per_symbol_metrics` — 3 trades across 2 symbols, verify correct grouping
- `test_monthly_returns` — equity curve spanning 3 months, verify monthly returns
- `test_benchmark_buy_and_hold` — known prices, verify benchmark return
- `test_trade_duration` — trades with known timestamps, verify duration stats

**Verify:** `cd engine && cargo test -p backtest-core -- metrics`

**Commit:** `feat: per-symbol breakdown, monthly returns, benchmark comparison, trade duration`

---

### Task F: Data Pipeline — Corporate Actions, Reference Symbols, Dividends (Features 15, 16, 17)

**Files:**
- Modify: `engine/crates/data/src/instruments.rs` — corporate_actions table
- Modify: `engine/crates/data/src/candles.rs` — price adjustment function
- Modify: `engine/crates/cli/src/commands/data.rs` — import-corporate-actions command
- Modify: `engine/crates/cli/src/commands/run.rs` — adjust bars, reference symbols
- Modify: `engine/crates/core/src/engine.rs` — reference symbol filtering
- Modify: `engine/crates/core/src/config.rs` — reference_symbols field

**Changes:**

1. **instruments.rs — Corporate actions table**:
   ```rust
   pub struct CorporateAction {
       pub symbol: String,
       pub exchange: String,
       pub date: String,           // YYYY-MM-DD
       pub action_type: String,    // "SPLIT", "BONUS", "DIVIDEND"
       pub ratio: f64,             // 2.0 for 1:2 split, 5.0 for ₹5 dividend
   }
   ```
   - Create `corporate_actions` table in `ensure_schema`
   - Add `insert_corporate_action()` and `get_corporate_actions(symbol, exchange) -> Vec<CorporateAction>`

2. **candles.rs or new module — Price adjustment**:
   ```rust
   pub fn adjust_for_corporate_actions(bars: &mut Vec<Bar>, actions: &[CorporateAction]) {
       for action in actions {
           let action_date_ms = parse_date_to_ms(&action.date);
           for bar in bars.iter_mut() {
               if bar.timestamp_ms < action_date_ms {
                   match action.action_type.as_str() {
                       "SPLIT" | "BONUS" => {
                           bar.open /= action.ratio;
                           bar.high /= action.ratio;
                           bar.low /= action.ratio;
                           bar.close /= action.ratio;
                           bar.volume = (bar.volume as f64 * action.ratio) as i64;
                       }
                       "DIVIDEND" => {
                           bar.open -= action.ratio;
                           bar.high -= action.ratio;
                           bar.low -= action.ratio;
                           bar.close -= action.ratio;
                       }
                       _ => {}
                   }
               }
           }
       }
   }
   ```

3. **data.rs — CLI import command**:
   - New subcommand: `backtest data import-corporate-actions --file actions.csv`
   - CSV format: `symbol,exchange,date,type,ratio`
   - Parse and insert into instruments.db

4. **run.rs — Apply adjustments + reference symbols**:
   - After reading bars, load corporate actions from InstrumentStore
   - Call `adjust_for_corporate_actions()` on each symbol's bars
   - Add `--reference-symbols` flag (comma-separated)
   - Load bars for reference symbols, include in bars_by_interval
   - Pass reference symbol list to engine via config

5. **config.rs**: Add `reference_symbols: Vec<String>` with `#[serde(default)]`

6. **engine.rs — Reference symbol filtering**:
   - In signal processing: skip signals where `signal.symbol` is in `config.reference_symbols`
   - Reference symbols' bars still appear in `MarketSnapshot.timeframes` and history

**Tests to add:**
- `test_split_adjustment` — bar at ₹1000 before 1:2 split becomes ₹500, volume doubles
- `test_dividend_adjustment` — bar at ₹1000 before ₹10 dividend becomes ₹990
- `test_corporate_actions_db_roundtrip` — insert and query back
- `test_reference_symbol_in_snapshot_but_not_traded` — reference symbol bars visible, signals ignored
- `test_import_csv` — parse CSV, verify DB entries

**Verify:** `cd engine && cargo test`

**Commit:** `feat: corporate actions, dividend adjustments, reference symbols`

---

## Execution Order & Dependencies

```
Task D (Calendar)           ── Independent, small, do first
Task E (Reporting)          ── Independent, do in parallel with D
Task A (Short + Squareoff)  ── Core position model change, do next
Task B (Order Mechanics)    ── Depends on A (product_type on Fill), do after A
Task C (Risk Controls)      ── Depends on A+B (short positions, exposure calc), do after B
Task F (Data Pipeline)      ── Mostly independent, do last (touches engine.rs which A-C modify)
```

Recommended parallel batches:
- **Batch 1**: D + E (independent, can run simultaneously)
- **Batch 2**: A (core position model, needs careful review)
- **Batch 3**: B + proto regen (depends on A)
- **Batch 4**: C (depends on A+B)
- **Batch 5**: F (data pipeline, after engine stabilizes)

## Final Verification

```bash
cd engine && cargo test                    # all Rust tests pass (expect ~130+)
cd strategies && pytest tests/ -v          # all Python tests pass (60+)

# Integration test with all features:
SYMBOLS="RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC"
backtest run --strategy rsi_daily_trend --symbols $SYMBOLS \
  --reference-symbols "NIFTY 50" \
  --from 2024-01-01 --to 2024-06-30 --capital 1000000 --interval 15minute \
  --max-drawdown 0.10 --daily-loss-limit 20000 --max-volume-pct 0.10 \
  --max-exposure 0.80

backtest results show <id>
# Verify: per-symbol table, monthly returns, benchmark/alpha, trade duration
```
