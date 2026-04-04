# Production-Grade Engine Gaps — Design Document

Date: 2026-04-04

## Goal

Build the backtesting engine to production-grade quality for equity research. 17 features across execution, simulation, risk, reporting, and data. Equities only, no F&O/options in this phase. Practical realism — no tick-level sim, market impact, or auction sessions.

## Scope

### In (17 features)

| # | Category | Feature |
|---|----------|---------|
| 1 | Execution | Short selling (negative positions) |
| 2 | Execution | MIS auto-squareoff at 3:20 PM IST |
| 3 | Execution | Order cancellation |
| 4 | Execution | Gap handling (fill at gap price) |
| 5 | Simulation | Volume constraints (max % of bar volume) |
| 6 | Simulation | Trading calendar (NSE holidays + weekends) |
| 7 | Risk | Max drawdown kill switch |
| 8 | Risk | Daily loss limit |
| 9 | Risk | Per-symbol position limit |
| 10 | Risk | Portfolio exposure limit |
| 11 | Reporting | Per-symbol breakdown |
| 12 | Reporting | Monthly returns table |
| 13 | Reporting | Benchmark comparison (buy-and-hold) |
| 14 | Reporting | Trade duration statistics |
| 15 | Data | Corporate actions (split/bonus adjustment) |
| 16 | Data | Index data as non-tradable reference |
| 17 | Data | Dividend adjustments |

### Out (deferred)

- Options/Greeks/chains, live/paper trading, tick-level simulation, market impact modeling, auction sessions, bracket/cover orders, partial fills, chart generation

---

## Feature Designs

### 1. Short Selling

**Current**: `PortfolioManager` rejects sells without a position. Positions are always positive.

**Change**: Positions can be negative (short).

- `InternalPosition.quantity` uses signed semantics (Rust `i32`)
- `apply_sell` without position → creates short: `quantity = -fill.quantity`, `avg_price = fill.fill_price`
- `apply_buy` against short → closes short (records `ClosedTrade` with `direction: Short`), excess creates long
- `ClosedTrade` gets `direction: Direction` enum (`Long | Short`)
- Short P&L = `(entry_price - exit_price) * qty` (profit when price drops)
- `equity()` = cash + sum(qty * current_price) — short positions have negative market value
- `update_prices` handles negative qty: unrealized PnL = `(avg_price - current_price) * abs(qty)` for shorts

**Files**: `portfolio.rs`, `types.rs`

### 2. MIS Auto-Squareoff

**Change**: At 3:20 PM IST, all MIS positions are force-closed.

- Add `product_type: ProductType` to `InternalPosition` (set from `Fill.product_type` or `Order.product_type`)
- In main loop, after processing each timestamp:
  - Parse timestamp to IST time-of-day
  - If time >= 15:20 IST, iterate positions where `product_type == MIS`
  - For each, submit a market order to close (buy to cover shorts, sell to close longs)
  - Tag resulting `ClosedTrade` with `auto_squareoff: true` for reporting
- Requires `Fill` to carry `product_type` — already available on the `Order` struct, propagate through to Fill

**Files**: `portfolio.rs`, `engine.rs`, `matching.rs`

### 3. Order Cancellation

**Change**: Strategies can cancel pending orders.

- Add `Cancel` variant to `Action` enum: `Action::Cancel`
- When `signal.action == Action::Cancel`, engine calls `matcher.cancel_orders_for_symbol(&signal.symbol)`
- `OrderMatcher` gets `cancel_orders_for_symbol(symbol: &str) -> Vec<Order>` — removes and returns cancelled orders
- Add `pending_orders: Vec<PendingOrder>` to `MarketSnapshot` so strategies see what's pending
- `PendingOrder` struct: symbol, side, quantity, order_type, limit_price, stop_price
- Proto: add `CANCEL = 3` to `Action` enum, add `repeated PendingOrderInfo pending_orders` to `BarEvent`
- Python: add `Cancel` to action map, add `PendingOrder` dataclass to `base.py`

**Files**: `types.rs`, `matching.rs`, `engine.rs`, `grpc_client.rs`, `strategy.proto`, `base.py`, `server.py`

### 4. Gap Handling

**Change**: Orders that gap through their trigger fill at `bar.open` instead of the trigger price.

- Limit buy at 1000, bar opens at 980 → fill at 980 (price improvement)
- Limit sell at 1000, bar opens at 1020 → fill at 1020 (price improvement)
- SL sell at 1000, bar opens at 950 → fill at 950 (slippage, worse than stop)
- SL buy at 1000, bar opens at 1050 → fill at 1050 (slippage, worse than stop)

Implementation in `matching.rs`:
- Market orders: unchanged (already fill at open ± slippage)
- Limit buy: `fill_price = min(bar.open, order.limit_price)` when `bar.low <= limit`
- Limit sell: `fill_price = max(bar.open, order.limit_price)` when `bar.high >= limit`
- SL sell: if `bar.open < order.stop_price` (gapped below), fill at `bar.open * (1 - slippage)` instead of stop_price
- SL buy: if `bar.open > order.stop_price` (gapped above), fill at `bar.open * (1 + slippage)`
- SL-M: already fills at bar.open with slippage (fixed in earlier audit)

**Files**: `matching.rs`

### 5. Volume Constraints

**Change**: Limit fill quantity to a percentage of bar volume.

- Add `max_volume_pct: f64` to `OrderMatcher` (default 0.10 = 10%)
- Add to `BacktestConfig` and CLI as `--max-volume-pct`
- During fill: `effective_qty = min(order.quantity, (bar.volume as f64 * self.max_volume_pct) as i32).max(1)`
- If clamped, create a new pending order for the remainder: `remaining = order.quantity - effective_qty`
- The remaining order stays in `pending_orders` for the next bar

**Files**: `matching.rs`, `config.rs`, `run.rs`

### 6. Trading Calendar

**Change**: NSE holiday calendar for validation and test data generation.

- New module `engine/crates/core/src/calendar.rs`
- `TradingCalendar` struct with hardcoded NSE holidays 2020-2027 (~120 dates)
- `is_trading_day(date: NaiveDate) -> bool` — false for Sat/Sun + holidays
- `next_trading_day(date: NaiveDate) -> NaiveDate`
- Test data generator (`data.rs`) uses calendar to skip non-trading days
- Engine can optionally validate that bars don't fall on holidays (warning only, don't reject)

**Files**: `calendar.rs` (new), `data.rs`

### 7. Max Drawdown Kill Switch

**Change**: Stop trading when portfolio drawdown exceeds a threshold.

- Add `max_drawdown_pct: Option<f64>` to `BacktestConfig` and CLI `--max-drawdown`
- Engine tracks `peak_equity` and computes `drawdown = (peak - current) / peak`
- When `drawdown > max_drawdown_pct`:
  - Force-close all positions at current bar's close
  - Set `killed = true` flag — skip all future `on_bar` calls
  - Record `kill_reason: String` in `BacktestResult`
- Reported in CLI output: `  Kill Switch:    Triggered at bar 234 (drawdown 10.5%)`

**Files**: `config.rs`, `engine.rs`, `run.rs`

### 8. Daily Loss Limit

**Change**: Limit losses per trading day.

- Add `daily_loss_limit: Option<f64>` to `BacktestConfig` and CLI `--daily-loss-limit`
- Engine tracks `day_start_equity` — reset at each new trading day (detect via timestamp date change)
- When `day_start_equity - current_equity > daily_loss_limit`:
  - Reject all new buy signals for the rest of the day
  - Force-close MIS positions
  - Log warning
- Resets at next day's first bar

**Files**: `config.rs`, `engine.rs`, `run.rs`

### 9. Per-Symbol Position Limit

**Change**: Clamp maximum position per symbol.

- Add `max_position_qty: Option<i32>` to `BacktestConfig` and CLI `--max-position-qty`
- During order submission in engine loop:
  - Current qty = portfolio's position for signal.symbol (0 if none)
  - If `abs(current_qty + order_qty) > max_position_qty`, clamp order qty
  - For shorts: same logic applies to negative side
- Generate `OrderRejection` if fully clamped to 0

**Files**: `config.rs`, `engine.rs`, `run.rs`

### 10. Portfolio Exposure Limit

**Change**: Cap total portfolio exposure as % of capital.

- Add `max_exposure_pct: Option<f64>` to `BacktestConfig` and CLI `--max-exposure`
- Total exposure = sum of `abs(qty * current_price)` for all positions
- Before submitting a buy/short order: check if `exposure + new_trade_value > capital * max_exposure_pct`
- If exceeded: reject with `EXPOSURE_LIMIT` reason
- Replaces `margin_available` which does similar but less cleanly. Keep both for backwards compat but document `max_exposure_pct` as preferred.

**Files**: `config.rs`, `engine.rs`, `run.rs`

### 11. Per-Symbol Breakdown

**Change**: Compute and report metrics per symbol.

- New struct `SymbolMetrics` in `metrics.rs`: total_trades, wins, losses, win_rate, total_pnl, avg_pnl, max_drawdown
- `MetricsReport` gets `per_symbol: HashMap<String, SymbolMetrics>`
- Group `ClosedTrade` list by symbol, compute stats per group
- Reporter saves `per_symbol` in `metrics.json`
- CLI `results show` prints per-symbol table

**Files**: `metrics.rs`, `reporter.rs`, CLI results display

### 12. Monthly Returns Table

**Change**: Compute returns per calendar month.

- New struct `MonthlyReturn { year: i32, month: u32, return_pct: f64, equity_start: f64, equity_end: f64 }`
- Group equity curve by YYYY-MM, compute `(end - start) / start` per month
- `MetricsReport` gets `monthly_returns: Vec<MonthlyReturn>`
- CLI prints monthly table

**Files**: `metrics.rs`, `reporter.rs`

### 13. Benchmark Comparison

**Change**: Compare strategy returns against buy-and-hold.

- Add `--benchmark` flag to CLI (default: `buy_and_hold`)
- Buy-and-hold benchmark: simulate buying equal-weight of all symbols at start, holding to end
  - `benchmark_return = (sum of final prices / sum of start prices) - 1`
  - Computed from `bars_by_interval` first/last bars per symbol
- `MetricsReport` gets `benchmark_return_pct: f64` and `alpha_pct: f64`
- CLI prints: `  Benchmark:      +5.2% (buy & hold)` and `  Alpha:          -2.1%`

**Files**: `metrics.rs`, `config.rs`, `run.rs`

### 14. Trade Duration Statistics

**Change**: Add duration metrics to trade statistics.

- Add to `TradeStatistics`: `avg_duration_bars`, `avg_duration_ms`, `min_duration_ms`, `max_duration_ms`
- Computed from `(exit_timestamp_ms - entry_timestamp_ms)` on each `ClosedTrade`
- Bars duration: count bars between entry and exit timestamps

**Files**: `metrics.rs`

### 15. Corporate Actions (Split/Bonus)

**Change**: Adjust historical prices for splits and bonuses.

- New struct `CorporateAction { symbol, exchange, date, action_type: Split|Bonus|Dividend, ratio: f64 }`
- New table in `instruments.db`: `corporate_actions (symbol TEXT, exchange TEXT, date TEXT, action_type TEXT, ratio REAL)`
- New CLI command: `backtest data import-corporate-actions --file actions.csv`
- CSV format: `symbol,exchange,date,type,ratio` (e.g., `RELIANCE,NSE,2024-09-05,SPLIT,2.0`)
- `CandleStore::read()` gets optional `adjust_corporate_actions: bool` parameter
- Adjustment: for each action, multiply pre-action OHLC by `1/ratio`, multiply volume by `ratio`
- Applied in `run.rs` after reading bars, before passing to engine
- `InstrumentStore` gets `get_corporate_actions(symbol, exchange) -> Vec<CorporateAction>`

**Files**: `instruments.rs`, `candles.rs` or `run.rs`, `data.rs` (CLI command)

### 16. Index as Non-Tradable Reference

**Change**: Strategies can see index data without trading it.

- Add `--reference-symbols` flag to CLI run (e.g., `--reference-symbols "NIFTY 50"`)
- CLI loads bars for reference symbols same as regular symbols
- Pass reference symbol list to engine alongside regular symbols
- Engine includes reference bars in `MarketSnapshot.timeframes` and `history`
- Engine ignores any signals where `signal.symbol` is a reference symbol (non-tradable)
- Strategies see index data in `snapshot.timeframes["day"]["NIFTY 50"]` for regime detection

**Files**: `run.rs`, `engine.rs`, `config.rs`

### 17. Dividend Adjustments

**Change**: Adjust prices for dividends.

- Handled via the corporate actions pipeline (feature 15)
- `action_type = Dividend`, `ratio = dividend_per_share`
- Adjustment: subtract dividend amount from all OHLC prices before ex-date
- Volume unchanged for dividends
- Combined with split/bonus adjustment — same table, same read path

**Files**: Same as feature 15

---

## Implementation Grouping

Features naturally cluster by the files they touch:

| Group | Features | Primary Files |
|-------|----------|---------------|
| A: Position Model | 1 (short), 2 (squareoff) | portfolio.rs, engine.rs |
| B: Order Mechanics | 3 (cancel), 4 (gaps), 5 (volume) | matching.rs, types.rs, proto |
| C: Risk Controls | 7 (kill switch), 8 (daily limit), 9 (symbol limit), 10 (exposure) | engine.rs, config.rs |
| D: Calendar | 6 (holidays) | calendar.rs (new), data.rs |
| E: Reporting | 11 (per-symbol), 12 (monthly), 13 (benchmark), 14 (duration) | metrics.rs, reporter.rs |
| F: Data Pipeline | 15 (corporate actions), 16 (reference), 17 (dividends) | instruments.rs, candles.rs, run.rs |

Groups A-D modify the engine core and should be done sequentially or with careful coordination. Group E (reporting) is fully independent. Group F (data) is mostly independent but feature 16 touches engine.rs.

---

## Verification

After each group:
```bash
cd engine && cargo test              # all Rust tests pass
cd strategies && pytest tests/ -v    # all Python tests pass
```

After all groups:
```bash
# Run all strategies on 10 stocks for 6 months
SYMBOLS="RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC"
backtest run --strategy rsi_daily_trend --symbols $SYMBOLS \
  --from 2024-01-01 --to 2024-06-30 --capital 1000000 --interval 15minute \
  --max-drawdown 0.10 --daily-loss-limit 20000 --max-volume-pct 0.10

# Verify per-symbol breakdown, monthly returns, benchmark in output
backtest results show <id>
```
