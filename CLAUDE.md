# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

### Rust Engine
```bash
cd engine
cargo build                          # build all crates
cargo build --release -p backtest-cli # release build of CLI binary
cargo test                           # run all tests (176 tests across workspace)
cargo test -p backtest-core           # test single crate (149 tests)
cargo test -p backtest-core -- matching  # test single module
```

### Python Strategy Server
```bash
cd strategies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./generate_proto.sh                  # regenerate gRPC stubs from proto
pytest tests/ -v                     # run strategy tests (320 tests)
python -m server.server              # start gRPC server on port 50051
```

### End-to-End
```bash
./tests/e2e_test.sh                  # full pipeline: build → generate data → start server → backtest → results
```

### CLI (after `cargo build --release -p backtest-cli`)
```bash
CLI=./engine/target/release/backtest
$CLI data fetch-instruments                    # fetch all instruments from Kite (must run first)
$CLI data fetch --symbol RELIANCE --from 2024-01-01 --to 2025-03-31 --interval day
$CLI data fetch --symbol RELIANCE --from 2025-03-01 --to 2025-03-31 --interval 5minute  # auto-chunks
$CLI data list
$CLI data import-corporate-actions --file actions.csv  # import split/bonus/dividend data
$CLI run --strategy sma_crossover --symbols RELIANCE --from 2024-01-01 --to 2025-03-31 --capital 1000000 --interval day --exchange NSE --params '{"fast_period": 10, "slow_period": 30}'
$CLI run --strategy rsi_daily_trend --symbols RELIANCE,INFY,TCS --from 2024-01-01 --to 2024-12-31 --capital 1000000 --interval 15minute --max-drawdown 0.10 --max-volume-pct 0.10 --max-exposure 0.80 --reference-symbols "NIFTY 50"
$CLI results list
$CLI results show <backtest_id>
```

## Architecture

Rust engine + Python strategy server communicating via gRPC (port 50051).

### Rust Workspace (`engine/crates/`)

| Crate | Purpose | Key types |
|-------|---------|-----------|
| `backtest-proto` | gRPC proto definitions + codegen | `BarEvent`, `Signal`, `StrategyService`, `DataRequirements` |
| `backtest-core` | Engine, matching, portfolio, costs, metrics, calendar | `BacktestEngine`, `OrderMatcher`, `PortfolioManager`, `ZerodhaCostModel`, `MetricsReport`, `MarketSnapshot`, `TradingCalendar` |
| `backtest-data` | Kite API client, SQLite instruments, Parquet candles | `KiteClient`, `InstrumentStore`, `CandleStore`, `CorporateAction` |
| `backtest-cli` | CLI binary (`backtest`) | Subcommands: `data`, `run`, `results` |

Dependency flow: `cli → data → core → proto`

### Python (`strategies/`)

- `strategies/base.py` — Abstract `Strategy` class with `required_data()`, `initialize()`, `on_bar()`, `on_complete()`. Also defines `MarketSnapshot`, `BarData`, `InstrumentInfo`, `FillInfo`, `OrderRejection`, `TradeInfo`, `SessionContext`, `PendingOrder`.
- `strategies/position_manager.py` — Shared `PositionManager` class handling all order lifecycle: entries (LIMIT/MARKET), engine SL-M stops, trailing stop ratcheting, profit targets, fill detection, DAY expiry re-submission, portfolio reconciliation. Deterministic strategies use this instead of managing orders directly.
- `strategies/indicators.py` — Shared technical indicators backed by pandas-ta: `compute_sma`, `compute_ema`, `compute_rsi`, `compute_atr`, `compute_macd`, `compute_bollinger`, `compute_adx`, `compute_obv`, `compute_obv_slope`, `compute_stochastic`, `compute_bbw`, `compute_zscore`, `compute_correlation`, `compute_cointegration`, `compute_halflife`, `compute_vwap`, `compute_vwap_bands`. All accept `list[float]`, return plain Python types. Strategies never import pandas-ta directly.
- `strategies/narrative_builder.py` — Translates raw indicator values into factual English narratives for LLM consumption (facts only, no interpretive conclusions). 6 functions: `build_symbol_narrative` (per-stock daily data + z-score), `build_intraday_narrative` (VWAP, VWAP bands, 15-min RSI/MACD/ATR, session gap, volume profile), `build_cross_stock_narrative` (correlation matrix, cointegration pairs, halflife, relative z-scores), `build_regime_narrative` (ADX/BBW market state), `build_portfolio_narrative` (drawdown/costs), `build_trade_history_narrative` (win rate, outcomes).
- `strategies/experience_manager.py` — `ExperienceManager` class for reflective learning. Trade journal with thesis/indicators/regime context, missed opportunity detector (flags big moves on unheld stocks), weekly LLM reflection cycle that produces evolving beliefs document (max 10 actionable beliefs), LLM post-mortem on each trade close. Persists to JSON across sessions.
- `strategies/llm_client.py` — `AzureOpenAIClient` wrapper. Reads env vars, calls Azure OpenAI REST API, retry with backoff on HTTP errors and network failures.
- `server/registry.py` — `@register("name")` decorator for strategy discovery
- `server/server.py` — gRPC server: handles `GetRequirements`, `Initialize`, `OnBar`, `OnComplete`
- Deterministic strategies go in `strategies/strategies/deterministic/`, LLM strategies in `strategies/strategies/llm/`, all decorated with `@register`
- **6 deterministic strategies**: `sma_crossover`, `rsi_daily_trend`, `donchian_breakout` (original), `confluence`, `pairs_trading`, `regime_adaptive` (new, pandas-ta powered).
- **2 intraday 5-min strategies**: `vwap_reversion` (VWAP band mean-reversion), `bollinger_squeeze` (volatility squeeze breakout). Both pure MIS with daily state resets.
- **6 advanced strategies**: `orb_breakout` (opening range breakout, 5min), `portfolio_combiner` (Donchian+RSI dynamic by ADX, 15min — **consistently profitable both years**), `intraday_momentum` (3-bar momentum burst, 5min), `time_adaptive` (mode switches by time-of-day, 5min), `relative_strength` (long top 3 / short bottom 3, 15min), `multi_tf_confirm` (daily+15min+5min triple confirmation).
- **3 ML-powered strategies**: `ml_classifier` (GradientBoosting with 20+ features, rolling training), `ou_mean_reversion` (Ornstein-Uhlenbeck process via statsmodels — **passed walk-forward, consistently profitable**), `ensemble_meta` (LogisticRegression meta-learner — failed walk-forward, overfits).
- **1 LLM strategy**: `llm_autonomous_trader` (thesis-driven 15-min multi-timeframe with reflective learning — see below).
- **Strategy tuning insights**: Mean-reversion strategies (RSI) need large positions (`risk_pct=0.3`) and long cooldowns to avoid overtrading. Trend-following strategies (Donchian) benefit from smaller positions (`risk_per_trade=0.01`) and wider stops (`atr_mult=2.0`) to survive drawdowns. Reducing risk_pct from 0.3→0.2 destroyed RSI's edge by generating 4x more trades. Confluence exit speed matters: exiting at score<1 (vs ≤0) improved returns by +6-9% across both years. Regime smoothing (3-bar confirmation) is necessary but insufficient — still 1,400 trades/year. Pairs trading returns near 0% regardless of parameters — NSE stock spreads too tight for mean-reversion after costs. **Parameters that work for trending year (2024) often fail in choppy year (2025) and vice versa.**
- **Walk-forward validation** (train 2024 → test 2025, 12 stocks, ₹1L): 4 strategies passed — Portfolio Combiner (+34%→+382%), RSI (+52%→+121%), OU (+39%→+5%), OU conservative (+10%→+3%). Ensemble Meta failed (+78%→-33% = overfitting). Tuned params available via `tune_and_validate.sh`.
- **Position sizing safety**: ATR-based sizing must be capped to available cash (`qty = min(qty, cash/price)`) to prevent leverage blow-ups at low capital. Without this cap, small 5-min ATR values produce position sizes exceeding account value.
- **Parallel execution**: `run_all_strategies.sh` runs all strategies in parallel, `tune_and_validate.sh` sweeps params with isolated servers, `sweep.sh` for custom sweeps. Server supports `--port` for multi-instance.

### Strategy Data Flow

**Strategy declares requirements → CLI loads data with warmup → Engine ticks active bars → Strategy decides → Engine executes.**

1. Engine calls `GetRequirements` RPC → strategy returns intervals + lookback per interval
2. CLI loads warmup bars (before `--from`) into pre-populated lookback buffers, and active bars (`--from` to `--to`) for the engine to tick through. Corporate action adjustments applied to both.
3. Engine ticks at the finest declared interval over active bars only
4. Each tick: `MarketSnapshot.timeframes` is `{interval → {symbol → BarData}}`. Coarser bars only appear when their candle closes. `history` contains lookback bars (pre-filled from warmup + accumulating during active period). `pending_orders` shows unfilled orders.
5. Strategy returns signals (BUY/SELL/CANCEL) → engine processes orders with risk controls

### Event Loop (`BacktestEngine::run`)

Per timestamp: check kill switch → MIS auto-squareoff at 15:20 IST → DAY order expiry at 15:30 IST → process pending orders (with gap handling, volume constraints, circuit limits) → update portfolio prices → check daily loss limit → build `MarketSnapshot` with all timeframes, lookback, fills (with costs), rejections, pending orders, trades, instruments → call strategy via gRPC → reject MIS orders after 15:15 IST → submit new orders with risk checks (CNC short restriction, margin, position limit, exposure limit). Market orders fill at next bar's open. Limit/SL orders handle gaps (fill at bar.open when gapped through). SL two-price model: trigger_price activates, limit_price fills. SL-M orders fill at market price with slippage. Force-close fills (kill switch, squareoff, daily loss) include slippage. Costs applied per fill via `ZerodhaCostModel`. Short selling supported (negative positions, MIS only).

### Data Storage

- **SQLite** (`./data/instruments.db`): Instrument metadata + corporate actions table (splits, bonuses, dividends)
- **Parquet** (`./data/{EXCHANGE}/{SYMBOL}/{INTERVAL}/data.parquet`): OHLCV+OI candles
- **Results** (`./results/{id}/`): `config.json`, `metrics.json` (includes per-symbol breakdown, monthly returns, benchmark), `trades.parquet` (with direction column), `equity_curve.parquet`

## Key Conventions

- Rust error handling: `anyhow::Result<T>` everywhere, propagate with `?`
- Rust tests: `#[cfg(test)] mod tests` inline in each module, `tempfile::TempDir` for file-based tests
- Proto changes require regenerating both Rust (`cargo build -p backtest-proto`) and Python (`cd strategies && ./generate_proto.sh`) stubs
- `generate_proto.sh` applies a `sed` fix for Python relative imports — don't remove it
- Kite access tokens expire daily; env vars `KITE_API_KEY` and `KITE_ACCESS_TOKEN` must be set for data fetching
- Azure OpenAI env vars for LLM strategies: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- LLM strategies use `AzureOpenAIClient` from `llm_client.py` directly. Use `max_completion_tokens` (not `max_tokens`) for newer Azure models.
- **LLM Autonomous Trader** (`strategies/llm/llm_autonomous_trader.py`): Thesis-driven AI portfolio manager. Multi-timeframe (15min + day). Uses 16 of 17 indicator functions. LLM call throttling: every `llm_interval_bars` (default 4 = hourly). History bootstrap from `snapshot.history` on first `on_bar`. Beliefs from ExperienceManager injected into system prompt. ~1,500-1,600 LLM calls/year.
- **ExperienceManager** (`strategies/experience_manager.py`): Reflective learning layer. Trade journal stores thesis + indicators + regime at entry. Missed opportunity detector flags >2x ATR moves on unheld stocks daily. Weekly reflection LLM call produces evolving beliefs (max 10). Post-mortem mini LLM call on each trade close. Persists to `strategies/data/llm_experience/experience.json`. Config: `experience_dir`, `reset_experience: true` for fresh start, `reflection_interval` (default 5 trading days).
- **Narrative builder** presents FACTS only — no interpretive conclusions ("oversold", "POTENTIAL LONG"). The LLM builds its own thesis from the data. Functions: `build_symbol_narrative` (daily), `build_intraday_narrative` (VWAP/bands/15m indicators), `build_cross_stock_narrative` (correlations/cointegration/halflife/z-scores), `build_regime_narrative`, `build_portfolio_narrative`, `build_trade_history_narrative`.
- Supported intervals: `minute`, `3minute`, `5minute`, `10minute`, `15minute`, `30minute`, `60minute`, `day`
- `fetch_candles_chunked` auto-splits requests to stay under Kite's 2000-candle limit
- The `StrategyClient` trait in `engine.rs` is the abstraction boundary — `GrpcStrategyClient` is the production impl, tests use mock impls
- Shared technical indicators in `strategies/indicators.py` — all strategies import from here, no duplication
- **PositionManager** (`strategies/position_manager.py`): shared order lifecycle class. Deterministic strategies delegate entries, exits, stop management, fill detection, DAY expiry re-submission, and reconciliation to this class. Strategies only implement trading logic (~120-150 lines each). New strategies should use PositionManager rather than managing orders directly.
- Short selling: positions can be negative. Selling without a position creates a short. Buying covers a short. `ClosedTrade.direction` is `Long` or `Short`. CNC shorts are rejected — use MIS for intraday shorts only.
- MIS auto-squareoff: all MIS positions are force-closed at 15:20 IST each trading day (with slippage). Happens BEFORE strategy on_bar call. New MIS orders rejected after 15:15 IST.
- Order cancellation: strategies can send `action=CANCEL` to remove pending limit/SL orders. `cancel_order_id` targets a specific order by ID. Pending orders visible in `MarketSnapshot.pending_orders` with `order_id`.
- Gap handling: limit/SL orders that gap through fill at `bar.open` (not the limit/stop price)
- SL two-price model: `trigger_price` activates the order, `limit_price` is the fill limit. If price gaps past the limit, the SL becomes a pending limit order. When `trigger_price=0`, falls back to `stop_price` behavior. Slippage applied to all SL fills.
- Volume constraints: fills clamped to `max_volume_pct` of bar volume (default 1.0 = unconstrained). Remainder stays pending.
- DAY validity: all pending DAY orders expire at 15:30 IST. IOC orders cancelled if unfilled on current bar. PositionManager handles re-submission of expired stops/targets.
- Risk controls (all optional via config/CLI flags):
  - `max_drawdown_pct`: kill switch — force-close all positions (with slippage) and stop trading when drawdown exceeds threshold
  - `daily_loss_limit`: reject new buys and close MIS after daily loss exceeds limit, resets next day
  - `max_position_qty`: clamp per-symbol position size
  - `max_exposure_pct`: reject buys that would push total exposure above % of capital
- Trading calendar: `TradingCalendar::nse()` with NSE holidays 2020-2027. Test data generator skips holidays.
- `CandleStore.write()` merges new bars with existing data (deduplicates by timestamp). Does not overwrite.
- Corporate actions: `instruments.db` has `corporate_actions` table. `adjust_for_corporate_actions()` adjusts pre-action OHLCV for splits/bonuses/dividends. Applied in CLI before engine run.
- Reference symbols: `--reference-symbols` loads non-tradable index data (e.g., NIFTY 50) into snapshots. Engine ignores signals for reference symbols.
- Metrics: sample std dev (N-1) for Sharpe/Sortino with configurable risk-free rate (default 7% for India), CAGR from actual equity curve, per-symbol breakdown, monthly returns, benchmark comparison (buy-and-hold alpha), trade duration stats.
- Reporting: `results show` displays per-symbol P&L table, monthly returns, benchmark/alpha, trade duration
- `ClosedTrade.pnl` is net of costs. `Fill.costs` tracks per-fill transaction costs passed to strategies via gRPC.
- `BarData` includes `timestamp_ms` for time-aware analysis in strategy history
- Lookback buffers are pre-populated from bars before `--from` date — strategies get full history from the first `on_bar` call, no warmup API calls
- Test data generator uses IST (UTC+5:30) timestamps and skips NSE holidays
- Zerodha charges zero brokerage on all equity trades (CNC and MIS). ₹20/order (per side) applies to F&O only.
- Cost model (`ZerodhaCostModel`): zero equity brokerage, STT (0.1% delivery both sides, 0.025% intraday sell-only), transaction charges (0.00307%), GST (18% on brokerage + txn + SEBI fees), SEBI fees (₹10/crore), stamp duty (0.015% delivery, 0.003% intraday — buy side only)
- CLI `run` command supports `--exchange` flag (default NSE) for BSE/MCX backtesting, `--risk-free-rate` (default 0.07)
- Instrument metadata is loaded from `instruments.db` and passed to the engine for correct cost model (instrument type) and strategy use
- Order modification is NOT supported. Use CANCEL + new order.
- Cover orders (CO) are NOT supported. Strategies pair entry + SL-M orders manually via PositionManager.
