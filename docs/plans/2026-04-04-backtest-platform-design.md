# Backtest Platform Design

**Date**: 2026-04-04
**Status**: Approved

## Goal

Build a platform to evaluate trading strategies (deterministic and LLM-based) against Indian market data (equities, F&O, commodities) sourced from Zerodha Kite Connect API. Long-term goal: power a trading bot with well-tested strategies.

## Architecture

**Approach**: Rust engine + Python strategy server communicating via gRPC.

```
CLI (Rust) ─┬─ Data Manager (Rust) ──► SQLite (metadata) + Parquet (candles)
             ├─ Backtest Engine (Rust) ──gRPC──► Strategy Server (Python)
             └─ Results Reporter (Rust) ──► JSON + Parquet (results)
```

### Components

1. **Data Manager (Rust)** — Fetches historical data from Zerodha Kite API, caches locally. Handles candle aggregation. Manages instrument metadata (symbols, expiries, lot sizes).

2. **Backtest Engine (Rust)** — Event-driven engine. Iterates through historical bars, sends market state to strategy server via gRPC, receives signals, simulates order execution with slippage, brokerage, and Indian tax/duty calculations.

3. **Strategy Server (Python)** — gRPC server hosting Python strategy implementations. Each strategy implements an abstract interface. Future LLM-based strategies live here too.

4. **Results Reporter (Rust)** — Computes performance metrics, outputs JSON/Parquet results.

## Data Model

### Instrument metadata (SQLite)

`instruments` table: tradingsymbol, exchange (NSE/BSE/MCX), instrument_type (EQ/FUT/OPT/COM), lot_size, tick_size, expiry, strike, option_type. Refreshed daily from Kite instrument dump.

### Market data (Parquet, partitioned by instrument + timeframe)

```
data/candles/{EXCHANGE}/{SYMBOL}/{TIMEFRAME}/{PERIOD}.parquet
```

Columns: `timestamp, open, high, low, close, volume, oi`

Supported timeframes: 1-minute, daily.

### Backtest results (JSON + Parquet)

```
results/{backtest_id}/config.json      — strategy params, date range, instruments
results/{backtest_id}/trades.parquet   — every trade with entry/exit, P&L
results/{backtest_id}/equity_curve.parquet — portfolio value over time
results/{backtest_id}/metrics.json     — summary stats
```

## gRPC Interface

```protobuf
service StrategyService {
  rpc Initialize(InitRequest) returns (InitResponse);
  rpc OnBar(BarEvent) returns (Signal);
  rpc OnComplete(CompleteRequest) returns (CompleteResponse);
}
```

- `BarEvent` contains: timestamp, symbol, OHLCV+OI, portfolio state (positions, cash)
- `Signal` contains: action (HOLD/BUY/SELL), symbol, quantity, order type (MARKET/LIMIT/SL/SL-M), limit/stop prices
- Strategies can return multiple signals per bar

### Python strategy interface

```python
class Strategy(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> None: ...

    @abstractmethod
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]: ...

    def on_complete(self) -> dict:
        return {}
```

## Backtest Engine Internals

### Event loop (per bar)

1. Send bar to strategy via gRPC, get signals
2. Process pending orders from previous bars (limit/SL orders)
3. Process new signals, create orders
4. Execute matching:
   - MARKET: fill at next bar's open
   - LIMIT: fill if price touches limit during bar
   - SL/SL-M: trigger if price crosses stop level
5. Apply transaction costs (Zerodha brokerage, STT, GST, SEBI fees, stamp duty)
6. Update portfolio state
7. Record trades

### Indian market specifics

- Trading hours: Pre-open (9:00-9:15), Normal (9:15-15:30) for NSE/BSE; MCX different
- Circuit limits: reject orders beyond upper/lower circuits
- Lot sizes: F&O trades in lots
- Expiry handling: auto-close positions on expiry day
- Settlement: T+1 for equity delivery

### Configurable parameters

- Start/end date, initial capital
- Strategy name + parameters
- Instruments, timeframe
- Slippage model (fixed %, fixed amount, none)
- Commission model (Zerodha default or custom)

## Performance Metrics

| Category | Metrics |
|----------|---------|
| Returns | Total return %, CAGR, daily/monthly returns |
| Risk | Max drawdown %, Sharpe, Sortino, Calmar, Volatility |
| Trading | Total trades, Win rate %, Avg win/loss, Profit factor, Avg holding period |
| Costs | Total brokerage, STT, taxes, Net P&L after costs |
| Drawdown | Max drawdown duration, Recovery time, Underwater curve |

## Project Structure

```
backtest-platform/
├── engine/                    # Rust workspace
│   ├── Cargo.toml
│   ├── crates/
│   │   ├── core/              # Backtest engine, matching, portfolio
│   │   ├── data/              # Data fetching (Kite API), storage
│   │   ├── proto/             # gRPC proto definitions + codegen
│   │   └── cli/               # CLI binary (clap)
├── strategies/                # Python strategy server
│   ├── pyproject.toml
│   ├── server/                # gRPC server
│   ├── strategies/            # User-defined strategies
│   │   ├── base.py            # Abstract Strategy class
│   │   └── examples/          # SMA crossover, RSI, etc.
│   └── tests/
├── ui/                        # TypeScript web UI (later)
├── data/                      # Local data cache (gitignored)
├── results/                   # Backtest results (gitignored)
└── docs/plans/
```

## CLI Commands

```bash
backtest data fetch --symbol RELIANCE --from 2023-01-01 --to 2024-12-31 --interval 1min
backtest data list
backtest run --strategy sma_crossover --symbols RELIANCE,TCS --from 2023-01-01 --to 2024-12-31 --capital 1000000 --interval day --params '{"fast_period": 10, "slow_period": 30}'
backtest results list
backtest results show <backtest_id>
backtest results export <backtest_id> --format csv
```

## Scope

### In scope (v1)
- Deterministic strategies with event-driven backtesting
- Kite API data fetching and caching
- Equities, F&O, commodities
- Indian market transaction cost model
- Performance metrics and CLI reporting

### Out of scope (v1)
- LLM-based strategies (architecture supports it, not implemented)
- Web UI (TypeScript, built later)
- Walk-forward analysis, Monte Carlo simulation
- Parameter optimization / sweeps
- Live trading / paper trading
- Multi-strategy portfolio backtesting

## Tech Stack

- **Engine**: Rust (arrow/parquet crates, tonic for gRPC, clap for CLI, rusqlite for SQLite, reqwest for HTTP)
- **Strategies**: Python 3.11+ (grpcio, abstract base classes)
- **UI (later)**: TypeScript, framework TBD
