# Backtest Platform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an Indian market backtesting platform with a Rust engine, Python strategy server (gRPC), and CLI interface.

**Architecture:** Rust workspace with 4 crates (core, data, proto, cli) communicating via gRPC with a Python strategy server. Data stored in SQLite (metadata) and Parquet (candles/results).

**Tech Stack:** Rust (tonic, prost, arrow, parquet, clap, rusqlite, reqwest, tokio), Python 3.11+ (grpcio, protobuf), protobuf

**Design doc:** `docs/plans/2026-04-04-backtest-platform-design.md`

---

## Phase 1: Project Scaffolding & gRPC Contract

**Goal:** Get the Rust workspace compiling, define the gRPC contract, and verify codegen works on both Rust and Python sides. No business logic yet — just the skeleton.

---

### Task 1: Scaffold Rust workspace

**Files:**
- Create: `engine/Cargo.toml` (workspace root)
- Create: `engine/crates/core/Cargo.toml`
- Create: `engine/crates/core/src/lib.rs`
- Create: `engine/crates/data/Cargo.toml`
- Create: `engine/crates/data/src/lib.rs`
- Create: `engine/crates/proto/Cargo.toml`
- Create: `engine/crates/proto/src/lib.rs`
- Create: `engine/crates/cli/Cargo.toml`
- Create: `engine/crates/cli/src/main.rs`
- Create: `.gitignore`

**Step 1: Create workspace Cargo.toml**

```toml
# engine/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/core",
    "crates/data",
    "crates/proto",
    "crates/cli",
]

[workspace.package]
edition = "2021"
version = "0.1.0"

[workspace.dependencies]
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
chrono = { version = "0.4", features = ["serde"] }
```

**Step 2: Create crate Cargo.tomls and empty src files**

`engine/crates/proto/Cargo.toml`:
```toml
[package]
name = "backtest-proto"
edition.workspace = true
version.workspace = true

[dependencies]
tonic = "0.14"
prost = "0.14"
tonic-prost = "0.14"

[build-dependencies]
tonic-prost-build = "0.14"
```

`engine/crates/core/Cargo.toml`:
```toml
[package]
name = "backtest-core"
edition.workspace = true
version.workspace = true

[dependencies]
backtest-proto = { path = "../proto" }
tokio.workspace = true
serde.workspace = true
serde_json.workspace = true
anyhow.workspace = true
chrono.workspace = true
```

`engine/crates/data/Cargo.toml`:
```toml
[package]
name = "backtest-data"
edition.workspace = true
version.workspace = true

[dependencies]
backtest-core = { path = "../core" }
anyhow.workspace = true
serde.workspace = true
serde_json.workspace = true
chrono.workspace = true
tokio.workspace = true
reqwest = { version = "0.12", features = ["json"] }
rusqlite = { version = "0.32", features = ["bundled"] }
arrow = "54"
parquet = { version = "54", features = ["arrow"] }
```

`engine/crates/cli/Cargo.toml`:
```toml
[package]
name = "backtest-cli"
edition.workspace = true
version.workspace = true

[[bin]]
name = "backtest"
path = "src/main.rs"

[dependencies]
backtest-core = { path = "../core" }
backtest-data = { path = "../data" }
backtest-proto = { path = "../proto" }
clap = { version = "4", features = ["derive"] }
tokio.workspace = true
anyhow.workspace = true
serde_json.workspace = true
```

Each `src/lib.rs` is an empty file. `cli/src/main.rs`:
```rust
fn main() {
    println!("backtest CLI");
}
```

**Step 3: Create .gitignore**

```gitignore
/target
/engine/target
/data/
/results/
*.parquet
*.db
.env
__pycache__/
*.pyc
*.egg-info/
.venv/
```

**Step 4: Build to verify workspace compiles**

Run: `cd engine && cargo build`
Expected: Compiles successfully with no errors.

**Step 5: Commit**

```bash
git add engine/ .gitignore
git commit -m "feat: scaffold Rust workspace with core, data, proto, cli crates"
```

---

### Task 2: Define protobuf contract and generate Rust code

**Files:**
- Create: `engine/crates/proto/proto/strategy.proto`
- Create: `engine/crates/proto/build.rs`
- Modify: `engine/crates/proto/src/lib.rs`

**Step 1: Write the proto definition**

```protobuf
// engine/crates/proto/proto/strategy.proto
syntax = "proto3";
package backtest;

service StrategyService {
  rpc Initialize(InitRequest) returns (InitResponse);
  rpc OnBar(BarEvent) returns (BarResponse);
  rpc OnComplete(CompleteRequest) returns (CompleteResponse);
}

// --- Initialize ---

message InitRequest {
  string strategy_name = 1;
  string config_json = 2;  // strategy parameters as JSON
  repeated string symbols = 3;
}

message InitResponse {
  bool success = 1;
  string error = 2;
}

// --- Bar Event ---

message BarEvent {
  int64 timestamp_ms = 1;
  string symbol = 2;
  double open = 3;
  double high = 4;
  double low = 5;
  double close = 6;
  int64 volume = 7;
  int64 oi = 8;
  PortfolioState portfolio = 9;
}

message PortfolioState {
  double cash = 1;
  double equity = 2;
  repeated PositionInfo positions = 3;
}

message PositionInfo {
  string symbol = 1;
  int32 quantity = 2;
  double avg_price = 3;
  double unrealized_pnl = 4;
}

message BarResponse {
  repeated Signal signals = 1;
}

message Signal {
  enum Action {
    HOLD = 0;
    BUY = 1;
    SELL = 2;
  }
  enum OrderType {
    MARKET = 0;
    LIMIT = 1;
    SL = 2;
    SL_M = 3;
  }
  Action action = 1;
  string symbol = 2;
  int32 quantity = 3;
  OrderType order_type = 4;
  double limit_price = 5;
  double stop_price = 6;
}

// --- Complete ---

message CompleteRequest {}

message CompleteResponse {
  string custom_metrics_json = 1;  // optional JSON from strategy
}
```

**Step 2: Write build.rs for protobuf codegen**

```rust
// engine/crates/proto/build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("proto/strategy.proto")?;
    Ok(())
}
```

**Step 3: Update proto lib.rs to re-export generated code**

```rust
// engine/crates/proto/src/lib.rs
pub mod backtest {
    tonic::include_proto!("backtest");
}
```

**Step 4: Build proto crate to verify codegen**

Run: `cd engine && cargo build -p backtest-proto`
Expected: Compiles. Generated Rust structs for all proto messages.

**Step 5: Commit**

```bash
git add engine/crates/proto/
git commit -m "feat: define gRPC strategy service proto and generate Rust code"
```

---

### Task 3: Scaffold Python strategy server

**Files:**
- Create: `strategies/pyproject.toml`
- Create: `strategies/server/__init__.py`
- Create: `strategies/server/server.py`
- Create: `strategies/strategies/__init__.py`
- Create: `strategies/strategies/base.py`
- Create: `strategies/tests/__init__.py`
- Create: `strategies/buf.gen.yaml` (or use grpc_tools)
- Create: `strategies/generate_proto.sh`

**Step 1: Create pyproject.toml**

```toml
# strategies/pyproject.toml
[project]
name = "backtest-strategies"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "grpcio>=1.68",
    "grpcio-tools>=1.68",
    "protobuf>=5",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
```

**Step 2: Create proto generation script**

```bash
#!/usr/bin/env bash
# strategies/generate_proto.sh
set -euo pipefail
PROTO_DIR="../engine/crates/proto/proto"
OUT_DIR="./server/generated"
mkdir -p "$OUT_DIR"
python -m grpc_tools.protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  strategy.proto
touch "$OUT_DIR/__init__.py"
echo "Proto generation complete."
```

**Step 3: Create the abstract Strategy base class**

```python
# strategies/strategies/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Bar:
    timestamp_ms: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float


@dataclass
class Portfolio:
    cash: float
    equity: float
    positions: list[Position]


@dataclass
class Signal:
    action: str    # "HOLD", "BUY", "SELL"
    symbol: str
    quantity: int
    order_type: str = "MARKET"    # "MARKET", "LIMIT", "SL", "SL_M"
    limit_price: float = 0.0
    stop_price: float = 0.0


class Strategy(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Called once with strategy parameters."""
        pass

    @abstractmethod
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        """Called on each new bar. Return list of signals."""
        pass

    def on_complete(self) -> dict:
        """Called at backtest end. Return any custom metrics."""
        return {}
```

**Step 4: Create empty server.py placeholder**

```python
# strategies/server/server.py
"""gRPC strategy server. Implemented in Phase 4."""
```

**Step 5: Set up virtual environment and install**

Run: `cd strategies && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: Installs successfully.

**Step 6: Generate Python proto code**

Run: `cd strategies && chmod +x generate_proto.sh && ./generate_proto.sh`
Expected: Files generated in `server/generated/`.

**Step 7: Commit**

```bash
git add strategies/
git commit -m "feat: scaffold Python strategy server with base class and proto codegen"
```

---

## Phase 2: Core Domain Types & Data Layer

**Goal:** Define the Rust domain types that the entire engine uses, build the SQLite instrument store, and implement Parquet candle storage. By the end, you can fetch data from Kite API and read it back from local storage.

---

### Task 4: Core domain types

**Files:**
- Create: `engine/crates/core/src/types.rs`
- Create: `engine/crates/core/src/config.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write tests for domain types**

```rust
// engine/crates/core/src/types.rs (tests at bottom)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_creation() {
        let bar = Bar {
            timestamp_ms: 1704067200000,
            symbol: "RELIANCE".into(),
            open: 2500.0,
            high: 2520.0,
            low: 2490.0,
            close: 2510.0,
            volume: 100000,
            oi: 0,
        };
        assert_eq!(bar.symbol, "RELIANCE");
        assert!(bar.high >= bar.low);
    }

    #[test]
    fn test_signal_market_buy() {
        let sig = Signal::market_buy("RELIANCE", 10);
        assert_eq!(sig.action, Action::Buy);
        assert_eq!(sig.order_type, OrderType::Market);
        assert_eq!(sig.quantity, 10);
    }

    #[test]
    fn test_portfolio_equity_calculation() {
        let pos = Position {
            symbol: "RELIANCE".into(),
            quantity: 10,
            avg_price: 2500.0,
            current_price: 2550.0,
        };
        let portfolio = Portfolio {
            cash: 100_000.0,
            positions: vec![pos],
        };
        // equity = cash + sum(quantity * current_price)
        assert_eq!(portfolio.equity(), 125_500.0);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd engine && cargo test -p backtest-core`
Expected: FAIL — types not defined yet.

**Step 3: Implement domain types**

```rust
// engine/crates/core/src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp_ms: i64,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub oi: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Hold,
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Sl,
    SlM,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub action: Action,
    pub symbol: String,
    pub quantity: i32,
    pub order_type: OrderType,
    pub limit_price: f64,
    pub stop_price: f64,
}

impl Signal {
    pub fn market_buy(symbol: &str, quantity: i32) -> Self {
        Self {
            action: Action::Buy,
            symbol: symbol.into(),
            quantity,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
        }
    }

    pub fn market_sell(symbol: &str, quantity: i32) -> Self {
        Self {
            action: Action::Sell,
            symbol: symbol.into(),
            quantity,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: i32,
    pub avg_price: f64,
    pub current_price: f64,
}

impl Position {
    pub fn market_value(&self) -> f64 {
        self.quantity as f64 * self.current_price
    }

    pub fn unrealized_pnl(&self) -> f64 {
        self.quantity as f64 * (self.current_price - self.avg_price)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub cash: f64,
    pub positions: Vec<Position>,
}

impl Portfolio {
    pub fn equity(&self) -> f64 {
        self.cash + self.positions.iter().map(|p| p.market_value()).sum::<f64>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Exchange {
    Nse,
    Bse,
    Mcx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstrumentType {
    Equity,
    FutureFO,
    OptionFO,
    Commodity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instrument {
    pub tradingsymbol: String,
    pub exchange: Exchange,
    pub instrument_type: InstrumentType,
    pub lot_size: i32,
    pub tick_size: f64,
    pub expiry: Option<String>,     // YYYY-MM-DD
    pub strike: Option<f64>,
    pub option_type: Option<String>, // CE or PE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    Minute,
    Day,
}

impl Interval {
    pub fn as_kite_str(&self) -> &str {
        match self {
            Interval::Minute => "minute",
            Interval::Day => "day",
        }
    }
}
```

```rust
// engine/crates/core/src/config.rs
use serde::{Deserialize, Serialize};
use crate::types::Interval;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub strategy_name: String,
    pub symbols: Vec<String>,
    pub start_date: String,     // YYYY-MM-DD
    pub end_date: String,       // YYYY-MM-DD
    pub initial_capital: f64,
    pub interval: Interval,
    pub strategy_params: serde_json::Value,
    pub slippage_pct: f64,      // e.g. 0.001 for 0.1%
}
```

```rust
// engine/crates/core/src/lib.rs
pub mod types;
pub mod config;
```

**Step 4: Run tests to verify they pass**

Run: `cd engine && cargo test -p backtest-core`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add engine/crates/core/
git commit -m "feat: add core domain types (Bar, Signal, Portfolio, Position, Instrument)"
```

---

### Task 5: Instrument metadata store (SQLite)

**Files:**
- Create: `engine/crates/data/src/instruments.rs`
- Modify: `engine/crates/data/src/lib.rs`

**Step 1: Write tests**

```rust
// engine/crates/data/src/instruments.rs (tests at bottom)
#[cfg(test)]
mod tests {
    use super::*;
    use backtest_core::types::{Exchange, InstrumentType};

    #[test]
    fn test_insert_and_query_instrument() {
        let store = InstrumentStore::in_memory().unwrap();
        let inst = Instrument {
            tradingsymbol: "RELIANCE".into(),
            exchange: Exchange::Nse,
            instrument_type: InstrumentType::Equity,
            lot_size: 1,
            tick_size: 0.05,
            expiry: None,
            strike: None,
            option_type: None,
        };
        store.upsert(&[inst]).unwrap();
        let result = store.find("RELIANCE", Exchange::Nse).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().lot_size, 1);
    }

    #[test]
    fn test_list_by_exchange() {
        let store = InstrumentStore::in_memory().unwrap();
        let instruments = vec![
            Instrument {
                tradingsymbol: "RELIANCE".into(),
                exchange: Exchange::Nse,
                instrument_type: InstrumentType::Equity,
                lot_size: 1, tick_size: 0.05,
                expiry: None, strike: None, option_type: None,
            },
            Instrument {
                tradingsymbol: "TCS".into(),
                exchange: Exchange::Nse,
                instrument_type: InstrumentType::Equity,
                lot_size: 1, tick_size: 0.05,
                expiry: None, strike: None, option_type: None,
            },
        ];
        store.upsert(&instruments).unwrap();
        let nse = store.list_by_exchange(Exchange::Nse).unwrap();
        assert_eq!(nse.len(), 2);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd engine && cargo test -p backtest-data`
Expected: FAIL.

**Step 3: Implement InstrumentStore**

Create `engine/crates/data/src/instruments.rs` with:
- `InstrumentStore` struct wrapping a `rusqlite::Connection`
- `InstrumentStore::open(path)` — opens/creates SQLite file
- `InstrumentStore::in_memory()` — for tests
- `upsert(&[Instrument])` — INSERT OR REPLACE
- `find(symbol, exchange) -> Option<Instrument>`
- `list_by_exchange(exchange) -> Vec<Instrument>`
- Creates table on init with: tradingsymbol, exchange, instrument_type, lot_size, tick_size, expiry, strike, option_type

**Step 4: Run tests**

Run: `cd engine && cargo test -p backtest-data`
Expected: PASS.

**Step 5: Commit**

```bash
git add engine/crates/data/
git commit -m "feat: add SQLite instrument metadata store"
```

---

### Task 6: Parquet candle storage (read/write)

**Files:**
- Create: `engine/crates/data/src/candles.rs`
- Modify: `engine/crates/data/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_and_read_candles() {
        let dir = TempDir::new().unwrap();
        let store = CandleStore::new(dir.path());
        let bars = vec![
            Bar {
                timestamp_ms: 1704067200000,
                symbol: "RELIANCE".into(),
                open: 2500.0, high: 2520.0, low: 2490.0, close: 2510.0,
                volume: 100000, oi: 0,
            },
            Bar {
                timestamp_ms: 1704067260000,
                symbol: "RELIANCE".into(),
                open: 2510.0, high: 2530.0, low: 2505.0, close: 2525.0,
                volume: 80000, oi: 0,
            },
        ];
        store.write("NSE", "RELIANCE", Interval::Minute, &bars).unwrap();
        let loaded = store.read("NSE", "RELIANCE", Interval::Minute, None, None).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].close, 2510.0);
    }

    #[test]
    fn test_read_with_time_filter() {
        let dir = TempDir::new().unwrap();
        let store = CandleStore::new(dir.path());
        let bars = vec![
            Bar { timestamp_ms: 1000, symbol: "X".into(), open: 1.0, high: 2.0, low: 0.5, close: 1.5, volume: 10, oi: 0 },
            Bar { timestamp_ms: 2000, symbol: "X".into(), open: 1.5, high: 2.5, low: 1.0, close: 2.0, volume: 20, oi: 0 },
            Bar { timestamp_ms: 3000, symbol: "X".into(), open: 2.0, high: 3.0, low: 1.5, close: 2.5, volume: 30, oi: 0 },
        ];
        store.write("NSE", "X", Interval::Minute, &bars).unwrap();
        let filtered = store.read("NSE", "X", Interval::Minute, Some(1500), Some(2500)).unwrap();
        assert_eq!(filtered.len(), 1); // only timestamp 2000
    }
}
```

Add `tempfile = "3"` to `engine/crates/data/Cargo.toml` `[dev-dependencies]`.

**Step 2: Run tests to verify they fail**

Run: `cd engine && cargo test -p backtest-data`
Expected: FAIL.

**Step 3: Implement CandleStore**

`engine/crates/data/src/candles.rs`:
- `CandleStore::new(base_path)` — base directory for data
- `write(exchange, symbol, interval, &[Bar])` — writes Parquet file to `{base}/{exchange}/{symbol}/{interval}/data.parquet`
- `read(exchange, symbol, interval, from_ms, to_ms) -> Vec<Bar>` — reads and optionally filters by timestamp range
- Uses `arrow` arrays: Int64Array (timestamp_ms, volume, oi), Float64Array (OHLC), Utf8Array (symbol)
- Uses `parquet::arrow::ArrowWriter` / `arrow_reader::ParquetRecordBatchReader`

**Step 4: Run tests**

Run: `cd engine && cargo test -p backtest-data`
Expected: All PASS.

**Step 5: Commit**

```bash
git add engine/crates/data/
git commit -m "feat: add Parquet candle storage with time-range filtering"
```

---

### Task 7: Kite API client (data fetching)

**Files:**
- Create: `engine/crates/data/src/kite.rs`
- Modify: `engine/crates/data/src/lib.rs`

**Step 1: Write the Kite API client struct**

The Kite Connect API requires:
- API key + access token (session-based, expires daily)
- Historical candles endpoint: `GET https://api.kite.trade/instruments/historical/{instrument_token}/{interval}?from=YYYY-MM-DD+HH:MM:SS&to=YYYY-MM-DD+HH:MM:SS`
- Instruments dump: `GET https://api.kite.trade/instruments` (returns CSV)

```rust
// engine/crates/data/src/kite.rs
use anyhow::Result;
use reqwest::Client;
use backtest_core::types::{Bar, Instrument, Interval};

pub struct KiteClient {
    client: Client,
    api_key: String,
    access_token: String,
    base_url: String,
}

impl KiteClient {
    pub fn new(api_key: String, access_token: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            access_token,
            base_url: "https://api.kite.trade".into(),
        }
    }

    /// Fetch all instruments from Kite (CSV endpoint).
    /// Returns parsed Instrument structs.
    pub async fn fetch_instruments(&self) -> Result<Vec<Instrument>> {
        // GET /instruments with Authorization header
        // Parse CSV response into Instrument structs
        todo!()
    }

    /// Fetch historical candles for a specific instrument token.
    /// Kite limits to ~2000 candles per request.
    pub async fn fetch_candles(
        &self,
        instrument_token: &str,
        interval: Interval,
        from: &str,  // "YYYY-MM-DD HH:MM:SS"
        to: &str,
    ) -> Result<Vec<Bar>> {
        // GET /instruments/historical/{token}/{interval}?from=...&to=...
        // Parse JSON response: {"data": {"candles": [[ts, o, h, l, c, v], ...]}}
        todo!()
    }
}
```

**Step 2: Implement fetch_instruments**

- GET `{base_url}/instruments` with header `Authorization: token {api_key}:{access_token}`
- Response is CSV with columns: instrument_token, exchange_token, tradingsymbol, name, last_price, expiry, strike, tick_size, lot_size, instrument_type, segment, exchange
- Parse CSV rows into `Instrument` structs (map exchange string to `Exchange` enum, instrument_type string to `InstrumentType` enum)

**Step 3: Implement fetch_candles**

- GET `{base_url}/instruments/historical/{instrument_token}/{interval}?from={from}&to={to}`
- Same auth header
- Response JSON: `{"status": "success", "data": {"candles": [[timestamp, open, high, low, close, volume, oi], ...]}}`
- Parse each candle array into a `Bar` struct
- Handle the 2000-candle limit by chunking date ranges in the caller (not in this function)

**Step 4: Write an integration test (gated behind a feature flag)**

```rust
#[cfg(test)]
#[cfg(feature = "kite-integration")]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_instruments_live() {
        let api_key = std::env::var("KITE_API_KEY").unwrap();
        let access_token = std::env::var("KITE_ACCESS_TOKEN").unwrap();
        let client = KiteClient::new(api_key, access_token);
        let instruments = client.fetch_instruments().await.unwrap();
        assert!(!instruments.is_empty());
    }
}
```

Add to `data/Cargo.toml`:
```toml
[features]
kite-integration = []
```

**Step 5: Commit**

```bash
git add engine/crates/data/
git commit -m "feat: add Kite Connect API client for instruments and candles"
```

---

## Phase 3: Python Strategy Server & Example Strategy

**Goal:** The Python gRPC server is fully functional. An SMA crossover example strategy is implemented and tested. You can start the server and call it from a Rust gRPC client.

---

### Task 8: Implement Python gRPC strategy server

**Files:**
- Modify: `strategies/server/server.py`
- Create: `strategies/server/registry.py`

**Step 1: Implement the strategy registry**

```python
# strategies/server/registry.py
from strategies.base import Strategy

_STRATEGIES: dict[str, type[Strategy]] = {}

def register(name: str):
    """Decorator to register a strategy class."""
    def decorator(cls: type[Strategy]):
        _STRATEGIES[name] = cls
        return cls
    return decorator

def get_strategy(name: str) -> Strategy:
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[name]()

def list_strategies() -> list[str]:
    return list(_STRATEGIES.keys())
```

**Step 2: Implement the gRPC server**

```python
# strategies/server/server.py
import json
import grpc
from concurrent import futures

from server.generated import strategy_pb2
from server.generated import strategy_pb2_grpc
from server.registry import get_strategy
from strategies.base import Bar, Portfolio, Position, Signal

class StrategyServicer(strategy_pb2_grpc.StrategyServiceServicer):
    def __init__(self):
        self.strategy = None

    def Initialize(self, request, context):
        try:
            self.strategy = get_strategy(request.strategy_name)
            config = json.loads(request.config_json) if request.config_json else {}
            self.strategy.initialize(config)
            return strategy_pb2.InitResponse(success=True, error="")
        except Exception as e:
            return strategy_pb2.InitResponse(success=False, error=str(e))

    def OnBar(self, request, context):
        bar = Bar(
            timestamp_ms=request.timestamp_ms,
            symbol=request.symbol,
            open=request.open,
            high=request.high,
            low=request.low,
            close=request.close,
            volume=request.volume,
            oi=request.oi,
        )
        portfolio = Portfolio(
            cash=request.portfolio.cash,
            equity=request.portfolio.equity,
            positions=[
                Position(p.symbol, p.quantity, p.avg_price, p.unrealized_pnl)
                for p in request.portfolio.positions
            ],
        )
        signals = self.strategy.on_bar(bar, portfolio)
        proto_signals = []
        for s in signals:
            action_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
            order_map = {"MARKET": 0, "LIMIT": 1, "SL": 2, "SL_M": 3}
            proto_signals.append(strategy_pb2.Signal(
                action=action_map[s.action],
                symbol=s.symbol,
                quantity=s.quantity,
                order_type=order_map[s.order_type],
                limit_price=s.limit_price,
                stop_price=s.stop_price,
            ))
        return strategy_pb2.BarResponse(signals=proto_signals)

    def OnComplete(self, request, context):
        metrics = self.strategy.on_complete() if self.strategy else {}
        return strategy_pb2.CompleteResponse(
            custom_metrics_json=json.dumps(metrics),
        )

def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    strategy_pb2_grpc.add_StrategyServiceServicer_to_server(
        StrategyServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Strategy server listening on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
```

**Step 3: Test manually**

Run: `cd strategies && source .venv/bin/activate && python -m server.server`
Expected: "Strategy server listening on port 50051" (Ctrl-C to stop).

**Step 4: Commit**

```bash
git add strategies/
git commit -m "feat: implement Python gRPC strategy server with registry"
```

---

### Task 9: SMA crossover example strategy

**Files:**
- Create: `strategies/strategies/examples/__init__.py`
- Create: `strategies/strategies/examples/sma_crossover.py`
- Create: `strategies/tests/test_sma_crossover.py`

**Step 1: Write unit tests for SMA crossover**

```python
# strategies/tests/test_sma_crossover.py
from strategies.base import Bar, Portfolio, Signal
from strategies.examples.sma_crossover import SmaCrossover

def make_bar(ts: int, close: float, symbol: str = "TEST") -> Bar:
    return Bar(ts, symbol, close, close, close, close, 1000, 0)

def empty_portfolio() -> Portfolio:
    return Portfolio(cash=100_000.0, equity=100_000.0, positions=[])

def test_no_signal_before_enough_bars():
    s = SmaCrossover()
    s.initialize({"fast_period": 3, "slow_period": 5})
    for i in range(4):
        signals = s.on_bar(make_bar(i, 100.0), empty_portfolio())
        assert signals == [] or all(sig.action == "HOLD" for sig in signals)

def test_buy_signal_on_golden_cross():
    s = SmaCrossover()
    s.initialize({"fast_period": 2, "slow_period": 3})
    # Slow prices: declining then rising = fast crosses above slow
    prices = [100, 95, 90, 92, 98, 105]  # fast(2) crosses above slow(3) at 105
    signals = []
    for i, p in enumerate(prices):
        signals = s.on_bar(make_bar(i, float(p)), empty_portfolio())
    # After a rising sequence, fast SMA > slow SMA → BUY
    assert any(sig.action == "BUY" for sig in signals)

def test_sell_signal_on_death_cross():
    s = SmaCrossover()
    s.initialize({"fast_period": 2, "slow_period": 3})
    prices = [90, 95, 100, 105, 102, 96, 88]
    signals = []
    for i, p in enumerate(prices):
        signals = s.on_bar(make_bar(i, float(p)), empty_portfolio())
    # fast SMA < slow SMA → SELL
    assert any(sig.action == "SELL" for sig in signals)
```

**Step 2: Run tests to verify they fail**

Run: `cd strategies && source .venv/bin/activate && pytest tests/test_sma_crossover.py -v`
Expected: FAIL — module not found.

**Step 3: Implement SMA crossover strategy**

```python
# strategies/strategies/examples/sma_crossover.py
from collections import deque
from server.registry import register
from strategies.base import Strategy, Bar, Portfolio, Signal


@register("sma_crossover")
class SmaCrossover(Strategy):
    def initialize(self, config: dict) -> None:
        self.fast_period = config.get("fast_period", 10)
        self.slow_period = config.get("slow_period", 30)
        self.prices: deque[float] = deque(maxlen=self.slow_period)
        self.prev_fast_above = None

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        self.prices.append(bar.close)
        if len(self.prices) < self.slow_period:
            return []

        fast_sma = sum(list(self.prices)[-self.fast_period:]) / self.fast_period
        slow_sma = sum(self.prices) / len(self.prices)
        fast_above = fast_sma > slow_sma

        signals = []
        if self.prev_fast_above is not None:
            if fast_above and not self.prev_fast_above:
                # Golden cross: fast crosses above slow → BUY
                signals.append(Signal(
                    action="BUY", symbol=bar.symbol, quantity=1,
                ))
            elif not fast_above and self.prev_fast_above:
                # Death cross: fast crosses below slow → SELL
                signals.append(Signal(
                    action="SELL", symbol=bar.symbol, quantity=1,
                ))

        self.prev_fast_above = fast_above
        return signals
```

**Step 4: Run tests**

Run: `cd strategies && pytest tests/test_sma_crossover.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add strategies/
git commit -m "feat: add SMA crossover example strategy with tests"
```

---

## Phase 4: Backtest Engine

**Goal:** The core backtest engine works end-to-end: it reads candle data, sends bars to the Python strategy server via gRPC, processes returning signals through an order matching engine with Indian market transaction costs, and tracks portfolio state.

---

### Task 10: Transaction cost model (Zerodha)

**Files:**
- Create: `engine/crates/core/src/costs.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity_intraday_costs() {
        let model = ZerodhaCostModel;
        // Buy 100 shares of RELIANCE at 2500 = 250,000 turnover
        let cost = model.calculate(TradeParams {
            instrument_type: InstrumentType::Equity,
            is_intraday: true,
            buy_value: 250_000.0,
            sell_value: 252_500.0,  // sold at 2525
            quantity: 100,
        });
        // Brokerage: min(20, 0.03% of turnover) per side
        // STT: 0.025% of sell side for intraday
        assert!(cost.total_brokerage > 0.0);
        assert!(cost.stt > 0.0);
        assert!(cost.total() > 0.0);
    }

    #[test]
    fn test_fo_costs() {
        let model = ZerodhaCostModel;
        let cost = model.calculate(TradeParams {
            instrument_type: InstrumentType::FutureFO,
            is_intraday: false,
            buy_value: 500_000.0,
            sell_value: 510_000.0,
            quantity: 25,
        });
        // F&O brokerage: flat ₹20 per order
        assert_eq!(cost.total_brokerage, 40.0); // 20 buy + 20 sell
    }
}
```

**Step 2: Implement ZerodhaCostModel**

Implement `TradeParams`, `TradeCosts` structs, and `ZerodhaCostModel` with `calculate()`:
- **Equity delivery**: Brokerage = 0, STT = 0.1% (buy+sell)
- **Equity intraday**: Brokerage = min(₹20, 0.03% of turnover) per side, STT = 0.025% of sell side
- **Futures**: Brokerage = ₹20/order, STT = 0.02% of sell side
- **Options**: Brokerage = ₹20/order, STT = 0.1% of sell side (on premium)
- **All**: Transaction charges (NSE: 0.00345%), GST (18% on brokerage + transaction charges), SEBI fees (₹10/crore), stamp duty (varies by state, use 0.015% of buy side)

**Step 3: Run tests**

Run: `cd engine && cargo test -p backtest-core -- costs`
Expected: PASS.

**Step 4: Commit**

```bash
git add engine/crates/core/src/costs.rs
git commit -m "feat: add Zerodha transaction cost model (brokerage, STT, GST, stamp duty)"
```

---

### Task 11: Order matching engine

**Files:**
- Create: `engine/crates/core/src/matching.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_order_fills_at_next_open() {
        let mut matcher = OrderMatcher::new(0.0); // no slippage
        let order = Order::market_buy("RELIANCE", 10);
        matcher.submit(order);

        let bar = Bar {
            timestamp_ms: 1000, symbol: "RELIANCE".into(),
            open: 2500.0, high: 2520.0, low: 2490.0, close: 2510.0,
            volume: 100000, oi: 0,
        };
        let fills = matcher.process_bar(&bar);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill_price, 2500.0); // filled at open
        assert_eq!(fills[0].quantity, 10);
    }

    #[test]
    fn test_limit_buy_fills_when_low_touches() {
        let mut matcher = OrderMatcher::new(0.0);
        let order = Order::limit_buy("RELIANCE", 10, 2480.0);
        matcher.submit(order);

        // Bar where low doesn't reach limit
        let bar1 = Bar {
            timestamp_ms: 1000, symbol: "RELIANCE".into(),
            open: 2500.0, high: 2520.0, low: 2490.0, close: 2510.0,
            volume: 100000, oi: 0,
        };
        assert!(matcher.process_bar(&bar1).is_empty());

        // Bar where low touches limit
        let bar2 = Bar {
            timestamp_ms: 2000, symbol: "RELIANCE".into(),
            open: 2495.0, high: 2510.0, low: 2475.0, close: 2505.0,
            volume: 100000, oi: 0,
        };
        let fills = matcher.process_bar(&bar2);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill_price, 2480.0);
    }

    #[test]
    fn test_stop_loss_triggers() {
        let mut matcher = OrderMatcher::new(0.0);
        let order = Order::stop_loss_sell("RELIANCE", 10, 2450.0);
        matcher.submit(order);

        let bar = Bar {
            timestamp_ms: 1000, symbol: "RELIANCE".into(),
            open: 2500.0, high: 2510.0, low: 2440.0, close: 2460.0,
            volume: 100000, oi: 0,
        };
        let fills = matcher.process_bar(&bar);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill_price, 2450.0); // SL triggered at stop price
    }

    #[test]
    fn test_slippage_applied() {
        let mut matcher = OrderMatcher::new(0.001); // 0.1% slippage
        let order = Order::market_buy("RELIANCE", 10);
        matcher.submit(order);

        let bar = Bar {
            timestamp_ms: 1000, symbol: "RELIANCE".into(),
            open: 2500.0, high: 2520.0, low: 2490.0, close: 2510.0,
            volume: 100000, oi: 0,
        };
        let fills = matcher.process_bar(&bar);
        assert_eq!(fills[0].fill_price, 2502.5); // 2500 + 0.1%
    }
}
```

**Step 2: Implement OrderMatcher**

- `Order` struct: symbol, side (Buy/Sell), quantity, order_type, limit_price, stop_price
- `Fill` struct: symbol, side, quantity, fill_price, timestamp_ms
- `OrderMatcher::new(slippage_pct)` — holds a `Vec<Order>` of pending orders
- `submit(order)` — adds to pending
- `process_bar(bar) -> Vec<Fill>` — iterates pending, checks fill conditions:
  - MARKET: fill at `bar.open ± slippage`
  - LIMIT BUY: fill if `bar.low <= limit_price`, at limit_price
  - LIMIT SELL: fill if `bar.high >= limit_price`, at limit_price
  - SL SELL: fill if `bar.low <= stop_price`, at stop_price
  - SL BUY: fill if `bar.high >= stop_price`, at stop_price

**Step 3: Run tests**

Run: `cd engine && cargo test -p backtest-core -- matching`
Expected: PASS.

**Step 4: Commit**

```bash
git add engine/crates/core/src/matching.rs
git commit -m "feat: add order matching engine with market, limit, and stop-loss orders"
```

---

### Task 12: Portfolio manager

**Files:**
- Create: `engine/crates/core/src/portfolio.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_creates_position() {
        let mut pm = PortfolioManager::new(1_000_000.0);
        pm.apply_fill(&Fill {
            symbol: "RELIANCE".into(),
            side: Side::Buy,
            quantity: 10,
            fill_price: 2500.0,
            timestamp_ms: 1000,
        }, 50.0); // ₹50 costs
        assert_eq!(pm.cash(), 1_000_000.0 - 25_000.0 - 50.0);
        assert_eq!(pm.position("RELIANCE").unwrap().quantity, 10);
    }

    #[test]
    fn test_sell_reduces_position_and_records_trade() {
        let mut pm = PortfolioManager::new(1_000_000.0);
        pm.apply_fill(&Fill {
            symbol: "RELIANCE".into(), side: Side::Buy,
            quantity: 10, fill_price: 2500.0, timestamp_ms: 1000,
        }, 50.0);
        pm.apply_fill(&Fill {
            symbol: "RELIANCE".into(), side: Side::Sell,
            quantity: 10, fill_price: 2600.0, timestamp_ms: 2000,
        }, 50.0);
        assert!(pm.position("RELIANCE").is_none());
        assert_eq!(pm.closed_trades().len(), 1);
        let trade = &pm.closed_trades()[0];
        assert_eq!(trade.pnl, 1000.0); // (2600-2500)*10
    }

    #[test]
    fn test_equity_curve_tracking() {
        let mut pm = PortfolioManager::new(1_000_000.0);
        pm.update_prices(&[("RELIANCE", 2500.0)].into_iter().collect(), 1000);
        pm.apply_fill(&Fill {
            symbol: "RELIANCE".into(), side: Side::Buy,
            quantity: 10, fill_price: 2500.0, timestamp_ms: 1000,
        }, 0.0);
        pm.update_prices(&[("RELIANCE", 2600.0)].into_iter().map(|(k, v)| (k.to_string(), v)).collect(), 2000);
        let curve = pm.equity_curve();
        assert_eq!(curve.len(), 2);
        assert!(curve[1].equity > curve[0].equity);
    }
}
```

**Step 2: Implement PortfolioManager**

- Tracks `cash`, `HashMap<String, Position>`, `Vec<ClosedTrade>`, `Vec<EquityPoint>`
- `apply_fill(fill, costs)` — updates cash, creates/modifies/closes positions
- `update_prices(prices_map, timestamp)` — updates current prices, records equity point
- `position(symbol) -> Option<&Position>`
- `closed_trades() -> &[ClosedTrade]`
- `equity_curve() -> &[EquityPoint]`
- `portfolio_state() -> Portfolio` — snapshot for gRPC

**Step 3: Run tests**

Run: `cd engine && cargo test -p backtest-core -- portfolio`
Expected: PASS.

**Step 4: Commit**

```bash
git add engine/crates/core/src/portfolio.rs
git commit -m "feat: add portfolio manager with position tracking and equity curve"
```

---

### Task 13: Backtest engine event loop

**Files:**
- Create: `engine/crates/core/src/engine.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write integration test (using mock strategy client)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockStrategyClient {
        /// Returns BUY on bar 5, SELL on bar 10
        call_count: std::cell::RefCell<usize>,
    }

    impl MockStrategyClient {
        fn new() -> Self { Self { call_count: std::cell::RefCell::new(0) } }
    }

    #[async_trait::async_trait]
    impl StrategyClient for MockStrategyClient {
        async fn initialize(&self, _name: &str, _config: &str, _symbols: &[String]) -> Result<()> { Ok(()) }
        async fn on_bar(&self, bar: &Bar, portfolio: &Portfolio) -> Result<Vec<Signal>> {
            let mut count = self.call_count.borrow_mut();
            *count += 1;
            if *count == 5 {
                Ok(vec![Signal::market_buy(&bar.symbol, 10)])
            } else if *count == 10 {
                Ok(vec![Signal::market_sell(&bar.symbol, 10)])
            } else {
                Ok(vec![])
            }
        }
        async fn on_complete(&self) -> Result<serde_json::Value> { Ok(serde_json::json!({})) }
    }

    #[tokio::test]
    async fn test_engine_runs_backtest() {
        let bars: Vec<Bar> = (0..15).map(|i| Bar {
            timestamp_ms: i * 60000,
            symbol: "TEST".into(),
            open: 100.0 + i as f64,
            high: 102.0 + i as f64,
            low: 99.0 + i as f64,
            close: 101.0 + i as f64,
            volume: 1000,
            oi: 0,
        }).collect();

        let client = MockStrategyClient::new();
        let config = BacktestConfig {
            strategy_name: "mock".into(),
            symbols: vec!["TEST".into()],
            start_date: "2024-01-01".into(),
            end_date: "2024-01-02".into(),
            initial_capital: 1_000_000.0,
            interval: Interval::Minute,
            strategy_params: serde_json::json!({}),
            slippage_pct: 0.0,
        };

        let result = BacktestEngine::run(config, bars, &client).await.unwrap();
        assert_eq!(result.trades.len(), 1); // one round-trip trade
        assert!(result.final_equity > 0.0);
        assert!(!result.equity_curve.is_empty());
    }
}
```

**Step 2: Define StrategyClient trait**

```rust
#[async_trait::async_trait]
pub trait StrategyClient: Send + Sync {
    async fn initialize(&self, name: &str, config: &str, symbols: &[String]) -> Result<()>;
    async fn on_bar(&self, bar: &Bar, portfolio: &Portfolio) -> Result<Vec<Signal>>;
    async fn on_complete(&self) -> Result<serde_json::Value>;
}
```

Add `async-trait = "0.1"` to `core/Cargo.toml`.

**Step 3: Implement BacktestEngine::run**

```rust
pub struct BacktestResult {
    pub trades: Vec<ClosedTrade>,
    pub equity_curve: Vec<EquityPoint>,
    pub final_equity: f64,
    pub config: BacktestConfig,
    pub custom_metrics: serde_json::Value,
}

pub struct BacktestEngine;

impl BacktestEngine {
    pub async fn run(
        config: BacktestConfig,
        bars: Vec<Bar>,
        strategy: &dyn StrategyClient,
    ) -> Result<BacktestResult> {
        strategy.initialize(
            &config.strategy_name,
            &serde_json::to_string(&config.strategy_params)?,
            &config.symbols,
        ).await?;

        let cost_model = ZerodhaCostModel;
        let mut portfolio = PortfolioManager::new(config.initial_capital);
        let mut matcher = OrderMatcher::new(config.slippage_pct);

        for bar in &bars {
            // 1. Process pending orders from previous bars
            let fills = matcher.process_bar(bar);
            for fill in &fills {
                let costs = cost_model.calculate(/* ... */);
                portfolio.apply_fill(fill, costs.total());
            }

            // 2. Update prices
            let mut prices = std::collections::HashMap::new();
            prices.insert(bar.symbol.clone(), bar.close);
            portfolio.update_prices(&prices, bar.timestamp_ms);

            // 3. Get strategy signals
            let signals = strategy.on_bar(bar, &portfolio.portfolio_state()).await?;

            // 4. Submit new orders
            for signal in signals {
                if signal.action != Action::Hold {
                    matcher.submit(Order::from_signal(&signal));
                }
            }
        }

        let custom_metrics = strategy.on_complete().await?;

        Ok(BacktestResult {
            trades: portfolio.closed_trades().to_vec(),
            equity_curve: portfolio.equity_curve().to_vec(),
            final_equity: portfolio.portfolio_state().equity(),
            config,
            custom_metrics,
        })
    }
}
```

**Step 4: Run tests**

Run: `cd engine && cargo test -p backtest-core -- engine`
Expected: PASS.

**Step 5: Commit**

```bash
git add engine/crates/core/
git commit -m "feat: add backtest engine event loop with mock strategy support"
```

---

### Task 14: gRPC strategy client (connects Rust engine to Python server)

**Files:**
- Create: `engine/crates/core/src/grpc_client.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Implement GrpcStrategyClient**

```rust
use backtest_proto::backtest::strategy_service_client::StrategyServiceClient;
use tonic::transport::Channel;

pub struct GrpcStrategyClient {
    client: StrategyServiceClient<Channel>,
}

impl GrpcStrategyClient {
    pub async fn connect(addr: &str) -> Result<Self> {
        let client = StrategyServiceClient::connect(addr.to_string()).await?;
        Ok(Self { client })
    }
}

#[async_trait::async_trait]
impl StrategyClient for GrpcStrategyClient {
    async fn initialize(&self, name: &str, config: &str, symbols: &[String]) -> Result<()> {
        let resp = self.client.clone().initialize(InitRequest {
            strategy_name: name.into(),
            config_json: config.into(),
            symbols: symbols.to_vec(),
        }).await?.into_inner();
        if !resp.success {
            anyhow::bail!("Strategy init failed: {}", resp.error);
        }
        Ok(())
    }

    async fn on_bar(&self, bar: &Bar, portfolio: &Portfolio) -> Result<Vec<Signal>> {
        // Convert Bar + Portfolio to proto BarEvent
        // Call gRPC, convert proto signals back to domain Signals
        todo!("implement proto conversion and gRPC call")
    }

    async fn on_complete(&self) -> Result<serde_json::Value> {
        let resp = self.client.clone().on_complete(CompleteRequest {}).await?.into_inner();
        let metrics: serde_json::Value = serde_json::from_str(&resp.custom_metrics_json)?;
        Ok(metrics)
    }
}
```

**Step 2: Implement Bar/Portfolio/Signal proto conversions**

Write `From<Bar> for proto::BarEvent`, `From<proto::Signal> for Signal`, etc.

**Step 3: Commit**

```bash
git add engine/crates/core/
git commit -m "feat: add gRPC strategy client connecting Rust engine to Python server"
```

---

## Phase 5: Results & Metrics

**Goal:** Compute all performance metrics from backtest results and output JSON/Parquet files.

---

### Task 15: Performance metrics calculator

**Files:**
- Create: `engine/crates/core/src/metrics.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let daily_returns = vec![0.01, -0.005, 0.008, 0.002, -0.003, 0.012, -0.001];
        let sharpe = calculate_sharpe(&daily_returns, 0.06); // 6% risk-free
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity_curve = vec![100.0, 110.0, 105.0, 95.0, 100.0, 115.0];
        let (dd_pct, peak_idx, trough_idx) = max_drawdown(&equity_curve);
        assert!((dd_pct - 0.1364).abs() < 0.01); // (110-95)/110
    }

    #[test]
    fn test_win_rate() {
        let trades = vec![
            ClosedTrade { pnl: 100.0, ..Default::default() },
            ClosedTrade { pnl: -50.0, ..Default::default() },
            ClosedTrade { pnl: 200.0, ..Default::default() },
            ClosedTrade { pnl: -30.0, ..Default::default() },
        ];
        let stats = trade_statistics(&trades);
        assert_eq!(stats.win_rate, 0.5);
        assert_eq!(stats.total_trades, 4);
        assert_eq!(stats.profit_factor, 300.0 / 80.0);
    }

    #[test]
    fn test_cagr() {
        let cagr = calculate_cagr(100_000.0, 150_000.0, 365); // 1 year, 50% return
        assert!((cagr - 0.5).abs() < 0.01);
    }
}
```

**Step 2: Implement metrics functions**

- `calculate_sharpe(daily_returns, risk_free_rate) -> f64`
- `calculate_sortino(daily_returns, risk_free_rate) -> f64`
- `max_drawdown(equity_values) -> (pct, peak_idx, trough_idx)`
- `max_drawdown_duration(equity_values) -> i64` (in bars)
- `calculate_cagr(start_value, end_value, days) -> f64`
- `trade_statistics(trades) -> TradeStats` (win_rate, avg_win, avg_loss, profit_factor, avg_holding_period)
- `calculate_volatility(daily_returns) -> f64` (annualized)
- `calculate_calmar(cagr, max_dd) -> f64`
- `MetricsReport` struct collecting all of the above
- `MetricsReport::compute(result: &BacktestResult) -> MetricsReport`

**Step 3: Run tests**

Run: `cd engine && cargo test -p backtest-core -- metrics`
Expected: PASS.

**Step 4: Commit**

```bash
git add engine/crates/core/src/metrics.rs
git commit -m "feat: add performance metrics (Sharpe, Sortino, max drawdown, CAGR, win rate)"
```

---

### Task 16: Results reporter (JSON + Parquet output)

**Files:**
- Create: `engine/crates/core/src/reporter.rs`
- Modify: `engine/crates/core/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load_results() {
        let dir = TempDir::new().unwrap();
        let result = BacktestResult { /* mock data */ };
        let reporter = Reporter::new(dir.path());
        let id = reporter.save(&result).unwrap();

        // Verify files created
        assert!(dir.path().join(&id).join("config.json").exists());
        assert!(dir.path().join(&id).join("trades.parquet").exists());
        assert!(dir.path().join(&id).join("equity_curve.parquet").exists());
        assert!(dir.path().join(&id).join("metrics.json").exists());

        // Verify metrics can be read back
        let metrics = reporter.load_metrics(&id).unwrap();
        assert!(metrics.sharpe_ratio.is_finite());
    }
}
```

**Step 2: Implement Reporter**

- `Reporter::new(base_path)` — points to `results/` directory
- `save(result) -> backtest_id` — generates UUID-based ID, writes:
  - `config.json` — serialized BacktestConfig
  - `trades.parquet` — ClosedTrade records (entry/exit time, price, P&L, costs)
  - `equity_curve.parquet` — EquityPoint records (timestamp, equity)
  - `metrics.json` — MetricsReport
- `list() -> Vec<(id, summary)>` — lists all saved backtests
- `load_metrics(id) -> MetricsReport`

**Step 3: Run tests**

Run: `cd engine && cargo test -p backtest-core -- reporter`
Expected: PASS.

**Step 4: Commit**

```bash
git add engine/crates/core/src/reporter.rs
git commit -m "feat: add results reporter with JSON and Parquet output"
```

---

## Phase 6: CLI

**Goal:** A working CLI binary that ties everything together. You can fetch data, run backtests, and view results from the command line.

---

### Task 17: CLI commands

**Files:**
- Modify: `engine/crates/cli/src/main.rs`
- Create: `engine/crates/cli/src/commands/mod.rs`
- Create: `engine/crates/cli/src/commands/data.rs`
- Create: `engine/crates/cli/src/commands/run.rs`
- Create: `engine/crates/cli/src/commands/results.rs`

**Step 1: Define CLI structure with clap**

```rust
// engine/crates/cli/src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "backtest", about = "Indian market backtesting platform")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage market data
    Data {
        #[command(subcommand)]
        action: DataCommands,
    },
    /// Run a backtest
    Run {
        #[arg(long)]
        strategy: String,
        #[arg(long, value_delimiter = ',')]
        symbols: Vec<String>,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, default_value = "1000000")]
        capital: f64,
        #[arg(long, default_value = "day")]
        interval: String,
        #[arg(long, default_value = "{}")]
        params: String,
        #[arg(long, default_value = "50051")]
        strategy_port: u16,
    },
    /// View backtest results
    Results {
        #[command(subcommand)]
        action: ResultsCommands,
    },
}

#[derive(Subcommand)]
enum DataCommands {
    /// Fetch data from Kite API
    Fetch {
        #[arg(long)]
        symbol: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, default_value = "day")]
        interval: String,
    },
    /// List cached data
    List,
}

#[derive(Subcommand)]
enum ResultsCommands {
    /// List all backtest results
    List,
    /// Show details of a specific backtest
    Show { id: String },
}
```

**Step 2: Implement the `data fetch` command**

- Reads `KITE_API_KEY` and `KITE_ACCESS_TOKEN` from environment
- Calls `KiteClient::fetch_candles()`, handles the 2000-candle chunking
- Writes to `CandleStore`
- Prints progress: "Fetched 2000 candles for RELIANCE (2023-01-01 to 2023-02-15)..."

**Step 3: Implement the `run` command**

- Parses config into `BacktestConfig`
- Reads candles from `CandleStore`
- Connects to Python strategy server via `GrpcStrategyClient`
- Runs `BacktestEngine::run()`
- Saves results via `Reporter`
- Prints summary metrics

**Step 4: Implement the `results list` and `results show` commands**

- `list`: scans results directory, prints table of backtest IDs with date, strategy, equity
- `show {id}`: loads metrics.json, prints formatted report

**Step 5: Build and test CLI**

Run: `cd engine && cargo build -p backtest-cli`
Run: `./target/debug/backtest --help`
Expected: Shows help with data, run, results subcommands.

**Step 6: Commit**

```bash
git add engine/crates/cli/
git commit -m "feat: add CLI with data fetch, backtest run, and results commands"
```

---

## Phase 7: End-to-End Integration Test

**Goal:** Verify the entire pipeline works: fetch/generate test data → start Python server → run backtest → check results.

---

### Task 18: End-to-end integration test

**Files:**
- Create: `tests/e2e_test.sh`

**Step 1: Write the integration test script**

```bash
#!/usr/bin/env bash
# tests/e2e_test.sh
set -euo pipefail

echo "=== Building Rust engine ==="
cd engine && cargo build --release && cd ..

echo "=== Setting up Python environment ==="
cd strategies
source .venv/bin/activate
./generate_proto.sh

echo "=== Starting strategy server in background ==="
python -m server.server &
SERVER_PID=$!
sleep 2

echo "=== Generating test data ==="
# Create synthetic RELIANCE daily data (or use a small real dataset if API key available)
cd ../engine
cargo run --release -p backtest-cli -- data list

echo "=== Running backtest ==="
cargo run --release -p backtest-cli -- run \
  --strategy sma_crossover \
  --symbols RELIANCE \
  --from 2023-01-01 \
  --to 2023-12-31 \
  --capital 1000000 \
  --interval day \
  --params '{"fast_period": 10, "slow_period": 30}'

echo "=== Viewing results ==="
cargo run --release -p backtest-cli -- results list

echo "=== Stopping strategy server ==="
kill $SERVER_PID

echo "=== All tests passed ==="
```

**Step 2: Create a test data generator**

Add a `generate-test-data` subcommand to the CLI that creates synthetic OHLCV data (random walk with trend) for testing without a Kite API key.

**Step 3: Run the integration test**

Run: `chmod +x tests/e2e_test.sh && ./tests/e2e_test.sh`
Expected: Full pipeline completes, results are printed.

**Step 4: Commit**

```bash
git add tests/
git commit -m "feat: add end-to-end integration test script"
```

---

## Dependency Graph

```
Phase 1: Scaffolding ──► Phase 2: Core + Data ──► Phase 4: Engine ──► Phase 6: CLI ──► Phase 7: E2E
                    └──► Phase 3: Python Server ──┘                └──► Phase 5: Metrics
```

Tasks within each phase are sequential. Phases 2 and 3 can be done in parallel.

## Summary

| Phase | Tasks | What you get |
|-------|-------|-------------|
| 1. Scaffolding | 1-3 | Compiling workspace, proto codegen, Python project |
| 2. Core + Data | 4-7 | Domain types, SQLite, Parquet, Kite API |
| 3. Python Server | 8-9 | Working gRPC strategy server, SMA example |
| 4. Engine | 10-14 | Transaction costs, matching, portfolio, event loop, gRPC client |
| 5. Metrics | 15-16 | Sharpe/drawdown/CAGR computation, JSON+Parquet output |
| 6. CLI | 17 | Fully functional `backtest` command |
| 7. E2E | 18 | End-to-end verification |
