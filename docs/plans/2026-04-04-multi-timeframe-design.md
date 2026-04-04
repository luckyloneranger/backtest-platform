# Multi-Timeframe Strategy Support — Design

**Date**: 2026-04-04
**Status**: Approved

## Problem

Strategies can only receive bars at a single interval per backtest run. A strategy needing both 5-minute bars for signals and daily bars for trend confirmation must run two separate backtests. There's no way to combine timeframes, and the CLI (not the strategy) decides the interval.

## Principle

**The strategy declares what data it needs. The engine provides it.**

## Solution

### 1. New RPC: GetRequirements

Before loading data or initializing, the engine calls `GetRequirements` to ask the strategy what intervals and lookback it needs.

```protobuf
rpc GetRequirements(RequirementsRequest) returns (DataRequirements);

message RequirementsRequest {
  string strategy_name = 1;
  string config_json = 2;
}

message DataRequirements {
  repeated IntervalRequirement intervals = 1;
}

message IntervalRequirement {
  string interval = 1;   // "minute", "5minute", "day"
  int32 lookback = 2;    // bars to keep in history
}
```

Python strategy declares:
```python
class Strategy(ABC):
    @abstractmethod
    def required_data(self) -> list[dict]:
        """Return [{"interval": "minute", "lookback": 50}, {"interval": "day", "lookback": 200}]"""
        pass
```

### 2. Multi-Timeframe BarEvent

```protobuf
message BarEvent {
  int64 timestamp_ms = 1;
  repeated TimeframeData timeframes = 2;      // interval → bars at this timestamp
  repeated TimeframeHistory history = 3;      // (symbol, interval) → lookback
  PortfolioState portfolio = 4;
  repeated InstrumentInfo instruments = 5;
  repeated FillInfo fills = 6;
  repeated OrderRejection rejections = 7;
  repeated TradeInfo closed_trades = 8;
  SessionContext context = 9;
}

message TimeframeData {
  string interval = 1;
  repeated BarData bars = 2;
}

message TimeframeHistory {
  string symbol = 1;
  string interval = 2;
  repeated BarData bars = 3;
}
```

### 3. Engine Behavior

- **Tick rate**: Finest declared interval (e.g., if strategy needs minute + day, tick every minute)
- **Coarser bars**: Included only when a new candle closes at that interval's boundary
- **Lookback**: Maintained per (symbol, interval) with separate buffer sizes
- **Data loading**: CLI loads candle data for each (symbol, interval) combination from CandleStore

### 4. Python MarketSnapshot

```python
@dataclass
class MarketSnapshot:
    timestamp_ms: int
    timeframes: dict[str, dict[str, BarData]]          # interval → symbol → bar
    history: dict[tuple[str, str], list[BarData]]       # (symbol, interval) → bars
    portfolio: Portfolio
    instruments: dict[str, InstrumentInfo]
    fills: list[FillInfo]
    rejections: list[OrderRejection]
    closed_trades: list[TradeInfo]
    context: SessionContext
```

### 5. Flow

```
1. CLI calls GetRequirements(strategy_name, config)
2. Gets back: [{interval: "minute", lookback: 50}, {interval: "day", lookback: 200}]
3. CLI loads CandleStore data for each (symbol, interval)
4. CLI calls Initialize(...)
5. Engine ticks at finest interval (minute)
6. Each tick: snapshot includes minute bar + day bar (only when new day candle closes)
7. Strategy processes snapshot, returns signals
```
