# Enhanced Strategy Interface — Design

**Date**: 2026-04-04
**Status**: Approved

## Problem

Strategies currently receive only the current bar (OHLCV+OI) and portfolio state per `on_bar` call. They have no access to historical bars, instrument metadata, order feedback, or trade history. This forces strategies to maintain their own price buffers, prevents cross-symbol analysis, and provides no feedback when orders are silently rejected by circuit limits or margin checks.

## Solution

Enrich the `BarEvent` proto message to include a full `MarketSnapshot` per bar:

| Data | Description |
|------|-------------|
| Multi-symbol bars | All symbols' bars at the same timestamp in one call |
| Lookback window | Last N bars (default 200) per symbol |
| Instrument metadata | lot_size, tick_size, expiry, strike, option_type, exchange, circuit limits |
| Fill feedback | Fills from previous bar (symbol, side, qty, price, costs) |
| Rejection feedback | Orders rejected with reasons (CIRCUIT_LIMIT, INSUFFICIENT_MARGIN) |
| Closed trade history | All completed trades in the current backtest |
| Session context | initial_capital, bar_number, total_bars, start/end dates, interval |

## Approach

Enrich the existing `BarEvent` proto message (Approach 1). Proto3 additive fields mean old strategies still compile (they ignore new fields). Single RPC per tick — no architectural change.

## Python Strategy Interface

```python
@dataclass
class MarketSnapshot:
    timestamp_ms: int
    bars: dict[str, BarData]
    history: dict[str, list[BarData]]
    portfolio: Portfolio
    instruments: dict[str, InstrumentInfo]
    fills: list[FillInfo]
    rejections: list[OrderRejection]
    closed_trades: list[TradeInfo]
    context: SessionContext

class Strategy(ABC):
    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None: ...
    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]: ...
    def on_complete(self) -> dict: ...
```

## Engine Changes

- Group bars by timestamp for multi-symbol calls
- Maintain per-symbol lookback buffers (VecDeque, capped at lookback_window)
- Track fills and rejections between bars
- Load instrument metadata at backtest start
- Add `lookback_window` to BacktestConfig (default 200)
- OrderMatcher returns rejection info instead of silently skipping
