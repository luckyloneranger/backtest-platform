from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BarData:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int
    timestamp_ms: int = 0


@dataclass
class InstrumentInfo:
    symbol: str
    exchange: str
    instrument_type: str     # "EQ", "FUT", "OPT", "COM"
    lot_size: int
    tick_size: float
    expiry: str
    strike: float
    option_type: str         # "CE", "PE", or ""
    circuit_limit_upper: float
    circuit_limit_lower: float


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
class FillInfo:
    symbol: str
    side: str                # "BUY", "SELL"
    quantity: int
    fill_price: float
    costs: float
    timestamp_ms: int


@dataclass
class OrderRejection:
    symbol: str
    side: str
    quantity: int
    reason: str              # "CIRCUIT_LIMIT", "INSUFFICIENT_MARGIN"


@dataclass
class TradeInfo:
    symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    pnl: float
    costs: float


@dataclass
class SessionContext:
    initial_capital: float
    bar_number: int
    total_bars: int
    start_date: str
    end_date: str
    intervals: list[str]
    lookback_window: int


@dataclass
class Signal:
    action: str              # "HOLD", "BUY", "SELL"
    symbol: str
    quantity: int
    order_type: str = "MARKET"    # "MARKET", "LIMIT", "SL", "SL_M"
    limit_price: float = 0.0
    stop_price: float = 0.0
    product_type: str = "CNC"     # "CNC" (delivery), "MIS" (intraday), "NRML" (F&O)


@dataclass
class MarketSnapshot:
    """Full market context sent to strategy per on_bar call."""
    timestamp_ms: int
    timeframes: dict[str, dict[str, BarData]]          # interval -> symbol -> bar
    history: dict[tuple[str, str], list[BarData]]       # (symbol, interval) -> bars
    portfolio: Portfolio
    instruments: dict[str, InstrumentInfo]
    fills: list[FillInfo]
    rejections: list[OrderRejection]
    closed_trades: list[TradeInfo]
    context: SessionContext


class Strategy(ABC):
    @abstractmethod
    def required_data(self) -> list[dict]:
        """Declare data requirements. Called before initialize().
        Return: [{"interval": "minute", "lookback": 50}, {"interval": "day", "lookback": 200}]
        """
        pass

    @abstractmethod
    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        """Called once with strategy parameters and instrument metadata."""
        pass

    @abstractmethod
    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Called on each timestamp with full market snapshot."""
        pass

    def on_complete(self) -> dict:
        """Called at backtest end. Return any custom metrics."""
        return {}
