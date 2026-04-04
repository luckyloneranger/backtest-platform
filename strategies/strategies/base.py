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
