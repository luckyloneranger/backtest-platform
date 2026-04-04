"""gRPC strategy server.

Hosts the StrategyService, delegating calls to the currently loaded
Python strategy via the registry.
"""

import json
import grpc
from concurrent import futures

from server.generated import strategy_pb2
from server.generated import strategy_pb2_grpc
from server.registry import get_strategy
from strategies.base import (
    BarData, InstrumentInfo, Position, Portfolio, FillInfo,
    OrderRejection, TradeInfo, SessionContext, MarketSnapshot, Signal,
)

# ------------------------------------------------------------------
# Import strategy modules here so that the @register decorators run.
# ------------------------------------------------------------------
import strategies.examples.sma_crossover  # noqa: F401


class StrategyServicer(strategy_pb2_grpc.StrategyServiceServicer):
    """Implements the gRPC StrategyService by delegating to a Python
    Strategy instance looked up from the registry."""

    def __init__(self):
        self.strategy = None

    def Initialize(self, request, context):
        try:
            self.strategy = get_strategy(request.strategy_name)
            config = json.loads(request.config_json) if request.config_json else {}

            # Convert proto instruments to Python
            instruments = {}
            for inst in request.instruments:
                instruments[inst.symbol] = InstrumentInfo(
                    symbol=inst.symbol,
                    exchange=inst.exchange,
                    instrument_type=inst.instrument_type,
                    lot_size=inst.lot_size,
                    tick_size=inst.tick_size,
                    expiry=inst.expiry,
                    strike=inst.strike,
                    option_type=inst.option_type,
                    circuit_limit_upper=inst.circuit_limit_upper,
                    circuit_limit_lower=inst.circuit_limit_lower,
                )

            self.strategy.initialize(config, instruments)
            return strategy_pb2.InitResponse(success=True, error="")
        except Exception as e:
            return strategy_pb2.InitResponse(success=False, error=str(e))

    def OnBar(self, request, context):
        if self.strategy is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Strategy not initialized. Call Initialize first.")
            return strategy_pb2.BarResponse()

        # Convert bars
        bars = {}
        for b in request.bars:
            bars[b.symbol] = BarData(
                symbol=b.symbol, open=b.open, high=b.high,
                low=b.low, close=b.close, volume=b.volume, oi=b.oi,
            )

        # Convert history
        history = {}
        for sh in request.history:
            history[sh.symbol] = [
                BarData(symbol=sh.symbol, open=b.open, high=b.high,
                        low=b.low, close=b.close, volume=b.volume, oi=b.oi)
                for b in sh.bars
            ]

        # Convert portfolio
        portfolio = Portfolio(
            cash=request.portfolio.cash,
            equity=request.portfolio.equity,
            positions=[
                Position(p.symbol, p.quantity, p.avg_price, p.unrealized_pnl)
                for p in request.portfolio.positions
            ],
        )

        # Convert instruments
        instruments = {}
        for inst in request.instruments:
            instruments[inst.symbol] = InstrumentInfo(
                symbol=inst.symbol, exchange=inst.exchange,
                instrument_type=inst.instrument_type, lot_size=inst.lot_size,
                tick_size=inst.tick_size, expiry=inst.expiry, strike=inst.strike,
                option_type=inst.option_type,
                circuit_limit_upper=inst.circuit_limit_upper,
                circuit_limit_lower=inst.circuit_limit_lower,
            )

        # Convert fills
        fills = [
            FillInfo(f.symbol, f.side, f.quantity, f.fill_price, f.costs, f.timestamp_ms)
            for f in request.fills
        ]

        # Convert rejections
        rejections = [
            OrderRejection(r.symbol, r.side, r.quantity, r.reason)
            for r in request.rejections
        ]

        # Convert closed trades
        closed_trades = [
            TradeInfo(t.symbol, t.quantity, t.entry_price, t.exit_price,
                      t.entry_timestamp_ms, t.exit_timestamp_ms, t.pnl, t.costs)
            for t in request.closed_trades
        ]

        # Convert context — in proto3 message fields are always present,
        # so we always construct SessionContext from whatever is there.
        ctx = SessionContext(
            initial_capital=request.context.initial_capital,
            bar_number=request.context.bar_number,
            total_bars=request.context.total_bars,
            start_date=request.context.start_date,
            end_date=request.context.end_date,
            interval=request.context.interval,
            lookback_window=request.context.lookback_window,
        )

        snapshot = MarketSnapshot(
            timestamp_ms=request.timestamp_ms,
            bars=bars,
            history=history,
            portfolio=portfolio,
            instruments=instruments,
            fills=fills,
            rejections=rejections,
            closed_trades=closed_trades,
            context=ctx,
        )

        signals = self.strategy.on_bar(snapshot)

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
    """Start the gRPC server and block until termination."""
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
