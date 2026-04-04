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
from strategies.base import Bar, Portfolio, Position, Signal

# ------------------------------------------------------------------
# Import strategy modules here so that the @register decorators run.
# Example (uncomment when strategies exist):
#   import strategies.sma_crossover  # noqa: F401
# ------------------------------------------------------------------


class StrategyServicer(strategy_pb2_grpc.StrategyServiceServicer):
    """Implements the gRPC StrategyService by delegating to a Python
    Strategy instance looked up from the registry."""

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
        if self.strategy is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Strategy not initialized. Call Initialize first.")
            return strategy_pb2.BarResponse()

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
