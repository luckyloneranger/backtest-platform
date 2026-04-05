"""LLM strategy base class.

Subclass LLMStrategy to build strategies that use an LLM for signal generation.
Override required_data(), initialize_strategy(), and build_prompt().
"""

import json
import logging
import re
from abc import abstractmethod

from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.llm_client import AzureOpenAIClient

logger = logging.getLogger(__name__)


class LLMStrategy(Strategy):
    """Base class for LLM-powered strategies.

    Handles Azure OpenAI client setup, snapshot formatting, and signal parsing.
    Subclasses implement build_prompt() to define the LLM interaction.
    """

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.client = AzureOpenAIClient()
        self.instruments = instruments
        self.llm_config = {
            "temperature": config.get("temperature", 0.0),
            "max_tokens": config.get("max_tokens", 512),
        }
        self.initialize_strategy(config, instruments)

    @abstractmethod
    def initialize_strategy(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        """Strategy-specific initialization. Called after LLM client is set up."""
        pass

    @abstractmethod
    def build_prompt(self, snapshot: MarketSnapshot) -> list[dict]:
        """Build the messages list for the LLM chat completion call."""
        pass

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Call LLM with the prompt and parse the response into signals."""
        try:
            messages = self.build_prompt(snapshot)
            response = self.client.chat_completion(messages, **self.llm_config)
            return self.parse_signals(response)
        except Exception:
            logger.exception("LLM call failed, returning no signals")
            return []

    def format_snapshot(self, snapshot: MarketSnapshot) -> str:
        """Convert a MarketSnapshot into a compact text summary for the LLM."""
        lines = [f"Timestamp: {snapshot.timestamp_ms}"]

        # Portfolio
        p = snapshot.portfolio
        lines.append(f"Portfolio: cash={p.cash:.2f}, equity={p.equity:.2f}")
        if p.positions:
            for pos in p.positions:
                lines.append(
                    f"  Position: {pos.symbol} qty={pos.quantity} "
                    f"avg={pos.avg_price:.2f} pnl={pos.unrealized_pnl:.2f}"
                )

        # Current bars
        for interval, bars in snapshot.timeframes.items():
            lines.append(f"Timeframe: {interval}")
            for symbol, bar in bars.items():
                lines.append(
                    f"  {symbol}: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
                    f"C={bar.close:.2f} V={bar.volume} OI={bar.oi}"
                )

        # Pending orders
        if snapshot.pending_orders:
            lines.append("Pending orders:")
            for po in snapshot.pending_orders:
                lines.append(f"  {po.symbol}: {po.order_type} {po.side} qty={po.quantity} "
                             f"limit={po.limit_price} stop={po.stop_price}")

        # Recent history (last 5 bars per symbol/interval)
        if snapshot.history:
            lines.append("Recent history:")
            for (symbol, interval), bars in snapshot.history.items():
                recent = bars[-5:]
                closes = [f"{b.close:.2f}" for b in recent]
                lines.append(
                    f"  {symbol}/{interval} last {len(recent)} closes: "
                    f"{', '.join(closes)}"
                )

        # Context
        ctx = snapshot.context
        lines.append(
            f"Bar {ctx.bar_number}/{ctx.total_bars}, "
            f"capital={ctx.initial_capital:.0f}"
        )

        return "\n".join(lines)

    def parse_signals(self, llm_response: str) -> list[Signal]:
        """Parse LLM response into a list of Signal objects.

        Expects a JSON array. Handles markdown code blocks.
        Returns empty list on parse failure.
        """
        VALID_ORDER_TYPES = {"MARKET", "LIMIT", "SL", "SL_M"}
        VALID_PRODUCT_TYPES = {"CNC", "MIS", "NRML"}
        # Try to extract JSON from markdown code block
        md_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", llm_response, re.DOTALL
        )
        text = md_match.group(1).strip() if md_match else llm_response.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response as JSON: %s",
                llm_response[:200],
            )
            return []

        if not isinstance(data, list):
            logger.warning("LLM response is not a JSON array: %s", type(data))
            return []

        signals = []
        for item in data:
            action = item.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD", "CANCEL"):
                continue
            if action == "HOLD":
                continue

            order_type = item.get("order_type", "MARKET").upper()
            if order_type not in VALID_ORDER_TYPES:
                order_type = "MARKET"

            product_type = item.get("product_type", "CNC").upper()
            if product_type not in VALID_PRODUCT_TYPES:
                product_type = "CNC"

            # Require limit_price for LIMIT orders
            if order_type == "LIMIT" and float(item.get("limit_price", 0.0)) <= 0:
                logger.warning("LIMIT order without limit_price, defaulting to MARKET")
                order_type = "MARKET"

            # Require stop_price for SL/SL_M orders
            if order_type in ("SL", "SL_M") and float(item.get("stop_price", 0.0)) <= 0:
                logger.warning("SL/SL_M order without stop_price, skipping signal")
                continue

            signals.append(Signal(
                action=action,
                symbol=item.get("symbol", ""),
                quantity=int(item.get("quantity", 0)),
                order_type=order_type,
                limit_price=float(item.get("limit_price", 0.0)),
                stop_price=float(item.get("stop_price", 0.0)),
                product_type=product_type,
            ))

        return signals
