"""LLM Signal Generator — direct signal generation via Azure OpenAI.

Sends OHLCV data and portfolio state to the LLM each bar and asks it
to return BUY/SELL signals as a JSON array.
"""

from server.registry import register
from strategies.base import MarketSnapshot, InstrumentInfo
from strategies.llm_base import LLMStrategy

DEFAULT_SYSTEM_PROMPT = """\
You are a quantitative trading assistant. You receive market data and portfolio state, \
and you must decide whether to BUY, SELL, or HOLD each symbol.

Respond ONLY with a JSON array of signals. Each signal is an object with:
- "action": "BUY" or "SELL" (omit symbols where you want to HOLD)
- "symbol": the stock symbol
- "quantity": number of shares (integer, positive)
- "product_type": "CNC" (delivery), "MIS" (intraday), or "NRML" (F&O)

Rules:
- Only trade symbols present in the market data
- Position size should not exceed {risk_pct:.0%} of available cash
- Return [] if no action should be taken
- Do NOT include any text outside the JSON array

Example response:
[{{"action": "BUY", "symbol": "RELIANCE", "quantity": 10, "product_type": "CNC"}}]
"""


@register("llm_signal_generator")
class LLMSignalGenerator(LLMStrategy):
    """Direct LLM signal generator — asks the LLM for trading decisions each bar."""

    def required_data(self) -> list[dict]:
        return [{"interval": "day", "lookback": 20}]

    def initialize_strategy(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.risk_pct = config.get("risk_pct", 0.2)
        self.system_prompt = config.get(
            "system_prompt",
            DEFAULT_SYSTEM_PROMPT.format(risk_pct=self.risk_pct),
        )

    def build_prompt(self, snapshot: MarketSnapshot) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.format_snapshot(snapshot)},
        ]

    def on_complete(self) -> dict:
        return {
            "strategy_type": "llm_signal_generator",
            "risk_pct": self.risk_pct,
        }
