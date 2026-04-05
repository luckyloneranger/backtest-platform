"""LLM Signal Generator — direct signal generation via Azure OpenAI.

Sends OHLCV data and portfolio state to the LLM each bar and asks it
to return BUY/SELL signals as a JSON array.
"""

from server.registry import register
from strategies.base import MarketSnapshot, InstrumentInfo
from strategies.llm_base import LLMStrategy

DEFAULT_SYSTEM_PROMPT = """\
You are a quantitative trading assistant. You receive market data, portfolio state, and pending orders. \
You must decide trading actions for each symbol.

Respond ONLY with a JSON array of signals. Each signal is an object with:
- "action": "BUY", "SELL", or "CANCEL"
- "symbol": the stock symbol
- "quantity": number of shares (integer, positive). Not needed for CANCEL.
- "order_type": "MARKET" (immediate fill), "LIMIT" (fill at limit_price or better), "SL_M" (stop-loss market: triggers at stop_price, fills at market)
- "limit_price": required for LIMIT orders — the maximum buy price or minimum sell price
- "stop_price": required for SL_M orders — the trigger price for the stop-loss
- "product_type": "CNC" (delivery, hold overnight/multi-day) or "MIS" (intraday, auto-closed at 3:20 PM IST)

Strategy guidance:
- Use LIMIT buy orders for mean-reversion entries (buy below current market price)
- Use MARKET orders for breakout entries (immediate execution needed)
- Use SL_M sell orders to set automatic stop-losses after entering a position
- Use CANCEL to remove pending limit or stop-loss orders for a symbol
- Use CNC for high-conviction multi-day trades, MIS for intraday trades
- Position size should not exceed {risk_pct:.0%} of available cash
- Return [] if no action should be taken
- Do NOT include any text outside the JSON array

Example response:
[
  {{"action": "BUY", "symbol": "RELIANCE", "quantity": 50, "order_type": "LIMIT", "limit_price": 1200.00, "product_type": "CNC"}},
  {{"action": "SELL", "symbol": "RELIANCE", "quantity": 50, "order_type": "SL_M", "stop_price": 1150.00, "product_type": "CNC"}},
  {{"action": "CANCEL", "symbol": "INFY"}}
]
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
