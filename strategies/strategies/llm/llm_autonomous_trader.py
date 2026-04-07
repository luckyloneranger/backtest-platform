"""LLM Autonomous Trader — thesis-driven AI portfolio manager.

Receives factual data summaries (not interpreted conclusions) and builds
its own investment theses. The LLM acts as a portfolio manager who must
articulate WHY before acting. Safety guardrails enforce risk limits.
"""

from collections import deque
import json
import logging
import re

from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.llm_client import AzureOpenAIClient, LLMClientError
from strategies.position_manager import PositionManager
from strategies.indicators import compute_features, compute_atr
from strategies.narrative_builder import (
    build_symbol_narrative,
    build_regime_narrative,
    build_portfolio_narrative,
    build_trade_history_narrative,
)
from server.registry import register

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a portfolio manager for a \u20b9{capital:.0f} Indian equity portfolio trading {n_stocks} NSE stocks.

YOUR PROCESS FOR EACH DECISION:

1. THESIS: For any stock you're considering, build an investment thesis.
   - What is your view? (bullish / bearish / neutral)
   - WHY do you hold this view? What's the story — not just "RSI is low" but WHY that matters for THIS stock right now.
   - What's the catalyst for the move you're expecting?

2. EVIDENCE: What in the data supports your thesis?
   - Which indicators confirm your thesis?
   - What's the price action telling you?
   - Does volume confirm or contradict?

3. COUNTER-THESIS: What would prove you wrong?
   - What's the bear case if you're bullish (or vice versa)?
   - What data would make you abandon this thesis?
   - Are there conflicting signals you're choosing to ignore? Be honest about them.

4. CONVICTION: Rate 1-10. Only act on 7 or above.
   - 1-3: Interesting but not tradeable
   - 4-6: Thesis exists but risks are too high or timing unclear
   - 7-8: Good setup with manageable risk
   - 9-10: High conviction, clear thesis with strong evidence

5. ACTION: If convicted (7+), what specifically?
   - Entry price, stop-loss level, target, position size
   - CNC (will hold for days/weeks) or MIS (exit today)

PORTFOLIO CONSTRAINTS:
- Capital: \u20b9{capital:.0f}
- Maximum {max_positions} positions at once
- Maximum 3% of capital at risk per trade
- Always set a stop-loss (SL_M order)

RESPOND with a JSON array. Each signal must include:
{{"action": "BUY"|"SELL"|"CANCEL", "symbol": "...", "quantity": N, \
"order_type": "MARKET"|"LIMIT"|"SL_M", "limit_price": 0.0, "stop_price": 0.0, \
"product_type": "CNC"|"MIS", \
"reasoning": "THESIS: ... EVIDENCE: ... CONVICTION: N/10"}}

Return [] if no thesis reaches conviction 7+. Sitting in cash is a valid decision.
"""


@register("llm_autonomous_trader")
class LLMAutonomousTrader(Strategy):
    """Thesis-driven AI portfolio manager that receives factual data
    summaries and builds its own investment theses, constrained by
    safety guardrails."""

    def required_data(self) -> list[dict]:
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.client = AzureOpenAIClient()
        self.instruments = instruments
        self.pm = PositionManager(max_pending_bars=1)

        # Config
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 1024)
        self.max_positions = config.get("max_positions", 4)
        self.max_daily_trades = config.get("max_daily_trades", 3)
        self.risk_pct = config.get("risk_pct", 0.03)
        self.auto_stop_pct = config.get("auto_stop_pct", 0.03)

        # Per-symbol data buffers
        self.closes: dict[str, deque] = {}
        self.highs: dict[str, deque] = {}
        self.lows: dict[str, deque] = {}
        self.volumes: dict[str, deque] = {}

        # Trade log for LLM memory
        self.trade_log: list[dict] = []
        self.total_costs: float = 0.0
        self.initial_capital: float = config.get("initial_capital", 100_000)
        self.peak_equity: float = self.initial_capital
        self.trades_today: int = 0
        self.last_bar_number: int = -1

        # Number of stocks (for system prompt)
        self.n_stocks = len(instruments) if instruments else 0

        # Format system prompt with capital and stock count
        self.system_prompt = SYSTEM_PROMPT.format(
            capital=self.initial_capital,
            max_positions=self.max_positions,
            n_stocks=self.n_stocks,
        )

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # Only act on daily bars
        if "day" not in snapshot.timeframes:
            return signals

        # Reset daily trade counter on new bar
        if snapshot.context.bar_number != self.last_bar_number:
            self.trades_today = 0
            self.last_bar_number = snapshot.context.bar_number

        # 1. Update buffers + compute indicators for all symbols
        all_features: dict[str, dict | None] = {}
        all_atrs: dict[str, float | None] = {}
        for symbol, bar in snapshot.timeframes["day"].items():
            self._update_buffers(symbol, bar)
            features = compute_features(
                list(self.closes[symbol]),
                list(self.highs[symbol]),
                list(self.lows[symbol]),
                list(self.volumes[symbol]),
            )
            all_features[symbol] = features
            all_atrs[symbol] = compute_atr(
                list(self.highs[symbol]),
                list(self.lows[symbol]),
                list(self.closes[symbol]),
                14,
            )

        # 2. Track closed trades for log
        for trade in snapshot.closed_trades:
            self._log_trade(trade)

        # Track costs from fills
        for fill in snapshot.fills:
            self.total_costs += fill.costs

        # Update peak equity
        self.peak_equity = max(self.peak_equity, snapshot.portfolio.equity)

        # 3. Build narrative dashboard
        narrative_parts: list[str] = []

        # Portfolio narrative
        narrative_parts.append(
            build_portfolio_narrative(
                snapshot.portfolio.cash,
                snapshot.portfolio.equity,
                self.peak_equity,
                snapshot.portfolio.positions,
                self.total_costs,
                len(self.trade_log),
            )
        )

        # Regime narrative (use average ADX/BBW across symbols)
        adx_values = [
            f.get("adx_14")
            for f in all_features.values()
            if f and f.get("adx_14") is not None
        ]
        avg_adx = sum(adx_values) / len(adx_values) if adx_values else None
        bbw_values = [
            f.get("bbw")
            for f in all_features.values()
            if f and f.get("bbw") is not None
        ]
        avg_bbw = sum(bbw_values) / len(bbw_values) if bbw_values else None
        narrative_parts.append(build_regime_narrative(avg_adx, avg_bbw, avg_bbw))

        # Per-symbol narratives
        for symbol, bar in snapshot.timeframes["day"].items():
            position = None
            for pos in snapshot.portfolio.positions:
                if pos.symbol == symbol:
                    pnl = (bar.close - pos.avg_price) * pos.quantity
                    position = {
                        "qty": pos.quantity,
                        "avg_price": pos.avg_price,
                        "unrealized_pnl": pnl,
                        "product_type": "CNC",
                    }
                    break
            narrative_parts.append(
                build_symbol_narrative(
                    symbol,
                    bar.close,
                    all_features.get(symbol),
                    all_atrs.get(symbol),
                    position,
                    snapshot.portfolio.equity,
                )
            )

        # Trade history narrative
        narrative_parts.append(
            build_trade_history_narrative(self.trade_log[-10:])
        )

        # Pending orders
        if snapshot.pending_orders:
            narrative_parts.append("\nPENDING ORDERS:")
            for po in snapshot.pending_orders:
                narrative_parts.append(
                    f"  {po.symbol}: {po.order_type} {po.side} qty={po.quantity} "
                    f"stop={po.stop_price}"
                )

        full_narrative = "\n".join(narrative_parts)

        # 4. Call LLM
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_narrative},
            ]
            response = self.client.chat_completion(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_signals = self._parse_llm_response(response)
        except (LLMClientError, Exception) as e:
            logger.warning("LLM call failed: %s — holding positions", e)
            return signals

        # 5. Apply guardrails + execute via PM
        for sig in raw_signals:
            if self.trades_today >= self.max_daily_trades:
                break

            validated = self._apply_guardrails(sig, snapshot, all_atrs)
            if validated:
                signals += validated
                if validated[0].action in ("BUY", "SELL"):
                    self.trades_today += 1

        return signals

    def on_complete(self) -> dict:
        return {
            "strategy_type": "llm_autonomous_trader",
            "total_trades": len(self.trade_log),
            "total_costs": self.total_costs,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_buffers(self, symbol: str, bar) -> None:
        """Append latest bar data to per-symbol deque buffers."""
        if symbol not in self.closes:
            self.closes[symbol] = deque(maxlen=300)
            self.highs[symbol] = deque(maxlen=300)
            self.lows[symbol] = deque(maxlen=300)
            self.volumes[symbol] = deque(maxlen=300)
        self.closes[symbol].append(bar.close)
        self.highs[symbol].append(bar.high)
        self.lows[symbol].append(bar.low)
        self.volumes[symbol].append(bar.volume)

    def _parse_llm_response(self, llm_response: str) -> list[dict]:
        """Parse LLM response into a list of raw signal dicts.

        Handles markdown code blocks and extracts reasoning.
        Returns empty list on parse failure.
        """
        # Strip markdown code blocks
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

        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            action = item.get("action", "").upper()
            if action not in ("BUY", "SELL", "CANCEL"):
                continue
            valid.append(item)

        return valid

    def _apply_guardrails(
        self,
        sig: dict,
        snapshot: MarketSnapshot,
        atrs: dict[str, float | None],
    ) -> list[Signal] | None:
        """Validate and constrain LLM signal. Returns list[Signal] or None."""
        # Validate symbol exists in our data
        known_symbols = set()
        for tf in snapshot.timeframes.values():
            known_symbols.update(tf.keys())

        symbol = sig.get("symbol", "")
        if symbol not in known_symbols:
            return None

        action = sig.get("action", "").upper()
        if action not in ("BUY", "SELL", "CANCEL"):
            return None

        if action == "CANCEL":
            return [Signal(action="CANCEL", symbol=symbol, quantity=0)]

        qty = int(sig.get("quantity", 0))
        if qty <= 0:
            return None

        # Get current price
        price = None
        for tf in snapshot.timeframes.values():
            if symbol in tf:
                price = tf[symbol].close
                break
        if not price or price <= 0:
            return None

        # Cap position size to available cash
        max_qty = int(snapshot.portfolio.cash / price)
        qty = min(qty, max_qty)
        if qty <= 0:
            return None

        # Max positions check
        open_positions = sum(
            1 for s in self.pm.states.values() if s.direction != "flat"
        )
        if (
            action == "BUY"
            and self.pm.is_flat(symbol)
            and open_positions >= self.max_positions
        ):
            return None

        # Drawdown scaling
        drawdown = (
            (self.peak_equity - snapshot.portfolio.equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        if drawdown > 0.10:
            qty = max(1, qty // 2)

        # Build the actual signal
        product_type = sig.get("product_type", "CNC").upper()
        if product_type not in ("CNC", "MIS"):
            product_type = "CNC"
        limit_price = float(sig.get("limit_price", 0.0))
        stop_price = float(sig.get("stop_price", 0.0))

        results: list[Signal] = []
        atr = atrs.get(symbol)

        if action == "BUY" and self.pm.is_flat(symbol):
            # Long entry
            if stop_price <= 0 and atr:
                stop_price = price - 2.0 * atr
            elif stop_price <= 0:
                stop_price = price * (1 - self.auto_stop_pct)
            results += self.pm.enter_long(
                symbol, qty, limit_price, product_type, stop_price
            )

        elif action == "SELL" and self.pm.is_flat(symbol):
            # Short entry — always MIS
            if stop_price <= 0 and atr:
                stop_price = price + 2.0 * atr
            elif stop_price <= 0:
                stop_price = price * (1 + self.auto_stop_pct)
            results += self.pm.enter_short(symbol, qty, limit_price, stop_price)

        elif action == "SELL" and self.pm.is_long(symbol):
            # Exit long
            results += self.pm.exit_position(symbol, qty=qty)

        elif action == "BUY" and self.pm.is_short(symbol):
            # Cover short
            results += self.pm.exit_position(symbol, qty=qty)

        # Log reasoning
        reasoning = sig.get("reasoning", "")
        if reasoning:
            logger.info("LLM %s %s: %s", action, symbol, reasoning)

        return results if results else None

    def _log_trade(self, trade) -> None:
        """Log a completed trade for the trade history narrative."""
        pnl_pct = (
            (trade.pnl / (trade.entry_price * trade.quantity) * 100)
            if trade.entry_price > 0 and trade.quantity > 0
            else 0
        )
        self.trade_log.append(
            {
                "symbol": trade.symbol,
                "side": "LONG"
                if trade.entry_price < trade.exit_price
                else "SHORT",
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_pct": pnl_pct,
                "reasoning": "",
                "bars_held": 0,
            }
        )
