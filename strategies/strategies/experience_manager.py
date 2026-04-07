"""Experience Manager — learning layer for LLM autonomous trader.

Tracks trade journal with decision context, detects missed opportunities
on unheld stocks, runs weekly LLM reflections to produce evolving beliefs,
and persists everything to disk across sessions.

The beliefs document is injected into the trading system prompt, making
the LLM organically adapt its behavior based on accumulated experience.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

POSTMORTEM_PROMPT = """\
You are reviewing a completed trade. Write a 1-sentence post-mortem.

Trade: {action} {symbol} at ₹{entry_price:.2f} on {entry_date}.
Thesis at entry: {thesis}
Indicators at entry: {indicators}
Market regime: {regime}
Exit: ₹{exit_price:.2f} on {exit_date}. P&L: {pnl_pct:+.1f}%.
Outcome: {outcome}.

One sentence: what happened and why?"""

REFLECTION_PROMPT = """\
You are reviewing your trading performance over the past week.

COMPLETED TRADES THIS PERIOD:
{trades_text}

MISSED OPPORTUNITIES THIS PERIOD:
{missed_text}

YOUR CURRENT BELIEFS:
{beliefs_text}

Update your beliefs based on this evidence.
- What patterns are working? What's failing?
- Which missed opportunities had clear signals you should have acted on?
- What should you do differently?

Respond with ONLY a numbered list of beliefs (max 10).
Each belief must be specific and actionable:
  Good: "Banking breakouts in trending regimes (ADX>25) work — 3 of 4 profitable."
  Bad: "Be more careful." (too vague)

Keep beliefs still supported by evidence. Remove contradicted ones. Add new ones."""


class ExperienceManager:
    """Manages experiential learning for LLM trading strategies.

    Components:
      - Trade journal: entries/exits with thesis and market context
      - Missed opportunity detector: big moves on unheld stocks
      - Weekly reflection: LLM reviews experience, updates beliefs
      - Persistence: all data saved/loaded from JSON files
    """

    def __init__(
        self,
        experience_dir: str | None = None,
        reflection_interval: int = 5,
    ):
        self.experience_dir = experience_dir
        self.reflection_interval = reflection_interval

        # State
        self.trade_journal: list[dict] = []
        self.open_trades: dict[str, dict] = {}
        self.missed_opportunities: list[dict] = []
        self.beliefs: list[str] = []
        self.reflection_log: list[dict] = []

        # Counters
        self.trading_days_since_reflection: int = 0
        self.last_reflection_date: str = ""

        # Load persisted state if available
        if self.experience_dir:
            self.load()

    # ------------------------------------------------------------------
    # Trade Journal
    # ------------------------------------------------------------------

    def record_entry(
        self,
        symbol: str,
        action: str,
        thesis: str,
        indicators: dict,
        price: float,
        date: str,
        regime: str,
    ) -> None:
        """Record a new trade entry with full decision context."""
        self.open_trades[symbol] = {
            "symbol": symbol,
            "action": action,
            "thesis": thesis,
            "entry_indicators": indicators,
            "entry_price": price,
            "entry_date": date,
            "entry_regime": regime,
        }

    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        date: str,
        llm_client=None,
    ) -> None:
        """Record trade exit. Optionally generate LLM post-mortem."""
        entry = self.open_trades.pop(symbol, None)
        if entry is None:
            return

        outcome = "WIN" if pnl > 0 else "LOSS"

        post_mortem = ""
        if llm_client is not None:
            try:
                prompt = POSTMORTEM_PROMPT.format(
                    action=entry["action"],
                    symbol=symbol,
                    entry_price=entry["entry_price"],
                    entry_date=entry["entry_date"],
                    thesis=entry["thesis"],
                    indicators=json.dumps(entry["entry_indicators"]),
                    regime=entry["entry_regime"],
                    exit_price=exit_price,
                    exit_date=date,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                )
                post_mortem = llm_client.chat_completion(
                    [{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100,
                )
            except Exception as e:
                logger.warning("Post-mortem LLM call failed: %s", e)

        self.trade_journal.append({
            **entry,
            "exit_price": exit_price,
            "exit_date": date,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
            "post_mortem": post_mortem,
        })

    # ------------------------------------------------------------------
    # Missed Opportunity Detection
    # ------------------------------------------------------------------

    def detect_missed_opportunities(
        self,
        all_features: dict[str, dict | None],
        all_atrs: dict[str, float | None],
        held_symbols: set[str],
        date: str,
    ) -> None:
        """Check unheld stocks for significant 5-day moves (> 2x normalized ATR)."""
        for symbol, features in all_features.items():
            if symbol in held_symbols or features is None:
                continue
            atr_norm = features.get("atr_norm", 0)
            ret_5 = features.get("ret_5")
            if ret_5 is None or atr_norm <= 0:
                continue

            if abs(ret_5) > 2 * atr_norm:
                move_pct = round(ret_5 * 100, 1)
                self.missed_opportunities.append({
                    "symbol": symbol,
                    "date": date,
                    "move_pct": move_pct,
                    "direction": "up" if move_pct > 0 else "down",
                    "indicators": {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in features.items()
                        if k in ("rsi_14", "adx_14", "close_sma_ratio", "macd_hist", "volume_zscore")
                    },
                    "regime": (
                        "trending" if (features.get("adx_14") or 0) > 25
                        else "ranging" if (features.get("adx_14") or 0) < 20
                        else "neutral"
                    ),
                })

    # ------------------------------------------------------------------
    # Weekly Reflection
    # ------------------------------------------------------------------

    def on_new_trading_day(self, date: str) -> None:
        """Track trading days for reflection interval."""
        self.trading_days_since_reflection += 1
        self.last_reflection_date = date

    def should_reflect(self) -> bool:
        """Check if it's time for a weekly reflection."""
        return self.trading_days_since_reflection >= self.reflection_interval

    def reflect(self, llm_client) -> bool:
        """Run a reflection cycle. Returns True if reflection happened."""
        if not self.should_reflect():
            return False

        recent_trades = self.trade_journal[-10:]
        recent_missed = self.missed_opportunities[-10:]

        if not recent_trades and not recent_missed:
            self.trading_days_since_reflection = 0
            return False

        trades_text = self._format_trades_for_reflection(recent_trades)
        missed_text = self._format_missed_for_reflection(recent_missed)
        beliefs_text = self._format_beliefs()

        prompt = REFLECTION_PROMPT.format(
            trades_text=trades_text or "No completed trades this period.",
            missed_text=missed_text or "No significant missed opportunities.",
            beliefs_text=beliefs_text or "No beliefs yet (first reflection).",
        )

        try:
            response = llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
            new_beliefs = self._parse_beliefs(response)
            if new_beliefs:
                self.beliefs = new_beliefs
                self.reflection_log.append({
                    "date": self.last_reflection_date,
                    "trades_reviewed": len(recent_trades),
                    "missed_reviewed": len(recent_missed),
                    "beliefs": new_beliefs,
                })
                logger.info("Reflection complete: %d beliefs updated", len(new_beliefs))
        except Exception as e:
            logger.warning("Reflection LLM call failed: %s", e)

        self.trading_days_since_reflection = 0
        return True

    def get_beliefs_narrative(self) -> str:
        """Return current beliefs as text for system prompt injection."""
        if not self.beliefs:
            return ""
        lines = ["\nYOUR LEARNED BELIEFS (updated weekly from experience):"]
        for i, belief in enumerate(self.beliefs, 1):
            lines.append(f"  {i}. {belief}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_trades_for_reflection(self, trades: list[dict]) -> str:
        lines = []
        for t in trades:
            lines.append(
                f"  {t.get('symbol')} {t.get('action')}: "
                f"\u20b9{t.get('entry_price', 0):.0f} \u2192 \u20b9{t.get('exit_price', 0):.0f} "
                f"= {t.get('pnl_pct', 0):+.1f}% ({t.get('outcome')}). "
                f"Regime: {t.get('entry_regime')}. "
                f"Thesis: {t.get('thesis', '')[:80]}. "
                f"Post-mortem: {t.get('post_mortem', 'N/A')}"
            )
        return "\n".join(lines)

    def _format_missed_for_reflection(self, missed: list[dict]) -> str:
        lines = []
        for m in missed:
            indicators = m.get("indicators", {})
            ind_str = ", ".join(f"{k}={v}" for k, v in indicators.items())
            lines.append(
                f"  {m.get('symbol')} moved {m.get('move_pct', 0):+.1f}% "
                f"({m.get('direction')}) on {m.get('date')}. "
                f"Regime: {m.get('regime')}. Indicators: {ind_str}"
            )
        return "\n".join(lines)

    def _format_beliefs(self) -> str:
        if not self.beliefs:
            return "None yet."
        return "\n".join(f"  {i}. {b}" for i, b in enumerate(self.beliefs, 1))

    def _parse_beliefs(self, response: str) -> list[str]:
        """Parse numbered list from LLM response."""
        beliefs = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned and len(cleaned) > 10:
                beliefs.append(cleaned)
        return beliefs[:10]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all experience data to disk."""
        if not self.experience_dir:
            return
        os.makedirs(self.experience_dir, exist_ok=True)

        data = {
            "beliefs": self.beliefs,
            "trade_journal": self.trade_journal,
            "open_trades": self.open_trades,
            "missed_opportunities": self.missed_opportunities,
            "reflection_log": self.reflection_log,
            "trading_days_since_reflection": self.trading_days_since_reflection,
            "last_reflection_date": self.last_reflection_date,
        }
        path = os.path.join(self.experience_dir, "experience.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Experience saved to %s", path)

    def load(self) -> None:
        """Load persisted experience from disk."""
        if not self.experience_dir:
            return
        path = os.path.join(self.experience_dir, "experience.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.beliefs = data.get("beliefs", [])
            self.trade_journal = data.get("trade_journal", [])
            self.open_trades = data.get("open_trades", {})
            self.missed_opportunities = data.get("missed_opportunities", [])
            self.reflection_log = data.get("reflection_log", [])
            self.trading_days_since_reflection = data.get("trading_days_since_reflection", 0)
            self.last_reflection_date = data.get("last_reflection_date", "")
            logger.info(
                "Experience loaded: %d beliefs, %d journal entries",
                len(self.beliefs), len(self.trade_journal),
            )
        except Exception as e:
            logger.warning("Failed to load experience: %s", e)
