"""Experience Manager — learning layer for LLM autonomous trader.

Tracks trade journal with decision context, detects missed opportunities
on unheld stocks, runs adaptive LLM reflections to produce evolving
regime-conditional beliefs that decay over time.

Beliefs are:
  - Tagged with regime: "trending", "ranging", "neutral", or "all"
  - Strength-decayed: lose 0.05 per reflection, removed at < 0.1
  - Re-affirmed beliefs reset to 1.0
  - Filtered by current regime when injected into system prompt

Missed opportunity detection uses confluence scoring
(3 of 4 indicators must align) to avoid FOMO noise.

Reflection interval adapts: faster when losing (3 days),
slower when winning (8 days).
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

DECAY_RATE = 0.05  # strength lost per reflection cycle
MIN_STRENGTH = 0.1  # beliefs below this are removed
CONFLUENCE_THRESHOLD = 3  # need 3 of 4 indicators aligned for missed opp

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

MISSED OPPORTUNITIES THIS PERIOD (indicator-confirmed only):
{missed_text}

YOUR CURRENT BELIEFS:
{beliefs_text}

CURRENT MARKET REGIME: {current_regime}

Update your beliefs based on this evidence.
- What patterns are working? What's failing?
- Which missed opportunities had clear signals you should have acted on?
- What should you do differently?

IMPORTANT: Each belief MUST start with a regime tag: [TRENDING], [RANGING], [NEUTRAL], or [ALL].
Respond with ONLY a numbered list of beliefs (max 10).
Each belief must be specific and actionable:
  Good: "1. [TRENDING] Continuation longs on large-caps work when ADX>25 and MACD positive."
  Good: "2. [ALL] Target 1-3 trades per month. Every trade costs money."
  Bad: "3. Be more careful." (too vague, no regime tag)

Keep beliefs still supported by evidence. Remove contradicted ones. Add new ones."""


class ExperienceManager:
    """Manages experiential learning for LLM trading strategies.

    Components:
      - Trade journal: entries/exits with thesis and market context
      - Missed opportunity detector: confluence-filtered big moves on unheld stocks
      - Adaptive reflection: interval scales by recent P&L
      - Regime-conditional beliefs: tagged, strength-decayed, filtered
      - Persistence: all data saved/loaded from JSON files
    """

    def __init__(
        self,
        experience_dir: str | None = None,
        reflection_interval: int = 5,
    ):
        self.experience_dir = experience_dir
        self.base_reflection_interval = reflection_interval

        # State
        self.trade_journal: list[dict] = []
        self.open_trades: dict[str, dict] = {}
        self.missed_opportunities: list[dict] = []
        self.beliefs: list[dict] = []  # [{text, regime, created_at, strength}]
        self.reflection_log: list[dict] = []

        # Counters
        self.trading_days_since_reflection: int = 0
        self.last_reflection_date: str = ""
        self.reflection_count: int = 0

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
    # Missed Opportunity Detection (confluence-filtered)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_entry_signal(features: dict, direction: str) -> bool:
        """Check if indicators supported an entry in the given direction.

        Requires 3 of 4 indicators aligned (confluence threshold).
        """
        score = 0
        rsi = features.get("rsi_14", 50)
        adx = features.get("adx_14", 0)
        macd = features.get("macd_hist", 0)
        sma_ratio = features.get("close_sma_ratio", 1.0)

        if direction == "up":
            if rsi < 45:
                score += 1  # not overbought, room to run
            if macd > 0:
                score += 1  # momentum positive
            if sma_ratio > 1.0:
                score += 1  # above average
            if adx > 20:
                score += 1  # some trend present
        else:
            if rsi > 55:
                score += 1  # not oversold
            if macd < 0:
                score += 1  # momentum negative
            if sma_ratio < 1.0:
                score += 1  # below average
            if adx > 20:
                score += 1  # some trend present

        return score >= CONFLUENCE_THRESHOLD

    def detect_missed_opportunities(
        self,
        all_features: dict[str, dict | None],
        all_atrs: dict[str, float | None],
        held_symbols: set[str],
        date: str,
    ) -> None:
        """Check unheld stocks for significant 5-day moves where indicators supported entry."""
        for symbol, features in all_features.items():
            if symbol in held_symbols or features is None:
                continue
            atr_norm = features.get("atr_norm", 0)
            ret_5 = features.get("ret_5")
            if ret_5 is None or atr_norm <= 0:
                continue

            # Must be a significant move (> 2x ATR)
            if abs(ret_5) <= 2 * atr_norm:
                continue

            direction = "up" if ret_5 > 0 else "down"

            # Confluence filter: indicators must have supported this direction
            if not self._has_entry_signal(features, direction):
                continue

            move_pct = round(ret_5 * 100, 1)
            self.missed_opportunities.append({
                "symbol": symbol,
                "date": date,
                "move_pct": move_pct,
                "direction": direction,
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
    # Belief Decay
    # ------------------------------------------------------------------

    def _decay_beliefs(self) -> None:
        """Decay all belief strengths and remove expired ones."""
        for belief in self.beliefs:
            belief["strength"] = round(max(0.0, belief["strength"] - DECAY_RATE), 4)
        self.beliefs = [b for b in self.beliefs if b["strength"] >= MIN_STRENGTH]

    # ------------------------------------------------------------------
    # Adaptive Reflection Interval
    # ------------------------------------------------------------------

    def _compute_reflection_interval(self) -> int:
        """Reflect faster when losing (every 3 days), slower when winning (every 8)."""
        recent = self.trade_journal[-5:]
        if not recent:
            return self.base_reflection_interval

        recent_pnl = sum(t.get("pnl", 0) for t in recent)
        if recent_pnl < 0:
            return max(3, self.base_reflection_interval - 2)
        elif recent_pnl > 0:
            return min(10, self.base_reflection_interval + 3)
        return self.base_reflection_interval

    # ------------------------------------------------------------------
    # Weekly Reflection
    # ------------------------------------------------------------------

    def on_new_trading_day(self, date: str) -> None:
        """Track trading days for reflection interval."""
        self.trading_days_since_reflection += 1
        self.last_reflection_date = date

    def should_reflect(self) -> bool:
        """Check if it's time for a reflection (adaptive interval)."""
        return self.trading_days_since_reflection >= self._compute_reflection_interval()

    def reflect(self, llm_client, current_regime: str = "neutral") -> bool:
        """Run a reflection cycle. Returns True if reflection happened."""
        if not self.should_reflect():
            return False

        recent_trades = self.trade_journal[-10:]
        recent_missed = self.missed_opportunities[-10:]

        if not recent_trades and not recent_missed:
            self.trading_days_since_reflection = 0
            return False

        # Decay existing beliefs before adding new ones
        self._decay_beliefs()

        trades_text = self._format_trades_for_reflection(recent_trades)
        missed_text = self._format_missed_for_reflection(recent_missed)
        beliefs_text = self._format_beliefs()

        prompt = REFLECTION_PROMPT.format(
            trades_text=trades_text or "No completed trades this period.",
            missed_text=missed_text or "No significant missed opportunities.",
            beliefs_text=beliefs_text or "No beliefs yet (first reflection).",
            current_regime=current_regime,
        )

        try:
            response = llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
            new_beliefs = self._parse_beliefs_with_regime(response)
            if new_beliefs:
                self.beliefs = new_beliefs
                self.reflection_count += 1
                self.reflection_log.append({
                    "date": self.last_reflection_date,
                    "reflection_number": self.reflection_count,
                    "trades_reviewed": len(recent_trades),
                    "missed_reviewed": len(recent_missed),
                    "regime": current_regime,
                    "beliefs": [b["text"] for b in new_beliefs],
                    "interval_used": self._compute_reflection_interval(),
                })
                logger.info(
                    "Reflection #%d: %d beliefs (interval=%d, regime=%s)",
                    self.reflection_count, len(new_beliefs),
                    self._compute_reflection_interval(), current_regime,
                )
        except Exception as e:
            logger.warning("Reflection LLM call failed: %s", e)

        self.trading_days_since_reflection = 0
        return True

    def get_beliefs_narrative(self, current_regime: str = "neutral") -> str:
        """Return beliefs filtered by current regime, with strength indicators."""
        visible = [
            b for b in self.beliefs
            if b["regime"] in (current_regime, "all")
            and b["strength"] >= MIN_STRENGTH
        ]
        if not visible:
            return ""

        lines = ["\nYOUR LEARNED BELIEFS (updated from experience, filtered for current regime):"]
        for i, belief in enumerate(visible, 1):
            # Show strength as stars: ★★★ (>0.7), ★★ (>0.4), ★ (>0.1)
            s = belief["strength"]
            stars = "\u2605\u2605\u2605" if s > 0.7 else "\u2605\u2605" if s > 0.4 else "\u2605"
            regime_tag = f"[{belief['regime'].upper()}]"
            lines.append(f"  {i}. {stars} {regime_tag} {belief['text']}")
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
        lines = []
        for i, b in enumerate(self.beliefs, 1):
            s = b["strength"]
            stars = "\u2605\u2605\u2605" if s > 0.7 else "\u2605\u2605" if s > 0.4 else "\u2605"
            lines.append(f"  {i}. {stars} [{b['regime'].upper()}] {b['text']} (strength: {s:.2f})")
        return "\n".join(lines)

    def _parse_beliefs_with_regime(self, response: str) -> list[dict]:
        """Parse numbered list with regime tags from LLM response."""
        beliefs = []
        regime_pattern = re.compile(r"\[(TRENDING|RANGING|NEUTRAL|ALL)\]\s*", re.IGNORECASE)

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if not cleaned or len(cleaned) < 10:
                continue

            # Extract regime tag
            match = regime_pattern.search(cleaned)
            if match:
                regime = match.group(1).lower()
                text = regime_pattern.sub("", cleaned).strip()
            else:
                regime = "all"
                text = cleaned

            if text:
                beliefs.append({
                    "text": text,
                    "regime": regime,
                    "created_at": self.reflection_count,
                    "strength": 1.0,
                })

        return beliefs[:10]

    # Legacy compatibility — parse flat beliefs
    def _parse_beliefs(self, response: str) -> list[str]:
        """Parse numbered list from LLM response (flat format for backward compat)."""
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
            "reflection_count": self.reflection_count,
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
            raw_beliefs = data.get("beliefs", [])
            # Handle both old (list[str]) and new (list[dict]) formats
            if raw_beliefs and isinstance(raw_beliefs[0], str):
                self.beliefs = [
                    {"text": b, "regime": "all", "created_at": 0, "strength": 1.0}
                    for b in raw_beliefs
                ]
            else:
                self.beliefs = raw_beliefs
            self.trade_journal = data.get("trade_journal", [])
            self.open_trades = data.get("open_trades", {})
            self.missed_opportunities = data.get("missed_opportunities", [])
            self.reflection_log = data.get("reflection_log", [])
            self.trading_days_since_reflection = data.get("trading_days_since_reflection", 0)
            self.last_reflection_date = data.get("last_reflection_date", "")
            self.reflection_count = data.get("reflection_count", 0)
            logger.info(
                "Experience loaded: %d beliefs, %d journal entries, reflection #%d",
                len(self.beliefs), len(self.trade_journal), self.reflection_count,
            )
        except Exception as e:
            logger.warning("Failed to load experience: %s", e)
