# LLM Reflective Learning Layer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add experiential learning to the LLM autonomous trader via a trade journal, missed opportunity detector, weekly reflection cycle, and evolving beliefs document that persists across sessions.

**Architecture:** New `ExperienceManager` class handles all learning: journaling entries/exits with context, detecting missed moves on unheld stocks, running weekly LLM reflection calls to produce updated beliefs, and persisting everything to JSON. The strategy's system prompt is augmented with the current beliefs on each LLM call.

**Tech Stack:** Python, JSON persistence, AzureOpenAIClient (existing), indicators.py (existing)

---

### Task 1: Create ExperienceManager with Trade Journal

**Files:**
- Create: `strategies/strategies/experience_manager.py`
- Test: `strategies/tests/test_experience_manager.py`

**Step 1: Write the failing tests**

```python
"""Tests for ExperienceManager — trade journal, missed opportunities, reflections."""

import json
import os
import tempfile
from unittest.mock import MagicMock

from strategies.experience_manager import ExperienceManager


def test_record_entry():
    em = ExperienceManager(experience_dir=None)
    em.record_entry(
        symbol="RELIANCE", action="BUY",
        thesis="THESIS: Strong momentum. CONVICTION: 8/10",
        indicators={"rsi": 35, "adx": 28, "zscore": -1.2},
        price=1200.0, date="2024-03-15", regime="trending",
    )
    assert len(em.open_trades) == 1
    assert em.open_trades["RELIANCE"]["entry_price"] == 1200.0
    assert em.open_trades["RELIANCE"]["thesis"] == "THESIS: Strong momentum. CONVICTION: 8/10"


def test_record_exit_without_llm():
    """Exit without LLM client produces journal entry with no post-mortem."""
    em = ExperienceManager(experience_dir=None)
    em.record_entry(
        symbol="RELIANCE", action="BUY",
        thesis="Test thesis", indicators={"rsi": 35},
        price=1200.0, date="2024-03-15", regime="trending",
    )
    em.record_exit(
        symbol="RELIANCE", exit_price=1250.0,
        pnl=500.0, pnl_pct=4.17, date="2024-03-22", llm_client=None,
    )
    assert len(em.trade_journal) == 1
    assert em.trade_journal[0]["outcome"] == "WIN"
    assert em.trade_journal[0]["pnl_pct"] == 4.17
    assert "RELIANCE" not in em.open_trades


def test_record_exit_with_llm_postmortem():
    """Exit with LLM client generates post-mortem."""
    em = ExperienceManager(experience_dir=None)
    em.record_entry(
        symbol="TCS", action="BUY", thesis="IT breakout",
        indicators={"rsi": 55}, price=3500.0, date="2024-04-01",
        regime="trending",
    )
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "Stopped out before recovery."
    em.record_exit(
        symbol="TCS", exit_price=3400.0, pnl=-1000.0, pnl_pct=-2.86,
        date="2024-04-08", llm_client=mock_client,
    )
    assert em.trade_journal[0]["post_mortem"] == "Stopped out before recovery."
    mock_client.chat_completion.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `cd strategies && pytest tests/test_experience_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategies.experience_manager'`

**Step 3: Write minimal ExperienceManager (journal only)**

```python
"""Experience Manager — learning layer for LLM autonomous trader.

Tracks trade journal, missed opportunities, and evolving beliefs.
Runs weekly reflections via separate LLM calls.
Persists to disk across sessions.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

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
        self.open_trades: dict[str, dict] = {}  # symbol -> entry record
        self.missed_opportunities: list[dict] = []
        self.beliefs: list[str] = []
        self.reflection_log: list[dict] = []

        # Counters
        self.trading_days_since_reflection: int = 0
        self.last_reflection_date: str = ""

        # Track previous 5-day closes for missed opp detection
        self._prev_closes: dict[str, list[float]] = {}

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
        """Record a new trade entry with full context."""
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

        # Generate post-mortem via LLM if client available
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

        journal_entry = {
            **entry,
            "exit_price": exit_price,
            "exit_date": date,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
            "post_mortem": post_mortem,
        }
        self.trade_journal.append(journal_entry)

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
        """Check unheld stocks for significant moves (>2x ATR in 5 days)."""
        for symbol, features in all_features.items():
            if symbol in held_symbols or features is None:
                continue
            atr = all_atrs.get(symbol)
            if atr is None or atr <= 0:
                continue

            # Track closes for 5-day move calculation
            close = features.get("ret_5")
            if close is None:
                continue

            move_pct = close * 100  # ret_5 is already a fraction
            atr_norm = features.get("atr_norm", 0)
            # Significant if absolute return > 2x normalized ATR
            if atr_norm > 0 and abs(close) > 2 * atr_norm:
                self.missed_opportunities.append({
                    "symbol": symbol,
                    "date": date,
                    "move_pct": round(move_pct, 1),
                    "direction": "up" if move_pct > 0 else "down",
                    "indicators": {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in (features or {}).items()
                        if k in ("rsi_14", "adx_14", "close_sma_ratio", "macd_hist", "volume_zscore")
                    },
                    "regime": "trending" if (features.get("adx_14") or 0) > 25
                             else "ranging" if (features.get("adx_14") or 0) < 20
                             else "neutral",
                })

    # ------------------------------------------------------------------
    # Weekly Reflection
    # ------------------------------------------------------------------

    def on_new_trading_day(self, date: str) -> None:
        """Call once per new trading day to track reflection interval."""
        self.trading_days_since_reflection += 1

    def should_reflect(self) -> bool:
        """Check if it's time for a weekly reflection."""
        return self.trading_days_since_reflection >= self.reflection_interval

    def reflect(self, llm_client) -> bool:
        """Run a reflection cycle. Returns True if reflection happened."""
        if not self.should_reflect():
            return False

        # Build context for reflection
        recent_trades = self.trade_journal[-10:]  # last 10 trades
        recent_missed = self.missed_opportunities[-10:]  # last 10 missed opps

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
                f"₹{t.get('entry_price', 0):.0f} → ₹{t.get('exit_price', 0):.0f} "
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
            # Strip leading number + period/paren
            import re
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned and len(cleaned) > 10:  # skip very short lines
                beliefs.append(cleaned)
        return beliefs[:10]  # max 10 beliefs

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
```

**Step 4: Run tests to verify they pass**

Run: `cd strategies && pytest tests/test_experience_manager.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add strategies/strategies/experience_manager.py strategies/tests/test_experience_manager.py
git commit -m "feat: ExperienceManager with trade journal and post-mortem"
```

---

### Task 2: Add Missed Opportunity Detection + Reflection Tests

**Files:**
- Test: `strategies/tests/test_experience_manager.py` (append)

**Step 1: Write additional tests**

Append to `test_experience_manager.py`:

```python
def test_detect_missed_opportunity():
    em = ExperienceManager(experience_dir=None)
    features = {
        "RELIANCE": {"ret_5": 0.08, "atr_norm": 0.02, "rsi_14": 42, "adx_14": 30,
                      "close_sma_ratio": 1.03, "macd_hist": 0.5, "volume_zscore": 1.2},
        "INFY": {"ret_5": 0.01, "atr_norm": 0.02, "rsi_14": 50, "adx_14": 20,
                  "close_sma_ratio": 1.0, "macd_hist": 0.0, "volume_zscore": 0.0},
    }
    atrs = {"RELIANCE": 25.0, "INFY": 15.0}
    em.detect_missed_opportunities(features, atrs, held_symbols=set(), date="2024-04-10")
    # RELIANCE moved 8% with atr_norm 2% -> 4x ATR -> should be detected
    assert len(em.missed_opportunities) == 1
    assert em.missed_opportunities[0]["symbol"] == "RELIANCE"
    assert em.missed_opportunities[0]["direction"] == "up"


def test_detect_missed_ignores_held():
    em = ExperienceManager(experience_dir=None)
    features = {"RELIANCE": {"ret_5": 0.10, "atr_norm": 0.02, "rsi_14": 50, "adx_14": 30}}
    atrs = {"RELIANCE": 25.0}
    em.detect_missed_opportunities(features, atrs, held_symbols={"RELIANCE"}, date="2024-04-10")
    assert len(em.missed_opportunities) == 0


def test_reflection_cycle():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    # Simulate 5 trading days
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    assert em.should_reflect()

    # Add a trade for reflection context
    em.record_entry("A", "BUY", "Test thesis", {"rsi": 30}, 100.0, "2024-01-01", "trending")
    em.record_exit("A", 110.0, 100.0, 10.0, "2024-01-05", llm_client=None)

    # Mock reflection LLM call
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = (
        "1. Trending market entries with RSI<40 work well.\n"
        "2. Stop-losses at 2x ATR are too tight for CNC trades.\n"
        "3. IT stocks move together — trade the leader."
    )
    result = em.reflect(mock_client)
    assert result is True
    assert len(em.beliefs) == 3
    assert "Trending market" in em.beliefs[0]
    assert em.trading_days_since_reflection == 0


def test_beliefs_narrative():
    em = ExperienceManager(experience_dir=None)
    em.beliefs = ["Banks work in trends.", "IT stocks correlate."]
    narrative = em.get_beliefs_narrative()
    assert "LEARNED BELIEFS" in narrative
    assert "Banks work in trends" in narrative
    assert "IT stocks correlate" in narrative


def test_beliefs_narrative_empty():
    em = ExperienceManager(experience_dir=None)
    assert em.get_beliefs_narrative() == ""


def test_persistence_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        em1 = ExperienceManager(experience_dir=tmpdir)
        em1.beliefs = ["Test belief 1", "Test belief 2"]
        em1.record_entry("X", "BUY", "thesis", {"rsi": 50}, 100.0, "2024-01-01", "neutral")
        em1.record_exit("X", 105.0, 50.0, 5.0, "2024-01-05", llm_client=None)
        em1.save()

        em2 = ExperienceManager(experience_dir=tmpdir)
        assert len(em2.beliefs) == 2
        assert em2.beliefs[0] == "Test belief 1"
        assert len(em2.trade_journal) == 1


def test_persistence_no_dir():
    """No experience_dir -> save/load are no-ops."""
    em = ExperienceManager(experience_dir=None)
    em.beliefs = ["test"]
    em.save()  # should not crash
    em.load()  # should not crash
```

**Step 2: Run all experience manager tests**

Run: `cd strategies && pytest tests/test_experience_manager.py -v`
Expected: 10 PASS

**Step 3: Commit**

```bash
git add strategies/tests/test_experience_manager.py
git commit -m "test: missed opportunity detection + reflection + persistence"
```

---

### Task 3: Integrate ExperienceManager into LLM Autonomous Trader

**Files:**
- Modify: `strategies/strategies/llm/llm_autonomous_trader.py`
- Test: `strategies/tests/test_llm_autonomous_trader.py` (append)

**Step 1: Write integration test**

Append to `test_llm_autonomous_trader.py`:

```python
def test_beliefs_injected_into_system_prompt():
    """Verify beliefs from ExperienceManager appear in system prompt."""
    s = _init_strategy({"initial_capital": 100_000})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    # Inject beliefs
    s.experience.beliefs = ["Banks work in trends.", "Avoid ICICIBANK mean-reversion."]

    captured = []
    def _capture(messages, **kwargs):
        captured.extend(messages)
        return json.dumps([])
    s.client.chat_completion = _capture

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    s.on_bar(snap)

    system_content = captured[0]["content"]
    assert "LEARNED BELIEFS" in system_content
    assert "Banks work in trends" in system_content


def test_experience_records_trade_entry():
    """When LLM places a BUY, ExperienceManager records the entry context."""
    s = _init_strategy()
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    s.client.chat_completion = MagicMock(
        return_value=json.dumps([{
            "action": "BUY", "symbol": "TEST", "quantity": 5,
            "order_type": "MARKET", "product_type": "CNC",
            "stop_price": 95.0,
            "reasoning": "THESIS: Strong momentum. CONVICTION: 8/10",
        }])
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    s.on_bar(snap)

    assert "TEST" in s.experience.open_trades
    assert "THESIS" in s.experience.open_trades["TEST"]["thesis"]
```

**Step 2: Modify `llm_autonomous_trader.py`**

Changes needed:
1. Import ExperienceManager
2. Initialize it in `initialize()`
3. Inject beliefs into system prompt before each LLM call
4. Record entries when guardrails approve a trade
5. Record exits when closed_trades arrive
6. Detect missed opportunities on new daily bars
7. Run reflection when due
8. Save on `on_complete()`

Key integration points in `on_bar()`:
- After `_apply_guardrails` approves a signal → `experience.record_entry()`
- After processing `snapshot.closed_trades` → `experience.record_exit()`
- After daily features recompute → `experience.detect_missed_opportunities()`
- After daily features recompute → `experience.maybe_reflect()`
- In `_build_full_narrative()` → append `experience.get_beliefs_narrative()`

**Step 3: Run tests**

Run: `cd strategies && pytest tests/test_llm_autonomous_trader.py tests/test_experience_manager.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add strategies/strategies/llm/llm_autonomous_trader.py strategies/strategies/experience_manager.py strategies/tests/
git commit -m "feat: integrate ExperienceManager into LLM autonomous trader"
```

---

### Task 4: Run Full Test Suite + Verify

**Step 1: Run all Python tests**

Run: `cd strategies && pytest tests/ -v`
Expected: 330+ tests PASS, 0 FAIL

**Step 2: Commit design doc**

```bash
git add docs/plans/2026-04-07-llm-reflective-learning-design.md docs/plans/2026-04-07-llm-reflective-learning-plan.md
git commit -m "docs: reflective learning layer design and implementation plan"
```
