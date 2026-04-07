"""Tests for ExperienceManager — conviction-scaled, regime-conditional, decay-aware learning."""

import json
import tempfile
from unittest.mock import MagicMock

from strategies.experience_manager import ExperienceManager, CONFLUENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Trade Journal (unchanged behavior)
# ---------------------------------------------------------------------------

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


def test_record_exit_without_llm():
    em = ExperienceManager(experience_dir=None)
    em.record_entry("RELIANCE", "BUY", "thesis", {"rsi": 35}, 1200.0, "2024-03-15", "trending")
    em.record_exit("RELIANCE", 1250.0, 500.0, 4.17, "2024-03-22", llm_client=None)
    assert len(em.trade_journal) == 1
    assert em.trade_journal[0]["outcome"] == "WIN"
    assert "RELIANCE" not in em.open_trades


def test_record_exit_loss():
    em = ExperienceManager(experience_dir=None)
    em.record_entry("TCS", "BUY", "thesis", {"rsi": 50}, 3500.0, "2024-04-01", "neutral")
    em.record_exit("TCS", 3400.0, -1000.0, -2.86, "2024-04-08", llm_client=None)
    assert em.trade_journal[0]["outcome"] == "LOSS"


def test_record_exit_with_llm_postmortem():
    em = ExperienceManager(experience_dir=None)
    em.record_entry("TCS", "BUY", "IT breakout", {"rsi": 55}, 3500.0, "2024-04-01", "trending")
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "Stopped out before recovery."
    em.record_exit("TCS", 3400.0, -1000.0, -2.86, "2024-04-08", llm_client=mock_client)
    assert em.trade_journal[0]["post_mortem"] == "Stopped out before recovery."


def test_record_exit_unknown_symbol():
    em = ExperienceManager(experience_dir=None)
    em.record_exit("UNKNOWN", 100.0, 0.0, 0.0, "2024-01-01", llm_client=None)
    assert len(em.trade_journal) == 0


# ---------------------------------------------------------------------------
# #6: Smart Missed Opportunity Detection (confluence-filtered)
# ---------------------------------------------------------------------------

def test_missed_opp_confluence_aligned():
    """Move > 2x ATR AND 3 of 4 indicators aligned → detected."""
    em = ExperienceManager(experience_dir=None)
    features = {
        "RELIANCE": {
            "ret_5": 0.08, "atr_norm": 0.02,
            "rsi_14": 42, "adx_14": 30, "close_sma_ratio": 1.03, "macd_hist": 0.5,
            "volume_zscore": 1.2,
        },
    }
    em.detect_missed_opportunities(features, {"RELIANCE": 25.0}, set(), "2024-04-10")
    # RSI<45 ✓, MACD>0 ✓, SMA>1.0 ✓, ADX>20 ✓ → 4/4 → detected
    assert len(em.missed_opportunities) == 1
    assert em.missed_opportunities[0]["symbol"] == "RELIANCE"


def test_missed_opp_confluence_not_aligned():
    """Big move but indicators DON'T support it → NOT detected."""
    em = ExperienceManager(experience_dir=None)
    features = {
        "INFY": {
            "ret_5": 0.08, "atr_norm": 0.02,
            # RSI 60 → not <45, MACD -0.5 → not >0, SMA 0.98 → not >1.0, ADX 25 → >20
            # Score: only 1/4 (ADX) → below threshold
            "rsi_14": 60, "adx_14": 25, "close_sma_ratio": 0.98, "macd_hist": -0.5,
        },
    }
    em.detect_missed_opportunities(features, {"INFY": 15.0}, set(), "2024-04-10")
    assert len(em.missed_opportunities) == 0


def test_missed_opp_small_move_ignored():
    """Move < 2x ATR → NOT detected regardless of indicators."""
    em = ExperienceManager(experience_dir=None)
    features = {
        "INFY": {
            "ret_5": 0.01, "atr_norm": 0.02,
            "rsi_14": 30, "adx_14": 30, "close_sma_ratio": 1.05, "macd_hist": 1.0,
        },
    }
    em.detect_missed_opportunities(features, {"INFY": 15.0}, set(), "2024-04-10")
    assert len(em.missed_opportunities) == 0


def test_missed_opp_held_stock_ignored():
    em = ExperienceManager(experience_dir=None)
    features = {
        "RELIANCE": {
            "ret_5": 0.10, "atr_norm": 0.02,
            "rsi_14": 40, "adx_14": 30, "close_sma_ratio": 1.05, "macd_hist": 0.5,
        },
    }
    em.detect_missed_opportunities(features, {"RELIANCE": 25.0}, {"RELIANCE"}, "2024-04-10")
    assert len(em.missed_opportunities) == 0


def test_has_entry_signal_up():
    """Confluence scoring for upward moves."""
    # All 4 aligned: RSI<45, MACD>0, SMA>1.0, ADX>20
    assert ExperienceManager._has_entry_signal(
        {"rsi_14": 40, "macd_hist": 0.5, "close_sma_ratio": 1.02, "adx_14": 25}, "up"
    ) is True

    # Only 2 aligned: RSI<45, ADX>20 (MACD negative, SMA below)
    assert ExperienceManager._has_entry_signal(
        {"rsi_14": 40, "macd_hist": -0.5, "close_sma_ratio": 0.98, "adx_14": 25}, "up"
    ) is False


def test_has_entry_signal_down():
    """Confluence scoring for downward moves."""
    # All 4: RSI>55, MACD<0, SMA<1.0, ADX>20
    assert ExperienceManager._has_entry_signal(
        {"rsi_14": 60, "macd_hist": -0.5, "close_sma_ratio": 0.97, "adx_14": 25}, "down"
    ) is True


# ---------------------------------------------------------------------------
# #9: Regime-Conditional Beliefs
# ---------------------------------------------------------------------------

def test_beliefs_regime_filtered():
    """Only beliefs matching current regime (or 'all') are shown."""
    em = ExperienceManager(experience_dir=None)
    em.beliefs = [
        {"text": "Breakouts work", "regime": "trending", "created_at": 0, "strength": 1.0},
        {"text": "Mean-reversion works", "regime": "ranging", "created_at": 0, "strength": 1.0},
        {"text": "Fewer trades win", "regime": "all", "created_at": 0, "strength": 1.0},
    ]

    trending_narrative = em.get_beliefs_narrative("trending")
    assert "Breakouts work" in trending_narrative
    assert "Fewer trades win" in trending_narrative
    assert "Mean-reversion works" not in trending_narrative

    ranging_narrative = em.get_beliefs_narrative("ranging")
    assert "Mean-reversion works" in ranging_narrative
    assert "Fewer trades win" in ranging_narrative
    assert "Breakouts work" not in ranging_narrative


def test_beliefs_narrative_empty():
    em = ExperienceManager(experience_dir=None)
    assert em.get_beliefs_narrative("trending") == ""


def test_beliefs_narrative_strength_stars():
    """Beliefs show strength indicators."""
    em = ExperienceManager(experience_dir=None)
    em.beliefs = [
        {"text": "Strong belief", "regime": "all", "created_at": 0, "strength": 0.9},
        {"text": "Medium belief", "regime": "all", "created_at": 0, "strength": 0.5},
        {"text": "Weak belief", "regime": "all", "created_at": 0, "strength": 0.2},
    ]
    narrative = em.get_beliefs_narrative("neutral")
    assert "\u2605\u2605\u2605" in narrative  # strong
    assert "\u2605\u2605" in narrative  # medium (also substring of strong, but that's ok)
    assert "Strong belief" in narrative
    assert "Weak belief" in narrative


def test_parse_beliefs_with_regime_tags():
    """LLM response with regime tags is parsed correctly."""
    em = ExperienceManager(experience_dir=None)
    response = (
        "1. [TRENDING] Continuation longs work when ADX>25.\n"
        "2. [RANGING] Mean-reversion entries need MACD confirmation.\n"
        "3. [ALL] Target 1-3 trades per month.\n"
        "4. No tag here, should default to all regime."
    )
    beliefs = em._parse_beliefs_with_regime(response)
    assert len(beliefs) == 4
    assert beliefs[0]["regime"] == "trending"
    assert beliefs[0]["text"] == "Continuation longs work when ADX>25."
    assert beliefs[1]["regime"] == "ranging"
    assert beliefs[2]["regime"] == "all"
    assert beliefs[3]["regime"] == "all"  # default when no tag
    assert all(b["strength"] == 1.0 for b in beliefs)


# ---------------------------------------------------------------------------
# #11: Belief Decay
# ---------------------------------------------------------------------------

def test_belief_decay():
    """Beliefs lose 0.05 strength per decay call."""
    em = ExperienceManager(experience_dir=None)
    em.beliefs = [
        {"text": "Fresh belief", "regime": "all", "created_at": 0, "strength": 1.0},
        {"text": "Old belief", "regime": "all", "created_at": 0, "strength": 0.15},
    ]
    em._decay_beliefs()
    assert em.beliefs[0]["strength"] == 0.95  # 1.0 - 0.05
    assert len(em.beliefs) == 2  # old belief still at 0.10, above MIN_STRENGTH

    # Decay again — old belief drops to 0.05 → removed
    em._decay_beliefs()
    assert len(em.beliefs) == 1
    assert em.beliefs[0]["text"] == "Fresh belief"


def test_belief_decay_removes_expired():
    """Beliefs below MIN_STRENGTH (0.1) are removed."""
    em = ExperienceManager(experience_dir=None)
    em.beliefs = [
        {"text": "Almost gone", "regime": "all", "created_at": 0, "strength": 0.11},
    ]
    em._decay_beliefs()  # 0.11 - 0.05 = 0.06 → below 0.1 → removed
    assert len(em.beliefs) == 0


def test_belief_survives_18_reflections():
    """A belief with no reinforcement survives ~18 decay cycles (0.05 * 18 = 0.90)."""
    em = ExperienceManager(experience_dir=None)
    em.beliefs = [
        {"text": "Long-lived belief", "regime": "all", "created_at": 0, "strength": 1.0},
    ]
    for _ in range(18):
        em._decay_beliefs()
    # After 18 decays: 1.0 - 18*0.05 = 0.10 → at threshold, still alive
    assert len(em.beliefs) == 1
    assert em.beliefs[0]["strength"] == 0.10

    # One more → removed
    em._decay_beliefs()
    assert len(em.beliefs) == 0


# ---------------------------------------------------------------------------
# #12: Adaptive Reflection Interval
# ---------------------------------------------------------------------------

def test_adaptive_interval_losing():
    """Losing trades → shorter interval (3 days)."""
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    em.trade_journal = [
        {"pnl": -100}, {"pnl": -200}, {"pnl": -50}, {"pnl": 10}, {"pnl": -80},
    ]
    assert em._compute_reflection_interval() == 3  # losing → 5 - 2 = 3


def test_adaptive_interval_winning():
    """Winning trades → longer interval (8 days)."""
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    em.trade_journal = [
        {"pnl": 100}, {"pnl": 200}, {"pnl": 50}, {"pnl": -10}, {"pnl": 80},
    ]
    assert em._compute_reflection_interval() == 8  # winning → 5 + 3 = 8


def test_adaptive_interval_no_trades():
    """No trades → default interval."""
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    assert em._compute_reflection_interval() == 5


def test_should_reflect_uses_adaptive_interval():
    """should_reflect() uses adaptive interval, not fixed."""
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    em.trade_journal = [{"pnl": -100}] * 5  # losing → interval = 3

    for _ in range(2):
        em.on_new_trading_day("2024-01-01")
    assert not em.should_reflect()  # 2 < 3

    em.on_new_trading_day("2024-01-03")
    assert em.should_reflect()  # 3 >= 3


# ---------------------------------------------------------------------------
# Reflection Cycle (updated for regime + decay)
# ---------------------------------------------------------------------------

def test_reflection_updates_beliefs_with_regime():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    em.record_entry("A", "BUY", "Test thesis", {"rsi": 30}, 100.0, "2024-01-01", "trending")
    em.record_exit("A", 110.0, 100.0, 10.0, "2024-01-05", llm_client=None)

    # Winning trade → adaptive interval = 8.  Need 8 trading days.
    for i in range(8):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")

    mock_client = MagicMock()
    mock_client.chat_completion.return_value = (
        "1. [TRENDING] Momentum entries with RSI<40 work well.\n"
        "2. [ALL] Stop-losses at 2x ATR are too tight for CNC trades.\n"
        "3. [RANGING] IT stocks mean-revert after 3-day selloffs."
    )
    result = em.reflect(mock_client, current_regime="trending")
    assert result is True
    assert len(em.beliefs) == 3
    assert em.beliefs[0]["regime"] == "trending"
    assert em.beliefs[1]["regime"] == "all"
    assert em.beliefs[2]["regime"] == "ranging"
    assert em.reflection_count == 1


def test_reflection_decays_before_adding():
    """Reflection decays old beliefs before adding new ones."""
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    em.beliefs = [
        {"text": "Old belief", "regime": "all", "created_at": 0, "strength": 0.15},
    ]
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    em.record_entry("A", "BUY", "thesis", {"rsi": 30}, 100.0, "2024-01-01", "trending")
    em.record_exit("A", 90.0, -100.0, -10.0, "2024-01-05", llm_client=None)

    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "1. [ALL] New belief after decay."
    em.reflect(mock_client, current_regime="neutral")

    # Old belief decayed to 0.10 → removed. Only new belief remains.
    assert len(em.beliefs) == 1
    assert em.beliefs[0]["text"] == "New belief after decay."


def test_reflection_no_data_resets_counter():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    mock_client = MagicMock()
    result = em.reflect(mock_client)
    assert result is False
    assert em.trading_days_since_reflection == 0


def test_reflection_llm_failure_doesnt_crash():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    em.record_entry("A", "BUY", "thesis", {"rsi": 30}, 100.0, "2024-01-01", "trending")
    em.record_exit("A", 90.0, -100.0, -10.0, "2024-01-05", llm_client=None)

    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = Exception("API down")
    em.reflect(mock_client)
    assert len(em.beliefs) == 0  # no crash, beliefs unchanged


# ---------------------------------------------------------------------------
# Persistence (updated for new belief structure)
# ---------------------------------------------------------------------------

def test_persistence_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        em1 = ExperienceManager(experience_dir=tmpdir)
        em1.beliefs = [
            {"text": "Test belief", "regime": "trending", "created_at": 1, "strength": 0.8},
        ]
        em1.reflection_count = 5
        em1.record_entry("X", "BUY", "thesis", {"rsi": 50}, 100.0, "2024-01-01", "neutral")
        em1.record_exit("X", 105.0, 50.0, 5.0, "2024-01-05", llm_client=None)
        em1.save()

        em2 = ExperienceManager(experience_dir=tmpdir)
        assert len(em2.beliefs) == 1
        assert em2.beliefs[0]["regime"] == "trending"
        assert em2.beliefs[0]["strength"] == 0.8
        assert em2.reflection_count == 5
        assert len(em2.trade_journal) == 1


def test_persistence_loads_old_flat_beliefs():
    """Backward compat: old format (list[str]) converted to new format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        path = os.path.join(tmpdir, "experience.json")
        with open(path, "w") as f:
            json.dump({"beliefs": ["Old flat belief 1", "Old flat belief 2"]}, f)

        em = ExperienceManager(experience_dir=tmpdir)
        assert len(em.beliefs) == 2
        assert em.beliefs[0]["text"] == "Old flat belief 1"
        assert em.beliefs[0]["regime"] == "all"
        assert em.beliefs[0]["strength"] == 1.0


def test_persistence_no_dir():
    em = ExperienceManager(experience_dir=None)
    em.beliefs = [{"text": "test", "regime": "all", "created_at": 0, "strength": 1.0}]
    em.save()  # no crash
    em.load()  # no crash


def test_persistence_missing_file():
    em = ExperienceManager(experience_dir="/tmp/nonexistent_experience_dir_xyz")
    em.load()
    assert len(em.beliefs) == 0
