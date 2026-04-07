"""Tests for ExperienceManager — trade journal, missed opportunities, reflections."""

import json
import tempfile
from unittest.mock import MagicMock

from strategies.experience_manager import ExperienceManager


# ---------------------------------------------------------------------------
# Trade Journal
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


def test_record_exit_loss():
    em = ExperienceManager(experience_dir=None)
    em.record_entry("TCS", "BUY", "thesis", {"rsi": 50}, 3500.0, "2024-04-01", "neutral")
    em.record_exit("TCS", 3400.0, -1000.0, -2.86, "2024-04-08", llm_client=None)
    assert em.trade_journal[0]["outcome"] == "LOSS"


def test_record_exit_with_llm_postmortem():
    """Exit with LLM client generates post-mortem."""
    em = ExperienceManager(experience_dir=None)
    em.record_entry("TCS", "BUY", "IT breakout", {"rsi": 55}, 3500.0, "2024-04-01", "trending")

    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "Stopped out before recovery."
    em.record_exit("TCS", 3400.0, -1000.0, -2.86, "2024-04-08", llm_client=mock_client)

    assert em.trade_journal[0]["post_mortem"] == "Stopped out before recovery."
    mock_client.chat_completion.assert_called_once()


def test_record_exit_unknown_symbol():
    """Exiting a symbol with no open entry is a no-op."""
    em = ExperienceManager(experience_dir=None)
    em.record_exit("UNKNOWN", 100.0, 0.0, 0.0, "2024-01-01", llm_client=None)
    assert len(em.trade_journal) == 0


# ---------------------------------------------------------------------------
# Missed Opportunity Detection
# ---------------------------------------------------------------------------

def test_detect_missed_opportunity():
    em = ExperienceManager(experience_dir=None)
    features = {
        "RELIANCE": {
            "ret_5": 0.08, "atr_norm": 0.02, "rsi_14": 42, "adx_14": 30,
            "close_sma_ratio": 1.03, "macd_hist": 0.5, "volume_zscore": 1.2,
        },
        "INFY": {
            "ret_5": 0.01, "atr_norm": 0.02, "rsi_14": 50, "adx_14": 20,
            "close_sma_ratio": 1.0, "macd_hist": 0.0, "volume_zscore": 0.0,
        },
    }
    atrs = {"RELIANCE": 25.0, "INFY": 15.0}
    em.detect_missed_opportunities(features, atrs, held_symbols=set(), date="2024-04-10")
    # RELIANCE moved 8% with atr_norm 2% -> 4x ATR -> detected
    assert len(em.missed_opportunities) == 1
    assert em.missed_opportunities[0]["symbol"] == "RELIANCE"
    assert em.missed_opportunities[0]["direction"] == "up"
    assert em.missed_opportunities[0]["regime"] == "trending"


def test_detect_missed_ignores_held():
    em = ExperienceManager(experience_dir=None)
    features = {"RELIANCE": {"ret_5": 0.10, "atr_norm": 0.02, "rsi_14": 50, "adx_14": 30}}
    atrs = {"RELIANCE": 25.0}
    em.detect_missed_opportunities(features, atrs, held_symbols={"RELIANCE"}, date="2024-04-10")
    assert len(em.missed_opportunities) == 0


def test_detect_missed_small_move_ignored():
    em = ExperienceManager(experience_dir=None)
    features = {"INFY": {"ret_5": 0.01, "atr_norm": 0.02, "rsi_14": 50, "adx_14": 20}}
    atrs = {"INFY": 15.0}
    em.detect_missed_opportunities(features, atrs, held_symbols=set(), date="2024-04-10")
    assert len(em.missed_opportunities) == 0


# ---------------------------------------------------------------------------
# Reflection Cycle
# ---------------------------------------------------------------------------

def test_reflection_triggers_after_interval():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    for i in range(4):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    assert not em.should_reflect()
    em.on_new_trading_day("2024-01-05")
    assert em.should_reflect()


def test_reflection_updates_beliefs():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")

    em.record_entry("A", "BUY", "Test thesis", {"rsi": 30}, 100.0, "2024-01-01", "trending")
    em.record_exit("A", 110.0, 100.0, 10.0, "2024-01-05", llm_client=None)

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
    assert len(em.reflection_log) == 1


def test_reflection_no_data_resets_counter():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    # No trades or missed opps
    mock_client = MagicMock()
    result = em.reflect(mock_client)
    assert result is False
    assert em.trading_days_since_reflection == 0
    mock_client.chat_completion.assert_not_called()


def test_reflection_llm_failure_doesnt_crash():
    em = ExperienceManager(experience_dir=None, reflection_interval=5)
    for i in range(5):
        em.on_new_trading_day(f"2024-01-{i+1:02d}")
    em.record_entry("A", "BUY", "thesis", {"rsi": 30}, 100.0, "2024-01-01", "trending")
    em.record_exit("A", 90.0, -100.0, -10.0, "2024-01-05", llm_client=None)

    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = Exception("API down")
    em.reflect(mock_client)
    # Should not crash; beliefs unchanged
    assert len(em.beliefs) == 0


# ---------------------------------------------------------------------------
# Beliefs Narrative
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

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


def test_persistence_missing_file():
    """Non-existent dir path -> load is no-op."""
    em = ExperienceManager(experience_dir="/tmp/nonexistent_experience_dir_xyz")
    em.load()
    assert len(em.beliefs) == 0


def test_parse_beliefs_handles_edge_cases():
    em = ExperienceManager(experience_dir=None)
    # Short lines should be filtered
    assert em._parse_beliefs("1. Short") == []
    # Normal lines should be parsed
    result = em._parse_beliefs("1. This is a valid belief about trading.\n2. Another valid belief here.")
    assert len(result) == 2
    # Max 10
    many = "\n".join(f"{i}. Belief number {i} is long enough to pass." for i in range(15))
    assert len(em._parse_beliefs(many)) == 10
