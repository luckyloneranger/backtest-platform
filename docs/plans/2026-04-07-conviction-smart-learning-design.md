# Conviction-Scaled Trading + Smart Learning System — Design Document

Date: 2026-04-07

## Goal

Six interconnected improvements to the LLM autonomous trader, addressing overtrading (29 trades → negative returns) and noisy learning (400 missed opps, flat beliefs, fixed intervals). Based on experimental data showing trade count is the strongest predictor of returns.

## Features

### 1. Conviction Threshold 8+ (was 7+) with Asymmetric Entry/Exit

**Problem:** LLM rates too many setups at 7/10. 29 trades in 2024 → -21%.

**Fix:** Entry requires conviction 8+. Exits only require 5+. System prompt updated with explicit conviction ladder.

### 2. Position Size Proportional to Conviction

**Problem:** Binary trade/don't-trade. LLM can't express uncertainty.

**Fix:** Conviction 8 → 50% of max position. 9 → 75%. 10 → 100%. Parsed from `reasoning` field via regex `CONVICTION: (\d+)/10`.

### 3. Smart Missed Opportunity Detection

**Problem:** Current detector flags ~400 missed opps/year. Most are noise (unforeseeable moves). Creates FOMO-driven beliefs.

**Fix:** Only flag when indicators AT THE TIME supported an entry. Confluence score: RSI + MACD + SMA position + ADX. Need 3 of 4 aligned with the move direction. Expected: ~400 → ~50-100 genuine misses.

### 4. Regime-Conditional Beliefs

**Problem:** Flat belief list. "Banking breakouts work" applies to trending markets but hurts in ranging ones. Cross-year transfer failed because beliefs were regime-agnostic.

**Fix:** Each belief tagged with regime: `[TRENDING]`, `[RANGING]`, `[NEUTRAL]`, or `[ALL]`. Reflection prompt requires tags. `get_beliefs_narrative(regime)` filters to show only relevant beliefs.

**Belief structure:**
```python
{"text": "...", "regime": "trending", "created_at_reflection": 5, "strength": 1.0}
```

### 5. Belief Decay

**Problem:** Old beliefs persist forever. Stale beliefs from a different market phase pollute decisions.

**Fix:** Each belief has `strength` (0.0-1.0). Decays by 0.05 per reflection cycle. Beliefs drop below 0.1 → removed. Re-affirmed beliefs reset to 1.0. A belief survives ~18 reflections (~90 trading days) without reinforcement.

### 6. Adaptive Reflection Interval

**Problem:** Fixed 5-day reflection. Should adapt faster when losing.

**Fix:** Recent P&L of last 5 trades determines interval:
- Losing → reflect every 3 days (faster learning from failures)
- Winning → reflect every 8 days (don't mess with what works)
- Neutral → default 5 days

## Architecture

```
SYSTEM PROMPT
├── Conviction ladder: 8+ entry, 5+ exit, scaling stated
├── Regime-filtered beliefs (only current regime + "all")
└── Belief strength shown as [★★★] / [★★] / [★]

GUARDRAILS
├── _extract_conviction() → parse from reasoning
├── Reject entries with conviction < 8
├── Scale qty: {8: 50%, 9: 75%, 10: 100%}
└── Exits allowed at conviction 5+

EXPERIENCE MANAGER
├── Beliefs: list[dict] with regime, strength, created_at
├── detect_missed_opportunities() → confluence filter (3 of 4 indicators)
├── reflect() → decay old beliefs, parse regime tags, adaptive interval
├── get_beliefs_narrative(regime) → filtered + strength-indicated
└── Persistence: updated JSON schema
```

## Files

| File | Changes |
|------|---------|
| `experience_manager.py` | Belief dicts (regime/strength/created), decay, confluence filter, adaptive interval, regime-filtered narrative |
| `llm_autonomous_trader.py` | System prompt (conviction 8+/scaling), `_extract_conviction()`, conviction-scaled qty, pass regime to beliefs |
| `test_experience_manager.py` | Tests for decay, regime filter, confluence, adaptive interval |
| `test_llm_autonomous_trader.py` | Tests for conviction parse, qty scaling, exit threshold |

## Verification

```bash
cd strategies && pytest tests/ -v
```

Run 2025 fresh with new system:
```bash
backtest run --strategy llm_autonomous_trader \
  --symbols RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC,KOTAKBANK,LT \
  --from 2025-01-01 --to 2025-12-31 --capital 100000 --interval 15minute \
  --params '{"reset_experience": true}'
```

Expected: fewer trades (target <10), conviction-scaled positions, cleaner beliefs.
