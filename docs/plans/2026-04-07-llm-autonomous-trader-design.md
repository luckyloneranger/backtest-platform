# LLM Autonomous Trader — Design Document

Date: 2026-04-07

## Goal

Build an LLM-powered autonomous trading strategy that receives pre-interpreted narrative analysis (not raw numbers) and makes all trading decisions. The LLM acts as a portfolio manager receiving daily analyst reports.

## Core Insight

LLMs understand language, not numbers. Instead of `RSI=32, ADX=18, MACD=-1.1`, we send:
```
RSI at 32 has entered oversold territory. ADX at 18 indicates a ranging market.
MACD is negative and falling, suggesting continued selling pressure.
CONFLICT: RSI says "buy the dip" but MACD says "momentum still down."
```

This forces the LLM to do what it does best: synthesize conflicting signals and make judgment calls.

## Architecture

```
Raw Bars → indicators.py (compute numbers)
         → narrative_builder.py (interpret numbers → English)
         → LLM prompt (system rules + narrative dashboard)
         → LLM response (JSON signals with reasoning)
         → Safety guardrails (validate + cap)
         → PositionManager (execute)
```

## New Module: `strategies/strategies/narrative_builder.py`

A pure function that transforms indicator values into qualitative English narratives.

### Per-Symbol Narrative

```python
def build_symbol_narrative(
    symbol: str,
    close: float,
    indicators: dict[str, float],  # from compute_features()
    position: dict | None,  # current position if any
    atr: float | None,
    capital: float,
) -> str:
```

Produces text like:
```
RELIANCE (₹1,245) — HOLD/PARTIAL EXIT candidate
You hold 8 shares bought at ₹1,205. Currently up ₹320 (+3.3%).
The stock is in a moderate uptrend — price is 3.2% above its 20-day average...
SIGNAL CONFLUENCE: 4 of 5 indicators bullish. No conflicts.
RISK: Stop at ₹1,170 would lose ₹600 (0.6% of portfolio).
```

**Interpretation rules (hardcoded):**

| Indicator | Bullish | Neutral | Bearish |
|-----------|---------|---------|---------|
| RSI | < 35 "oversold" | 35-65 "neutral" | > 65 "overbought" |
| MACD hist | > 0 and rising "accelerating" | near 0 "flat" | < 0 and falling "selling pressure" |
| ADX | > 25 "trending" | 20-25 "transitioning" | < 20 "ranging/no trend" |
| BB %B | < 0.2 "near lower support" | 0.2-0.8 "mid-range" | > 0.8 "near upper resistance" |
| OBV slope | positive "buying pressure" | flat "no conviction" | negative "distribution" |
| Price vs SMA | > SMA "above average" | near SMA "at average" | < SMA "below average" |

**Confluence scoring:**
- Count signals: each indicator votes +1 (bullish), 0 (neutral), -1 (bearish)
- Score 3+ in either direction = "strong convergence"
- Mixed signals = explicitly list the CONFLICTS
- Action suggestion: strong bull = "POTENTIAL LONG", strong bear = "POTENTIAL SHORT", mixed = "NO ACTION"

**Risk calculation:**
- Compute stop price (2x ATR below/above entry)
- Compute dollar risk per position
- Express as % of portfolio
- Flag if risk exceeds 3% of capital

### Regime Narrative

```python
def build_regime_narrative(adx: float, bbw: float, avg_bbw: float) -> str:
```

Produces:
```
The broad market is TRENDING (ADX=28.4). Volatility is normal.
Trend-following strategies historically outperform in this regime.
```

### Trade History Narrative

```python
def build_trade_history_narrative(recent_trades: list[dict]) -> str:
```

Produces:
```
Your last 3 trades: 1 win (+5.1%), 2 losses (-1.3%, -1.2%).
Win rate: 50%. Profit factor: 2.3.
LESSON: Last RSI mean-reversion entry failed because MACD was still falling.
Wait for MACD to flatten before mean-reversion entries.
```

Analyzes recent losses to find patterns and suggests adjustments.

### Portfolio Narrative

```python
def build_portfolio_narrative(cash, equity, peak_equity, positions, costs) -> str:
```

Produces:
```
Portfolio: ₹97,450 equity (from ₹1,00,000). Drawdown: -2.55%.
1 open position. Cash: ₹85,230. Costs year-to-date: ₹1,240 (1.2%).
```

## Strategy: `strategies/deterministic/llm_autonomous_trader.py`

**Register:** `@register("llm_autonomous_trader")`
**Interval:** day, lookback 200

### System Prompt (encodes everything we learned)

```
You are an autonomous portfolio manager for a ₹1,00,000 Indian equity portfolio.
You trade 12 NSE stocks: RELIANCE, INFY, TCS, BAJFINANCE, HINDUNILVR, BHARTIARTL,
SBIN, ICICIBANK, HDFCBANK, ITC, KOTAKBANK, LT.

RULES YOU MUST FOLLOW:
1. FEWER TRADES WIN. Target 2-4 trades per month maximum. Every trade costs money.
2. NEVER enter without a stop-loss. Submit an SL_M order for every entry.
3. Risk no more than 3% of capital per trade.
4. Maximum 4 open positions at any time.
5. In TRENDING markets (ADX>25): favor breakout entries, wider stops, let winners run.
6. In RANGING markets (ADX<20): favor mean-reversion entries, tighter stops.
7. Use CNC for high-conviction multi-day trades. Use MIS for uncertain/intraday trades.
8. Cut losses at -2% to -3%. Trail stops on winners.
9. If drawdown exceeds -10%, reduce position sizes by 50% until recovery.
10. If unsure, do NOTHING. Return []. Sitting in cash is a valid strategy.

RESPOND with a JSON array of signals. Each signal must include:
- "action": "BUY", "SELL", or "CANCEL"
- "symbol", "quantity", "order_type", "product_type"
- "limit_price" (for LIMIT), "stop_price" (for SL_M)
- "reasoning": explain WHY in 1-2 sentences

Return [] if no action should be taken today.
```

### on_bar Flow

```python
def on_bar(self, snapshot):
    self.pm.increment_bars()
    signals = self.pm.process_fills(snapshot)
    signals += self.pm.resubmit_expired(snapshot)
    self.pm.reconcile(snapshot)

    if "day" not in snapshot.timeframes:
        return signals

    # 1. Compute all indicators for all symbols
    all_indicators = {}
    for symbol, bar in snapshot.timeframes["day"].items():
        self._update_buffers(symbol, bar)
        features = compute_features(closes, highs, lows, volumes)
        all_indicators[symbol] = features

    # 2. Build narrative dashboard
    narrative = build_portfolio_narrative(...)
    narrative += build_regime_narrative(...)
    for symbol in all_symbols:
        narrative += build_symbol_narrative(symbol, indicators, position, ...)
    narrative += build_trade_history_narrative(self.trade_log)

    # 3. Call LLM
    messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": narrative},
    ]
    response = self.client.chat_completion(messages, temperature=0.3, max_tokens=1024)

    # 4. Parse + validate + apply guardrails
    raw_signals = self.parse_signals(response)
    validated = self._apply_guardrails(raw_signals, snapshot)

    # 5. Execute via PositionManager
    for sig in validated:
        if sig.action == "BUY":
            signals += self.pm.enter_long(...)
        elif sig.action == "SELL":
            signals += self.pm.exit_position(...)
        # ... etc

    # 6. Log reasoning for post-analysis
    self._log_reasoning(raw_signals, response)

    return signals
```

### Safety Guardrails (`_apply_guardrails`)

Applied AFTER LLM response, BEFORE execution:

1. **Position size cap**: `qty = min(qty, cash / price)` — no leverage
2. **Max 4 positions**: Reject entries if already holding 4 symbols
3. **Auto stop-loss**: If LLM submits a BUY without a corresponding SL_M, auto-add one at -3%
4. **Daily trade limit**: Max 3 new entries per day
5. **Drawdown scaling**: If drawdown > 10%, halve all quantities
6. **Cost tracking**: Accumulate costs, include in next day's narrative
7. **Signal validation**: Reject empty symbols, zero qty, invalid order types

### Trade Log (Memory Within Session)

```python
self.trade_log = []  # list of dicts: {symbol, side, entry, exit, pnl, reasoning, date}
```

When a trade closes (detected via `snapshot.closed_trades`), log it with the LLM's reasoning from entry. This feeds back into `build_trade_history_narrative()` so the LLM can learn from its own decisions within the backtest.

## Backtesting Considerations

- At daily bars, the LLM is called once per bar = 250 calls/year
- Temperature 0.3 for more deterministic responses (not 0.0 to avoid exact repetition)
- Each call ~2-3 seconds = ~10 minutes for a full year backtest
- Results are non-deterministic — run 3x and average for reliable assessment
- Can run in "narrative-only" mode (skip LLM call, log the narrative) for debugging

## Files

| Action | File |
|--------|------|
| CREATE | `strategies/strategies/narrative_builder.py` (~200 lines) |
| CREATE | `strategies/strategies/deterministic/llm_autonomous_trader.py` (~250 lines) |
| CREATE | `strategies/tests/test_narrative_builder.py` |
| CREATE | `strategies/tests/test_llm_autonomous_trader.py` |
| MODIFY | `strategies/server/server.py` — add import |

## What Makes This Different From All Previous Strategies

| Previous strategies | This strategy |
|--------------------|---------------|
| Fixed rules: "if RSI < 35 then buy" | LLM reasons: "RSI oversold but MACD falling — wait" |
| Same behavior regardless of history | Adapts based on recent trade outcomes |
| Binary signals (buy/sell/hold) | Graduated conviction with reasoning |
| No awareness of portfolio context | Considers drawdown, cost budget, open positions |
| Numbers in, numbers out | Narratives in, judgment out |

## Verification

```bash
cd strategies && pytest tests/ -v

# Run at ₹1L on 2025 (single year — 250 LLM calls):
backtest run --strategy llm_autonomous_trader --symbols RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC,KOTAKBANK,LT --from 2025-01-01 --to 2025-12-31 --capital 100000 --interval day --max-drawdown 0.25

# Compare against Portfolio Combiner and OU Mean Reversion baselines
```
