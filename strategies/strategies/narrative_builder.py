"""Narrative builder: translates raw indicator values into qualitative English
narratives for LLM-based strategies.

Four public functions:
  - build_symbol_narrative   -- per-symbol indicator interpretation
  - build_regime_narrative   -- market regime description
  - build_portfolio_narrative -- portfolio-level summary
  - build_trade_history_narrative -- recent trade history with lessons
"""

from __future__ import annotations


def build_symbol_narrative(
    symbol: str,
    close: float,
    features: dict[str, float] | None,
    atr: float | None,
    position: dict | None,  # {qty, avg_price, unrealized_pnl, product_type} or None
    capital: float,
) -> str:
    """Build a qualitative narrative for one symbol."""
    if features is None:
        return f"{symbol} (₹{close}) — INSUFFICIENT DATA for analysis."

    lines = [f"\n{symbol} (₹{close:.2f})"]

    # Position status
    if position:
        pnl_pct = (close - position["avg_price"]) / position["avg_price"] * 100
        lines.append(
            f"  You hold {position['qty']} shares @ ₹{position['avg_price']:.2f}. "
            f"P&L: {'+'if pnl_pct > 0 else ''}{pnl_pct:.1f}%"
        )

    # RSI interpretation
    rsi = features.get("rsi_14")
    if rsi is not None:
        if rsi < 25:
            rsi_text = f"RSI at {rsi:.0f} — deeply oversold, strong mean-reversion buy signal"
        elif rsi < 35:
            rsi_text = f"RSI at {rsi:.0f} — oversold territory, potential buy opportunity"
        elif rsi < 45:
            rsi_text = f"RSI at {rsi:.0f} — slightly below neutral, mild bullish lean"
        elif rsi < 55:
            rsi_text = f"RSI at {rsi:.0f} — neutral, no momentum signal"
        elif rsi < 65:
            rsi_text = f"RSI at {rsi:.0f} — slightly above neutral, mild bearish lean"
        elif rsi < 75:
            rsi_text = f"RSI at {rsi:.0f} — overbought territory, consider taking profits"
        else:
            rsi_text = f"RSI at {rsi:.0f} — deeply overbought, mean-reversion sell signal"
        lines.append(f"  {rsi_text}")

    # ADX interpretation
    adx = features.get("adx_14")
    if adx is not None:
        if adx > 30:
            adx_text = f"ADX at {adx:.0f} — strong trend in place, favor trend-following"
        elif adx > 25:
            adx_text = f"ADX at {adx:.0f} — trending, breakout strategies viable"
        elif adx > 20:
            adx_text = f"ADX at {adx:.0f} — transitioning, uncertain regime"
        else:
            adx_text = f"ADX at {adx:.0f} — ranging/no trend, mean-reversion strategies preferred"
        lines.append(f"  {adx_text}")

    # MACD interpretation
    macd_hist = features.get("macd_hist")
    macd_slope = features.get("macd_hist_slope")
    if macd_hist is not None:
        direction = "positive" if macd_hist > 0 else "negative"
        if macd_slope is not None:
            if macd_slope > 0:
                momentum = "and accelerating"
            elif macd_slope < -0.5:
                momentum = "but decelerating"
            else:
                momentum = "and stable"
        else:
            momentum = ""
        lines.append(f"  MACD histogram is {direction} {momentum}")

    # Bollinger %B interpretation
    bb_pct = features.get("bb_pct_b")
    if bb_pct is not None:
        if bb_pct > 0.9:
            bb_text = "Price at upper Bollinger Band — near resistance, may face selling pressure"
        elif bb_pct > 0.7:
            bb_text = "Price in upper portion of Bollinger Bands — bullish but approaching resistance"
        elif bb_pct < 0.1:
            bb_text = "Price at lower Bollinger Band — near support, potential bounce zone"
        elif bb_pct < 0.3:
            bb_text = "Price in lower portion of Bollinger Bands — bearish but approaching support"
        else:
            bb_text = "Price in middle of Bollinger Bands — no extreme signal"
        lines.append(f"  {bb_text}")

    # OBV interpretation
    obv_slope = features.get("obv_slope_10")
    if obv_slope is not None:
        if obv_slope > 0:
            lines.append("  Volume confirms move (OBV rising — buying pressure)")
        elif obv_slope < 0:
            lines.append("  Volume diverges (OBV falling — distribution/selling)")
        else:
            lines.append("  Volume neutral (OBV flat — no strong conviction)")

    # Price vs SMA
    sma_ratio = features.get("close_sma_ratio")
    if sma_ratio is not None:
        pct_from_sma = (sma_ratio - 1.0) * 100
        if abs(pct_from_sma) < 0.5:
            lines.append("  Price at 20-day average (within 0.5%)")
        else:
            above_below = "above" if pct_from_sma > 0 else "below"
            lines.append(f"  Price is {abs(pct_from_sma):.1f}% {above_below} its 20-day average")

    # Confluence scoring
    bullish = 0
    bearish = 0
    if rsi is not None:
        if rsi < 35:
            bullish += 1
        elif rsi > 65:
            bearish += 1
    if macd_hist is not None:
        if macd_hist > 0:
            bullish += 1
        elif macd_hist < 0:
            bearish += 1
    if bb_pct is not None:
        if bb_pct < 0.2:
            bullish += 1
        elif bb_pct > 0.8:
            bearish += 1
    if obv_slope is not None:
        if obv_slope > 0:
            bullish += 1
        elif obv_slope < 0:
            bearish += 1
    if adx is not None and adx > 25 and sma_ratio is not None:
        if sma_ratio > 1.0:
            bullish += 1
        elif sma_ratio < 1.0:
            bearish += 1

    if bullish >= 3:
        lines.append(f"  CONFLUENCE: {bullish} of 5 indicators bullish — POTENTIAL LONG ENTRY")
    elif bearish >= 3:
        lines.append(f"  CONFLUENCE: {bearish} of 5 indicators bearish — POTENTIAL SHORT ENTRY")
    elif bullish > 0 and bearish > 0:
        lines.append(
            f"  CONFLUENCE: Mixed signals ({bullish} bullish, {bearish} bearish) — CONFLICTING"
        )
    else:
        lines.append("  CONFLUENCE: No strong signals — NO ACTION")

    # Risk calculation
    if atr is not None and close > 0:
        stop_distance = 2.0 * atr
        stop_pct = stop_distance / close * 100
        max_shares = int(capital * 0.03 / stop_distance) if stop_distance > 0 else 0
        lines.append(
            f"  RISK: 2x ATR stop = ₹{stop_distance:.2f} ({stop_pct:.1f}% from entry). "
            f"Max position at 3% risk: {max_shares} shares"
        )

    # Action suggestion
    if position:
        if bullish >= 3:
            suggestion = "HOLD — trend continues in your favor"
        elif bearish >= 3:
            suggestion = "CONSIDER EXIT — signals turning against you"
        else:
            suggestion = "HOLD — no clear exit signal"
    else:
        if bullish >= 3:
            suggestion = "POTENTIAL LONG ENTRY"
        elif bearish >= 3:
            suggestion = "POTENTIAL SHORT ENTRY"
        else:
            suggestion = "NO ACTION — wait for clearer setup"
    lines.append(f"  SUGGESTION: {suggestion}")

    return "\n".join(lines)


def build_regime_narrative(
    adx: float | None, bbw: float | None, avg_bbw: float | None
) -> str:
    """Build a market regime narrative from ADX and Bollinger Band Width.

    Returns 2-3 sentences describing the current regime and which strategies
    are favored.
    """
    parts: list[str] = []

    # Determine regime from ADX
    if adx is not None:
        if adx > 30:
            parts.append(f"MARKET REGIME: TRENDING (ADX={adx:.0f}).")
            parts.append(
                "Trend-following strategies (breakouts, momentum) are favored."
            )
        elif adx > 20:
            parts.append(f"MARKET REGIME: TRANSITIONING (ADX={adx:.0f}).")
            parts.append(
                "Market is between trending and ranging — use smaller positions and tighter stops."
            )
        else:
            parts.append(f"MARKET REGIME: RANGING (ADX={adx:.0f}).")
            parts.append(
                "Mean-reversion strategies (RSI oversold/overbought) are favored."
            )
    else:
        parts.append("MARKET REGIME: UNKNOWN (ADX unavailable).")

    # Volatility overlay from BBW
    if bbw is not None and avg_bbw is not None and avg_bbw > 0:
        ratio = bbw / avg_bbw
        if ratio > 1.5:
            parts.append(
                "VOLATILE (high BBW). Consider reducing position sizes or staying flat."
            )
        elif ratio < 0.6:
            parts.append(
                "COMPRESSED volatility (low BBW). A breakout move may be imminent."
            )
    elif bbw is not None:
        if bbw > 0.10:
            parts.append(
                "VOLATILE (high BBW). Consider reducing position sizes or staying flat."
            )
        elif bbw < 0.03:
            parts.append(
                "COMPRESSED volatility (low BBW). A breakout move may be imminent."
            )

    return " ".join(parts)


def build_portfolio_narrative(
    cash: float,
    equity: float,
    initial_capital: float,
    positions: list,  # list of Position objects or dicts
    total_costs: float,
    trade_count: int,
) -> str:
    """Build a portfolio-level summary narrative.

    Includes total value, drawdown from initial, transaction cost impact,
    and open position count.
    """
    total_value = cash + equity
    total_return_pct = (total_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
    drawdown_pct = max(0.0, (initial_capital - total_value) / initial_capital * 100) if initial_capital > 0 else 0.0
    cost_pct = total_costs / initial_capital * 100 if initial_capital > 0 else 0.0
    open_count = len(positions) if positions else 0
    cash_pct = cash / total_value * 100 if total_value > 0 else 100.0

    lines = [
        "PORTFOLIO SUMMARY:",
        f"  Total value: ₹{total_value:,.2f} ({'+' if total_return_pct >= 0 else ''}{total_return_pct:.1f}% from start)",
        f"  Cash: ₹{cash:,.2f} ({cash_pct:.0f}% available)",
        f"  Open positions: {open_count}",
        f"  Total trades executed: {trade_count}",
        f"  Transaction costs: ₹{total_costs:,.2f} ({cost_pct:.2f}% of capital)",
    ]

    if drawdown_pct > 0:
        if drawdown_pct > 10:
            lines.append(
                f"  DRAWDOWN: {drawdown_pct:.1f}% from initial capital — DANGER ZONE. "
                "Consider reducing exposure."
            )
        elif drawdown_pct > 5:
            lines.append(
                f"  DRAWDOWN: {drawdown_pct:.1f}% from initial capital — caution advised."
            )
        else:
            lines.append(f"  DRAWDOWN: {drawdown_pct:.1f}% from initial capital — within normal range.")
    else:
        lines.append(f"  No drawdown from initial capital — portfolio is in profit.")

    return "\n".join(lines)


def build_trade_history_narrative(trade_log: list[dict]) -> str:
    """Build a narrative from recent trade history.

    Each entry: {symbol, side, entry_price, exit_price, pnl, pnl_pct,
                 reasoning, bars_held}
    Returns summary of last 5 trades, win rate, avg win/loss, profit factor,
    and a LESSON derived from pattern matching on recent losses.
    """
    if not trade_log:
        return "TRADE HISTORY: No completed trades yet."

    lines = ["TRADE HISTORY:"]

    # Last 5 trades
    recent = trade_log[-5:]
    lines.append(f"  Last {len(recent)} trade(s):")
    for t in recent:
        outcome = "WIN" if t.get("pnl", 0) > 0 else "LOSS"
        pnl_pct = t.get("pnl_pct", 0.0)
        held = t.get("bars_held", "?")
        reasoning = t.get("reasoning", "")
        reason_text = f" ({reasoning})" if reasoning else ""
        lines.append(
            f"    {t.get('symbol','?')} {t.get('side','?')}: "
            f"₹{t.get('entry_price', 0):.2f} → ₹{t.get('exit_price', 0):.2f} "
            f"= {'+' if pnl_pct > 0 else ''}{pnl_pct:.1f}% [{outcome}] "
            f"held {held} bars{reason_text}"
        )

    # Statistics
    wins = [t for t in trade_log if t.get("pnl", 0) > 0]
    losses = [t for t in trade_log if t.get("pnl", 0) <= 0]
    total = len(trade_log)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    avg_win = sum(t.get("pnl_pct", 0) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.get("pnl_pct", 0) for t in losses) / len(losses) if losses else 0
    gross_wins = sum(t.get("pnl", 0) for t in wins)
    gross_losses = abs(sum(t.get("pnl", 0) for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    lines.append(
        f"  Win rate: {win_rate:.0f}% ({len(wins)}W / {len(losses)}L out of {total})"
    )
    lines.append(f"  Avg win: {'+' if avg_win >= 0 else ''}{avg_win:.1f}%  |  Avg loss: {avg_loss:.1f}%")
    if profit_factor == float("inf"):
        lines.append("  Profit factor: inf (no losses)")
    else:
        lines.append(f"  Profit factor: {profit_factor:.2f}")

    # LESSON: pattern matching on recent losses
    lesson = _derive_lesson(trade_log)
    if lesson:
        lines.append(f"  LESSON: {lesson}")

    return "\n".join(lines)


def _derive_lesson(trade_log: list[dict]) -> str:
    """Analyze recent losses for recurring patterns."""
    recent_losses = [t for t in trade_log[-10:] if t.get("pnl", 0) <= 0]
    if len(recent_losses) < 2:
        return ""

    # Pattern 1: all losses on one side
    sides = [t.get("side", "") for t in recent_losses]
    long_losses = sides.count("BUY") + sides.count("LONG")
    short_losses = sides.count("SELL") + sides.count("SHORT")
    if long_losses > 0 and short_losses == 0:
        return (
            f"All {long_losses} recent losses were LONG entries. "
            "Consider reducing long bias or requiring stronger confirmation."
        )
    if short_losses > 0 and long_losses == 0:
        return (
            f"All {short_losses} recent losses were SHORT entries. "
            "Consider reducing short bias or requiring stronger confirmation."
        )

    # Pattern 2: quick stops (held <= 3 bars)
    quick_stops = [t for t in recent_losses if t.get("bars_held", 999) <= 3]
    if len(quick_stops) >= 2:
        return (
            f"{len(quick_stops)} of {len(recent_losses)} losses were stopped out within 3 bars. "
            "Stops may be too tight — consider widening or using time-based exits."
        )

    # Pattern 3: losses with high RSI at entry (if reasoning mentions it)
    overbought_losses = [
        t for t in recent_losses
        if "overbought" in t.get("reasoning", "").lower()
        or "rsi" in t.get("reasoning", "").lower()
    ]
    if len(overbought_losses) >= 2:
        return (
            f"{len(overbought_losses)} losses had RSI-related entries. "
            "Entries may be in overbought/oversold conditions that did not reverse."
        )

    return ""
