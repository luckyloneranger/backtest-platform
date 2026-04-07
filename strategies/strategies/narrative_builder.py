"""Narrative builder: translates raw indicator values into factual English
summaries for LLM-based strategies.

Presents FACTS without interpretation, conclusions, or suggestions.
The LLM builds its own thesis from the data.

Four public functions:
  - build_symbol_narrative       -- per-symbol factual data summary
  - build_regime_narrative       -- market-wide regime facts
  - build_portfolio_narrative    -- portfolio-level summary
  - build_trade_history_narrative -- recent trade outcomes
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
    """Build a factual narrative for one symbol.

    Presents raw indicator values, relative context, and risk math
    without interpretation or trading suggestions.
    """
    if features is None:
        return f"{symbol} (₹{close}) — INSUFFICIENT DATA for analysis."

    lines = [f"\n{symbol} (₹{close:.2f})"]

    # --- Position status (factual P&L) ---
    if position:
        pnl_pct = (close - position["avg_price"]) / position["avg_price"] * 100
        lines.append(
            f"  Position: {position['qty']} shares @ ₹{position['avg_price']:.2f}. "
            f"Unrealized P&L: {'+'if pnl_pct > 0 else ''}{pnl_pct:.1f}%."
        )
    else:
        lines.append("  Position: Not held.")

    # --- Price action ---
    price_parts: list[str] = []
    # Weekly return
    ret_5 = features.get("ret_5")
    if ret_5 is not None:
        pct = ret_5 * 100
        price_parts.append(f"{'+'if pct > 0 else ''}{pct:.1f}% this week")
    # Distance from 20-day SMA
    sma_ratio = features.get("close_sma_ratio")
    if sma_ratio is not None:
        pct_from_sma = (sma_ratio - 1.0) * 100
        if abs(pct_from_sma) < 0.5:
            price_parts.append("at 20-day average")
        else:
            above_below = "above" if pct_from_sma > 0 else "below"
            price_parts.append(f"{abs(pct_from_sma):.1f}% {above_below} 20-day average")
    if price_parts:
        lines.append(f"  Price action: {'. '.join(price_parts)}.")

    # --- Momentum ---
    momentum_parts: list[str] = []
    rsi = features.get("rsi_14")
    if rsi is not None:
        momentum_parts.append(f"RSI {rsi:.0f}")
    macd_hist = features.get("macd_hist")
    macd_slope = features.get("macd_hist_slope")
    if macd_hist is not None:
        hist_text = f"MACD histogram {macd_hist:+.2f}"
        if macd_slope is not None:
            if macd_slope > 0:
                hist_text += ", rising"
            elif macd_slope < 0:
                hist_text += ", falling"
            else:
                hist_text += ", flat"
        momentum_parts.append(hist_text)
    stoch_k = features.get("stoch_k")
    stoch_d = features.get("stoch_d")
    if stoch_k is not None and stoch_d is not None:
        momentum_parts.append(f"Stochastic %K={stoch_k:.0f} %D={stoch_d:.0f}")
    if momentum_parts:
        lines.append(f"  Momentum: {'. '.join(momentum_parts)}.")

    # --- Trend ---
    trend_parts: list[str] = []
    adx = features.get("adx_14")
    if adx is not None:
        trend_parts.append(f"ADX {adx:.0f}")
    if sma_ratio is not None:
        if sma_ratio > 1.0:
            trend_parts.append("Price above 20-day SMA")
        elif sma_ratio < 1.0:
            trend_parts.append("Price below 20-day SMA")
        else:
            trend_parts.append("Price at 20-day SMA")
    ema_ratio = features.get("close_ema_ratio")
    if ema_ratio is not None:
        ema_pct = (ema_ratio - 1.0) * 100
        trend_parts.append(f"{'+'if ema_pct > 0 else ''}{ema_pct:.1f}% from 20-day EMA")
    if trend_parts:
        lines.append(f"  Trend: {'. '.join(trend_parts)}.")

    # --- Volatility ---
    vol_parts: list[str] = []
    bb_pct = features.get("bb_pct_b")
    if bb_pct is not None:
        bb_percentile = bb_pct * 100
        vol_parts.append(f"Bollinger Band position {bb_percentile:.0f}th percentile")
    bbw = features.get("bbw")
    if bbw is not None:
        vol_parts.append(f"BBW {bbw:.4f}")
    if atr is not None:
        vol_parts.append(f"ATR ₹{atr:.2f}")
    atr_norm = features.get("atr_norm")
    if atr_norm is not None:
        vol_parts.append(f"ATR/price {atr_norm:.3f}")
    if vol_parts:
        lines.append(f"  Volatility: {'. '.join(vol_parts)}.")

    # --- Volume ---
    volume_parts: list[str] = []
    vol_z = features.get("volume_zscore")
    if vol_z is not None:
        if abs(vol_z) > 0.1:
            volume_parts.append(f"Volume z-score {vol_z:+.1f}")
        else:
            volume_parts.append("Volume near average")
    obv_slope = features.get("obv_slope_10")
    if obv_slope is not None:
        if obv_slope > 0:
            volume_parts.append("OBV rising")
        elif obv_slope < 0:
            volume_parts.append("OBV declining")
        else:
            volume_parts.append("OBV flat")
    if volume_parts:
        lines.append(f"  Volume: {'. '.join(volume_parts)}.")

    # --- Returns context ---
    ret_parts: list[str] = []
    ret_1 = features.get("ret_1")
    if ret_1 is not None:
        ret_parts.append(f"1-day: {'+'if ret_1 * 100 > 0 else ''}{ret_1 * 100:.1f}%")
    if ret_5 is not None:
        ret_parts.append(f"5-day: {'+'if ret_5 * 100 > 0 else ''}{ret_5 * 100:.1f}%")
    ret_10 = features.get("ret_10")
    if ret_10 is not None:
        ret_parts.append(f"10-day: {'+'if ret_10 * 100 > 0 else ''}{ret_10 * 100:.1f}%")
    ret_20 = features.get("ret_20")
    if ret_20 is not None:
        ret_parts.append(f"20-day: {'+'if ret_20 * 100 > 0 else ''}{ret_20 * 100:.1f}%")
    if ret_parts:
        lines.append(f"  Returns: {'. '.join(ret_parts)}.")

    # --- Risk math (pure calculation) ---
    if atr is not None and close > 0:
        stop_distance = 2.0 * atr
        stop_pct = stop_distance / close * 100
        max_shares = int(capital * 0.03 / stop_distance) if stop_distance > 0 else 0
        lines.append(
            f"  Risk math: 2x ATR stop = ₹{stop_distance:.2f} ({stop_pct:.1f}% from entry). "
            f"At 3% capital risk, max {max_shares} shares."
        )

    return "\n".join(lines)


def build_regime_narrative(
    adx: float | None, bbw: float | None, avg_bbw: float | None
) -> str:
    """Build a factual market regime narrative.

    Presents ADX and BBW values without recommending strategies or actions.
    """
    parts: list[str] = []

    if adx is not None:
        parts.append(f"Average ADX across portfolio: {adx:.1f}.")
    else:
        parts.append("ADX data unavailable.")

    if bbw is not None and avg_bbw is not None and avg_bbw > 0:
        ratio = bbw / avg_bbw
        if ratio > 1.5:
            parts.append(
                f"Bollinger Band Width is {ratio:.1f}x above average — elevated volatility."
            )
        elif ratio < 0.6:
            parts.append(
                f"Bollinger Band Width is {ratio:.1f}x below average — compressed volatility."
            )
        else:
            parts.append("Bollinger Band Width is at average levels.")
    elif bbw is not None:
        parts.append(f"Bollinger Band Width: {bbw:.4f}.")

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
    and open position count. Factual only — no advice.
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
        lines.append(f"  Drawdown: {drawdown_pct:.1f}% from initial capital.")
    else:
        lines.append("  No drawdown from initial capital — portfolio is in profit.")

    return "\n".join(lines)


def build_trade_history_narrative(trade_log: list[dict]) -> str:
    """Build a factual narrative from recent trade history.

    Each entry: {symbol, side, entry_price, exit_price, pnl, pnl_pct,
                 reasoning, bars_held}
    Returns summary of last 5 trades, win rate, avg win/loss, profit factor.
    No lessons or advice — just outcomes.
    """
    if not trade_log:
        return "TRADE HISTORY: No completed trades yet."

    lines = ["TRADE HISTORY:"]

    # Last 5 trades — factual outcomes
    recent = trade_log[-5:]
    lines.append(f"  Last {len(recent)} trade(s):")
    for t in recent:
        pnl = t.get("pnl", 0)
        pnl_pct = t.get("pnl_pct", 0.0)
        held = t.get("bars_held", "?")
        lines.append(
            f"    {t.get('symbol','?')} {t.get('side','?')}: "
            f"₹{t.get('entry_price', 0):.2f} → ₹{t.get('exit_price', 0):.2f} "
            f"= {'+'if pnl_pct > 0 else ''}{pnl_pct:.1f}% "
            f"(₹{'+'if pnl > 0 else ''}{pnl:.0f}), "
            f"held {held} bars"
        )

    # Statistics — factual only
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

    return "\n".join(lines)
