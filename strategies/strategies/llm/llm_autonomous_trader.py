"""LLM Autonomous Trader — thesis-driven AI portfolio manager (15-min multi-timeframe).

Receives factual data summaries (not interpreted conclusions) and builds
its own investment theses. Operates on 15-min bars with daily context.
Safety guardrails enforce risk limits.

Multi-timeframe data:
  - Daily bars: trend, regime detection, cross-stock analysis
  - 15-min bars: VWAP, intraday momentum, precise entry/exit timing
"""

from collections import deque
import json
import logging
import re

from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.llm_client import AzureOpenAIClient, LLMClientError
from strategies.position_manager import PositionManager
from strategies.indicators import compute_features, compute_atr, compute_zscore
from strategies.experience_manager import ExperienceManager
from strategies.narrative_builder import (
    build_symbol_narrative,
    build_intraday_narrative,
    build_cross_stock_narrative,
    build_regime_narrative,
    build_portfolio_narrative,
    build_trade_history_narrative,
)
from server.registry import register

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a portfolio manager for a ₹{capital:.0f} Indian equity portfolio trading {n_stocks} NSE stocks.

You receive a multi-timeframe dashboard updated every hour:
- PORTFOLIO: Current holdings, P&L, drawdown, costs
- MARKET REGIME: ADX trend strength, Bollinger Band Width volatility
- PER-STOCK DAILY: Trend, momentum, volatility, volume, returns, z-score
- PER-STOCK INTRADAY: VWAP position, intraday momentum, session context
- CROSS-STOCK: Correlations, cointegrated pairs, relative z-scores
- TRADE HISTORY: Recent outcomes, win rate, profit factor

YOUR PROCESS FOR EACH DECISION:

1. THESIS: For any stock you're considering, build an investment thesis.
   - What is your view? (bullish / bearish / neutral)
   - WHY do you hold this view? What's the story — not just "RSI is low" but WHY that matters for THIS stock right now.
   - What's the catalyst for the move you're expecting?

2. EVIDENCE: What in the data supports your thesis?
   - Which indicators confirm your thesis?
   - What's the price action telling you?
   - Does volume confirm or contradict?
   - What does VWAP position tell you about intraday fair value?
   - Are any stocks cointegrated / correlated that support a relative value thesis?

3. COUNTER-THESIS: What would prove you wrong?
   - What's the bear case if you're bullish (or vice versa)?
   - What data would make you abandon this thesis?
   - Are there conflicting signals you're choosing to ignore? Be honest about them.

4. CONVICTION: Rate 1-10. Position size scales with conviction.
   - 1-6: No action. Not tradeable. Do NOT include in response.
   - 7: Watchlist only. DO NOT trade. Interesting but not ready.
   - 8: Moderate conviction → enter at HALF position size
   - 9: High conviction → enter at THREE-QUARTER position size
   - 10: Maximum conviction → enter at FULL position size

   For EXITS (cutting losses, taking profits, closing positions):
   - 5+: Sufficient to exit. Protecting capital requires less conviction than deploying it.

   You MUST include "CONVICTION: N/10" in your reasoning field.

5. ACTION: If convicted (8+ for entries, 5+ for exits), what specifically?
   - Entry price, stop-loss level, target, position size
   - CNC (will hold for days/weeks) or MIS (exit today)

MULTI-TIMEFRAME USAGE:
- Use daily context for DIRECTION (which side of the market)
- Use intraday context for TIMING (when to enter/exit)
- Use cross-stock analysis for SELECTION (which stocks, pair trades)

PORTFOLIO CONSTRAINTS:
- Capital: ₹{capital:.0f}
- Maximum {max_positions} positions at once
- Maximum 3% of capital at risk per trade
- Always set a stop-loss (SL_M order)

RESPOND with a JSON array. Each signal must include:
{{"action": "BUY"|"SELL"|"CANCEL", "symbol": "...", "quantity": N, \
"order_type": "MARKET"|"LIMIT"|"SL_M", "limit_price": 0.0, "stop_price": 0.0, \
"product_type": "CNC"|"MIS", \
"reasoning": "THESIS: ... EVIDENCE: ... CONVICTION: N/10"}}

Return [] if no thesis reaches conviction 8+ for entries. Sitting in cash is a valid decision.
Most days, the right answer is []. The best traders trade rarely.
"""


@register("llm_autonomous_trader")
class LLMAutonomousTrader(Strategy):
    """Thesis-driven AI portfolio manager operating on 15-min bars with
    daily context. Uses all 17 indicator functions for comprehensive analysis."""

    def required_data(self) -> list[dict]:
        return [
            {"interval": "15minute", "lookback": 200},
            {"interval": "day", "lookback": 200},
        ]

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
        self.llm_interval_bars = config.get("llm_interval_bars", 4)  # call LLM every N bars

        # Experience manager (reflective learning layer)
        exp_dir = config.get("experience_dir", "strategies/data/llm_experience")
        if config.get("reset_experience"):
            exp_dir = None  # fresh start, no persistence
        self.experience = ExperienceManager(
            experience_dir=exp_dir,
            reflection_interval=config.get("reflection_interval", 5),
        )

        # Daily buffers (append once per day, maxlen=300)
        self.daily_closes: dict[str, deque] = {}
        self.daily_highs: dict[str, deque] = {}
        self.daily_lows: dict[str, deque] = {}
        self.daily_volumes: dict[str, deque] = {}

        # 15-min rolling buffers (for RSI, MACD, ATR on 15-min — don't reset)
        self.m15_closes: dict[str, deque] = {}
        self.m15_highs: dict[str, deque] = {}
        self.m15_lows: dict[str, deque] = {}
        self.m15_volumes: dict[str, deque] = {}

        # Intraday buffers (reset each trading day — for VWAP)
        self.intraday_closes: dict[str, list] = {}
        self.intraday_highs: dict[str, list] = {}
        self.intraday_lows: dict[str, list] = {}
        self.intraday_volumes: dict[str, list] = {}

        # Track previous daily close for session gap
        self.prev_daily_close: dict[str, float] = {}

        # Trade log for LLM memory
        self.trade_log: list[dict] = []
        self.total_costs: float = 0.0
        self.initial_capital: float = config.get("initial_capital", 100_000)
        self.peak_equity: float = self.initial_capital
        self.trades_today: int = 0

        # Throttling state
        self.bar_count: int = 0
        self.last_daily_date: str = ""
        self._history_bootstrapped: bool = False

        # Cached cross-stock narrative (computed once per day)
        self._cross_stock_cache: str = ""
        # Cached daily features + z-scores (computed once per day)
        self._daily_features: dict[str, dict | None] = {}
        self._daily_atrs: dict[str, float | None] = {}
        self._daily_zscores: dict[str, float | None] = {}

        # Number of stocks (for system prompt)
        self.n_stocks = len(instruments) if instruments else 0

        # Format system prompt
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

        # Bootstrap buffers from lookback history on first call
        if not self._history_bootstrapped:
            self._bootstrap_from_history(snapshot)
            self._history_bootstrapped = True

        # Track closed trades
        for trade in snapshot.closed_trades:
            self._log_trade(trade)
            # Record exit in experience manager
            pnl_pct = (
                (trade.pnl / (trade.entry_price * trade.quantity) * 100)
                if trade.entry_price > 0 and trade.quantity > 0 else 0
            )
            self.experience.record_exit(
                symbol=trade.symbol,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_pct=pnl_pct,
                date=self.last_daily_date or self._extract_date(snapshot),
                llm_client=self.client,
            )
        for fill in snapshot.fills:
            self.total_costs += fill.costs
        self.peak_equity = max(self.peak_equity, snapshot.portfolio.equity)

        # Process daily bars (update daily buffers, compute daily features)
        has_new_daily = False
        if "day" in snapshot.timeframes:
            current_date = self._extract_date(snapshot)
            if current_date != self.last_daily_date:
                has_new_daily = True
                self.last_daily_date = current_date
                # Reset intraday buffers on new trading day
                self.intraday_closes.clear()
                self.intraday_highs.clear()
                self.intraday_lows.clear()
                self.intraday_volumes.clear()
                self.trades_today = 0

            for symbol, bar in snapshot.timeframes["day"].items():
                self._update_daily_buffers(symbol, bar)

            # Recompute daily features + cross-stock (once per day)
            if has_new_daily:
                self._recompute_daily_analysis(snapshot)

                # Experience learning: new day tracking, missed opps, reflection
                self.experience.on_new_trading_day(current_date)
                held = {s for s in self.pm.states if self.pm.states[s].direction != "flat"}
                self.experience.detect_missed_opportunities(
                    self._daily_features, self._daily_atrs, held, current_date,
                )
                self.experience.reflect(self.client, current_regime=self._get_current_regime())

        # Process 15-min bars
        has_15m = "15minute" in snapshot.timeframes
        if has_15m:
            for symbol, bar in snapshot.timeframes["15minute"].items():
                self._update_m15_buffers(symbol, bar)
                self._update_intraday_buffers(symbol, bar)

        # Throttle LLM calls: only call every N 15-min bars
        if has_15m:
            self.bar_count += 1
        should_call_llm = has_15m and (self.bar_count % self.llm_interval_bars == 0)

        # Fallback: if no 15-min data but we have daily, call LLM on daily
        if not has_15m and has_new_daily:
            should_call_llm = True

        if not should_call_llm:
            return signals

        # Build narrative dashboard
        narrative = self._build_full_narrative(snapshot)

        # Inject learned beliefs into system prompt (regime-filtered)
        active_prompt = self.system_prompt
        current_regime = self._get_current_regime()
        beliefs = self.experience.get_beliefs_narrative(current_regime)
        if beliefs:
            active_prompt += "\n" + beliefs

        # Call LLM
        try:
            messages = [
                {"role": "system", "content": active_prompt},
                {"role": "user", "content": narrative},
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

        # Apply guardrails + execute via PM
        for sig in raw_signals:
            if self.trades_today >= self.max_daily_trades:
                break
            validated = self._apply_guardrails(sig, snapshot)
            if validated:
                signals += validated
                if validated[0].action in ("BUY", "SELL"):
                    self.trades_today += 1
                    # Record entry in experience manager
                    symbol = sig.get("symbol", "")
                    reasoning = sig.get("reasoning", "")
                    features = self._daily_features.get(symbol, {}) or {}
                    regime = (
                        "trending" if (features.get("adx_14") or 0) > 25
                        else "ranging" if (features.get("adx_14") or 0) < 20
                        else "neutral"
                    )
                    price = None
                    for tf in snapshot.timeframes.values():
                        if symbol in tf:
                            price = tf[symbol].close
                            break
                    if price:
                        self.experience.record_entry(
                            symbol=symbol,
                            action=sig.get("action", ""),
                            thesis=reasoning,
                            indicators={
                                k: round(v, 2) if isinstance(v, float) else v
                                for k, v in features.items()
                                if k in ("rsi_14", "adx_14", "close_sma_ratio",
                                         "macd_hist", "volume_zscore", "bbw")
                            },
                            price=price,
                            date=self.last_daily_date or self._extract_date(snapshot),
                            regime=regime,
                        )

        return signals

    def on_complete(self) -> dict:
        self.experience.save()
        return {
            "strategy_type": "llm_autonomous_trader",
            "total_trades": len(self.trade_log),
            "total_costs": self.total_costs,
            "beliefs": self.experience.beliefs,
            "reflections": len(self.experience.reflection_log),
            "missed_opportunities": len(self.experience.missed_opportunities),
        }

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _bootstrap_from_history(self, snapshot: MarketSnapshot) -> None:
        """Pre-populate buffers from lookback history on first on_bar call.

        snapshot.history is dict[(symbol, interval) -> list[BarData]].
        This gives us the 200 warmup bars declared in required_data().
        """
        if not snapshot.history:
            return

        for (symbol, interval), bars in snapshot.history.items():
            if interval == "day":
                for bar in bars:
                    self._update_daily_buffers(symbol, bar)
            elif interval == "15minute":
                for bar in bars:
                    self._update_m15_buffers(symbol, bar)

        # Compute daily analysis from bootstrapped data
        self._recompute_daily_analysis(snapshot)

    def _update_daily_buffers(self, symbol: str, bar) -> None:
        if symbol not in self.daily_closes:
            self.daily_closes[symbol] = deque(maxlen=300)
            self.daily_highs[symbol] = deque(maxlen=300)
            self.daily_lows[symbol] = deque(maxlen=300)
            self.daily_volumes[symbol] = deque(maxlen=300)
        self.daily_closes[symbol].append(bar.close)
        self.daily_highs[symbol].append(bar.high)
        self.daily_lows[symbol].append(bar.low)
        self.daily_volumes[symbol].append(bar.volume)
        # Track prev close for session gap
        if len(self.daily_closes[symbol]) >= 2:
            self.prev_daily_close[symbol] = list(self.daily_closes[symbol])[-2]
        else:
            self.prev_daily_close[symbol] = bar.close

    def _update_m15_buffers(self, symbol: str, bar) -> None:
        if symbol not in self.m15_closes:
            self.m15_closes[symbol] = deque(maxlen=200)
            self.m15_highs[symbol] = deque(maxlen=200)
            self.m15_lows[symbol] = deque(maxlen=200)
            self.m15_volumes[symbol] = deque(maxlen=200)
        self.m15_closes[symbol].append(bar.close)
        self.m15_highs[symbol].append(bar.high)
        self.m15_lows[symbol].append(bar.low)
        self.m15_volumes[symbol].append(bar.volume)

    def _update_intraday_buffers(self, symbol: str, bar) -> None:
        if symbol not in self.intraday_closes:
            self.intraday_closes[symbol] = []
            self.intraday_highs[symbol] = []
            self.intraday_lows[symbol] = []
            self.intraday_volumes[symbol] = []
        self.intraday_closes[symbol].append(bar.close)
        self.intraday_highs[symbol].append(bar.high)
        self.intraday_lows[symbol].append(bar.low)
        self.intraday_volumes[symbol].append(bar.volume)

    def _extract_date(self, snapshot: MarketSnapshot) -> str:
        """Extract date string from snapshot timestamp for day change detection."""
        import datetime
        ts_s = snapshot.timestamp_ms / 1000.0
        dt = datetime.datetime.fromtimestamp(ts_s, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def _get_current_regime(self) -> str:
        """Determine current market regime from average ADX across portfolio."""
        adx_values = [
            f.get("adx_14")
            for f in self._daily_features.values()
            if f and f.get("adx_14") is not None
        ]
        if not adx_values:
            return "neutral"
        avg_adx = sum(adx_values) / len(adx_values)
        if avg_adx > 25:
            return "trending"
        elif avg_adx < 20:
            return "ranging"
        return "neutral"

    # ------------------------------------------------------------------
    # Analysis computation
    # ------------------------------------------------------------------

    def _recompute_daily_analysis(self, snapshot: MarketSnapshot) -> None:
        """Recompute daily features, z-scores, and cross-stock analysis."""
        self._daily_features.clear()
        self._daily_atrs.clear()
        self._daily_zscores.clear()

        for symbol in self.daily_closes:
            closes = list(self.daily_closes[symbol])
            highs = list(self.daily_highs[symbol])
            lows = list(self.daily_lows[symbol])
            volumes = list(self.daily_volumes[symbol])

            self._daily_features[symbol] = compute_features(closes, highs, lows, volumes)
            self._daily_atrs[symbol] = compute_atr(highs, lows, closes, 14)
            self._daily_zscores[symbol] = compute_zscore(closes, 20)

        # Cross-stock analysis
        all_closes = {s: list(self.daily_closes[s]) for s in self.daily_closes}
        symbols = list(self.daily_closes.keys())
        self._cross_stock_cache = build_cross_stock_narrative(all_closes, symbols)

    # ------------------------------------------------------------------
    # Narrative builder
    # ------------------------------------------------------------------

    def _build_full_narrative(self, snapshot: MarketSnapshot) -> str:
        """Assemble the complete multi-timeframe narrative dashboard."""
        parts: list[str] = []

        # Portfolio narrative
        parts.append(
            build_portfolio_narrative(
                snapshot.portfolio.cash,
                snapshot.portfolio.equity,
                self.peak_equity,
                snapshot.portfolio.positions,
                self.total_costs,
                len(self.trade_log),
            )
        )

        # Regime narrative
        adx_values = [
            f.get("adx_14")
            for f in self._daily_features.values()
            if f and f.get("adx_14") is not None
        ]
        avg_adx = sum(adx_values) / len(adx_values) if adx_values else None
        bbw_values = [
            f.get("bbw")
            for f in self._daily_features.values()
            if f and f.get("bbw") is not None
        ]
        avg_bbw = sum(bbw_values) / len(bbw_values) if bbw_values else None
        parts.append(build_regime_narrative(avg_adx, avg_bbw, avg_bbw))

        # Per-symbol: daily narrative + intraday narrative
        # Determine which symbols to show (from any available timeframe)
        all_symbols: set[str] = set()
        for tf in snapshot.timeframes.values():
            all_symbols.update(tf.keys())

        for symbol in sorted(all_symbols):
            # Get current price from best available source
            price = None
            for tf_key in ("15minute", "day"):
                if tf_key in snapshot.timeframes and symbol in snapshot.timeframes[tf_key]:
                    price = snapshot.timeframes[tf_key][symbol].close
                    break
            if price is None:
                continue

            # Position info
            position = None
            for pos in snapshot.portfolio.positions:
                if pos.symbol == symbol:
                    pnl = (price - pos.avg_price) * pos.quantity
                    position = {
                        "qty": pos.quantity,
                        "avg_price": pos.avg_price,
                        "unrealized_pnl": pnl,
                        "product_type": "CNC",
                    }
                    break

            # Daily symbol narrative
            parts.append(
                build_symbol_narrative(
                    symbol,
                    price,
                    self._daily_features.get(symbol),
                    self._daily_atrs.get(symbol),
                    position,
                    snapshot.portfolio.equity,
                    zscore=self._daily_zscores.get(symbol),
                )
            )

            # Intraday narrative (if we have 15-min data for this symbol)
            if symbol in self.intraday_closes and self.intraday_closes[symbol]:
                parts.append(
                    build_intraday_narrative(
                        symbol,
                        self.intraday_closes[symbol],
                        self.intraday_highs[symbol],
                        self.intraday_lows[symbol],
                        self.intraday_volumes[symbol],
                        list(self.m15_closes.get(symbol, [])),
                        list(self.m15_highs.get(symbol, [])),
                        list(self.m15_lows.get(symbol, [])),
                        self.prev_daily_close.get(symbol),
                    )
                )

        # Cross-stock narrative (cached, updated daily)
        if self._cross_stock_cache:
            parts.append(self._cross_stock_cache)

        # Trade history
        parts.append(build_trade_history_narrative(self.trade_log[-10:]))

        # Pending orders
        if snapshot.pending_orders:
            parts.append("\nPENDING ORDERS:")
            for po in snapshot.pending_orders:
                parts.append(
                    f"  {po.symbol}: {po.order_type} {po.side} qty={po.quantity} "
                    f"stop={po.stop_price}"
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLM parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, llm_response: str) -> list[dict]:
        """Parse LLM response into a list of raw signal dicts."""
        md_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", llm_response, re.DOTALL
        )
        text = md_match.group(1).strip() if md_match else llm_response.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response as JSON: %s", llm_response[:200],
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

    # ------------------------------------------------------------------
    # Safety guardrails
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_conviction(reasoning: str) -> int:
        """Parse conviction score from reasoning text. Returns 0-10."""
        match = re.search(r"CONVICTION:\s*(\d+)\s*/\s*10", reasoning, re.IGNORECASE)
        if match:
            return min(10, max(0, int(match.group(1))))
        # Fallback: look for just a number after "conviction"
        match = re.search(r"conviction[:\s]+(\d+)", reasoning, re.IGNORECASE)
        if match:
            return min(10, max(0, int(match.group(1))))
        return 0

    def _apply_guardrails(
        self,
        sig: dict,
        snapshot: MarketSnapshot,
    ) -> list[Signal] | None:
        """Validate and constrain LLM signal with conviction-scaled sizing."""
        known_symbols: set[str] = set()
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

        # Parse conviction from reasoning
        reasoning = sig.get("reasoning", "")
        conviction = self._extract_conviction(reasoning)

        # Determine if this is an entry or exit
        is_entry = (
            (action == "BUY" and self.pm.is_flat(symbol))
            or (action == "SELL" and self.pm.is_flat(symbol))
        )
        is_exit = (
            (action == "SELL" and self.pm.is_long(symbol))
            or (action == "BUY" and self.pm.is_short(symbol))
        )

        # Conviction threshold: 8+ for entries, 5+ for exits
        if is_entry and conviction < 8:
            logger.info("Rejected %s %s: conviction %d < 8", action, symbol, conviction)
            return None
        if is_exit and conviction < 5:
            logger.info("Rejected exit %s %s: conviction %d < 5", action, symbol, conviction)
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

        # Conviction-scaled position sizing for entries
        if is_entry:
            scale = {8: 0.5, 9: 0.75, 10: 1.0}.get(conviction, 0.5)
            qty = max(1, int(qty * scale))

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
        atr = self._daily_atrs.get(symbol)

        if action == "BUY" and self.pm.is_flat(symbol):
            if stop_price <= 0 and atr:
                stop_price = price - 2.0 * atr
            elif stop_price <= 0:
                stop_price = price * (1 - self.auto_stop_pct)
            results += self.pm.enter_long(
                symbol, qty, limit_price, product_type, stop_price
            )

        elif action == "SELL" and self.pm.is_flat(symbol):
            if stop_price <= 0 and atr:
                stop_price = price + 2.0 * atr
            elif stop_price <= 0:
                stop_price = price * (1 + self.auto_stop_pct)
            results += self.pm.enter_short(symbol, qty, limit_price, stop_price)

        elif action == "SELL" and self.pm.is_long(symbol):
            results += self.pm.exit_position(symbol, qty=qty)

        elif action == "BUY" and self.pm.is_short(symbol):
            results += self.pm.exit_position(symbol, qty=qty)

        # Log reasoning
        reasoning = sig.get("reasoning", "")
        if reasoning:
            logger.info("LLM %s %s: %s", action, symbol, reasoning)

        return results if results else None

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

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
