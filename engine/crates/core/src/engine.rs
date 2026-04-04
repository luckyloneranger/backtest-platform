use std::collections::{HashMap, VecDeque};

use anyhow::Result;
use async_trait::async_trait;

use crate::config::BacktestConfig;
use crate::costs::{TradeParams, ZerodhaCostModel};
use crate::matching::{Fill, Order, OrderMatcher, OrderRejection, Side};
use crate::portfolio::{ClosedTrade, EquityPoint, PortfolioManager};
use crate::types::{Action, Bar, InstrumentType, Portfolio, ProductType, Signal};

// ── InstrumentData ─────────────────────────────────────────────────────────

/// Instrument data sent to strategies at initialization and on each bar.
#[derive(Debug, Clone)]
pub struct InstrumentData {
    pub symbol: String,
    pub exchange: String,
    pub instrument_type: String,
    pub lot_size: i32,
    pub tick_size: f64,
    pub expiry: String,
    pub strike: f64,
    pub option_type: String,
    pub circuit_limit_upper: f64,
    pub circuit_limit_lower: f64,
}

// ── IntervalRequirement ────────────────────────────────────────────────────

/// Describes a timeframe interval that a strategy requires, along with
/// how many historical bars to retain for that interval.
#[derive(Debug, Clone)]
pub struct IntervalRequirement {
    pub interval: String,    // "minute", "5minute", "day"
    pub lookback: usize,
}

// ── SessionContext ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionContext {
    pub initial_capital: f64,
    pub bar_number: i32,
    pub total_bars: i32,
    pub start_date: String,
    pub end_date: String,
    pub intervals: Vec<String>,
    pub lookback_window: i32,
}

// ── MarketSnapshot ─────────────────────────────────────────────────────────

/// Full market snapshot sent to strategy per timestamp.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub timestamp_ms: i64,
    /// Bars grouped by interval -> symbol -> bar. Only intervals that have
    /// a bar at this timestamp are included.
    pub timeframes: HashMap<String, HashMap<String, Bar>>,
    /// Lookback history keyed by (symbol, interval) -> bars.
    pub history: HashMap<(String, String), Vec<Bar>>,
    pub portfolio: Portfolio,
    pub instruments: Vec<InstrumentData>,
    pub fills: Vec<Fill>,
    pub rejections: Vec<OrderRejection>,
    pub closed_trades: Vec<ClosedTrade>,
    pub context: SessionContext,
}

// ── StrategyClient trait ────────────────────────────────────────────────────

#[async_trait]
pub trait StrategyClient: Send + Sync {
    /// Query the strategy for its data requirements (intervals + lookback).
    async fn get_requirements(&self, name: &str, config: &str) -> Result<Vec<IntervalRequirement>>;

    async fn initialize(
        &self,
        name: &str,
        config: &str,
        symbols: &[String],
        instruments: &[InstrumentData],
    ) -> Result<()>;

    async fn on_bar(&self, snapshot: &MarketSnapshot) -> Result<Vec<Signal>>;

    async fn on_complete(&self) -> Result<serde_json::Value>;
}

// ── BacktestResult ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub trades: Vec<ClosedTrade>,
    pub equity_curve: Vec<EquityPoint>,
    pub final_equity: f64,
    pub initial_capital: f64,
    pub config: BacktestConfig,
    pub custom_metrics: serde_json::Value,
}

// ── Helper: interval -> bars per day ────────────────────────────────────────

/// Maps a Kite-style interval string to the number of bars in a single
/// Indian trading session (9:15 to 15:30 = 375 minutes).
fn parse_interval_bars_per_day(interval: &str) -> usize {
    match interval {
        "minute" => 375,
        "3minute" => 125,
        "5minute" => 75,
        "10minute" => 38,
        "15minute" => 25,
        "30minute" => 13,
        "60minute" => 7,
        "day" => 1,
        _ => 1,
    }
}

// ── BacktestEngine ──────────────────────────────────────────────────────────

pub struct BacktestEngine;

impl BacktestEngine {
    pub async fn run(
        config: BacktestConfig,
        bars_by_interval: HashMap<String, Vec<Bar>>,
        strategy: &dyn StrategyClient,
        instruments: Vec<InstrumentData>,
        requirements: &[IntervalRequirement],
    ) -> Result<BacktestResult> {
        // 1. Initialize strategy
        strategy
            .initialize(
                &config.strategy_name,
                &serde_json::to_string(&config.strategy_params)?,
                &config.symbols,
                &instruments,
            )
            .await?;

        // 2. Set up components
        let cost_model = ZerodhaCostModel;
        let mut portfolio = PortfolioManager::new(config.initial_capital);
        let mut matcher = OrderMatcher::new(config.slippage_pct);

        // Build instrument type lookup from provided instruments
        let instrument_type_map: HashMap<String, InstrumentType> = instruments
            .iter()
            .filter_map(|inst| {
                let itype = match inst.instrument_type.as_str() {
                    "EQ" => Some(InstrumentType::Equity),
                    "FUT" => Some(InstrumentType::FutureFO),
                    "OPT" => Some(InstrumentType::OptionFO),
                    "COM" => Some(InstrumentType::Commodity),
                    _ => None,
                };
                itype.map(|t| (inst.symbol.clone(), t))
            })
            .collect();

        // 3. Determine the finest interval (most bars per day = finest granularity)
        let finest_interval_str = requirements
            .iter()
            .max_by_key(|r| parse_interval_bars_per_day(&r.interval))
            .map(|r| r.interval.clone())
            .unwrap_or_else(|| "day".to_string());

        // 4. Get the finest interval bars (these drive the tick timeline)
        let finest_bars = bars_by_interval
            .get(&finest_interval_str)
            .ok_or_else(|| {
                anyhow::anyhow!("no bars for finest interval {}", finest_interval_str)
            })?;

        // 5. Index all intervals' bars by timestamp for O(1) lookup
        let mut bar_index: HashMap<String, HashMap<i64, Vec<Bar>>> = HashMap::new();
        // For coarser intervals, keep sorted timestamp list for "most recent" lookup
        let mut sorted_timestamps: HashMap<String, Vec<i64>> = HashMap::new();
        for (interval, bars) in &bars_by_interval {
            let mut ts_map: HashMap<i64, Vec<Bar>> = HashMap::new();
            for bar in bars {
                ts_map
                    .entry(bar.timestamp_ms)
                    .or_default()
                    .push(bar.clone());
            }
            let mut ts_list: Vec<i64> = ts_map.keys().cloned().collect();
            ts_list.sort();
            sorted_timestamps.insert(interval.clone(), ts_list);
            bar_index.insert(interval.clone(), ts_map);
        }

        // Track the last emitted coarser-interval timestamp per interval
        // so we only emit a new bar when a new candle appears
        let mut last_coarse_ts: HashMap<String, i64> = HashMap::new();

        // 6. Lookback buffers per (symbol, interval)
        let mut lookback: HashMap<(String, String), VecDeque<Bar>> = HashMap::new();
        let lookback_sizes: HashMap<String, usize> = requirements
            .iter()
            .map(|r| (r.interval.clone(), r.lookback))
            .collect();

        // Collect all interval names for SessionContext
        let interval_names: Vec<String> = requirements.iter().map(|r| r.interval.clone()).collect();

        // Track fills and rejections for next on_bar call
        let mut last_fills: Vec<Fill> = Vec::new();
        let mut last_rejections: Vec<OrderRejection> = Vec::new();

        // 7. Group finest bars by timestamp
        let grouped = group_bars_by_timestamp(finest_bars.clone());
        debug_assert!(
            grouped.windows(2).all(|w| w[0].0 <= w[1].0),
            "bars are not sorted by timestamp — group_bars_by_timestamp requires sorted input"
        );
        let total_bars = grouped.len() as i32;

        // 8. Process each timestamp group
        for (bar_idx, (timestamp, finest_bar_group)) in grouped.iter().enumerate() {
            // a. Process pending orders using finest interval bars
            let mut current_fills = Vec::new();
            let mut current_rejections = Vec::new();
            for bar in finest_bar_group {
                let (fills, rejects) = matcher.process_bar(bar);
                for fill in &fills {
                    let trade_value = fill.quantity as f64 * fill.fill_price;
                    let is_intraday = fill.product_type == ProductType::Mis;
                    let inst_type = instrument_type_map
                        .get(&fill.symbol)
                        .copied()
                        .unwrap_or(InstrumentType::Equity);
                    let params = TradeParams {
                        instrument_type: inst_type,
                        is_intraday,
                        buy_value: if fill.side == Side::Buy {
                            trade_value
                        } else {
                            0.0
                        },
                        sell_value: if fill.side == Side::Sell {
                            trade_value
                        } else {
                            0.0
                        },
                        quantity: fill.quantity,
                    };
                    let costs = cost_model.calculate(&params);
                    portfolio.apply_fill(fill, costs.total());
                }
                current_fills.extend(fills);
                current_rejections.extend(rejects);
            }

            // b. Update portfolio with current prices (from finest interval bars)
            let mut prices = HashMap::new();
            for bar in finest_bar_group {
                prices.insert(bar.symbol.clone(), bar.close);
            }
            portfolio.update_prices(&prices, *timestamp);

            // c. Build timeframes map and update lookback buffers
            let mut timeframes: HashMap<String, HashMap<String, Bar>> = HashMap::new();
            for req in requirements {
                if let Some(ts_map) = bar_index.get(&req.interval) {
                    // For the finest interval, exact match
                    // For coarser intervals, find the most recent bar at or before this timestamp
                    let matching_ts = if req.interval == finest_interval_str {
                        // Exact match for finest
                        if ts_map.contains_key(timestamp) {
                            Some(*timestamp)
                        } else {
                            None
                        }
                    } else {
                        // Binary search for most recent coarser bar <= current timestamp
                        if let Some(ts_list) = sorted_timestamps.get(&req.interval) {
                            match ts_list.binary_search(timestamp) {
                                Ok(idx) => Some(ts_list[idx]),
                                Err(idx) if idx > 0 => Some(ts_list[idx - 1]),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    };

                    if let Some(bar_ts) = matching_ts {
                        // Only emit if this is a NEW bar we haven't sent before
                        let prev_ts = last_coarse_ts.get(&req.interval).copied();
                        let is_new = prev_ts.map_or(true, |prev| bar_ts > prev);

                        if is_new || req.interval == finest_interval_str {
                            if let Some(bars_at_ts) = ts_map.get(&bar_ts) {
                                let symbol_map: HashMap<String, Bar> = bars_at_ts
                                    .iter()
                                    .map(|b| (b.symbol.clone(), b.clone()))
                                    .collect();
                                timeframes.insert(req.interval.clone(), symbol_map);

                                // Update lookback for this interval
                                if is_new {
                                    last_coarse_ts.insert(req.interval.clone(), bar_ts);
                                    for bar in bars_at_ts {
                                        let key = (bar.symbol.clone(), req.interval.clone());
                                        let max_len = lookback_sizes
                                            .get(&req.interval)
                                            .copied()
                                            .unwrap_or(200);
                                        let buf =
                                            lookback.entry(key).or_insert_with(VecDeque::new);
                                        buf.push_back(bar.clone());
                                        if buf.len() > max_len {
                                            buf.pop_front();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // d. Build history from lookback buffers
            let history: HashMap<(String, String), Vec<Bar>> = lookback
                .iter()
                .map(|(k, buf)| (k.clone(), buf.iter().cloned().collect()))
                .collect();

            // e. Build snapshot
            let snapshot = MarketSnapshot {
                timestamp_ms: *timestamp,
                timeframes,
                history,
                portfolio: portfolio.portfolio_state(),
                instruments: instruments.clone(),
                fills: last_fills.clone(),
                rejections: last_rejections.clone(),
                closed_trades: portfolio.closed_trades().to_vec(),
                context: SessionContext {
                    initial_capital: config.initial_capital,
                    bar_number: bar_idx as i32,
                    total_bars,
                    start_date: config.start_date.clone(),
                    end_date: config.end_date.clone(),
                    intervals: interval_names.clone(),
                    lookback_window: config.lookback_window as i32,
                },
            };

            // f. Get strategy signals
            let signals = strategy.on_bar(&snapshot).await?;

            // g. Submit new orders (with margin check for buys only)
            for signal in signals {
                if signal.action != Action::Hold {
                    // Bug 3 fix: margin check only applies to Buy orders.
                    // Sells reduce exposure and should never be margin-blocked.
                    if signal.action == Action::Buy {
                        if let Some(max_margin) = config.margin_available {
                            // Bug 4 fix: use the correct symbol's price, not the first bar's price
                            let price = finest_bar_group
                                .iter()
                                .find(|b| b.symbol == signal.symbol)
                                .map(|b| b.close)
                                .unwrap_or(finest_bar_group[0].close);
                            let trade_value = signal.quantity as f64 * price;
                            let current_exposure = portfolio.equity() - portfolio.cash();
                            if current_exposure + trade_value > max_margin {
                                current_rejections.push(OrderRejection {
                                    symbol: signal.symbol.clone(),
                                    side: Side::Buy,
                                    quantity: signal.quantity,
                                    reason: "INSUFFICIENT_MARGIN".into(),
                                });
                                continue;
                            }
                        }
                    }
                    matcher.submit(Order::from_signal(&signal));
                }
            }

            // Save for next iteration
            last_fills = current_fills;
            last_rejections = current_rejections;
        }

        // 9. Complete
        let custom_metrics = strategy.on_complete().await?;

        Ok(BacktestResult {
            trades: portfolio.closed_trades().to_vec(),
            equity_curve: portfolio.equity_curve().to_vec(),
            final_equity: portfolio.equity(),
            initial_capital: config.initial_capital,
            config,
            custom_metrics,
        })
    }
}

/// Group bars by timestamp. Bars must be pre-sorted by timestamp_ms.
/// Collects consecutive bars with the same timestamp into groups.
fn group_bars_by_timestamp(bars: Vec<Bar>) -> Vec<(i64, Vec<Bar>)> {
    let mut grouped: Vec<(i64, Vec<Bar>)> = Vec::new();
    for bar in bars {
        if let Some(last) = grouped.last_mut() {
            if last.0 == bar.timestamp_ms {
                last.1.push(bar);
                continue;
            }
        }
        grouped.push((bar.timestamp_ms, vec![bar]));
    }
    grouped
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Interval;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockStrategyClient {
        call_count: AtomicUsize,
    }

    impl MockStrategyClient {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl StrategyClient for MockStrategyClient {
        async fn get_requirements(&self, _name: &str, _config: &str) -> Result<Vec<IntervalRequirement>> {
            Ok(vec![IntervalRequirement {
                interval: "minute".into(),
                lookback: 200,
            }])
        }

        async fn initialize(
            &self,
            _name: &str,
            _config: &str,
            _symbols: &[String],
            _instruments: &[InstrumentData],
        ) -> Result<()> {
            Ok(())
        }

        async fn on_bar(&self, snapshot: &MarketSnapshot) -> Result<Vec<Signal>> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
            // Get the first symbol from the first available timeframe
            let symbol = snapshot
                .timeframes
                .values()
                .next()
                .and_then(|m| m.keys().next().cloned())
                .unwrap_or_default();
            if count == 5 {
                Ok(vec![Signal::market_buy(&symbol, 10)])
            } else if count == 10 {
                Ok(vec![Signal::market_sell(&symbol, 10)])
            } else {
                Ok(vec![])
            }
        }

        async fn on_complete(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }

    #[tokio::test]
    async fn test_engine_runs_backtest() {
        let bars: Vec<Bar> = (0..15)
            .map(|i| Bar {
                timestamp_ms: i * 60000,
                symbol: "TEST".into(),
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 99.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000,
                oi: 0,
            })
            .collect();

        let client = MockStrategyClient::new();
        let config = BacktestConfig {
            strategy_name: "mock".into(),
            symbols: vec!["TEST".into()],
            start_date: "2024-01-01".into(),
            end_date: "2024-01-02".into(),
            initial_capital: 1_000_000.0,
            interval: Interval::Minute,
            strategy_params: serde_json::json!({}),
            slippage_pct: 0.0,
            margin_available: None,
            lookback_window: 200,
        };

        let mut bars_by_interval = HashMap::new();
        bars_by_interval.insert("minute".to_string(), bars);
        let requirements = vec![IntervalRequirement {
            interval: "minute".into(),
            lookback: 200,
        }];

        let result =
            BacktestEngine::run(config, bars_by_interval, &client, vec![], &requirements)
                .await
                .unwrap();
        assert_eq!(result.trades.len(), 1); // one round-trip trade (buy bar 5, sell bar 10)
        assert!(result.final_equity > 0.0);
        assert!(!result.equity_curve.is_empty());
        assert_eq!(result.initial_capital, 1_000_000.0);
    }

    #[tokio::test]
    async fn test_engine_no_trades_when_hold() {
        // Strategy that never trades
        struct HoldStrategy;

        #[async_trait]
        impl StrategyClient for HoldStrategy {
            async fn get_requirements(&self, _: &str, _: &str) -> Result<Vec<IntervalRequirement>> {
                Ok(vec![IntervalRequirement {
                    interval: "day".into(),
                    lookback: 200,
                }])
            }
            async fn initialize(
                &self,
                _: &str,
                _: &str,
                _: &[String],
                _: &[InstrumentData],
            ) -> Result<()> {
                Ok(())
            }
            async fn on_bar(&self, _: &MarketSnapshot) -> Result<Vec<Signal>> {
                Ok(vec![])
            }
            async fn on_complete(&self) -> Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
        }

        let bars: Vec<Bar> = (0..5)
            .map(|i| Bar {
                timestamp_ms: i * 60000,
                symbol: "TEST".into(),
                open: 100.0,
                high: 102.0,
                low: 99.0,
                close: 101.0,
                volume: 1000,
                oi: 0,
            })
            .collect();

        let mut bars_by_interval = HashMap::new();
        bars_by_interval.insert("day".to_string(), bars);
        let requirements = vec![IntervalRequirement {
            interval: "day".into(),
            lookback: 200,
        }];

        let result = BacktestEngine::run(
            BacktestConfig {
                strategy_name: "hold".into(),
                symbols: vec!["TEST".into()],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-02".into(),
                initial_capital: 1_000_000.0,
                interval: Interval::Day,
                strategy_params: serde_json::json!({}),
                slippage_pct: 0.0,
                margin_available: None,
                lookback_window: 200,
            },
            bars_by_interval,
            &HoldStrategy,
            vec![],
            &requirements,
        )
        .await
        .unwrap();

        assert!(result.trades.is_empty());
        assert_eq!(result.final_equity, 1_000_000.0);
    }

    #[tokio::test]
    async fn test_engine_margin_limit() {
        // Strategy that tries to buy 1000 shares at ~100 each = 100,000 value
        // But margin_available is only 50,000
        struct BigBuyStrategy;

        #[async_trait]
        impl StrategyClient for BigBuyStrategy {
            async fn get_requirements(&self, _: &str, _: &str) -> Result<Vec<IntervalRequirement>> {
                Ok(vec![IntervalRequirement {
                    interval: "minute".into(),
                    lookback: 200,
                }])
            }
            async fn initialize(
                &self,
                _: &str,
                _: &str,
                _: &[String],
                _: &[InstrumentData],
            ) -> Result<()> {
                Ok(())
            }
            async fn on_bar(&self, snapshot: &MarketSnapshot) -> Result<Vec<Signal>> {
                let symbol = snapshot
                    .timeframes
                    .values()
                    .next()
                    .and_then(|m| m.keys().next().cloned())
                    .unwrap_or_default();
                Ok(vec![Signal::market_buy(&symbol, 1000)])
            }
            async fn on_complete(&self) -> Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
        }

        let bars: Vec<Bar> = (0..5)
            .map(|i| Bar {
                timestamp_ms: i * 60000,
                symbol: "TEST".into(),
                open: 100.0,
                high: 102.0,
                low: 99.0,
                close: 101.0,
                volume: 1000,
                oi: 0,
            })
            .collect();

        let mut bars_by_interval = HashMap::new();
        bars_by_interval.insert("minute".to_string(), bars);
        let requirements = vec![IntervalRequirement {
            interval: "minute".into(),
            lookback: 200,
        }];

        let result = BacktestEngine::run(
            BacktestConfig {
                strategy_name: "big_buy".into(),
                symbols: vec!["TEST".into()],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-02".into(),
                initial_capital: 1_000_000.0,
                interval: Interval::Minute,
                strategy_params: serde_json::json!({}),
                slippage_pct: 0.0,
                margin_available: Some(50_000.0),
                lookback_window: 200,
            },
            bars_by_interval,
            &BigBuyStrategy,
            vec![],
            &requirements,
        )
        .await
        .unwrap();

        // With 50K margin, the 100K trade should be blocked
        assert!(result.trades.is_empty());
    }

    #[tokio::test]
    async fn test_margin_check_does_not_block_sells() {
        // Strategy that buys on bar 2, then sells on bar 5.
        // With tight margin, the sell should NOT be blocked even though margin is full.
        struct BuySellStrategy {
            call_count: AtomicUsize,
        }

        impl BuySellStrategy {
            fn new() -> Self {
                Self {
                    call_count: AtomicUsize::new(0),
                }
            }
        }

        #[async_trait]
        impl StrategyClient for BuySellStrategy {
            async fn get_requirements(&self, _: &str, _: &str) -> Result<Vec<IntervalRequirement>> {
                Ok(vec![IntervalRequirement {
                    interval: "minute".into(),
                    lookback: 200,
                }])
            }
            async fn initialize(
                &self,
                _: &str,
                _: &str,
                _: &[String],
                _: &[InstrumentData],
            ) -> Result<()> {
                Ok(())
            }
            async fn on_bar(&self, snapshot: &MarketSnapshot) -> Result<Vec<Signal>> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
                let symbol = snapshot
                    .timeframes
                    .values()
                    .next()
                    .and_then(|m| m.keys().next().cloned())
                    .unwrap_or_default();
                if count == 2 {
                    // Buy 100 shares — this will use up margin
                    Ok(vec![Signal::market_buy(&symbol, 100)])
                } else if count == 5 {
                    // Sell 100 shares — should NOT be blocked by margin
                    Ok(vec![Signal::market_sell(&symbol, 100)])
                } else {
                    Ok(vec![])
                }
            }
            async fn on_complete(&self) -> Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
        }

        let bars: Vec<Bar> = (0..10)
            .map(|i| Bar {
                timestamp_ms: i * 60000,
                symbol: "TEST".into(),
                open: 100.0,
                high: 102.0,
                low: 99.0,
                close: 101.0,
                volume: 1000,
                oi: 0,
            })
            .collect();

        let mut bars_by_interval = HashMap::new();
        bars_by_interval.insert("minute".to_string(), bars);
        let requirements = vec![IntervalRequirement {
            interval: "minute".into(),
            lookback: 200,
        }];

        let result = BacktestEngine::run(
            BacktestConfig {
                strategy_name: "buy_sell".into(),
                symbols: vec!["TEST".into()],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-02".into(),
                initial_capital: 1_000_000.0,
                interval: Interval::Minute,
                strategy_params: serde_json::json!({}),
                slippage_pct: 0.0,
                // Set margin to allow the buy (100 * 101 = 10100) but would block another buy
                margin_available: Some(11_000.0),
                lookback_window: 200,
            },
            bars_by_interval,
            &BuySellStrategy::new(),
            vec![],
            &requirements,
        )
        .await
        .unwrap();

        // The sell should have gone through (not margin-blocked), producing a closed trade
        assert_eq!(
            result.trades.len(),
            1,
            "sell should not be blocked by margin check"
        );
    }

    #[tokio::test]
    async fn test_margin_check_uses_correct_symbol_price() {
        // Strategy that tries to buy EXPENSIVE on bar 2.
        // We have two symbols: CHEAP (close=10) and EXPENSIVE (close=500).
        // Margin is 1000. If the engine incorrectly uses the first bar's price (CHEAP=10),
        // it would allow the buy. Correct behavior uses EXPENSIVE=500 and blocks it.
        struct MultiSymbolBuyStrategy {
            call_count: AtomicUsize,
        }

        impl MultiSymbolBuyStrategy {
            fn new() -> Self {
                Self {
                    call_count: AtomicUsize::new(0),
                }
            }
        }

        #[async_trait]
        impl StrategyClient for MultiSymbolBuyStrategy {
            async fn get_requirements(&self, _: &str, _: &str) -> Result<Vec<IntervalRequirement>> {
                Ok(vec![IntervalRequirement {
                    interval: "minute".into(),
                    lookback: 200,
                }])
            }
            async fn initialize(
                &self,
                _: &str,
                _: &str,
                _: &[String],
                _: &[InstrumentData],
            ) -> Result<()> {
                Ok(())
            }
            async fn on_bar(&self, _snapshot: &MarketSnapshot) -> Result<Vec<Signal>> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count == 2 {
                    // Try to buy 10 shares of EXPENSIVE at ~500 = 5000 > margin 1000
                    Ok(vec![Signal::market_buy("EXPENSIVE", 10)])
                } else {
                    Ok(vec![])
                }
            }
            async fn on_complete(&self) -> Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
        }

        // Create bars for both symbols at same timestamps
        let mut bars: Vec<Bar> = Vec::new();
        for i in 0..5i64 {
            bars.push(Bar {
                timestamp_ms: i * 60000,
                symbol: "CHEAP".into(),
                open: 10.0,
                high: 12.0,
                low: 9.0,
                close: 10.0,
                volume: 1000,
                oi: 0,
            });
            bars.push(Bar {
                timestamp_ms: i * 60000,
                symbol: "EXPENSIVE".into(),
                open: 500.0,
                high: 520.0,
                low: 490.0,
                close: 500.0,
                volume: 1000,
                oi: 0,
            });
        }

        let mut bars_by_interval = HashMap::new();
        bars_by_interval.insert("minute".to_string(), bars);
        let requirements = vec![IntervalRequirement {
            interval: "minute".into(),
            lookback: 200,
        }];

        let result = BacktestEngine::run(
            BacktestConfig {
                strategy_name: "multi_sym".into(),
                symbols: vec!["CHEAP".into(), "EXPENSIVE".into()],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-02".into(),
                initial_capital: 1_000_000.0,
                interval: Interval::Minute,
                strategy_params: serde_json::json!({}),
                slippage_pct: 0.0,
                // Margin 1000: 10 * 500 = 5000 should be blocked
                margin_available: Some(1_000.0),
                lookback_window: 200,
            },
            bars_by_interval,
            &MultiSymbolBuyStrategy::new(),
            vec![],
            &requirements,
        )
        .await
        .unwrap();

        // The buy for EXPENSIVE should have been margin-rejected
        assert!(
            result.trades.is_empty(),
            "buy for EXPENSIVE should be blocked — margin uses correct symbol price"
        );
    }
}
