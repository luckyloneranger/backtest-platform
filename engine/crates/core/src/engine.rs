use std::collections::{HashMap, VecDeque};

use anyhow::Result;
use async_trait::async_trait;

use crate::config::BacktestConfig;
use crate::costs::{TradeParams, ZerodhaCostModel};
use crate::matching::{Fill, Order, OrderMatcher, OrderRejection, Side};
use crate::portfolio::{ClosedTrade, EquityPoint, PortfolioManager};
use crate::types::{Action, Bar, InstrumentType, Portfolio, Signal};

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

// ── SessionContext ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionContext {
    pub initial_capital: f64,
    pub bar_number: i32,
    pub total_bars: i32,
    pub start_date: String,
    pub end_date: String,
    pub interval: String,
    pub lookback_window: i32,
}

// ── MarketSnapshot ─────────────────────────────────────────────────────────

/// Full market snapshot sent to strategy per timestamp.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub timestamp_ms: i64,
    pub bars: HashMap<String, Bar>,
    pub history: HashMap<String, Vec<Bar>>,
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

// ── BacktestEngine ──────────────────────────────────────────────────────────

pub struct BacktestEngine;

impl BacktestEngine {
    pub async fn run(
        config: BacktestConfig,
        bars: Vec<Bar>,
        strategy: &dyn StrategyClient,
        instruments: Vec<InstrumentData>,
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

        // 3. Lookback buffers per symbol
        let mut lookback: HashMap<String, VecDeque<Bar>> = HashMap::new();

        // Track fills and rejections for next on_bar call
        let mut last_fills: Vec<Fill> = Vec::new();
        let mut last_rejections: Vec<OrderRejection> = Vec::new();

        // 4. Group bars by timestamp
        let grouped = group_bars_by_timestamp(bars);
        let total_bars = grouped.len() as i32;

        // 5. Process each timestamp group
        for (bar_idx, (timestamp, bar_group)) in grouped.iter().enumerate() {
            // a. Process pending orders for each bar in this group
            let mut current_fills = Vec::new();
            let mut current_rejections = Vec::new();
            for bar in bar_group {
                let (fills, rejects) = matcher.process_bar(bar);
                for fill in &fills {
                    let trade_value = fill.quantity as f64 * fill.fill_price;
                    let params = TradeParams {
                        instrument_type: InstrumentType::Equity, // default for now
                        is_intraday: true,                       // default for now
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

            // b. Update portfolio with current prices
            let mut prices = HashMap::new();
            for bar in bar_group {
                prices.insert(bar.symbol.clone(), bar.close);
            }
            portfolio.update_prices(&prices, *timestamp);

            // c. Update lookback buffers
            for bar in bar_group {
                let buf = lookback
                    .entry(bar.symbol.clone())
                    .or_insert_with(VecDeque::new);
                buf.push_back(bar.clone());
                if buf.len() > config.lookback_window {
                    buf.pop_front();
                }
            }

            // d. Build snapshot
            let bars_map: HashMap<String, Bar> = bar_group
                .iter()
                .map(|b| (b.symbol.clone(), b.clone()))
                .collect();
            let history: HashMap<String, Vec<Bar>> = lookback
                .iter()
                .map(|(sym, buf)| (sym.clone(), buf.iter().cloned().collect()))
                .collect();

            let snapshot = MarketSnapshot {
                timestamp_ms: *timestamp,
                bars: bars_map,
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
                    interval: config.interval.as_kite_str().to_string(),
                    lookback_window: config.lookback_window as i32,
                },
            };

            // e. Get strategy signals
            let signals = strategy.on_bar(&snapshot).await?;

            // f. Submit new orders (with margin check)
            for signal in signals {
                if signal.action != Action::Hold {
                    if let Some(max_margin) = config.margin_available {
                        let trade_value = signal.quantity as f64 * bar_group[0].close;
                        let current_exposure = portfolio.equity() - portfolio.cash();
                        if current_exposure + trade_value > max_margin {
                            current_rejections.push(OrderRejection {
                                symbol: signal.symbol.clone(),
                                side: if signal.action == Action::Buy {
                                    Side::Buy
                                } else {
                                    Side::Sell
                                },
                                quantity: signal.quantity,
                                reason: "INSUFFICIENT_MARGIN".into(),
                            });
                            continue;
                        }
                    }
                    matcher.submit(Order::from_signal(&signal));
                }
            }

            // Save for next iteration
            last_fills = current_fills;
            last_rejections = current_rejections;
        }

        // 6. Complete
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
            // Get the first symbol from the current bars
            let symbol = snapshot
                .bars
                .keys()
                .next()
                .cloned()
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

        let result = BacktestEngine::run(config, bars, &client, vec![])
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
            bars,
            &HoldStrategy,
            vec![],
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
                    .bars
                    .keys()
                    .next()
                    .cloned()
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
            bars,
            &BigBuyStrategy,
            vec![],
        )
        .await
        .unwrap();

        // With 50K margin, the 100K trade should be blocked
        assert!(result.trades.is_empty());
    }
}
