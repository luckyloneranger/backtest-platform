use anyhow::Result;
use async_trait::async_trait;

use crate::config::BacktestConfig;
use crate::costs::{TradeParams, ZerodhaCostModel};
use crate::matching::{Order, OrderMatcher, Side};
use crate::portfolio::{ClosedTrade, EquityPoint, PortfolioManager};
use crate::types::{Action, Bar, InstrumentType, Portfolio, Signal};

// ── StrategyClient trait ────────────────────────────────────────────────────

#[async_trait]
pub trait StrategyClient: Send + Sync {
    async fn initialize(&self, name: &str, config: &str, symbols: &[String]) -> Result<()>;
    async fn on_bar(&self, bar: &Bar, portfolio: &Portfolio) -> Result<Vec<Signal>>;
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
    ) -> Result<BacktestResult> {
        // 1. Initialize strategy
        strategy
            .initialize(
                &config.strategy_name,
                &serde_json::to_string(&config.strategy_params)?,
                &config.symbols,
            )
            .await?;

        // 2. Set up components
        let cost_model = ZerodhaCostModel;
        let mut portfolio = PortfolioManager::new(config.initial_capital);
        let mut matcher = OrderMatcher::new(config.slippage_pct);

        // 3. Process each bar
        for bar in &bars {
            // a. Process pending orders against this bar
            let fills = matcher.process_bar(bar);
            for fill in &fills {
                // Calculate costs for each fill
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

            // b. Update portfolio with current prices
            let mut prices = std::collections::HashMap::new();
            prices.insert(bar.symbol.clone(), bar.close);
            portfolio.update_prices(&prices, bar.timestamp_ms);

            // c. Get strategy signals for this bar
            let signals = strategy.on_bar(bar, &portfolio.portfolio_state()).await?;

            // d. Convert signals to orders and submit to matcher
            for signal in signals {
                if signal.action != Action::Hold {
                    matcher.submit(Order::from_signal(&signal));
                }
            }
        }

        // 4. Complete
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
        ) -> Result<()> {
            Ok(())
        }

        async fn on_bar(&self, bar: &Bar, _portfolio: &Portfolio) -> Result<Vec<Signal>> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 5 {
                Ok(vec![Signal::market_buy(&bar.symbol, 10)])
            } else if count == 10 {
                Ok(vec![Signal::market_sell(&bar.symbol, 10)])
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
        };

        let result = BacktestEngine::run(config, bars, &client).await.unwrap();
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
            ) -> Result<()> {
                Ok(())
            }
            async fn on_bar(&self, _: &Bar, _: &Portfolio) -> Result<Vec<Signal>> {
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
            },
            bars,
            &HoldStrategy,
        )
        .await
        .unwrap();

        assert!(result.trades.is_empty());
        assert_eq!(result.final_equity, 1_000_000.0);
    }
}
