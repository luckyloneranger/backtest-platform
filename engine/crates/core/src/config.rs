use serde::{Deserialize, Serialize};

use crate::types::Interval;

/// Top-level configuration for a single backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub strategy_name: String,
    pub symbols: Vec<String>,
    pub start_date: String,
    pub end_date: String,
    pub initial_capital: f64,
    pub interval: Interval,
    pub strategy_params: serde_json::Value,
    pub slippage_pct: f64,
    /// If set, reject trades whose notional value would push total exposure
    /// above this margin limit.
    #[serde(default)]
    pub margin_available: Option<f64>,
    /// Number of historical bars to keep per symbol for strategy lookback.
    #[serde(default = "default_lookback")]
    pub lookback_window: usize,
    /// Maximum fraction of bar volume that can be filled per order (0.0–1.0).
    /// Default 1.0 means no constraint.
    #[serde(default = "default_max_volume_pct")]
    pub max_volume_pct: f64,
    /// Maximum portfolio drawdown (fraction, e.g. 0.05 = 5%) before kill switch triggers.
    /// When breached, all positions are force-closed and trading stops.
    #[serde(default)]
    pub max_drawdown_pct: Option<f64>,
    /// Maximum loss (absolute rupees) allowed per trading day.
    /// When breached, MIS positions are force-closed and new buys are rejected.
    #[serde(default)]
    pub daily_loss_limit: Option<f64>,
    /// Maximum position quantity allowed per symbol.
    /// Orders are clamped or rejected to stay within this limit.
    #[serde(default)]
    pub max_position_qty: Option<i32>,
    /// Maximum portfolio exposure as a fraction of initial capital (e.g. 0.8 = 80%).
    /// Buy orders that would push exposure above this are rejected.
    #[serde(default)]
    pub max_exposure_pct: Option<f64>,
    /// Non-tradable reference symbols (e.g. NIFTY 50 index). Their data appears
    /// in MarketSnapshot but signals for them are silently dropped.
    #[serde(default)]
    pub reference_symbols: Vec<String>,
}

fn default_lookback() -> usize {
    200
}

fn default_max_volume_pct() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backtest_config_creation() {
        let config = BacktestConfig {
            strategy_name: "sma_crossover".into(),
            symbols: vec!["RELIANCE".into(), "INFY".into()],
            start_date: "2024-01-01".into(),
            end_date: "2024-12-31".into(),
            initial_capital: 1_000_000.0,
            interval: Interval::Day,
            strategy_params: serde_json::json!({"fast": 10, "slow": 30}),
            slippage_pct: 0.05,
            margin_available: None,
            lookback_window: 200,
            max_volume_pct: 1.0,
            max_drawdown_pct: None,
            daily_loss_limit: None,
            max_position_qty: None,
            max_exposure_pct: None,
            reference_symbols: vec![],
        };

        assert_eq!(config.strategy_name, "sma_crossover");
        assert_eq!(config.symbols.len(), 2);
        assert!((config.initial_capital - 1_000_000.0).abs() < f64::EPSILON);
        assert_eq!(config.interval, Interval::Day);
        assert!((config.slippage_pct - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn backtest_config_serde_roundtrip() {
        let config = BacktestConfig {
            strategy_name: "momentum".into(),
            symbols: vec!["TCS".into()],
            start_date: "2024-06-01".into(),
            end_date: "2024-06-30".into(),
            initial_capital: 500_000.0,
            interval: Interval::Minute,
            strategy_params: serde_json::json!({}),
            slippage_pct: 0.0,
            margin_available: None,
            lookback_window: 200,
            max_volume_pct: 1.0,
            max_drawdown_pct: None,
            daily_loss_limit: None,
            max_position_qty: None,
            max_exposure_pct: None,
            reference_symbols: vec![],
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let deser: BacktestConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deser.strategy_name, "momentum");
        assert_eq!(deser.symbols, vec!["TCS"]);
        assert_eq!(deser.interval, Interval::Minute);
    }
}
