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
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let deser: BacktestConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deser.strategy_name, "momentum");
        assert_eq!(deser.symbols, vec!["TCS"]);
        assert_eq!(deser.interval, Interval::Minute);
    }
}
