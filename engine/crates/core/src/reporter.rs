use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde::{Deserialize, Serialize};

use crate::engine::BacktestResult;
use crate::metrics::MetricsReport;
use crate::portfolio::{ClosedTrade, EquityPoint};

// ── BacktestSummary ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSummary {
    pub id: String,
    pub strategy_name: String,
    pub start_date: String,
    pub end_date: String,
    pub final_equity: f64,
}

// ── ConfigSnapshot (for config.json) ───────────────────────────────────────

/// Subset of backtest information persisted in config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigSnapshot {
    strategy_name: String,
    symbols: Vec<String>,
    start_date: String,
    end_date: String,
    initial_capital: f64,
    interval: String,
    strategy_params: serde_json::Value,
    slippage_pct: f64,
    final_equity: f64,
}

// ── Reporter ───────────────────────────────────────────────────────────────

pub struct Reporter {
    base_path: PathBuf,
}

impl Reporter {
    pub fn new(base_path: &Path) -> Self {
        Self {
            base_path: base_path.to_path_buf(),
        }
    }

    /// Save backtest results to disk. Returns the generated backtest ID.
    ///
    /// Creates a directory `{base_path}/{id}/` with the following files:
    /// - `config.json`        — backtest configuration + final equity
    /// - `trades.parquet`     — closed trade records
    /// - `equity_curve.parquet` — equity curve time series
    /// - `metrics.json`       — computed performance metrics
    pub fn save(&self, result: &BacktestResult) -> Result<String> {
        // 1. Generate a short ID (first 12 hex chars of UUID v4, dashes stripped)
        let id = uuid::Uuid::new_v4().to_string().replace('-', "");
        let id = id[..12].to_string();

        // 2. Create the output directory
        let dir = self.base_path.join(&id);
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create output directory: {}", dir.display()))?;

        // 3. Write config.json
        let config_snapshot = ConfigSnapshot {
            strategy_name: result.config.strategy_name.clone(),
            symbols: result.config.symbols.clone(),
            start_date: result.config.start_date.clone(),
            end_date: result.config.end_date.clone(),
            initial_capital: result.config.initial_capital,
            interval: format!("{:?}", result.config.interval),
            strategy_params: result.config.strategy_params.clone(),
            slippage_pct: result.config.slippage_pct,
            final_equity: result.final_equity,
        };
        let config_json = serde_json::to_string_pretty(&config_snapshot)
            .context("failed to serialize config snapshot")?;
        fs::write(dir.join("config.json"), config_json)
            .context("failed to write config.json")?;

        // 4. Write trades.parquet
        Self::write_trades_parquet(&dir.join("trades.parquet"), &result.trades)
            .context("failed to write trades.parquet")?;

        // 5. Write equity_curve.parquet
        Self::write_equity_curve_parquet(
            &dir.join("equity_curve.parquet"),
            &result.equity_curve,
        )
        .context("failed to write equity_curve.parquet")?;

        // 6. Write metrics.json
        let metrics = MetricsReport::compute(result);
        let metrics_json = serde_json::to_string_pretty(&metrics)
            .context("failed to serialize metrics report")?;
        fs::write(dir.join("metrics.json"), metrics_json)
            .context("failed to write metrics.json")?;

        Ok(id)
    }

    /// List all saved backtest IDs with basic info, sorted by ID.
    pub fn list(&self) -> Result<Vec<BacktestSummary>> {
        let mut summaries = Vec::new();

        if !self.base_path.exists() {
            return Ok(summaries);
        }

        let entries = fs::read_dir(&self.base_path)
            .with_context(|| format!("failed to read directory: {}", self.base_path.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let config_path = path.join("config.json");
            if !config_path.exists() {
                continue;
            }

            let id = entry
                .file_name()
                .to_string_lossy()
                .to_string();

            let config_data = fs::read_to_string(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?;

            let snapshot: ConfigSnapshot = serde_json::from_str(&config_data)
                .with_context(|| format!("failed to parse {}", config_path.display()))?;

            summaries.push(BacktestSummary {
                id,
                strategy_name: snapshot.strategy_name,
                start_date: snapshot.start_date,
                end_date: snapshot.end_date,
                final_equity: snapshot.final_equity,
            });
        }

        summaries.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(summaries)
    }

    /// Load the metrics report for a specific backtest run.
    pub fn load_metrics(&self, id: &str) -> Result<MetricsReport> {
        let metrics_path = self.base_path.join(id).join("metrics.json");
        let data = fs::read_to_string(&metrics_path)
            .with_context(|| format!("failed to read {}", metrics_path.display()))?;
        let report: MetricsReport = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse {}", metrics_path.display()))?;
        Ok(report)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn write_trades_parquet(path: &Path, trades: &[ClosedTrade]) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("symbol", DataType::Utf8, false),
            Field::new("side", DataType::Utf8, false),
            Field::new("quantity", DataType::Int32, false),
            Field::new("entry_price", DataType::Float64, false),
            Field::new("exit_price", DataType::Float64, false),
            Field::new("entry_timestamp_ms", DataType::Int64, false),
            Field::new("exit_timestamp_ms", DataType::Int64, false),
            Field::new("pnl", DataType::Float64, false),
            Field::new("costs", DataType::Float64, false),
            Field::new("direction", DataType::Utf8, false),
        ]));

        let symbols: Vec<&str> = trades.iter().map(|t| t.symbol.as_str()).collect();
        let sides: Vec<&str> = trades.iter().map(|t| t.side.as_str()).collect();
        let quantities: Vec<i32> = trades.iter().map(|t| t.quantity).collect();
        let entry_prices: Vec<f64> = trades.iter().map(|t| t.entry_price).collect();
        let exit_prices: Vec<f64> = trades.iter().map(|t| t.exit_price).collect();
        let entry_timestamps: Vec<i64> = trades.iter().map(|t| t.entry_timestamp_ms).collect();
        let exit_timestamps: Vec<i64> = trades.iter().map(|t| t.exit_timestamp_ms).collect();
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let costs: Vec<f64> = trades.iter().map(|t| t.costs).collect();
        let directions: Vec<&str> = trades
            .iter()
            .map(|t| match t.direction {
                crate::types::Direction::Long => "LONG",
                crate::types::Direction::Short => "SHORT",
            })
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(symbols)),
                Arc::new(StringArray::from(sides)),
                Arc::new(Int32Array::from(quantities)),
                Arc::new(Float64Array::from(entry_prices)),
                Arc::new(Float64Array::from(exit_prices)),
                Arc::new(Int64Array::from(entry_timestamps)),
                Arc::new(Int64Array::from(exit_timestamps)),
                Arc::new(Float64Array::from(pnls)),
                Arc::new(Float64Array::from(costs)),
                Arc::new(StringArray::from(directions)),
            ],
        )
        .context("failed to create record batch for trades")?;

        let file = fs::File::create(path)
            .with_context(|| format!("failed to create file: {}", path.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema, None)
            .context("failed to create parquet writer for trades")?;
        writer
            .write(&batch)
            .context("failed to write trades batch")?;
        writer.close().context("failed to close trades writer")?;

        Ok(())
    }

    fn write_equity_curve_parquet(path: &Path, equity_curve: &[EquityPoint]) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("equity", DataType::Float64, false),
        ]));

        let timestamps: Vec<i64> = equity_curve.iter().map(|p| p.timestamp_ms).collect();
        let equities: Vec<f64> = equity_curve.iter().map(|p| p.equity).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)),
                Arc::new(Float64Array::from(equities)),
            ],
        )
        .context("failed to create record batch for equity curve")?;

        let file = fs::File::create(path)
            .with_context(|| format!("failed to create file: {}", path.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema, None)
            .context("failed to create parquet writer for equity curve")?;
        writer
            .write(&batch)
            .context("failed to write equity curve batch")?;
        writer
            .close()
            .context("failed to close equity curve writer")?;

        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BacktestConfig;
    use crate::types::Interval;

    /// Build a minimal BacktestResult with some mock trades and equity curve.
    fn mock_backtest_result(strategy_name: &str) -> BacktestResult {
        let trades = vec![
            ClosedTrade {
                symbol: "RELIANCE".into(),
                side: "BUY".into(),
                quantity: 10,
                entry_price: 2500.0,
                exit_price: 2600.0,
                entry_timestamp_ms: 1_000,
                exit_timestamp_ms: 2_000,
                pnl: 1000.0,
                costs: 50.0,
                direction: crate::types::Direction::Long,
            },
            ClosedTrade {
                symbol: "INFY".into(),
                side: "BUY".into(),
                quantity: 20,
                entry_price: 1500.0,
                exit_price: 1450.0,
                entry_timestamp_ms: 3_000,
                exit_timestamp_ms: 4_000,
                pnl: -1000.0,
                costs: 60.0,
                direction: crate::types::Direction::Long,
            },
        ];

        let equity_curve = vec![
            EquityPoint {
                timestamp_ms: 0,
                equity: 1_000_000.0,
            },
            EquityPoint {
                timestamp_ms: 60_000,
                equity: 1_001_000.0,
            },
            EquityPoint {
                timestamp_ms: 86_400_000,
                equity: 1_000_500.0,
            },
            EquityPoint {
                timestamp_ms: 86_460_000,
                equity: 1_000_000.0,
            },
        ];

        BacktestResult {
            trades,
            equity_curve,
            final_equity: 1_000_000.0,
            initial_capital: 1_000_000.0,
            config: BacktestConfig {
                strategy_name: strategy_name.into(),
                symbols: vec!["RELIANCE".into(), "INFY".into()],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-31".into(),
                initial_capital: 1_000_000.0,
                interval: Interval::Minute,
                strategy_params: serde_json::json!({"fast": 10, "slow": 30}),
                slippage_pct: 0.05,
                margin_available: None,
                lookback_window: 200,
            },
            custom_metrics: serde_json::json!({"custom_key": 42}),
            benchmark_return_pct: None,
        }
    }

    #[test]
    fn test_save_and_load_results() {
        let tmp = tempfile::TempDir::new().unwrap();
        let reporter = Reporter::new(tmp.path());

        let result = mock_backtest_result("sma_crossover");
        let id = reporter.save(&result).unwrap();

        // ID should be 12 characters
        assert_eq!(id.len(), 12);

        // Verify all 4 files exist
        let dir = tmp.path().join(&id);
        assert!(dir.join("config.json").exists(), "config.json must exist");
        assert!(
            dir.join("trades.parquet").exists(),
            "trades.parquet must exist"
        );
        assert!(
            dir.join("equity_curve.parquet").exists(),
            "equity_curve.parquet must exist"
        );
        assert!(dir.join("metrics.json").exists(), "metrics.json must exist");

        // Verify config.json contents
        let config_data = fs::read_to_string(dir.join("config.json")).unwrap();
        let snapshot: serde_json::Value = serde_json::from_str(&config_data).unwrap();
        assert_eq!(snapshot["strategy_name"], "sma_crossover");
        assert_eq!(snapshot["start_date"], "2024-01-01");
        assert_eq!(snapshot["end_date"], "2024-01-31");
        assert_eq!(snapshot["final_equity"], 1_000_000.0);

        // Load metrics back and verify fields
        let metrics = reporter.load_metrics(&id).unwrap();
        assert_eq!(metrics.initial_capital, 1_000_000.0);
        assert_eq!(metrics.final_equity, 1_000_000.0);
        assert_eq!(metrics.trade_stats.total_trades, 2);
        assert_eq!(metrics.trade_stats.winning_trades, 1);
        assert_eq!(metrics.trade_stats.losing_trades, 1);
        assert!((metrics.trade_stats.win_rate - 0.5).abs() < 1e-6);
        assert!((metrics.total_return_pct - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_list_results() {
        let tmp = tempfile::TempDir::new().unwrap();
        let reporter = Reporter::new(tmp.path());

        // Save two results with different strategy names
        let r1 = mock_backtest_result("strategy_alpha");
        let r2 = mock_backtest_result("strategy_beta");

        let id1 = reporter.save(&r1).unwrap();
        let id2 = reporter.save(&r2).unwrap();

        // Listing should return 2 entries
        let summaries = reporter.list().unwrap();
        assert_eq!(summaries.len(), 2);

        // Find each entry by ID (list is sorted by ID)
        let s1 = summaries.iter().find(|s| s.id == id1).unwrap();
        assert_eq!(s1.strategy_name, "strategy_alpha");
        assert_eq!(s1.start_date, "2024-01-01");
        assert_eq!(s1.end_date, "2024-01-31");
        assert_eq!(s1.final_equity, 1_000_000.0);

        let s2 = summaries.iter().find(|s| s.id == id2).unwrap();
        assert_eq!(s2.strategy_name, "strategy_beta");
    }

    #[test]
    fn test_list_empty_directory() {
        let tmp = tempfile::TempDir::new().unwrap();
        let reporter = Reporter::new(tmp.path());
        let summaries = reporter.list().unwrap();
        assert!(summaries.is_empty());
    }

    #[test]
    fn test_list_nonexistent_directory() {
        let reporter = Reporter::new(Path::new("/tmp/nonexistent_backtest_dir_xyz_12345"));
        let summaries = reporter.list().unwrap();
        assert!(summaries.is_empty());
    }

    #[test]
    fn test_load_metrics_nonexistent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let reporter = Reporter::new(tmp.path());
        let result = reporter.load_metrics("nonexistent_id");
        assert!(result.is_err());
    }

    #[test]
    fn test_save_empty_trades_and_equity() {
        let tmp = tempfile::TempDir::new().unwrap();
        let reporter = Reporter::new(tmp.path());

        let result = BacktestResult {
            trades: vec![],
            equity_curve: vec![],
            final_equity: 1_000_000.0,
            initial_capital: 1_000_000.0,
            config: BacktestConfig {
                strategy_name: "empty".into(),
                symbols: vec![],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-01".into(),
                initial_capital: 1_000_000.0,
                interval: Interval::Day,
                strategy_params: serde_json::json!({}),
                slippage_pct: 0.0,
                margin_available: None,
                lookback_window: 200,
            },
            custom_metrics: serde_json::json!({}),
            benchmark_return_pct: None,
        };

        let id = reporter.save(&result).unwrap();
        let dir = tmp.path().join(&id);
        assert!(dir.join("trades.parquet").exists());
        assert!(dir.join("equity_curve.parquet").exists());

        let metrics = reporter.load_metrics(&id).unwrap();
        assert_eq!(metrics.trade_stats.total_trades, 0);
    }
}
