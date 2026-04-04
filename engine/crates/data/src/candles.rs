use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use backtest_core::types::{Bar, Interval};

/// Reads and writes candle (Bar) data as Parquet files.
///
/// File layout: `{base_path}/{exchange}/{symbol}/{interval_str}/data.parquet`
pub struct CandleStore {
    base_path: PathBuf,
}

impl CandleStore {
    /// Create a new CandleStore rooted at `base_path`.
    pub fn new(base_path: &Path) -> Self {
        Self {
            base_path: base_path.to_path_buf(),
        }
    }

    /// Build the directory path for a given exchange/symbol/interval.
    fn dir_path(&self, exchange: &str, symbol: &str, interval: Interval) -> PathBuf {
        self.base_path
            .join(exchange)
            .join(symbol)
            .join(interval.as_kite_str())
    }

    /// Build the full file path for the Parquet data file.
    fn file_path(&self, exchange: &str, symbol: &str, interval: Interval) -> PathBuf {
        self.dir_path(exchange, symbol, interval).join("data.parquet")
    }

    /// Returns the Arrow schema used for candle Parquet files.
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("volume", DataType::Int64, false),
            Field::new("oi", DataType::Int64, false),
        ]))
    }

    /// Write bars to a Parquet file, merging with any existing data.
    /// New bars override existing bars on timestamp collision.
    /// Creates directories as needed.
    pub fn write(
        &self,
        exchange: &str,
        symbol: &str,
        interval: Interval,
        bars: &[Bar],
    ) -> Result<()> {
        // Read existing bars (empty vec if file doesn't exist)
        let existing = self.read(exchange, symbol, interval, None, None).unwrap_or_default();

        // Merge: existing first, then new bars (new override on collision)
        let mut merged: BTreeMap<i64, Bar> = BTreeMap::new();
        for bar in existing {
            merged.insert(bar.timestamp_ms, bar);
        }
        for bar in bars {
            merged.insert(bar.timestamp_ms, bar.clone());
        }

        let sorted_bars: Vec<Bar> = merged.into_values().collect();
        self.write_internal(exchange, symbol, interval, &sorted_bars)
    }

    /// Internal write that directly writes bars to a Parquet file, truncating any existing file.
    fn write_internal(
        &self,
        exchange: &str,
        symbol: &str,
        interval: Interval,
        bars: &[Bar],
    ) -> Result<()> {
        let dir = self.dir_path(exchange, symbol, interval);
        fs::create_dir_all(&dir)?;

        let file_path = self.file_path(exchange, symbol, interval);
        let file = fs::File::create(&file_path)?;

        let schema = Self::schema();

        let timestamp_ms: Int64Array = bars.iter().map(|b| b.timestamp_ms).collect();
        let symbols: StringArray = bars.iter().map(|b| Some(b.symbol.as_str())).collect();
        let open: Float64Array = bars.iter().map(|b| b.open).collect();
        let high: Float64Array = bars.iter().map(|b| b.high).collect();
        let low: Float64Array = bars.iter().map(|b| b.low).collect();
        let close: Float64Array = bars.iter().map(|b| b.close).collect();
        let volume: Int64Array = bars.iter().map(|b| b.volume).collect();
        let oi: Int64Array = bars.iter().map(|b| b.oi).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(timestamp_ms),
                Arc::new(symbols),
                Arc::new(open),
                Arc::new(high),
                Arc::new(low),
                Arc::new(close),
                Arc::new(volume),
                Arc::new(oi),
            ],
        )?;

        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Read bars from a Parquet file, optionally filtering by timestamp range.
    ///
    /// - `from_ms`: inclusive lower bound on `timestamp_ms` (if `Some`)
    /// - `to_ms`: exclusive upper bound on `timestamp_ms` (if `Some`)
    ///
    /// Returns an empty `Vec` if the file does not exist.
    pub fn read(
        &self,
        exchange: &str,
        symbol: &str,
        interval: Interval,
        from_ms: Option<i64>,
        to_ms: Option<i64>,
    ) -> Result<Vec<Bar>> {
        let file_path = self.file_path(exchange, symbol, interval);

        if !file_path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&file_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        let mut bars = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;

            let ts_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("timestamp_ms column should be Int64Array");
            let sym_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("symbol column should be StringArray");
            let open_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("open column should be Float64Array");
            let high_array = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("high column should be Float64Array");
            let low_array = batch
                .column(4)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("low column should be Float64Array");
            let close_array = batch
                .column(5)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("close column should be Float64Array");
            let vol_array = batch
                .column(6)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("volume column should be Int64Array");
            let oi_array = batch
                .column(7)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("oi column should be Int64Array");

            for i in 0..batch.num_rows() {
                let ts = ts_array.value(i);

                // Apply timestamp filtering: from_ms inclusive, to_ms exclusive
                if let Some(from) = from_ms {
                    if ts < from {
                        continue;
                    }
                }
                if let Some(to) = to_ms {
                    if ts >= to {
                        continue;
                    }
                }

                bars.push(Bar {
                    timestamp_ms: ts,
                    symbol: sym_array.value(i).to_string(),
                    open: open_array.value(i),
                    high: high_array.value(i),
                    low: low_array.value(i),
                    close: close_array.value(i),
                    volume: vol_array.value(i),
                    oi: oi_array.value(i),
                });
            }
        }

        Ok(bars)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use backtest_core::types::{Bar, Interval};
    use tempfile::TempDir;

    /// Helper: create a Bar with the given values.
    fn make_bar(timestamp_ms: i64, symbol: &str, open: f64, close: f64, volume: i64) -> Bar {
        Bar {
            timestamp_ms,
            symbol: symbol.to_string(),
            open,
            high: open + 10.0,
            low: open - 10.0,
            close,
            volume,
            oi: 0,
        }
    }

    #[test]
    fn test_write_and_read_candles() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let store = CandleStore::new(tmp.path());

        let bars = vec![
            make_bar(1_000, "RELIANCE", 2450.0, 2465.0, 100_000),
            make_bar(2_000, "RELIANCE", 2465.0, 2480.0, 120_000),
        ];

        store
            .write("NSE", "RELIANCE", Interval::Minute, &bars)
            .expect("write failed");

        let read_bars = store
            .read("NSE", "RELIANCE", Interval::Minute, None, None)
            .expect("read failed");

        assert_eq!(read_bars.len(), 2);

        // Verify first bar
        assert_eq!(read_bars[0].timestamp_ms, 1_000);
        assert_eq!(read_bars[0].symbol, "RELIANCE");
        assert!((read_bars[0].open - 2450.0).abs() < f64::EPSILON);
        assert!((read_bars[0].high - 2460.0).abs() < f64::EPSILON);
        assert!((read_bars[0].low - 2440.0).abs() < f64::EPSILON);
        assert!((read_bars[0].close - 2465.0).abs() < f64::EPSILON);
        assert_eq!(read_bars[0].volume, 100_000);
        assert_eq!(read_bars[0].oi, 0);

        // Verify second bar
        assert_eq!(read_bars[1].timestamp_ms, 2_000);
        assert_eq!(read_bars[1].symbol, "RELIANCE");
        assert!((read_bars[1].open - 2465.0).abs() < f64::EPSILON);
        assert!((read_bars[1].close - 2480.0).abs() < f64::EPSILON);
        assert_eq!(read_bars[1].volume, 120_000);
    }

    #[test]
    fn test_read_with_time_filter() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let store = CandleStore::new(tmp.path());

        let bars = vec![
            make_bar(1_000, "INFY", 1400.0, 1410.0, 50_000),
            make_bar(2_000, "INFY", 1410.0, 1420.0, 60_000),
            make_bar(3_000, "INFY", 1420.0, 1430.0, 70_000),
        ];

        store
            .write("NSE", "INFY", Interval::Day, &bars)
            .expect("write failed");

        // from_ms=1500 (inclusive), to_ms=2500 (exclusive) → only bar at 2000
        let filtered = store
            .read("NSE", "INFY", Interval::Day, Some(1500), Some(2500))
            .expect("read failed");

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].timestamp_ms, 2_000);
        assert_eq!(filtered[0].symbol, "INFY");
        assert!((filtered[0].open - 1410.0).abs() < f64::EPSILON);
        assert!((filtered[0].close - 1420.0).abs() < f64::EPSILON);
        assert_eq!(filtered[0].volume, 60_000);
    }

    #[test]
    fn test_read_nonexistent_file_returns_empty() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let store = CandleStore::new(tmp.path());

        let bars = store
            .read("NSE", "DOESNOTEXIST", Interval::Minute, None, None)
            .expect("read should succeed even if file doesn't exist");

        assert!(bars.is_empty());
    }

    #[test]
    fn test_write_merges_not_overwrites() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let store = CandleStore::new(tmp.path());

        // First write: bars at ts=1000 and ts=2000
        let batch1 = vec![
            make_bar(1_000, "RELIANCE", 2450.0, 2465.0, 100_000),
            make_bar(2_000, "RELIANCE", 2465.0, 2480.0, 120_000),
        ];
        store
            .write("NSE", "RELIANCE", Interval::Day, &batch1)
            .expect("first write failed");

        // Second write: bars at ts=3000 and ts=4000
        let batch2 = vec![
            make_bar(3_000, "RELIANCE", 2480.0, 2495.0, 130_000),
            make_bar(4_000, "RELIANCE", 2495.0, 2510.0, 140_000),
        ];
        store
            .write("NSE", "RELIANCE", Interval::Day, &batch2)
            .expect("second write failed");

        // Read all — should have all 4 bars
        let all_bars = store
            .read("NSE", "RELIANCE", Interval::Day, None, None)
            .expect("read failed");

        assert_eq!(all_bars.len(), 4);
        assert_eq!(all_bars[0].timestamp_ms, 1_000);
        assert_eq!(all_bars[1].timestamp_ms, 2_000);
        assert_eq!(all_bars[2].timestamp_ms, 3_000);
        assert_eq!(all_bars[3].timestamp_ms, 4_000);

        // Verify data integrity of merged bars
        assert!((all_bars[0].open - 2450.0).abs() < f64::EPSILON);
        assert!((all_bars[2].open - 2480.0).abs() < f64::EPSILON);
        assert_eq!(all_bars[3].volume, 140_000);
    }

    #[test]
    fn test_write_merge_overrides_on_collision() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let store = CandleStore::new(tmp.path());

        // First write: bar at ts=1000
        let batch1 = vec![make_bar(1_000, "INFY", 1400.0, 1410.0, 50_000)];
        store
            .write("NSE", "INFY", Interval::Minute, &batch1)
            .expect("first write failed");

        // Second write: bar at same ts=1000 with different data
        let batch2 = vec![make_bar(1_000, "INFY", 1500.0, 1520.0, 80_000)];
        store
            .write("NSE", "INFY", Interval::Minute, &batch2)
            .expect("second write failed");

        // Read — should have 1 bar with the new data
        let bars = store
            .read("NSE", "INFY", Interval::Minute, None, None)
            .expect("read failed");

        assert_eq!(bars.len(), 1);
        assert!((bars[0].open - 1500.0).abs() < f64::EPSILON);
        assert!((bars[0].close - 1520.0).abs() < f64::EPSILON);
        assert_eq!(bars[0].volume, 80_000);
    }
}
