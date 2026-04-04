use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;

use backtest_core::types::{Bar, Exchange, Instrument, InstrumentType, Interval};

// ── KiteClient ──────────────────────────────────────────────────────────────

/// HTTP client for the Zerodha Kite Connect API.
///
/// Supports fetching the full instrument list (CSV) and historical candle
/// data (JSON) for a given instrument token.
pub struct KiteClient {
    client: reqwest::Client,
    api_key: String,
    access_token: String,
    base_url: String,
}

impl KiteClient {
    /// Create a new `KiteClient` with the given API key and access token.
    pub fn new(api_key: String, access_token: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            access_token,
            base_url: "https://api.kite.trade".to_string(),
        }
    }

    /// Build the `Authorization` header value.
    fn auth_header(&self) -> String {
        format!("token {}:{}", self.api_key, self.access_token)
    }

    /// Fetch all instruments from the Kite instruments CSV endpoint.
    ///
    /// `GET {base_url}/instruments`
    ///
    /// Only instruments on NSE, BSE, and MCX are returned; other exchanges
    /// are silently skipped.
    pub async fn fetch_instruments(&self) -> Result<Vec<Instrument>> {
        let url = format!("{}/instruments", self.base_url);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .context("failed to send instruments request")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "instruments endpoint returned HTTP {}: {}",
                status.as_u16(),
                body
            );
        }

        let csv_text = resp
            .text()
            .await
            .context("failed to read instruments response body")?;

        parse_instruments_csv(&csv_text)
    }

    /// Fetch historical candles for a specific instrument token.
    ///
    /// `GET {base_url}/instruments/historical/{instrument_token}/{interval}?from=…&to=…&oi=1[&continuous=1]`
    ///
    /// Kite limits responses to ~2000 candles per request.
    /// `from` and `to` use the format `YYYY-MM-DD+HH:MM:SS` or `YYYY-MM-DD`.
    ///
    /// `oi=1` is always sent so that open-interest data is included for F&O
    /// instruments (equity candles simply return `0`).
    ///
    /// When `continuous` is `true`, `continuous=1` is appended so Kite returns
    /// a continuous futures series stitched across expiries.
    pub async fn fetch_candles(
        &self,
        instrument_token: &str,
        symbol: &str,
        interval: Interval,
        from: &str,
        to: &str,
        continuous: bool,
    ) -> Result<Vec<Bar>> {
        let url = format!(
            "{}/instruments/historical/{}/{}",
            self.base_url,
            instrument_token,
            interval.as_kite_str()
        );

        let mut params: Vec<(&str, &str)> = vec![("from", from), ("to", to), ("oi", "1")];
        if continuous {
            params.push(("continuous", "1"));
        }

        let resp = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .query(&params)
            .send()
            .await
            .context("failed to send candles request")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "candles endpoint returned HTTP {}: {}",
                status.as_u16(),
                body
            );
        }

        let json_text = resp
            .text()
            .await
            .context("failed to read candles response body")?;

        parse_candles_json(&json_text, symbol)
    }

    /// Maximum candles returned by a single Kite historical-data request.
    const KITE_MAX_CANDLES: usize = 2000;

    /// Fetch historical candles with automatic chunking to stay under
    /// Kite's ~2000-candle-per-request limit.
    ///
    /// Splits the date range into chunks based on the interval's `bars_per_day()`,
    /// makes sequential API calls with a rate-limit pause between them, and
    /// concatenates the results.
    pub async fn fetch_candles_chunked(
        &self,
        instrument_token: &str,
        symbol: &str,
        interval: Interval,
        from: &str,  // "YYYY-MM-DD"
        to: &str,    // "YYYY-MM-DD"
        continuous: bool,
    ) -> Result<Vec<Bar>> {
        let from_date = NaiveDate::parse_from_str(from, "%Y-%m-%d")
            .context("invalid 'from' date format, expected YYYY-MM-DD")?;
        let to_date = NaiveDate::parse_from_str(to, "%Y-%m-%d")
            .context("invalid 'to' date format, expected YYYY-MM-DD")?;

        let bars_per_day = interval.bars_per_day();
        let days_per_chunk = if bars_per_day <= 1 {
            2000 // day interval: 2000 days per chunk (~8 years)
        } else {
            (Self::KITE_MAX_CANDLES / bars_per_day).max(1)
        };

        let mut all_bars = Vec::new();
        let mut chunk_start = from_date;
        let mut chunk_num = 0u32;

        while chunk_start <= to_date {
            let chunk_end_raw =
                chunk_start + chrono::Duration::days(days_per_chunk as i64 - 1);
            let chunk_end = if chunk_end_raw > to_date {
                to_date
            } else {
                chunk_end_raw
            };

            let from_str = chunk_start.format("%Y-%m-%d").to_string();
            let to_str = chunk_end.format("%Y-%m-%d").to_string();

            // Rate limit: sleep between requests (Kite allows ~3 req/sec for historical data)
            if chunk_num > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(350)).await;
            }

            let bars = self
                .fetch_candles(instrument_token, symbol, interval, &from_str, &to_str, continuous)
                .await?;

            let count = bars.len();
            all_bars.extend(bars);

            if count > 0 {
                eprintln!("  Fetched {} candles ({} to {})", count, from_str, to_str);
            }

            chunk_start = chunk_end + chrono::Duration::days(1);
            chunk_num += 1;
        }

        Ok(all_bars)
    }
}

// ── CSV Instrument Parsing ──────────────────────────────────────────────────

/// Row shape for the Kite instruments CSV.
///
/// Columns: instrument_token, exchange_token, tradingsymbol, name,
/// last_price, expiry, strike, tick_size, lot_size, instrument_type,
/// segment, exchange
#[derive(Debug, Deserialize)]
struct CsvInstrumentRow {
    instrument_token: String,
    #[allow(dead_code)]
    exchange_token: String,
    tradingsymbol: String,
    name: String,
    #[allow(dead_code)]
    last_price: String,
    expiry: String,
    strike: String,
    tick_size: String,
    lot_size: String,
    instrument_type: String,
    segment: String,
    exchange: String,
}

/// Parse the Kite instruments CSV text into a `Vec<Instrument>`.
///
/// Rows whose exchange is not NSE, BSE, or MCX are skipped.
/// Rows whose instrument type cannot be mapped are also skipped.
pub fn parse_instruments_csv(csv_text: &str) -> Result<Vec<Instrument>> {
    let mut reader = csv::Reader::from_reader(csv_text.as_bytes());
    let mut instruments = Vec::new();

    for result in reader.deserialize::<CsvInstrumentRow>() {
        let row = result.context("failed to deserialize instruments CSV row")?;

        let exchange = match row.exchange.as_str() {
            "NSE" => Exchange::Nse,
            "BSE" => Exchange::Bse,
            "MCX" => Exchange::Mcx,
            _ => continue, // skip unsupported exchanges
        };

        let (instrument_type, option_type) = match row.instrument_type.as_str() {
            "EQ" => (InstrumentType::Equity, None),
            "FUT" => (InstrumentType::FutureFO, None),
            "CE" => (InstrumentType::OptionFO, Some("CE".to_string())),
            "PE" => (InstrumentType::OptionFO, Some("PE".to_string())),
            "COM" => (InstrumentType::Commodity, None),
            _ => continue, // skip unknown instrument types
        };

        let expiry = if row.expiry.is_empty() {
            None
        } else {
            Some(row.expiry.clone())
        };

        let strike = if row.strike.is_empty() {
            None
        } else {
            row.strike.parse::<f64>().ok()
        };

        let lot_size = row
            .lot_size
            .parse::<i32>()
            .unwrap_or(1);

        let tick_size = row
            .tick_size
            .parse::<f64>()
            .unwrap_or(0.05);

        instruments.push(Instrument {
            instrument_token: Some(row.instrument_token),
            tradingsymbol: row.tradingsymbol,
            name: if row.name.is_empty() { None } else { Some(row.name) },
            exchange,
            instrument_type,
            lot_size,
            tick_size,
            expiry,
            strike,
            option_type,
            segment: if row.segment.is_empty() { None } else { Some(row.segment) },
        });
    }

    Ok(instruments)
}

// ── JSON Candle Parsing ─────────────────────────────────────────────────────

/// Top-level response from the Kite historical-candles endpoint.
#[derive(Debug, Deserialize)]
struct KiteCandleResponse {
    #[allow(dead_code)]
    status: String,
    data: KiteCandleData,
}

/// Inner data object containing the candles array.
#[derive(Debug, Deserialize)]
struct KiteCandleData {
    candles: Vec<Vec<serde_json::Value>>,
}

/// Parse the Kite historical-candles JSON response into a `Vec<Bar>`.
///
/// Each candle is an array: `[timestamp_string, open, high, low, close, volume, oi]`.
/// The timestamp is ISO 8601 with offset, e.g. `"2024-01-02T09:15:00+0530"`.
pub fn parse_candles_json(json_text: &str, symbol: &str) -> Result<Vec<Bar>> {
    let response: KiteCandleResponse =
        serde_json::from_str(json_text).context("failed to parse candles JSON")?;

    let mut bars = Vec::with_capacity(response.data.candles.len());

    for candle in &response.data.candles {
        if candle.len() < 6 {
            continue; // skip malformed rows (need at least ts, o, h, l, c, v)
        }

        let ts_str = candle[0]
            .as_str()
            .context("candle timestamp should be a string")?;

        let timestamp_ms = parse_kite_timestamp(ts_str)?;

        let open = candle[1]
            .as_f64()
            .context("candle open should be a number")?;
        let high = candle[2]
            .as_f64()
            .context("candle high should be a number")?;
        let low = candle[3]
            .as_f64()
            .context("candle low should be a number")?;
        let close = candle[4]
            .as_f64()
            .context("candle close should be a number")?;
        let volume = candle[5]
            .as_i64()
            .context("candle volume should be an integer")?;
        // OI is optional — equity candles have 6 elements, F&O have 7
        let oi = candle.get(6).and_then(|v| v.as_i64()).unwrap_or(0);

        bars.push(Bar {
            timestamp_ms,
            symbol: symbol.to_string(),
            open,
            high,
            low,
            close,
            volume,
            oi,
        });
    }

    Ok(bars)
}

/// Parse a Kite-style ISO 8601 timestamp into milliseconds since UNIX epoch.
///
/// Kite returns timestamps like `"2024-01-02T09:15:00+0530"` (no colon in
/// the UTC offset). `chrono::DateTime::parse_from_str` handles this with the
/// `%z` format specifier.
fn parse_kite_timestamp(ts: &str) -> Result<i64> {
    let dt = chrono::DateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S%z")
        .with_context(|| format!("failed to parse timestamp: {ts}"))?;
    Ok(dt.timestamp_millis())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- CSV instrument parsing tests --

    #[test]
    fn parse_csv_equity_instruments() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
408065,1594,INFY,INFOSYS,1500.0,,,0.05,1,EQ,NSE,NSE
738561,2885,RELIANCE,RELIANCE INDUSTRIES,2450.0,,,0.05,1,EQ,NSE,NSE
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        assert_eq!(instruments.len(), 2);

        assert_eq!(instruments[0].tradingsymbol, "INFY");
        assert_eq!(instruments[0].exchange, Exchange::Nse);
        assert_eq!(instruments[0].instrument_type, InstrumentType::Equity);
        assert_eq!(instruments[0].lot_size, 1);
        assert!((instruments[0].tick_size - 0.05).abs() < f64::EPSILON);
        assert_eq!(instruments[0].expiry, None);
        assert_eq!(instruments[0].strike, None);
        assert_eq!(instruments[0].option_type, None);

        assert_eq!(instruments[1].tradingsymbol, "RELIANCE");
        assert_eq!(instruments[1].exchange, Exchange::Nse);
        assert_eq!(instruments[1].instrument_type, InstrumentType::Equity);
    }

    #[test]
    fn parse_csv_futures_instrument() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
11536386,45064,NIFTY24APRFUT,NIFTY,22500.0,2024-04-25,,0.05,50,FUT,NFO-FUT,NSE
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        assert_eq!(instruments.len(), 1);
        assert_eq!(instruments[0].tradingsymbol, "NIFTY24APRFUT");
        assert_eq!(instruments[0].instrument_type, InstrumentType::FutureFO);
        assert_eq!(instruments[0].lot_size, 50);
        assert_eq!(instruments[0].expiry, Some("2024-04-25".to_string()));
        assert_eq!(instruments[0].strike, None);
        assert_eq!(instruments[0].option_type, None);
    }

    #[test]
    fn parse_csv_options_instruments() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
11550210,45118,NIFTY24APR22500CE,NIFTY,150.0,2024-04-25,22500,0.05,50,CE,NFO-OPT,NSE
11550466,45119,NIFTY24APR22500PE,NIFTY,100.0,2024-04-25,22500,0.05,50,PE,NFO-OPT,NSE
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        assert_eq!(instruments.len(), 2);

        // CE option
        assert_eq!(instruments[0].tradingsymbol, "NIFTY24APR22500CE");
        assert_eq!(instruments[0].instrument_type, InstrumentType::OptionFO);
        assert_eq!(instruments[0].option_type, Some("CE".to_string()));
        assert_eq!(instruments[0].strike, Some(22500.0));
        assert_eq!(instruments[0].expiry, Some("2024-04-25".to_string()));
        assert_eq!(instruments[0].lot_size, 50);

        // PE option
        assert_eq!(instruments[1].tradingsymbol, "NIFTY24APR22500PE");
        assert_eq!(instruments[1].instrument_type, InstrumentType::OptionFO);
        assert_eq!(instruments[1].option_type, Some("PE".to_string()));
        assert_eq!(instruments[1].strike, Some(22500.0));
    }

    #[test]
    fn parse_csv_mcx_commodity() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
53496839,208972,GOLDM24JUN,GOLDM,72000.0,2024-06-05,,1,10,COM,MCX,MCX
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        assert_eq!(instruments.len(), 1);
        assert_eq!(instruments[0].tradingsymbol, "GOLDM24JUN");
        assert_eq!(instruments[0].exchange, Exchange::Mcx);
        assert_eq!(instruments[0].instrument_type, InstrumentType::Commodity);
        assert_eq!(instruments[0].lot_size, 10);
        assert!((instruments[0].tick_size - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_csv_skips_unsupported_exchange() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
12345,100,SOME,SOME COMPANY,100.0,,,0.05,1,EQ,CDS,CDS
408065,1594,INFY,INFOSYS,1500.0,,,0.05,1,EQ,NSE,NSE
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        // Only NSE row should survive; CDS is skipped
        assert_eq!(instruments.len(), 1);
        assert_eq!(instruments[0].tradingsymbol, "INFY");
    }

    #[test]
    fn parse_csv_skips_unknown_instrument_type() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
408065,1594,INFY,INFOSYS,1500.0,,,0.05,1,EQ,NSE,NSE
999999,9999,SOMETHING,FOO,100,,,0.05,1,UNKNOWN,NSE,NSE
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        assert_eq!(instruments.len(), 1);
        assert_eq!(instruments[0].tradingsymbol, "INFY");
    }

    #[test]
    fn parse_csv_bse_exchange() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
128029,500209,INFOSYS,INFOSYS,1500.0,,,0.05,1,EQ,BSE,BSE
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        assert_eq!(instruments.len(), 1);
        assert_eq!(instruments[0].exchange, Exchange::Bse);
    }

    #[test]
    fn parse_csv_empty_input() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");
        assert!(instruments.is_empty());
    }

    // -- JSON candle parsing tests --

    #[test]
    fn parse_json_candles_success() {
        let json = r#"{
            "status": "success",
            "data": {
                "candles": [
                    ["2024-01-02T09:15:00+0530", 2450.0, 2475.0, 2440.0, 2465.0, 1000000, 0],
                    ["2024-01-02T09:16:00+0530", 2465.0, 2480.0, 2460.0, 2478.0, 800000, 0]
                ]
            }
        }"#;

        let bars = parse_candles_json(json, "RELIANCE").expect("parse should succeed");

        assert_eq!(bars.len(), 2);

        // First bar
        assert_eq!(bars[0].symbol, "RELIANCE");
        assert!((bars[0].open - 2450.0).abs() < f64::EPSILON);
        assert!((bars[0].high - 2475.0).abs() < f64::EPSILON);
        assert!((bars[0].low - 2440.0).abs() < f64::EPSILON);
        assert!((bars[0].close - 2465.0).abs() < f64::EPSILON);
        assert_eq!(bars[0].volume, 1_000_000);
        assert_eq!(bars[0].oi, 0);

        // Verify timestamp: 2024-01-02T09:15:00+0530 = 2024-01-02T03:45:00Z
        // = 1704167100 seconds since epoch = 1704167100000 ms
        assert_eq!(bars[0].timestamp_ms, 1704167100000);

        // Second bar
        assert_eq!(bars[1].symbol, "RELIANCE");
        assert!((bars[1].open - 2465.0).abs() < f64::EPSILON);
        assert!((bars[1].close - 2478.0).abs() < f64::EPSILON);
        assert_eq!(bars[1].volume, 800_000);
    }

    #[test]
    fn parse_json_candles_with_oi() {
        let json = r#"{
            "status": "success",
            "data": {
                "candles": [
                    ["2024-01-02T09:15:00+0530", 22500.0, 22550.0, 22480.0, 22530.0, 50000, 12500000]
                ]
            }
        }"#;

        let bars = parse_candles_json(json, "NIFTY24APRFUT").expect("parse should succeed");

        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].symbol, "NIFTY24APRFUT");
        assert_eq!(bars[0].oi, 12_500_000);
    }

    #[test]
    fn parse_json_candles_empty_array() {
        let json = r#"{
            "status": "success",
            "data": {
                "candles": []
            }
        }"#;

        let bars = parse_candles_json(json, "RELIANCE").expect("parse should succeed");
        assert!(bars.is_empty());
    }

    #[test]
    fn parse_json_candles_skips_malformed_row() {
        let json = r#"{
            "status": "success",
            "data": {
                "candles": [
                    ["2024-01-02T09:15:00+0530", 2450.0, 2475.0, 2440.0, 2465.0, 1000000, 0],
                    ["2024-01-02T09:16:00+0530", 2465.0],
                    ["2024-01-02T09:17:00+0530", 2478.0, 2490.0, 2470.0, 2485.0, 900000, 0]
                ]
            }
        }"#;

        let bars = parse_candles_json(json, "RELIANCE").expect("parse should succeed");
        // The malformed row (only 2 elements) should be skipped
        assert_eq!(bars.len(), 2);
        assert!((bars[0].open - 2450.0).abs() < f64::EPSILON);
        assert!((bars[1].open - 2478.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_kite_timestamp_ist() {
        // 2024-01-02T09:15:00+0530 = 2024-01-02T03:45:00 UTC
        let ms = parse_kite_timestamp("2024-01-02T09:15:00+0530").expect("parse should succeed");
        assert_eq!(ms, 1704167100000);
    }

    #[test]
    fn parse_kite_timestamp_utc() {
        // 2024-01-02T03:45:00+0000
        let ms = parse_kite_timestamp("2024-01-02T03:45:00+0000").expect("parse should succeed");
        assert_eq!(ms, 1704167100000);
    }

    // -- Chunk size calculation tests --

    #[test]
    fn test_chunk_size_calculation() {
        use backtest_core::types::Interval;
        // For Minute: 2000 / 375 = 5 days per chunk
        assert_eq!(2000 / Interval::Minute.bars_per_day(), 5);
        // For Day: bars_per_day=1, special case -> 2000 days
        assert_eq!(Interval::Day.bars_per_day(), 1);
        // For Minute5: 2000 / 75 = 26 days per chunk
        assert_eq!(2000 / Interval::Minute5.bars_per_day(), 26);
        // For Minute15: 2000 / 25 = 80 days per chunk
        assert_eq!(2000 / Interval::Minute15.bars_per_day(), 80);
    }

    // -- Mixed / multi-exchange CSV tests --

    #[test]
    fn parse_csv_mixed_exchanges_and_types() {
        let csv = "\
instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
408065,1594,INFY,INFOSYS,1500.0,,,0.05,1,EQ,NSE,NSE
128029,500209,INFOSYS,INFOSYS,1500.0,,,0.05,1,EQ,BSE,BSE
11536386,45064,NIFTY24APRFUT,NIFTY,22500.0,2024-04-25,,0.05,50,FUT,NFO-FUT,NSE
11550210,45118,NIFTY24APR22500CE,NIFTY,150.0,2024-04-25,22500,0.05,50,CE,NFO-OPT,NSE
53496839,208972,GOLDM24JUN,GOLDM,72000.0,2024-06-05,,1,10,COM,MCX,MCX
999,999,USDINR,USD/INR,83.5,,,0.0025,1,FUT,CDS-FUT,CDS
";
        let instruments = parse_instruments_csv(csv).expect("parse should succeed");

        // CDS row should be skipped (unsupported exchange)
        assert_eq!(instruments.len(), 5);

        // Verify the instrument types are correct
        let types: Vec<InstrumentType> = instruments.iter().map(|i| i.instrument_type).collect();
        assert_eq!(
            types,
            vec![
                InstrumentType::Equity,
                InstrumentType::Equity,
                InstrumentType::FutureFO,
                InstrumentType::OptionFO,
                InstrumentType::Commodity,
            ]
        );
    }
}

// ── Feature-gated integration test ──────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "kite-integration")]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_instruments_from_kite() {
        let api_key = std::env::var("KITE_API_KEY").expect("KITE_API_KEY env var must be set");
        let access_token =
            std::env::var("KITE_ACCESS_TOKEN").expect("KITE_ACCESS_TOKEN env var must be set");

        let client = KiteClient::new(api_key, access_token);
        let instruments = client
            .fetch_instruments()
            .await
            .expect("fetch_instruments should succeed");

        // Kite typically returns thousands of instruments
        assert!(
            !instruments.is_empty(),
            "expected non-empty instrument list"
        );

        // Verify at least one NSE equity is present
        let has_nse_equity = instruments.iter().any(|i| {
            i.exchange == Exchange::Nse && i.instrument_type == InstrumentType::Equity
        });
        assert!(has_nse_equity, "expected at least one NSE equity instrument");

        println!(
            "Fetched {} instruments from Kite Connect",
            instruments.len()
        );
    }
}
