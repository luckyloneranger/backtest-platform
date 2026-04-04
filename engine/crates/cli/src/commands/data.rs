use std::path::Path;

use anyhow::{Context, Result};
use chrono::{FixedOffset, NaiveDate};
use clap::Subcommand;
use rand::Rng;

use backtest_core::calendar::TradingCalendar;
use backtest_core::types::{Bar, Exchange, Interval};
use backtest_data::candles::CandleStore;
use backtest_data::instruments::{CorporateAction, InstrumentStore};
use backtest_data::kite::KiteClient;

#[derive(Subcommand)]
pub enum DataCommands {
    /// Fetch data from Kite API
    Fetch {
        #[arg(long)]
        symbol: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, default_value = "day")]
        interval: String,
        /// Kite instrument token (auto-resolved from instruments.db if omitted)
        #[arg(long)]
        token: Option<String>,
        #[arg(long, default_value = "NSE")]
        exchange: String,
        /// Fetch continuous futures data across expiries
        #[arg(long, default_value_t = false)]
        continuous: bool,
    },
    /// List cached data
    List,
    /// Generate synthetic test data for testing without Kite API
    GenerateTestData {
        #[arg(long)]
        symbol: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, default_value = "day")]
        interval: String,
        #[arg(long, default_value = "1000.0")]
        start_price: f64,
    },
    /// Fetch and store all instrument metadata from Kite API
    FetchInstruments,
    /// Import corporate actions (splits, bonuses, dividends) from a CSV file
    ImportCorporateActions {
        /// Path to CSV file (format: symbol,exchange,date,type,ratio)
        #[arg(long)]
        file: String,
    },
}

/// Parse an interval string into the Interval enum (delegates to shared helper).
fn parse_interval(s: &str) -> Result<Interval> {
    super::parse_interval(s)
}

pub async fn handle(cmd: DataCommands) -> Result<()> {
    match cmd {
        DataCommands::Fetch {
            symbol,
            from,
            to,
            interval,
            token,
            exchange,
            continuous,
        } => handle_fetch(&symbol, &from, &to, &interval, &token, &exchange, continuous).await,
        DataCommands::List => handle_list(),
        DataCommands::GenerateTestData {
            symbol,
            from,
            to,
            interval,
            start_price,
        } => handle_generate_test_data(&symbol, &from, &to, &interval, start_price),
        DataCommands::FetchInstruments => handle_fetch_instruments().await,
        DataCommands::ImportCorporateActions { file } => handle_import_corporate_actions(&file),
    }
}

fn parse_exchange(s: &str) -> Result<Exchange> {
    match s {
        "NSE" => Ok(Exchange::Nse),
        "BSE" => Ok(Exchange::Bse),
        "MCX" => Ok(Exchange::Mcx),
        _ => anyhow::bail!("unsupported exchange '{}'. Use: NSE, BSE, MCX", s),
    }
}

async fn handle_fetch(
    symbol: &str,
    from: &str,
    to: &str,
    interval: &str,
    token: &Option<String>,
    exchange: &str,
    continuous: bool,
) -> Result<()> {
    let api_key =
        std::env::var("KITE_API_KEY").context("KITE_API_KEY environment variable not set")?;
    let access_token = std::env::var("KITE_ACCESS_TOKEN")
        .context("KITE_ACCESS_TOKEN environment variable not set")?;

    let resolved_token = match token {
        Some(t) => t.clone(),
        None => {
            let db_path = Path::new("./data/instruments.db");
            if !db_path.exists() {
                anyhow::bail!(
                    "No instrument token provided and ./data/instruments.db not found. \
                     Run 'backtest data fetch-instruments' first, or pass --token."
                );
            }
            let store = InstrumentStore::open(db_path)?;
            let exchange_enum = parse_exchange(exchange)?;
            store
                .find_token(symbol, exchange_enum)?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No instrument_token found for {}:{}. \
                         Run 'backtest data fetch-instruments' first, or pass --token.",
                        exchange,
                        symbol
                    )
                })?
        }
    };

    let interval_enum = parse_interval(interval)?;
    let client = KiteClient::new(api_key, access_token);
    let bars = client
        .fetch_candles_chunked(&resolved_token, symbol, interval_enum, from, to, continuous)
        .await?;

    let store = CandleStore::new(Path::new("./data"));
    store.write(exchange, symbol, interval_enum, &bars)?;

    println!("Fetched {} candles for {}", bars.len(), symbol);
    Ok(())
}

async fn handle_fetch_instruments() -> Result<()> {
    let api_key =
        std::env::var("KITE_API_KEY").context("KITE_API_KEY environment variable not set")?;
    let access_token = std::env::var("KITE_ACCESS_TOKEN")
        .context("KITE_ACCESS_TOKEN environment variable not set")?;

    let client = KiteClient::new(api_key, access_token);
    let instruments = client.fetch_instruments().await?;

    // Ensure the data directory exists
    std::fs::create_dir_all("./data").context("failed to create ./data directory")?;

    let store = InstrumentStore::open(Path::new("./data/instruments.db"))?;
    store.upsert(&instruments)?;

    println!(
        "Stored {} instruments in ./data/instruments.db",
        instruments.len()
    );
    Ok(())
}

fn handle_list() -> Result<()> {
    let data_path = Path::new("./data");
    if !data_path.exists() {
        println!("No cached data found. (./data directory does not exist)");
        return Ok(());
    }

    let mut found = false;

    // Walk: ./data/{exchange}/{symbol}/{interval}/data.parquet
    let exchanges = std::fs::read_dir(data_path).context("failed to read ./data directory")?;

    for exchange_entry in exchanges {
        let exchange_entry = exchange_entry?;
        if !exchange_entry.path().is_dir() {
            continue;
        }
        let exchange = exchange_entry.file_name().to_string_lossy().to_string();

        let symbols = std::fs::read_dir(exchange_entry.path())?;
        for symbol_entry in symbols {
            let symbol_entry = symbol_entry?;
            if !symbol_entry.path().is_dir() {
                continue;
            }
            let symbol = symbol_entry.file_name().to_string_lossy().to_string();

            let intervals = std::fs::read_dir(symbol_entry.path())?;
            for interval_entry in intervals {
                let interval_entry = interval_entry?;
                if !interval_entry.path().is_dir() {
                    continue;
                }
                let interval_str = interval_entry.file_name().to_string_lossy().to_string();

                let parquet_path = interval_entry.path().join("data.parquet");
                if !parquet_path.exists() {
                    continue;
                }

                // Read the bars to get the date range
                let interval_enum = match parse_interval(&interval_str) {
                    Ok(i) => i,
                    Err(_) => continue, // skip unknown interval directories
                };

                let store = CandleStore::new(data_path);
                let bars = store.read(&exchange, &symbol, interval_enum, None, None)?;

                if bars.is_empty() {
                    println!(
                        "  {}/{}/{}: (empty)",
                        exchange, symbol, interval_str
                    );
                } else {
                    let first_ts = bars.first().unwrap().timestamp_ms;
                    let last_ts = bars.last().unwrap().timestamp_ms;
                    let first_dt = chrono::DateTime::from_timestamp_millis(first_ts)
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_else(|| "?".to_string());
                    let last_dt = chrono::DateTime::from_timestamp_millis(last_ts)
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_else(|| "?".to_string());

                    println!(
                        "  {}/{}/{}: {} bars, {} to {}",
                        exchange,
                        symbol,
                        interval_str,
                        bars.len(),
                        first_dt,
                        last_dt,
                    );
                }

                found = true;
            }
        }
    }

    if !found {
        println!("No cached data found.");
    }

    Ok(())
}

fn handle_generate_test_data(
    symbol: &str,
    from: &str,
    to: &str,
    interval: &str,
    start_price: f64,
) -> Result<()> {
    let interval_enum = parse_interval(interval)?;

    let from_date =
        NaiveDate::parse_from_str(from, "%Y-%m-%d").context("invalid --from date format")?;
    let to_date =
        NaiveDate::parse_from_str(to, "%Y-%m-%d").context("invalid --to date format")?;

    if to_date <= from_date {
        anyhow::bail!("--to date must be after --from date");
    }

    let mut bars = Vec::new();
    let mut rng = rand::thread_rng();
    let mut prev_close = start_price;

    let ist = FixedOffset::east_opt(19800).unwrap(); // IST = UTC+5:30
    let calendar = TradingCalendar::nse();

    match interval_enum {
        Interval::Day => {
            let mut current_date = from_date;
            while current_date <= to_date {
                // Skip non-trading days (weekends + NSE holidays)
                if !calendar.is_trading_day(current_date) {
                    current_date += chrono::Duration::days(1);
                    continue;
                }

                let timestamp_ms = current_date
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_local_timezone(ist)
                    .unwrap()
                    .timestamp_millis();

                let bar = generate_bar(&mut rng, symbol, timestamp_ms, prev_close);
                prev_close = bar.close;
                bars.push(bar);

                current_date += chrono::Duration::days(1);
            }
        }
        _ => {
            // All intraday intervals
            let step_minutes: u32 = match interval_enum {
                Interval::Minute => 1,
                Interval::Minute3 => 3,
                Interval::Minute5 => 5,
                Interval::Minute10 => 10,
                Interval::Minute15 => 15,
                Interval::Minute30 => 30,
                Interval::Minute60 => 60,
                Interval::Day => unreachable!(),
            };

            let mut current_date = from_date;
            while current_date <= to_date {
                // Skip non-trading days (weekends + NSE holidays)
                if !calendar.is_trading_day(current_date) {
                    current_date += chrono::Duration::days(1);
                    continue;
                }

                // Indian market hours: 9:15 AM to 3:30 PM IST
                let mut hour = 9;
                let mut minute = 15;

                while hour < 15 || (hour == 15 && minute <= 30) {
                    let timestamp_ms = current_date
                        .and_hms_opt(hour, minute, 0)
                        .unwrap()
                        .and_local_timezone(ist)
                        .unwrap()
                        .timestamp_millis();

                    let bar = generate_bar(&mut rng, symbol, timestamp_ms, prev_close);
                    prev_close = bar.close;
                    bars.push(bar);

                    minute += step_minutes;
                    if minute >= 60 {
                        hour += minute / 60;
                        minute %= 60;
                    }
                }

                current_date += chrono::Duration::days(1);
            }
        }
    }

    let store = CandleStore::new(Path::new("./data"));
    store.write("NSE", symbol, interval_enum, &bars)?;

    println!(
        "Generated {} synthetic candles for {} ({} to {})",
        bars.len(),
        symbol,
        from,
        to,
    );
    Ok(())
}

/// Generate a single synthetic OHLCV bar using a random walk from `prev_close`.
fn generate_bar(rng: &mut impl Rng, symbol: &str, timestamp_ms: i64, prev_close: f64) -> Bar {
    // Open = prev_close * (1 + random(-0.02, 0.02))
    let open_change: f64 = rng.gen_range(-0.02..0.02);
    let open = prev_close * (1.0 + open_change);

    // High and low within 2% of open
    let high_offset: f64 = rng.gen_range(0.0..0.02);
    let low_offset: f64 = rng.gen_range(0.0..0.02);
    let high = open * (1.0 + high_offset);
    let low = open * (1.0 - low_offset);

    // Close = random between low and high
    let close = rng.gen_range(low..=high);

    // Volume = random 50000-200000
    let volume: i64 = rng.gen_range(50_000..=200_000);

    Bar {
        timestamp_ms,
        symbol: symbol.to_string(),
        open,
        high,
        low,
        close,
        volume,
        oi: 0,
    }
}

fn handle_import_corporate_actions(file: &str) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read file: {}", file))?;

    // Ensure the data directory exists
    std::fs::create_dir_all("./data").context("failed to create ./data directory")?;

    let store = InstrumentStore::open(Path::new("./data/instruments.db"))?;
    let mut count = 0;

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Skip header line if present
        if line_no == 0 && line.to_lowercase().starts_with("symbol") {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 5 {
            eprintln!(
                "Warning: skipping line {} (expected 5 fields, got {}): {}",
                line_no + 1,
                parts.len(),
                line
            );
            continue;
        }

        let ratio: f64 = parts[4].trim().parse().with_context(|| {
            format!("invalid ratio on line {}: {}", line_no + 1, parts[4])
        })?;

        let action = CorporateAction {
            symbol: parts[0].trim().to_string(),
            exchange: parts[1].trim().to_string(),
            date: parts[2].trim().to_string(),
            action_type: parts[3].trim().to_string(),
            ratio,
        };

        store.insert_corporate_action(&action)?;
        count += 1;
    }

    println!("Imported {} corporate actions into ./data/instruments.db", count);
    Ok(())
}
