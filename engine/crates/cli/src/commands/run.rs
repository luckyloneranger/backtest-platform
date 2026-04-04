use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;

use backtest_core::config::BacktestConfig;
use backtest_core::engine::{BacktestEngine, IntervalRequirement, StrategyClient};
use backtest_core::grpc_client::GrpcStrategyClient;
use backtest_core::reporter::Reporter;
use backtest_core::types::Interval;
use backtest_data::candles::CandleStore;

#[derive(Parser)]
pub struct RunArgs {
    /// Strategy name (must match the registered strategy on the Python server)
    #[arg(long)]
    pub strategy: String,

    /// Comma-separated list of symbols to backtest
    #[arg(long, value_delimiter = ',')]
    pub symbols: Vec<String>,

    /// Start date (YYYY-MM-DD)
    #[arg(long)]
    pub from: String,

    /// End date (YYYY-MM-DD)
    #[arg(long)]
    pub to: String,

    /// Initial capital
    #[arg(long, default_value = "1000000")]
    pub capital: f64,

    /// Candle interval (day or minute)
    #[arg(long, default_value = "day")]
    pub interval: String,

    /// Strategy parameters as JSON string
    #[arg(long, default_value = "{}")]
    pub params: String,

    /// Port of the Python strategy gRPC server
    #[arg(long, default_value = "50051")]
    pub strategy_port: u16,

    /// Slippage percentage (e.g., 0.001 = 0.1%)
    #[arg(long, default_value = "0.001")]
    pub slippage: f64,

    /// Number of historical bars to keep per symbol for strategy lookback
    #[arg(long, default_value = "200")]
    pub lookback: usize,
}

/// Parse an interval string into the Interval enum (delegates to shared helper).
fn parse_interval(s: &str) -> Result<Interval> {
    super::parse_interval(s)
}

pub async fn handle(args: RunArgs) -> Result<()> {
    let interval = parse_interval(&args.interval)?;

    let strategy_params: serde_json::Value = serde_json::from_str(&args.params)
        .context("failed to parse --params as JSON")?;

    let config = BacktestConfig {
        strategy_name: args.strategy.clone(),
        symbols: args.symbols.clone(),
        start_date: args.from.clone(),
        end_date: args.to.clone(),
        initial_capital: args.capital,
        interval,
        strategy_params,
        slippage_pct: args.slippage,
        margin_available: None,
        lookback_window: args.lookback,
    };

    // Connect to Python strategy server
    let addr = format!("http://[::1]:{}", args.strategy_port);
    let strategy = GrpcStrategyClient::connect(&addr)
        .await
        .with_context(|| {
            format!(
                "failed to connect to strategy server at {}. Is the Python strategy server running?",
                addr
            )
        })?;

    // Query the strategy for its data requirements
    let config_json = serde_json::to_string(&config.strategy_params)?;
    let requirements = strategy
        .get_requirements(&args.strategy, &config_json)
        .await
        .unwrap_or_else(|_| {
            // Fallback: use the CLI interval with default lookback
            vec![IntervalRequirement {
                interval: args.interval.clone(),
                lookback: args.lookback,
            }]
        });

    // Read candles from CandleStore at ./data/ for each required interval
    let store = CandleStore::new(Path::new("./data"));
    let mut bars_by_interval: HashMap<String, Vec<backtest_core::types::Bar>> = HashMap::new();

    for req in &requirements {
        let req_interval = parse_interval(&req.interval)?;
        let mut interval_bars = Vec::new();

        for symbol in &args.symbols {
            let bars = store.read("NSE", symbol, req_interval, None, None)?;
            if bars.is_empty() {
                eprintln!(
                    "Warning: no data found for {} (interval={}). Run 'backtest data fetch' or 'backtest data generate-test-data' first.",
                    symbol,
                    req.interval,
                );
            }
            interval_bars.extend(bars);
        }

        // Sort bars by timestamp for proper event ordering
        interval_bars.sort_by_key(|b| b.timestamp_ms);
        bars_by_interval.insert(req.interval.clone(), interval_bars);
    }

    let any_data = bars_by_interval.values().any(|v| !v.is_empty());
    if !any_data {
        anyhow::bail!("no candle data found for any symbol/interval. Cannot run backtest.");
    }

    // Run backtest
    println!(
        "Running backtest: strategy={}, symbols={}, {} to {}",
        args.strategy,
        args.symbols.join(","),
        args.from,
        args.to,
    );

    let result =
        BacktestEngine::run(config, bars_by_interval, &strategy, vec![], &requirements).await?;

    // Save results
    let reporter = Reporter::new(Path::new("./results"));
    let id = reporter.save(&result)?;

    // Compute metrics for display
    let metrics = reporter.load_metrics(&id)?;

    // Print summary
    println!();
    println!("=== Backtest Complete ===");
    println!("  ID:             {}", id);
    println!("  Strategy:       {}", args.strategy);
    println!("  Period:         {} to {}", args.from, args.to);
    println!("  Initial Capital:{:.2}", result.initial_capital);
    println!("  Final Equity:   {:.2}", result.final_equity);
    println!(
        "  Return:         {:.2}%",
        metrics.total_return_pct * 100.0
    );
    println!("  Trades:         {}", metrics.trade_stats.total_trades);
    println!("  Sharpe Ratio:   {:.4}", metrics.sharpe_ratio);
    println!("  Max Drawdown:   {:.2}%", metrics.max_drawdown_pct * 100.0);
    println!("  Results saved to ./results/{}/", id);

    Ok(())
}
