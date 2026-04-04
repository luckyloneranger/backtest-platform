use std::path::Path;

use anyhow::Result;
use clap::Subcommand;

use backtest_core::reporter::Reporter;

#[derive(Subcommand)]
pub enum ResultsCommands {
    /// List all saved backtest results
    List,
    /// Show detailed metrics for a specific backtest run
    Show {
        /// Backtest ID (8-character hex string)
        id: String,
    },
}

pub fn handle(cmd: ResultsCommands) -> Result<()> {
    match cmd {
        ResultsCommands::List => handle_list(),
        ResultsCommands::Show { id } => handle_show(&id),
    }
}

fn handle_list() -> Result<()> {
    let reporter = Reporter::new(Path::new("./results"));
    let summaries = reporter.list()?;

    if summaries.is_empty() {
        println!("No backtest results found. Run 'backtest run' first.");
        return Ok(());
    }

    println!(
        "{:<10} {:<20} {:<12} {:<12} {:>14}",
        "ID", "Strategy", "From", "To", "Final Equity"
    );
    println!("{}", "-".repeat(70));

    for s in &summaries {
        println!(
            "{:<10} {:<20} {:<12} {:<12} {:>14.2}",
            s.id, s.strategy_name, s.start_date, s.end_date, s.final_equity,
        );
    }

    println!();
    println!("{} result(s) found.", summaries.len());

    Ok(())
}

fn handle_show(id: &str) -> Result<()> {
    let reporter = Reporter::new(Path::new("./results"));
    let metrics = reporter.load_metrics(id)?;

    println!("=== Backtest Report: {} ===", id);
    println!();
    println!("--- Capital ---");
    println!("  Initial Capital:  {:.2}", metrics.initial_capital);
    println!("  Final Equity:     {:.2}", metrics.final_equity);
    println!(
        "  Total Return:     {:.2}%",
        metrics.total_return_pct * 100.0
    );
    println!("  CAGR:             {:.2}%", metrics.cagr * 100.0);
    println!();
    println!("--- Risk Metrics ---");
    println!("  Sharpe Ratio:     {:.4}", metrics.sharpe_ratio);
    println!("  Sortino Ratio:    {:.4}", metrics.sortino_ratio);
    println!("  Calmar Ratio:     {:.4}", metrics.calmar_ratio);
    println!(
        "  Max Drawdown:     {:.2}%",
        metrics.max_drawdown_pct * 100.0
    );
    println!(
        "  Max DD Duration:  {} bars",
        metrics.max_drawdown_duration
    );
    println!("  Volatility:       {:.2}%", metrics.volatility * 100.0);
    println!();
    println!("--- Trade Statistics ---");
    println!("  Total Trades:     {}", metrics.trade_stats.total_trades);
    println!(
        "  Winning Trades:   {}",
        metrics.trade_stats.winning_trades
    );
    println!("  Losing Trades:    {}", metrics.trade_stats.losing_trades);
    println!(
        "  Win Rate:         {:.2}%",
        metrics.trade_stats.win_rate * 100.0
    );
    println!("  Avg Win:          {:.2}", metrics.trade_stats.avg_win);
    println!("  Avg Loss:         {:.2}", metrics.trade_stats.avg_loss);
    println!(
        "  Profit Factor:    {:.4}",
        metrics.trade_stats.profit_factor
    );
    println!("  Avg PnL:          {:.2}", metrics.trade_stats.avg_pnl);
    println!("  Total PnL:        {:.2}", metrics.trade_stats.total_pnl);
    println!(
        "  Total Costs:      {:.2}",
        metrics.trade_stats.total_costs
    );

    Ok(())
}
