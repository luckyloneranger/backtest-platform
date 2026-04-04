use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(name = "backtest", about = "Indian market backtesting platform")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage market data
    Data {
        #[command(subcommand)]
        action: commands::data::DataCommands,
    },
    /// Run a backtest
    Run(commands::run::RunArgs),
    /// View backtest results
    Results {
        #[command(subcommand)]
        action: commands::results::ResultsCommands,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Data { action } => commands::data::handle(action).await?,
        Commands::Run(args) => commands::run::handle(args).await?,
        Commands::Results { action } => commands::results::handle(action)?,
    }
    Ok(())
}
