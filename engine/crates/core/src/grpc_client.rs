use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use backtest_proto::backtest::{
    strategy_service_client::StrategyServiceClient, BarEvent, CompleteRequest, InitRequest,
    PortfolioState, PositionInfo,
};

use crate::engine::StrategyClient;
use crate::types::{Action, Bar, OrderType, Portfolio, Signal};

// ── GrpcStrategyClient ─────────────────────────────────────────────────────

/// A `StrategyClient` implementation that delegates to a remote Python gRPC
/// strategy server via `StrategyServiceClient`.
///
/// We wrap the tonic client in a `tokio::sync::Mutex` because the generated
/// `StrategyServiceClient` requires `&mut self` for RPC calls while our
/// `StrategyClient` trait uses `&self`. Cloning the inner tonic client is
/// cheap (it shares the underlying HTTP/2 connection), so we clone inside
/// the lock to keep the critical section short.
pub struct GrpcStrategyClient {
    client: Mutex<StrategyServiceClient<Channel>>,
}

impl GrpcStrategyClient {
    /// Connect to a running Python strategy gRPC server at the given address.
    ///
    /// # Arguments
    /// * `addr` - The server address, e.g. `"http://127.0.0.1:50051"`.
    pub async fn connect(addr: &str) -> Result<Self> {
        let client = StrategyServiceClient::connect(addr.to_string()).await?;
        Ok(Self {
            client: Mutex::new(client),
        })
    }
}

#[async_trait]
impl StrategyClient for GrpcStrategyClient {
    async fn initialize(&self, name: &str, config: &str, symbols: &[String]) -> Result<()> {
        let mut client = self.client.lock().await.clone();

        let resp = client
            .initialize(InitRequest {
                strategy_name: name.into(),
                config_json: config.into(),
                symbols: symbols.to_vec(),
            })
            .await?
            .into_inner();

        if !resp.success {
            anyhow::bail!("Strategy initialization failed: {}", resp.error);
        }
        Ok(())
    }

    async fn on_bar(&self, bar: &Bar, portfolio: &Portfolio) -> Result<Vec<Signal>> {
        let mut client = self.client.lock().await.clone();

        // Convert domain Portfolio -> proto PortfolioState
        let proto_portfolio = PortfolioState {
            cash: portfolio.cash,
            equity: portfolio.equity(),
            positions: portfolio
                .positions
                .iter()
                .map(|p| PositionInfo {
                    symbol: p.symbol.clone(),
                    quantity: p.quantity,
                    avg_price: p.avg_price,
                    unrealized_pnl: p.unrealized_pnl(),
                })
                .collect(),
        };

        // Convert domain Bar -> proto BarEvent
        let bar_event = BarEvent {
            timestamp_ms: bar.timestamp_ms,
            symbol: bar.symbol.clone(),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            oi: bar.oi,
            portfolio: Some(proto_portfolio),
        };

        let resp = client.on_bar(bar_event).await?.into_inner();

        // Convert proto Signal -> domain Signal
        let signals = resp
            .signals
            .into_iter()
            .map(|s| {
                let action = match s.action() {
                    backtest_proto::backtest::signal::Action::Hold => Action::Hold,
                    backtest_proto::backtest::signal::Action::Buy => Action::Buy,
                    backtest_proto::backtest::signal::Action::Sell => Action::Sell,
                };
                let order_type = match s.order_type() {
                    backtest_proto::backtest::signal::OrderType::Market => OrderType::Market,
                    backtest_proto::backtest::signal::OrderType::Limit => OrderType::Limit,
                    backtest_proto::backtest::signal::OrderType::Sl => OrderType::Sl,
                    backtest_proto::backtest::signal::OrderType::SlM => OrderType::SlM,
                };
                Signal {
                    action,
                    symbol: s.symbol,
                    quantity: s.quantity,
                    order_type,
                    limit_price: s.limit_price,
                    stop_price: s.stop_price,
                }
            })
            .collect();

        Ok(signals)
    }

    async fn on_complete(&self) -> Result<serde_json::Value> {
        let mut client = self.client.lock().await.clone();

        let resp = client
            .on_complete(CompleteRequest {})
            .await?
            .into_inner();

        let metrics: serde_json::Value =
            serde_json::from_str(&resp.custom_metrics_json).unwrap_or(serde_json::json!({}));
        Ok(metrics)
    }
}
