use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use backtest_proto::backtest::{
    strategy_service_client::StrategyServiceClient, BarData, BarEvent, CompleteRequest,
    FillInfo, InitRequest, InstrumentInfo, OrderRejection as ProtoOrderRejection,
    PortfolioState, PositionInfo, SessionContext as ProtoSessionContext,
    SymbolHistory, TradeInfo,
};

use crate::engine::{InstrumentData, MarketSnapshot, StrategyClient};
use crate::matching::Side;
use crate::types::{Action, OrderType, Signal};

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
    async fn initialize(
        &self,
        name: &str,
        config: &str,
        symbols: &[String],
        instruments: &[InstrumentData],
    ) -> Result<()> {
        let mut client = self.client.lock().await.clone();

        let proto_instruments: Vec<InstrumentInfo> = instruments
            .iter()
            .map(|inst| InstrumentInfo {
                symbol: inst.symbol.clone(),
                exchange: inst.exchange.clone(),
                instrument_type: inst.instrument_type.clone(),
                lot_size: inst.lot_size,
                tick_size: inst.tick_size,
                expiry: inst.expiry.clone(),
                strike: inst.strike,
                option_type: inst.option_type.clone(),
                circuit_limit_upper: inst.circuit_limit_upper,
                circuit_limit_lower: inst.circuit_limit_lower,
            })
            .collect();

        let resp = client
            .initialize(InitRequest {
                strategy_name: name.into(),
                config_json: config.into(),
                symbols: symbols.to_vec(),
                instruments: proto_instruments,
            })
            .await?
            .into_inner();

        if !resp.success {
            anyhow::bail!("Strategy initialization failed: {}", resp.error);
        }
        Ok(())
    }

    async fn on_bar(&self, snapshot: &MarketSnapshot) -> Result<Vec<Signal>> {
        let mut client = self.client.lock().await.clone();

        // Convert domain Portfolio -> proto PortfolioState
        let proto_portfolio = PortfolioState {
            cash: snapshot.portfolio.cash,
            equity: snapshot.portfolio.equity(),
            positions: snapshot
                .portfolio
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

        // Convert domain bars -> proto BarData
        let proto_bars: Vec<BarData> = snapshot
            .bars
            .values()
            .map(|b| BarData {
                symbol: b.symbol.clone(),
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
                volume: b.volume,
                oi: b.oi,
            })
            .collect();

        // Convert history -> proto SymbolHistory
        let proto_history: Vec<SymbolHistory> = snapshot
            .history
            .iter()
            .map(|(sym, bars)| SymbolHistory {
                symbol: sym.clone(),
                bars: bars
                    .iter()
                    .map(|b| BarData {
                        symbol: b.symbol.clone(),
                        open: b.open,
                        high: b.high,
                        low: b.low,
                        close: b.close,
                        volume: b.volume,
                        oi: b.oi,
                    })
                    .collect(),
            })
            .collect();

        // Convert instruments -> proto InstrumentInfo
        let proto_instruments: Vec<InstrumentInfo> = snapshot
            .instruments
            .iter()
            .map(|inst| InstrumentInfo {
                symbol: inst.symbol.clone(),
                exchange: inst.exchange.clone(),
                instrument_type: inst.instrument_type.clone(),
                lot_size: inst.lot_size,
                tick_size: inst.tick_size,
                expiry: inst.expiry.clone(),
                strike: inst.strike,
                option_type: inst.option_type.clone(),
                circuit_limit_upper: inst.circuit_limit_upper,
                circuit_limit_lower: inst.circuit_limit_lower,
            })
            .collect();

        // Convert fills -> proto FillInfo
        let proto_fills: Vec<FillInfo> = snapshot
            .fills
            .iter()
            .map(|f| FillInfo {
                symbol: f.symbol.clone(),
                side: match f.side {
                    Side::Buy => "BUY".into(),
                    Side::Sell => "SELL".into(),
                },
                quantity: f.quantity,
                fill_price: f.fill_price,
                costs: 0.0, // costs are tracked in portfolio, not in fills
                timestamp_ms: f.timestamp_ms,
            })
            .collect();

        // Convert rejections -> proto OrderRejection
        let proto_rejections: Vec<ProtoOrderRejection> = snapshot
            .rejections
            .iter()
            .map(|r| ProtoOrderRejection {
                symbol: r.symbol.clone(),
                side: match r.side {
                    Side::Buy => "BUY".into(),
                    Side::Sell => "SELL".into(),
                },
                quantity: r.quantity,
                reason: r.reason.clone(),
            })
            .collect();

        // Convert closed trades -> proto TradeInfo
        let proto_trades: Vec<TradeInfo> = snapshot
            .closed_trades
            .iter()
            .map(|t| TradeInfo {
                symbol: t.symbol.clone(),
                quantity: t.quantity,
                entry_price: t.entry_price,
                exit_price: t.exit_price,
                entry_timestamp_ms: t.entry_timestamp_ms,
                exit_timestamp_ms: t.exit_timestamp_ms,
                pnl: t.pnl,
                costs: t.costs,
            })
            .collect();

        // Convert context -> proto SessionContext
        let proto_context = ProtoSessionContext {
            initial_capital: snapshot.context.initial_capital,
            bar_number: snapshot.context.bar_number,
            total_bars: snapshot.context.total_bars,
            start_date: snapshot.context.start_date.clone(),
            end_date: snapshot.context.end_date.clone(),
            interval: snapshot.context.interval.clone(),
            lookback_window: snapshot.context.lookback_window,
        };

        // Build enriched BarEvent
        let bar_event = BarEvent {
            timestamp_ms: snapshot.timestamp_ms,
            bars: proto_bars,
            history: proto_history,
            portfolio: Some(proto_portfolio),
            instruments: proto_instruments,
            fills: proto_fills,
            rejections: proto_rejections,
            closed_trades: proto_trades,
            context: Some(proto_context),
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
