use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use backtest_proto::backtest::{
    strategy_service_client::StrategyServiceClient, BarData, BarEvent, CompleteRequest,
    FillInfo, InitRequest, InstrumentInfo, OrderRejection as ProtoOrderRejection,
    PendingOrderInfo as ProtoPendingOrderInfo,
    PortfolioState, PositionInfo, RequirementsRequest,
    SessionContext as ProtoSessionContext, TimeframeData, TimeframeHistory,
    TradeInfo,
};

use crate::engine::{InstrumentData, IntervalRequirement, MarketSnapshot, StrategyClient};
use crate::matching::Side;
use crate::types::{Action, OrderType, OrderValidity, ProductType, Signal};

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

/// Helper to convert a domain `Bar` into a proto `BarData`.
fn bar_to_proto(b: &crate::types::Bar) -> BarData {
    BarData {
        symbol: b.symbol.clone(),
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
        volume: b.volume,
        oi: b.oi,
        timestamp_ms: b.timestamp_ms,
    }
}

#[async_trait]
impl StrategyClient for GrpcStrategyClient {
    async fn get_requirements(
        &self,
        name: &str,
        config: &str,
    ) -> Result<Vec<IntervalRequirement>> {
        let mut client = self.client.lock().await.clone();
        let resp = client
            .get_requirements(RequirementsRequest {
                strategy_name: name.into(),
                config_json: config.into(),
            })
            .await?
            .into_inner();

        Ok(resp
            .intervals
            .into_iter()
            .map(|r| IntervalRequirement {
                interval: r.interval,
                lookback: r.lookback as usize,
            })
            .collect())
    }

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

        // Convert timeframes -> proto TimeframeData
        let proto_timeframes: Vec<TimeframeData> = snapshot
            .timeframes
            .iter()
            .map(|(interval, symbol_bars)| TimeframeData {
                interval: interval.clone(),
                bars: symbol_bars.values().map(|b| bar_to_proto(b)).collect(),
            })
            .collect();

        // Convert history -> proto TimeframeHistory
        let proto_history: Vec<TimeframeHistory> = snapshot
            .history
            .iter()
            .map(|((symbol, interval), bars)| TimeframeHistory {
                symbol: symbol.clone(),
                interval: interval.clone(),
                bars: bars.iter().map(|b| bar_to_proto(b)).collect(),
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
                costs: f.costs,
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

        // Convert context -> proto SessionContext (intervals instead of interval)
        let proto_context = ProtoSessionContext {
            initial_capital: snapshot.context.initial_capital,
            bar_number: snapshot.context.bar_number,
            total_bars: snapshot.context.total_bars,
            start_date: snapshot.context.start_date.clone(),
            end_date: snapshot.context.end_date.clone(),
            intervals: snapshot.context.intervals.clone(),
            lookback_window: snapshot.context.lookback_window,
        };

        // Convert pending orders -> proto PendingOrderInfo
        let proto_pending_orders: Vec<ProtoPendingOrderInfo> = snapshot
            .pending_orders
            .iter()
            .map(|po| ProtoPendingOrderInfo {
                symbol: po.symbol.clone(),
                side: match po.side {
                    Side::Buy => "BUY".into(),
                    Side::Sell => "SELL".into(),
                },
                quantity: po.quantity,
                order_type: match po.order_type {
                    OrderType::Market => "MARKET".into(),
                    OrderType::Limit => "LIMIT".into(),
                    OrderType::Sl => "SL".into(),
                    OrderType::SlM => "SL_M".into(),
                },
                limit_price: po.limit_price,
                stop_price: po.stop_price,
                order_id: po.order_id,
            })
            .collect();

        // Build enriched BarEvent with multi-timeframe data
        let bar_event = BarEvent {
            timestamp_ms: snapshot.timestamp_ms,
            timeframes: proto_timeframes,
            history: proto_history,
            portfolio: Some(proto_portfolio),
            instruments: proto_instruments,
            fills: proto_fills,
            rejections: proto_rejections,
            closed_trades: proto_trades,
            context: Some(proto_context),
            pending_orders: proto_pending_orders,
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
                    backtest_proto::backtest::signal::Action::Cancel => Action::Cancel,
                };
                let order_type = match s.order_type() {
                    backtest_proto::backtest::signal::OrderType::Market => OrderType::Market,
                    backtest_proto::backtest::signal::OrderType::Limit => OrderType::Limit,
                    backtest_proto::backtest::signal::OrderType::Sl => OrderType::Sl,
                    backtest_proto::backtest::signal::OrderType::SlM => OrderType::SlM,
                };
                let product_type = match s.product_type() {
                    backtest_proto::backtest::signal::ProductType::Cnc => ProductType::Cnc,
                    backtest_proto::backtest::signal::ProductType::Mis => ProductType::Mis,
                    backtest_proto::backtest::signal::ProductType::Nrml => ProductType::Nrml,
                };
                let validity = match s.validity() {
                    backtest_proto::backtest::signal::OrderValidity::Day => OrderValidity::Day,
                    backtest_proto::backtest::signal::OrderValidity::Ioc => OrderValidity::Ioc,
                };
                Signal {
                    action,
                    symbol: s.symbol,
                    quantity: s.quantity,
                    order_type,
                    limit_price: s.limit_price,
                    stop_price: s.stop_price,
                    product_type,
                    trigger_price: s.trigger_price,
                    validity,
                    cancel_order_id: s.cancel_order_id,
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
