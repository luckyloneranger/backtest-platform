use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::matching::{Fill, Side};
use crate::types::{Direction, Portfolio, Position, ProductType};

// ── ClosedTrade ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedTrade {
    pub symbol: String,
    pub side: String, // "BUY" then "SELL" or vice versa
    pub quantity: i32,
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_timestamp_ms: i64,
    pub exit_timestamp_ms: i64,
    pub pnl: f64,   // net P&L after costs: (exit - entry) * qty - total_costs
    pub costs: f64,  // total transaction costs for this round-trip
    pub direction: Direction,
}

impl Default for ClosedTrade {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            side: String::new(),
            quantity: 0,
            entry_price: 0.0,
            exit_price: 0.0,
            entry_timestamp_ms: 0,
            exit_timestamp_ms: 0,
            pnl: 0.0,
            costs: 0.0,
            direction: Direction::Long,
        }
    }
}

// ── EquityPoint ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp_ms: i64,
    pub equity: f64,
}

// ── InternalPosition ────────────────────────────────────────────────────────

/// Internal position tracking (richer than the types::Position which is for
/// external reporting).
#[derive(Debug, Clone)]
pub struct InternalPosition {
    pub symbol: String,
    pub quantity: i32,        // always positive; direction indicated by is_short
    pub avg_price: f64,
    pub current_price: f64,
    pub entry_timestamp_ms: i64,
    pub total_costs: f64,     // accumulated costs
    pub is_short: bool,
    pub product_type: ProductType,
}

// ── PortfolioManager ────────────────────────────────────────────────────────

pub struct PortfolioManager {
    cash: f64,
    positions: HashMap<String, InternalPosition>,
    closed_trades: Vec<ClosedTrade>,
    equity_curve: Vec<EquityPoint>,
}

impl PortfolioManager {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            positions: HashMap::new(),
            closed_trades: Vec::new(),
            equity_curve: Vec::new(),
        }
    }

    /// Apply a fill (buy or sell) and its associated costs.
    ///
    /// - BUY: deduct (qty * price + costs) from cash, add/increase position
    /// - SELL: add (qty * price - costs) to cash, reduce/close position
    /// - When a position is fully closed, record a ClosedTrade
    pub fn apply_fill(&mut self, fill: &Fill, costs: f64) {
        match fill.side {
            Side::Buy => self.apply_buy(fill, costs),
            Side::Sell => self.apply_sell(fill, costs),
        }
    }

    fn apply_buy(&mut self, fill: &Fill, costs: f64) {
        if let Some(pos) = self.positions.get_mut(&fill.symbol) {
            if pos.is_short {
                // Closing (fully or partially) a short position
                let close_qty = fill.quantity.min(pos.quantity);
                // Debit cash: we buy back shares at fill_price + costs
                self.cash -= close_qty as f64 * fill.fill_price + costs;

                if close_qty >= pos.quantity {
                    // Full close of short
                    let closed_qty = pos.quantity;
                    let gross_pnl = (pos.avg_price - fill.fill_price) * closed_qty as f64;
                    let total_costs = pos.total_costs + costs;
                    let pnl = gross_pnl - total_costs;

                    self.closed_trades.push(ClosedTrade {
                        symbol: fill.symbol.clone(),
                        side: "SELL".to_string(),
                        quantity: closed_qty,
                        entry_price: pos.avg_price,
                        exit_price: fill.fill_price,
                        entry_timestamp_ms: pos.entry_timestamp_ms,
                        exit_timestamp_ms: fill.timestamp_ms,
                        pnl,
                        costs: total_costs,
                        direction: Direction::Short,
                    });
                    self.positions.remove(&fill.symbol);
                } else {
                    // Partial close of short
                    let closed_qty = close_qty;
                    let gross_pnl = (pos.avg_price - fill.fill_price) * closed_qty as f64;
                    let entry_cost_portion =
                        pos.total_costs * (closed_qty as f64 / pos.quantity as f64);
                    let total_costs = entry_cost_portion + costs;
                    let pnl = gross_pnl - total_costs;

                    self.closed_trades.push(ClosedTrade {
                        symbol: fill.symbol.clone(),
                        side: "SELL".to_string(),
                        quantity: closed_qty,
                        entry_price: pos.avg_price,
                        exit_price: fill.fill_price,
                        entry_timestamp_ms: pos.entry_timestamp_ms,
                        exit_timestamp_ms: fill.timestamp_ms,
                        pnl,
                        costs: total_costs,
                        direction: Direction::Short,
                    });

                    pos.total_costs -= entry_cost_portion;
                    pos.quantity -= closed_qty;
                }
            } else {
                // Existing long position: average up
                self.cash -= fill.quantity as f64 * fill.fill_price + costs;
                let old_qty = pos.quantity;
                let new_qty = old_qty + fill.quantity;
                pos.avg_price = (old_qty as f64 * pos.avg_price
                    + fill.quantity as f64 * fill.fill_price)
                    / new_qty as f64;
                pos.quantity = new_qty;
                pos.total_costs += costs;
            }
        } else {
            // New long position
            self.cash -= fill.quantity as f64 * fill.fill_price + costs;
            self.positions.insert(
                fill.symbol.clone(),
                InternalPosition {
                    symbol: fill.symbol.clone(),
                    quantity: fill.quantity,
                    avg_price: fill.fill_price,
                    current_price: fill.fill_price,
                    entry_timestamp_ms: fill.timestamp_ms,
                    total_costs: costs,
                    is_short: false,
                    product_type: fill.product_type,
                },
            );
        }
    }

    fn apply_sell(&mut self, fill: &Fill, costs: f64) {
        if let Some(pos) = self.positions.get_mut(&fill.symbol) {
            if pos.is_short {
                // Adding to an existing short position: average down
                self.cash += fill.quantity as f64 * fill.fill_price - costs;
                let old_qty = pos.quantity;
                let new_qty = old_qty + fill.quantity;
                pos.avg_price = (old_qty as f64 * pos.avg_price
                    + fill.quantity as f64 * fill.fill_price)
                    / new_qty as f64;
                pos.quantity = new_qty;
                pos.total_costs += costs;
            } else {
                // Closing (fully or partially) a long position
                // Bug 5 fix: clamp effective quantity to position size to prevent overselling.
                let effective_qty = fill.quantity.min(pos.quantity);
                if effective_qty < fill.quantity {
                    eprintln!(
                        "WARNING: sell qty {} for {} clamped to position qty {}",
                        fill.quantity, fill.symbol, pos.quantity
                    );
                }

                // Credit cash only for the effective (clamped) quantity
                self.cash += effective_qty as f64 * fill.fill_price - costs;

                let should_remove = if effective_qty >= pos.quantity {
                    // Full close
                    let closed_qty = pos.quantity;
                    let gross_pnl = (fill.fill_price - pos.avg_price) * closed_qty as f64;
                    let total_costs = pos.total_costs + costs;
                    // Bug 7 fix: pnl is net of costs
                    let pnl = gross_pnl - total_costs;

                    self.closed_trades.push(ClosedTrade {
                        symbol: fill.symbol.clone(),
                        side: "BUY".to_string(),
                        quantity: closed_qty,
                        entry_price: pos.avg_price,
                        exit_price: fill.fill_price,
                        entry_timestamp_ms: pos.entry_timestamp_ms,
                        exit_timestamp_ms: fill.timestamp_ms,
                        pnl,
                        costs: total_costs,
                        direction: Direction::Long,
                    });
                    true
                } else {
                    // Partial close
                    let closed_qty = effective_qty;
                    let gross_pnl = (fill.fill_price - pos.avg_price) * closed_qty as f64;
                    // Allocate costs proportionally
                    let entry_cost_portion =
                        pos.total_costs * (closed_qty as f64 / pos.quantity as f64);
                    let total_costs = entry_cost_portion + costs;
                    // Bug 7 fix: pnl is net of costs
                    let pnl = gross_pnl - total_costs;

                    self.closed_trades.push(ClosedTrade {
                        symbol: fill.symbol.clone(),
                        side: "BUY".to_string(),
                        quantity: closed_qty,
                        entry_price: pos.avg_price,
                        exit_price: fill.fill_price,
                        entry_timestamp_ms: pos.entry_timestamp_ms,
                        exit_timestamp_ms: fill.timestamp_ms,
                        pnl,
                        costs: total_costs,
                        direction: Direction::Long,
                    });

                    pos.total_costs -= entry_cost_portion;
                    pos.quantity -= closed_qty;
                    false
                };

                if should_remove {
                    self.positions.remove(&fill.symbol);
                }
            }
        } else {
            // No position exists: create a SHORT position
            // Credit cash with short sale proceeds
            self.cash += fill.quantity as f64 * fill.fill_price - costs;
            self.positions.insert(
                fill.symbol.clone(),
                InternalPosition {
                    symbol: fill.symbol.clone(),
                    quantity: fill.quantity,
                    avg_price: fill.fill_price,
                    current_price: fill.fill_price,
                    entry_timestamp_ms: fill.timestamp_ms,
                    total_costs: costs,
                    is_short: true,
                    product_type: fill.product_type,
                },
            );
        }
    }

    /// Update current prices for all positions and record an equity point.
    pub fn update_prices(&mut self, prices: &HashMap<String, f64>, timestamp_ms: i64) {
        for (symbol, price) in prices {
            if let Some(pos) = self.positions.get_mut(symbol) {
                pos.current_price = *price;
            }
        }
        let eq = self.equity();
        self.equity_curve.push(EquityPoint {
            timestamp_ms,
            equity: eq,
        });
    }

    /// Get current cash.
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Get a position by symbol (None if no position).
    pub fn position(&self, symbol: &str) -> Option<&InternalPosition> {
        self.positions.get(symbol)
    }

    /// Get all closed trades.
    pub fn closed_trades(&self) -> &[ClosedTrade] {
        &self.closed_trades
    }

    /// Get equity curve.
    pub fn equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    /// Compute current total equity (cash + long market values - short market values).
    /// For long positions: value = qty * current_price.
    /// For short positions: value = -(qty * current_price) because we owe shares.
    /// (Short sale proceeds are already in cash.)
    pub fn equity(&self) -> f64 {
        self.cash
            + self
                .positions
                .values()
                .map(|p| {
                    if p.is_short {
                        -(p.quantity as f64 * p.current_price)
                    } else {
                        p.quantity as f64 * p.current_price
                    }
                })
                .sum::<f64>()
    }

    /// Build a Portfolio snapshot for sending to strategy via gRPC.
    /// Short positions are reported with negative quantity.
    pub fn portfolio_state(&self) -> Portfolio {
        let positions: Vec<Position> = self
            .positions
            .values()
            .map(|p| Position {
                symbol: p.symbol.clone(),
                quantity: if p.is_short { -p.quantity } else { p.quantity },
                avg_price: p.avg_price,
                current_price: p.current_price,
            })
            .collect();

        Portfolio {
            cash: self.cash,
            positions,
        }
    }

    /// Returns a snapshot of all open positions as (symbol, quantity, is_short, product_type).
    /// Used by the engine for MIS auto-squareoff.
    pub fn positions_snapshot(&self) -> Vec<(String, i32, bool, ProductType)> {
        self.positions
            .values()
            .map(|p| (p.symbol.clone(), p.quantity, p.is_short, p.product_type))
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProductType;

    fn buy_fill(symbol: &str, qty: i32, price: f64, ts: i64) -> Fill {
        Fill {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity: qty,
            fill_price: price,
            timestamp_ms: ts,
            product_type: ProductType::Cnc,
            costs: 0.0,
        }
    }

    fn sell_fill(symbol: &str, qty: i32, price: f64, ts: i64) -> Fill {
        Fill {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity: qty,
            fill_price: price,
            timestamp_ms: ts,
            product_type: ProductType::Cnc,
            costs: 0.0,
        }
    }

    #[test]
    fn test_buy_creates_position() {
        let mut pm = PortfolioManager::new(1_000_000.0);
        let fill = buy_fill("RELIANCE", 10, 2500.0, 1_000);
        pm.apply_fill(&fill, 50.0);

        // cash = 1_000_000 - (10 * 2500 + 50) = 1_000_000 - 25_050 = 974_950
        assert!((pm.cash() - 974_950.0).abs() < f64::EPSILON);

        let pos = pm.position("RELIANCE").expect("position should exist");
        assert_eq!(pos.quantity, 10);
        assert!((pos.avg_price - 2500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sell_reduces_position_and_records_trade() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy 10 at 2500
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2500.0, 1_000), 50.0);

        // Sell 10 at 2600
        pm.apply_fill(&sell_fill("RELIANCE", 10, 2600.0, 2_000), 50.0);

        // Position should be closed
        assert!(pm.position("RELIANCE").is_none());

        // Should have one closed trade
        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);

        let trade = &trades[0];
        assert_eq!(trade.symbol, "RELIANCE");
        assert_eq!(trade.quantity, 10);
        assert!((trade.entry_price - 2500.0).abs() < f64::EPSILON);
        assert!((trade.exit_price - 2600.0).abs() < f64::EPSILON);
        // pnl = (2600 - 2500) * 10 - (50 buy costs + 50 sell costs) = 1000 - 100 = 900 (net)
        assert!((trade.pnl - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_equity_curve_tracking() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy 10 at 2500
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2500.0, 1_000), 0.0);

        // Price goes up to 2600
        let mut prices = HashMap::new();
        prices.insert("RELIANCE".to_string(), 2600.0);
        pm.update_prices(&prices, 2_000);

        assert_eq!(pm.equity_curve().len(), 1);

        // equity = cash + position market value
        // cash = 1_000_000 - 25_000 = 975_000
        // position value = 10 * 2600 = 26_000
        // total = 975_000 + 26_000 = 1_001_000
        let eq = pm.equity_curve()[0].equity;
        assert!((eq - 1_001_000.0).abs() < f64::EPSILON);
        assert!(eq > 1_000_000.0); // equity increased

        // Price goes up further to 2700
        prices.insert("RELIANCE".to_string(), 2700.0);
        pm.update_prices(&prices, 3_000);

        assert_eq!(pm.equity_curve().len(), 2);
        let eq2 = pm.equity_curve()[1].equity;
        // equity = 975_000 + 10 * 2700 = 975_000 + 27_000 = 1_002_000
        assert!((eq2 - 1_002_000.0).abs() < f64::EPSILON);
        assert!(eq2 > eq); // equity kept increasing
    }

    #[test]
    fn test_portfolio_state() {
        let mut pm = PortfolioManager::new(1_000_000.0);
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2500.0, 1_000), 0.0);

        let portfolio = pm.portfolio_state();

        // cash after buy = 1_000_000 - 25_000 = 975_000
        assert!((portfolio.cash - 975_000.0).abs() < f64::EPSILON);
        assert_eq!(portfolio.positions.len(), 1);

        let pos = &portfolio.positions[0];
        assert_eq!(pos.symbol, "RELIANCE");
        assert_eq!(pos.quantity, 10);
        assert!((pos.avg_price - 2500.0).abs() < f64::EPSILON);
        assert!((pos.current_price - 2500.0).abs() < f64::EPSILON);

        // Verify equity method on Portfolio struct
        // equity = 975_000 + 10 * 2500 = 1_000_000
        assert!((portfolio.equity() - 1_000_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_partial_close() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy 10 at 2500
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2500.0, 1_000), 0.0);

        // Sell 5 at 2600 (partial close)
        pm.apply_fill(&sell_fill("RELIANCE", 5, 2600.0, 2_000), 0.0);

        // Remaining position: 5 shares
        let pos = pm.position("RELIANCE").expect("position should still exist");
        assert_eq!(pos.quantity, 5);
        assert!((pos.avg_price - 2500.0).abs() < f64::EPSILON);

        // One closed trade for 5 shares
        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 5);
        // pnl = (2600 - 2500) * 5 = 500
        assert!((trades[0].pnl - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_average_up_on_second_buy() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy 10 at 2500
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2500.0, 1_000), 0.0);

        // Buy 10 more at 2600 (average up)
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2600.0, 2_000), 0.0);

        let pos = pm.position("RELIANCE").expect("position should exist");
        assert_eq!(pos.quantity, 20);
        // avg = (10*2500 + 10*2600) / 20 = 51000 / 20 = 2550
        assert!((pos.avg_price - 2550.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_closed_trade_default() {
        let trade = ClosedTrade::default();
        assert_eq!(trade.symbol, "");
        assert_eq!(trade.quantity, 0);
        assert!((trade.pnl - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sell_without_position_creates_short() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Sell without any position — should create a short position
        pm.apply_fill(&sell_fill("RELIANCE", 10, 2600.0, 1_000), 50.0);

        // Cash should increase by sale proceeds minus costs: 10 * 2600 - 50 = 25950
        assert!((pm.cash() - 1_025_950.0).abs() < f64::EPSILON);
        // No closed trades yet
        assert!(pm.closed_trades().is_empty());
        // Short position should exist
        let pos = pm.position("RELIANCE").expect("short position should exist");
        assert_eq!(pos.quantity, 10);
        assert!(pos.is_short);
        assert!((pos.avg_price - 2600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_oversell_clamped_to_position_qty() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy 5 at 2500
        pm.apply_fill(&buy_fill("RELIANCE", 5, 2500.0, 1_000), 0.0);

        // Try to sell 10 (more than held) — should be clamped to 5
        pm.apply_fill(&sell_fill("RELIANCE", 10, 2600.0, 2_000), 0.0);

        // Position should be fully closed
        assert!(pm.position("RELIANCE").is_none());

        // Closed trade should have quantity = 5 (clamped), not 10
        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 5);

        // Cash should reflect only 5 shares sold, not 10
        // Initial: 1_000_000
        // After buy: 1_000_000 - 5*2500 = 987_500
        // After sell 5 at 2600: 987_500 + 5*2600 = 1_000_500
        assert!((pm.cash() - 1_000_500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trade_pnl_is_net_of_costs() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy 10 at 2500 with costs=100
        pm.apply_fill(&buy_fill("RELIANCE", 10, 2500.0, 1_000), 100.0);

        // Sell 10 at 2600 with costs=100
        pm.apply_fill(&sell_fill("RELIANCE", 10, 2600.0, 2_000), 100.0);

        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);

        let trade = &trades[0];
        // gross_pnl = (2600 - 2500) * 10 = 1000
        // total_costs = 100 (buy) + 100 (sell) = 200
        // net_pnl = 1000 - 200 = 800
        assert!((trade.pnl - 800.0).abs() < f64::EPSILON);
        assert!((trade.costs - 200.0).abs() < f64::EPSILON);
    }

    // ── Short selling tests ─────────────────────────────────────────────

    fn sell_fill_mis(symbol: &str, qty: i32, price: f64, ts: i64) -> Fill {
        Fill {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity: qty,
            fill_price: price,
            timestamp_ms: ts,
            product_type: ProductType::Mis,
            costs: 0.0,
        }
    }

    fn buy_fill_mis(symbol: &str, qty: i32, price: f64, ts: i64) -> Fill {
        Fill {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity: qty,
            fill_price: price,
            timestamp_ms: ts,
            product_type: ProductType::Mis,
            costs: 0.0,
        }
    }

    #[test]
    fn test_short_sell_creates_short_position() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Sell 10 shares at 100 — no existing position, should create short
        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 0.0);

        let pos = pm.position("TEST").expect("short position should exist");
        assert_eq!(pos.quantity, 10);
        assert!(pos.is_short);
        assert!((pos.avg_price - 100.0).abs() < f64::EPSILON);

        // Cash should increase by short sale proceeds: 10 * 100 = 1000
        assert!((pm.cash() - 1_001_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_buy_closes_short_position() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Short 10 at 100
        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 0.0);

        // Buy 10 at 90 (cover the short)
        pm.apply_fill(&buy_fill("TEST", 10, 90.0, 2_000), 0.0);

        // Position should be closed
        assert!(pm.position("TEST").is_none());

        // Should have one closed trade with direction=Short
        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].direction, Direction::Short);
        assert_eq!(trades[0].quantity, 10);
        assert!((trades[0].entry_price - 100.0).abs() < f64::EPSILON);
        assert!((trades[0].exit_price - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_short_pnl_correct() {
        // Short at 100, cover at 90 => profit
        let mut pm = PortfolioManager::new(1_000_000.0);

        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 5.0); // entry costs
        pm.apply_fill(&buy_fill("TEST", 10, 90.0, 2_000), 5.0);   // exit costs

        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);
        // gross_pnl = (100 - 90) * 10 = 100
        // total_costs = 5 + 5 = 10
        // net_pnl = 100 - 10 = 90
        assert!((trades[0].pnl - 90.0).abs() < f64::EPSILON);
        assert!(trades[0].pnl > 0.0, "short pnl should be positive when price drops");
    }

    #[test]
    fn test_short_pnl_loss() {
        // Short at 100, cover at 110 => loss
        let mut pm = PortfolioManager::new(1_000_000.0);

        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 5.0);
        pm.apply_fill(&buy_fill("TEST", 10, 110.0, 2_000), 5.0);

        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);
        // gross_pnl = (100 - 110) * 10 = -100
        // total_costs = 5 + 5 = 10
        // net_pnl = -100 - 10 = -110
        assert!((trades[0].pnl - (-110.0)).abs() < f64::EPSILON);
        assert!(trades[0].pnl < 0.0, "short pnl should be negative when price rises");
    }

    #[test]
    fn test_short_equity_calculation() {
        // Short position, price rises => equity falls
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Short 10 at 100 (no costs for simplicity)
        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 0.0);
        // Cash should be 1_000_000 + 10*100 = 1_001_000

        // Price rises to 110
        let mut prices = HashMap::new();
        prices.insert("TEST".to_string(), 110.0);
        pm.update_prices(&prices, 2_000);

        // equity = cash - short_market_value = 1_001_000 - 10*110 = 1_001_000 - 1_100 = 999_900
        let eq = pm.equity();
        assert!((eq - 999_900.0).abs() < f64::EPSILON);
        assert!(eq < 1_000_000.0, "equity should fall when short position price rises");
    }

    #[test]
    fn test_short_equity_price_drop() {
        // Short position, price drops => equity rises
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Short 10 at 100
        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 0.0);

        // Price drops to 90
        let mut prices = HashMap::new();
        prices.insert("TEST".to_string(), 90.0);
        pm.update_prices(&prices, 2_000);

        // equity = cash - short_market_value = 1_001_000 - 10*90 = 1_001_000 - 900 = 1_000_100
        let eq = pm.equity();
        assert!((eq - 1_000_100.0).abs() < f64::EPSILON);
        assert!(eq > 1_000_000.0, "equity should rise when short position price drops");
    }

    #[test]
    fn test_portfolio_state_reports_negative_qty_for_shorts() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Short 10 at 100
        pm.apply_fill(&sell_fill("TEST", 10, 100.0, 1_000), 0.0);

        let portfolio = pm.portfolio_state();
        assert_eq!(portfolio.positions.len(), 1);
        let pos = &portfolio.positions[0];
        assert_eq!(pos.symbol, "TEST");
        assert_eq!(pos.quantity, -10, "short positions should report negative quantity");
        assert!((pos.avg_price - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_positions_snapshot() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy a CNC position
        pm.apply_fill(&buy_fill("LONG", 10, 100.0, 1_000), 0.0);

        // Short an MIS position
        pm.apply_fill(&sell_fill_mis("SHORT", 5, 200.0, 2_000), 0.0);

        let snapshot = pm.positions_snapshot();
        assert_eq!(snapshot.len(), 2);

        // Find each position in snapshot
        let long_pos = snapshot.iter().find(|(s, _, _, _)| s == "LONG");
        let short_pos = snapshot.iter().find(|(s, _, _, _)| s == "SHORT");

        assert!(long_pos.is_some());
        let (_, qty, is_short, pt) = long_pos.unwrap();
        assert_eq!(*qty, 10);
        assert!(!is_short);
        assert_eq!(*pt, ProductType::Cnc);

        assert!(short_pos.is_some());
        let (_, qty, is_short, pt) = short_pos.unwrap();
        assert_eq!(*qty, 5);
        assert!(*is_short);
        assert_eq!(*pt, ProductType::Mis);
    }

    #[test]
    fn test_closed_trade_direction_long() {
        let mut pm = PortfolioManager::new(1_000_000.0);

        // Buy then sell — long trade
        pm.apply_fill(&buy_fill("TEST", 10, 100.0, 1_000), 0.0);
        pm.apply_fill(&sell_fill("TEST", 10, 110.0, 2_000), 0.0);

        let trades = pm.closed_trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].direction, Direction::Long);
    }
}
