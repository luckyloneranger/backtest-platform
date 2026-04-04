use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::matching::{Fill, Side};
use crate::types::{Portfolio, Position};

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
    pub pnl: f64,   // (exit - entry) * qty for long, (entry - exit) * qty for short
    pub costs: f64,  // total transaction costs for this round-trip
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
    pub quantity: i32,        // positive = long, negative = short
    pub avg_price: f64,
    pub current_price: f64,
    pub entry_timestamp_ms: i64,
    pub total_costs: f64,     // accumulated costs
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
        self.cash -= fill.quantity as f64 * fill.fill_price + costs;

        if let Some(pos) = self.positions.get_mut(&fill.symbol) {
            // Existing long position: average up
            let old_qty = pos.quantity;
            let new_qty = old_qty + fill.quantity;
            pos.avg_price = (old_qty as f64 * pos.avg_price
                + fill.quantity as f64 * fill.fill_price)
                / new_qty as f64;
            pos.quantity = new_qty;
            pos.total_costs += costs;
        } else {
            // New position
            self.positions.insert(
                fill.symbol.clone(),
                InternalPosition {
                    symbol: fill.symbol.clone(),
                    quantity: fill.quantity,
                    avg_price: fill.fill_price,
                    current_price: fill.fill_price,
                    entry_timestamp_ms: fill.timestamp_ms,
                    total_costs: costs,
                },
            );
        }
    }

    fn apply_sell(&mut self, fill: &Fill, costs: f64) {
        self.cash += fill.quantity as f64 * fill.fill_price - costs;

        let should_remove = if let Some(pos) = self.positions.get_mut(&fill.symbol) {
            if fill.quantity >= pos.quantity {
                // Full close
                let closed_qty = pos.quantity;
                let pnl =
                    (fill.fill_price - pos.avg_price) * closed_qty as f64;
                let total_costs = pos.total_costs + costs;

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
                });
                true
            } else {
                // Partial close
                let closed_qty = fill.quantity;
                let pnl =
                    (fill.fill_price - pos.avg_price) * closed_qty as f64;
                // Allocate costs proportionally
                let entry_cost_portion =
                    pos.total_costs * (closed_qty as f64 / pos.quantity as f64);
                let total_costs = entry_cost_portion + costs;

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
                });

                pos.total_costs -= entry_cost_portion;
                pos.quantity -= closed_qty;
                false
            }
        } else {
            // Selling without a position — open short (not fully modeled yet,
            // but handle gracefully)
            false
        };

        if should_remove {
            self.positions.remove(&fill.symbol);
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

    /// Compute current total equity (cash + sum of position market values).
    pub fn equity(&self) -> f64 {
        self.cash
            + self
                .positions
                .values()
                .map(|p| p.quantity as f64 * p.current_price)
                .sum::<f64>()
    }

    /// Build a Portfolio snapshot for sending to strategy via gRPC.
    pub fn portfolio_state(&self) -> Portfolio {
        let positions: Vec<Position> = self
            .positions
            .values()
            .map(|p| Position {
                symbol: p.symbol.clone(),
                quantity: p.quantity,
                avg_price: p.avg_price,
                current_price: p.current_price,
            })
            .collect();

        Portfolio {
            cash: self.cash,
            positions,
        }
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
        // pnl = (2600 - 2500) * 10 = 1000
        assert!((trade.pnl - 1000.0).abs() < f64::EPSILON);
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
}
