use std::collections::HashMap;

use crate::types::{Action, Bar, OrderType, ProductType, Signal};

// ── Side ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

// ── Order ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Order {
    pub symbol: String,
    pub side: Side,
    pub quantity: i32,
    pub order_type: OrderType,
    pub limit_price: f64,
    pub stop_price: f64,
    pub product_type: ProductType,
}

impl Order {
    pub fn market_buy(symbol: &str, qty: i32) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity: qty,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
            product_type: ProductType::Cnc,
        }
    }

    pub fn market_sell(symbol: &str, qty: i32) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity: qty,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
            product_type: ProductType::Cnc,
        }
    }

    pub fn limit_buy(symbol: &str, qty: i32, price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity: qty,
            order_type: OrderType::Limit,
            limit_price: price,
            stop_price: 0.0,
            product_type: ProductType::Cnc,
        }
    }

    pub fn limit_sell(symbol: &str, qty: i32, price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity: qty,
            order_type: OrderType::Limit,
            limit_price: price,
            stop_price: 0.0,
            product_type: ProductType::Cnc,
        }
    }

    pub fn stop_loss_sell(symbol: &str, qty: i32, stop: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity: qty,
            order_type: OrderType::Sl,
            limit_price: 0.0,
            stop_price: stop,
            product_type: ProductType::Cnc,
        }
    }

    pub fn stop_loss_buy(symbol: &str, qty: i32, stop: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity: qty,
            order_type: OrderType::Sl,
            limit_price: 0.0,
            stop_price: stop,
            product_type: ProductType::Cnc,
        }
    }

    /// Convert a `Signal` into an `Order`.
    pub fn from_signal(signal: &Signal) -> Self {
        let side = match signal.action {
            Action::Buy => Side::Buy,
            Action::Sell => Side::Sell,
            Action::Hold => Side::Buy, // Hold signals shouldn't generate orders,
                                       // but we default to Buy for safety.
        };
        Self {
            symbol: signal.symbol.clone(),
            side,
            quantity: signal.quantity,
            order_type: signal.order_type,
            limit_price: signal.limit_price,
            stop_price: signal.stop_price,
            product_type: signal.product_type,
        }
    }
}

// ── Fill ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Fill {
    pub symbol: String,
    pub side: Side,
    pub quantity: i32,
    pub fill_price: f64,
    pub timestamp_ms: i64,
    pub product_type: ProductType,
}

// ── CircuitLimits ──────────────────────────────────────────────────────────

/// Per-symbol circuit limit bounds.
#[derive(Debug, Clone)]
pub struct CircuitLimits {
    pub lower: f64,
    pub upper: f64,
}

// ── OrderRejection ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OrderRejection {
    pub symbol: String,
    pub side: Side,
    pub quantity: i32,
    pub reason: String,
}

// ── OrderMatcher ────────────────────────────────────────────────────────────

pub struct OrderMatcher {
    pending: Vec<Order>,
    slippage_pct: f64,
    circuit_limits: HashMap<String, CircuitLimits>,
}

impl OrderMatcher {
    pub fn new(slippage_pct: f64) -> Self {
        Self {
            pending: Vec::new(),
            slippage_pct,
            circuit_limits: HashMap::new(),
        }
    }

    /// Set circuit limits for a symbol. If set, fills outside these bounds are rejected.
    pub fn set_circuit_limits(&mut self, symbol: &str, limits: CircuitLimits) {
        self.circuit_limits.insert(symbol.to_string(), limits);
    }

    /// Check if a fill price is within circuit limits for the symbol.
    fn within_circuit_limits(&self, symbol: &str, price: f64) -> bool {
        match self.circuit_limits.get(symbol) {
            Some(limits) => price >= limits.lower && price <= limits.upper,
            None => true, // no limits = no restriction
        }
    }

    pub fn submit(&mut self, order: Order) {
        self.pending.push(order);
    }

    pub fn process_bar(&mut self, bar: &Bar) -> (Vec<Fill>, Vec<OrderRejection>) {
        let mut fills = Vec::new();
        let mut rejections = Vec::new();
        let mut remaining = Vec::new();

        let orders: Vec<Order> = self.pending.drain(..).collect();

        for order in orders {
            // Only process orders that match the bar's symbol.
            if order.symbol != bar.symbol {
                remaining.push(order);
                continue;
            }

            match (&order.order_type, &order.side) {
                // ── Market orders: fill at bar.open adjusted for slippage ────
                (OrderType::Market, Side::Buy) => {
                    let price = bar.open * (1.0 + self.slippage_pct);
                    if self.within_circuit_limits(&order.symbol, price) {
                        fills.push(Fill {
                            symbol: order.symbol,
                            side: Side::Buy,
                            quantity: order.quantity,
                            fill_price: price,
                            timestamp_ms: bar.timestamp_ms,
                            product_type: order.product_type,
                        });
                    } else {
                        rejections.push(OrderRejection {
                            symbol: order.symbol,
                            side: order.side,
                            quantity: order.quantity,
                            reason: "CIRCUIT_LIMIT".into(),
                        });
                    }
                }
                (OrderType::Market, Side::Sell) => {
                    let price = bar.open * (1.0 - self.slippage_pct);
                    if self.within_circuit_limits(&order.symbol, price) {
                        fills.push(Fill {
                            symbol: order.symbol,
                            side: Side::Sell,
                            quantity: order.quantity,
                            fill_price: price,
                            timestamp_ms: bar.timestamp_ms,
                            product_type: order.product_type,
                        });
                    } else {
                        rejections.push(OrderRejection {
                            symbol: order.symbol,
                            side: order.side,
                            quantity: order.quantity,
                            reason: "CIRCUIT_LIMIT".into(),
                        });
                    }
                }

                // ── Limit orders ────────────────────────────────────────────
                (OrderType::Limit, Side::Buy) => {
                    if bar.low <= order.limit_price {
                        let price = order.limit_price;
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Buy,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                            });
                        } else {
                            rejections.push(OrderRejection {
                                symbol: order.symbol,
                                side: order.side,
                                quantity: order.quantity,
                                reason: "CIRCUIT_LIMIT".into(),
                            });
                        }
                    } else {
                        remaining.push(order);
                    }
                }
                (OrderType::Limit, Side::Sell) => {
                    if bar.high >= order.limit_price {
                        let price = order.limit_price;
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Sell,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                            });
                        } else {
                            rejections.push(OrderRejection {
                                symbol: order.symbol,
                                side: order.side,
                                quantity: order.quantity,
                                reason: "CIRCUIT_LIMIT".into(),
                            });
                        }
                    } else {
                        remaining.push(order);
                    }
                }

                // ── Stop-loss orders (SL): fill at stop_price ───────────────
                (OrderType::Sl, Side::Sell) => {
                    if bar.low <= order.stop_price {
                        let price = order.stop_price;
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Sell,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                            });
                        } else {
                            rejections.push(OrderRejection {
                                symbol: order.symbol,
                                side: order.side,
                                quantity: order.quantity,
                                reason: "CIRCUIT_LIMIT".into(),
                            });
                        }
                    } else {
                        remaining.push(order);
                    }
                }
                (OrderType::Sl, Side::Buy) => {
                    if bar.high >= order.stop_price {
                        let price = order.stop_price;
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Buy,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                            });
                        } else {
                            rejections.push(OrderRejection {
                                symbol: order.symbol,
                                side: order.side,
                                quantity: order.quantity,
                                reason: "CIRCUIT_LIMIT".into(),
                            });
                        }
                    } else {
                        remaining.push(order);
                    }
                }

                // ── Stop-loss market (SL-M): trigger like SL, fill at stop_price ─
                (OrderType::SlM, Side::Sell) => {
                    if bar.low <= order.stop_price {
                        let price = order.stop_price;
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Sell,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                            });
                        } else {
                            rejections.push(OrderRejection {
                                symbol: order.symbol,
                                side: order.side,
                                quantity: order.quantity,
                                reason: "CIRCUIT_LIMIT".into(),
                            });
                        }
                    } else {
                        remaining.push(order);
                    }
                }
                (OrderType::SlM, Side::Buy) => {
                    if bar.high >= order.stop_price {
                        let price = order.stop_price;
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Buy,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                            });
                        } else {
                            rejections.push(OrderRejection {
                                symbol: order.symbol,
                                side: order.side,
                                quantity: order.quantity,
                                reason: "CIRCUIT_LIMIT".into(),
                            });
                        }
                    } else {
                        remaining.push(order);
                    }
                }
            }
        }

        self.pending = remaining;
        (fills, rejections)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test bar.
    fn make_bar(symbol: &str, open: f64, high: f64, low: f64, close: f64, ts: i64) -> Bar {
        Bar {
            timestamp_ms: ts,
            symbol: symbol.to_string(),
            open,
            high,
            low,
            close,
            volume: 100_000,
            oi: 0,
        }
    }

    #[test]
    fn test_market_order_fills_at_next_open() {
        let mut matcher = OrderMatcher::new(0.0); // no slippage
        matcher.submit(Order::market_buy("RELIANCE", 10));

        let bar = make_bar("RELIANCE", 2500.0, 2520.0, 2480.0, 2510.0, 1_000);
        let (fills, rejections) = matcher.process_bar(&bar);

        assert!(rejections.is_empty());
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].symbol, "RELIANCE");
        assert_eq!(fills[0].side, Side::Buy);
        assert_eq!(fills[0].quantity, 10);
        assert!((fills[0].fill_price - 2500.0).abs() < f64::EPSILON); // bar.open
        assert_eq!(fills[0].timestamp_ms, 1_000);
    }

    #[test]
    fn test_limit_buy_fills_when_low_touches() {
        let mut matcher = OrderMatcher::new(0.0);
        matcher.submit(Order::limit_buy("RELIANCE", 10, 2480.0));

        // Bar1: low=2490, does NOT touch limit_price=2480 -> no fill
        let bar1 = make_bar("RELIANCE", 2500.0, 2520.0, 2490.0, 2510.0, 1_000);
        let (fills1, _) = matcher.process_bar(&bar1);
        assert!(fills1.is_empty());

        // Bar2: low=2475, touches limit_price=2480 -> fill at 2480
        let bar2 = make_bar("RELIANCE", 2495.0, 2510.0, 2475.0, 2500.0, 2_000);
        let (fills2, _) = matcher.process_bar(&bar2);

        assert_eq!(fills2.len(), 1);
        assert_eq!(fills2[0].side, Side::Buy);
        assert!((fills2[0].fill_price - 2480.0).abs() < f64::EPSILON);
        assert_eq!(fills2[0].timestamp_ms, 2_000);
    }

    #[test]
    fn test_limit_sell_fills_when_high_touches() {
        let mut matcher = OrderMatcher::new(0.0);
        matcher.submit(Order::limit_sell("RELIANCE", 10, 2550.0));

        let bar = make_bar("RELIANCE", 2500.0, 2560.0, 2490.0, 2540.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, Side::Sell);
        assert!((fills[0].fill_price - 2550.0).abs() < f64::EPSILON);
        assert_eq!(fills[0].timestamp_ms, 1_000);
    }

    #[test]
    fn test_stop_loss_triggers() {
        let mut matcher = OrderMatcher::new(0.0);
        matcher.submit(Order::stop_loss_sell("RELIANCE", 10, 2450.0));

        let bar = make_bar("RELIANCE", 2470.0, 2480.0, 2440.0, 2460.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, Side::Sell);
        assert!((fills[0].fill_price - 2450.0).abs() < f64::EPSILON);
        assert_eq!(fills[0].timestamp_ms, 1_000);
    }

    #[test]
    fn test_slippage_applied() {
        let mut matcher = OrderMatcher::new(0.001); // 0.1% slippage
        matcher.submit(Order::market_buy("RELIANCE", 10));

        let bar = make_bar("RELIANCE", 2500.0, 2520.0, 2480.0, 2510.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        // fill_price = 2500 * (1 + 0.001) = 2502.5
        assert!((fills[0].fill_price - 2502.5).abs() < 1e-10);
    }

    #[test]
    fn test_unfilled_orders_remain_pending() {
        let mut matcher = OrderMatcher::new(0.0);
        matcher.submit(Order::limit_buy("RELIANCE", 10, 2400.0));

        // Bar where low=2450, does not touch 2400
        let bar = make_bar("RELIANCE", 2500.0, 2520.0, 2450.0, 2510.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert!(fills.is_empty());
        // The order should still be pending -- submit another bar that touches it
        let bar2 = make_bar("RELIANCE", 2420.0, 2430.0, 2390.0, 2410.0, 2_000);
        let (fills2, _) = matcher.process_bar(&bar2);
        assert_eq!(fills2.len(), 1);
        assert!((fills2[0].fill_price - 2400.0).abs() < f64::EPSILON);
    }

    // ── Circuit limit tests ─────────────────────────────────────────────

    #[test]
    fn test_circuit_limit_rejects_market_order() {
        let mut matcher = OrderMatcher::new(0.0);
        matcher.set_circuit_limits(
            "TEST",
            CircuitLimits {
                lower: 90.0,
                upper: 110.0,
            },
        );
        matcher.submit(Order::market_buy("TEST", 10));
        // Bar opens at 115 which is above upper circuit
        let bar = Bar {
            timestamp_ms: 1000,
            symbol: "TEST".into(),
            open: 115.0,
            high: 120.0,
            low: 112.0,
            close: 118.0,
            volume: 100000,
            oi: 0,
        };
        let (fills, rejections) = matcher.process_bar(&bar);
        assert!(
            fills.is_empty(),
            "order should be rejected - fill price above circuit upper"
        );
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].reason, "CIRCUIT_LIMIT");
        assert_eq!(rejections[0].symbol, "TEST");
        assert_eq!(rejections[0].quantity, 10);
    }

    #[test]
    fn test_circuit_limit_allows_within_range() {
        let mut matcher = OrderMatcher::new(0.0);
        matcher.set_circuit_limits(
            "TEST",
            CircuitLimits {
                lower: 90.0,
                upper: 110.0,
            },
        );
        matcher.submit(Order::market_buy("TEST", 10));
        let bar = Bar {
            timestamp_ms: 1000,
            symbol: "TEST".into(),
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 100000,
            oi: 0,
        };
        let (fills, rejections) = matcher.process_bar(&bar);
        assert!(rejections.is_empty());
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill_price, 100.0);
    }

    #[test]
    fn test_no_circuit_limits_allows_all() {
        let mut matcher = OrderMatcher::new(0.0);
        // No circuit limits set
        matcher.submit(Order::market_buy("TEST", 10));
        let bar = Bar {
            timestamp_ms: 1000,
            symbol: "TEST".into(),
            open: 5000.0,
            high: 5100.0,
            low: 4900.0,
            close: 5050.0,
            volume: 100000,
            oi: 0,
        };
        let (fills, rejections) = matcher.process_bar(&bar);
        assert!(rejections.is_empty());
        assert_eq!(fills.len(), 1); // should fill normally
    }
}
