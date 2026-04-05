use std::collections::HashMap;

use crate::types::{Action, Bar, OrderType, OrderValidity, ProductType, Signal};

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
    pub trigger_price: f64,
    pub validity: OrderValidity,
    pub order_id: u64,
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
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
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
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
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
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
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
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
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
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
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
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
        }
    }

    pub fn stop_loss_market_sell(symbol: &str, qty: i32, stop: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity: qty,
            order_type: OrderType::SlM,
            limit_price: 0.0,
            stop_price: stop,
            product_type: ProductType::Cnc,
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
        }
    }

    pub fn stop_loss_market_buy(symbol: &str, qty: i32, stop: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity: qty,
            order_type: OrderType::SlM,
            limit_price: 0.0,
            stop_price: stop,
            product_type: ProductType::Cnc,
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            order_id: 0,
        }
    }

    /// Convert a `Signal` into an `Order`.
    pub fn from_signal(signal: &Signal) -> Self {
        let side = match signal.action {
            Action::Buy => Side::Buy,
            Action::Sell => Side::Sell,
            Action::Hold => panic!("Hold signals should not be converted to orders"),
            Action::Cancel => panic!("Cancel signals should not be converted to orders"),
        };
        Self {
            symbol: signal.symbol.clone(),
            side,
            quantity: signal.quantity,
            order_type: signal.order_type,
            limit_price: signal.limit_price,
            stop_price: signal.stop_price,
            product_type: signal.product_type,
            trigger_price: signal.trigger_price,
            validity: signal.validity,
            order_id: 0,
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
    pub costs: f64,
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

// ── PendingOrderInfo ──────────────────────────────────────────────────────

/// Lightweight read-only view of a pending order, sent to strategies.
#[derive(Debug, Clone)]
pub struct PendingOrderInfo {
    pub symbol: String,
    pub side: Side,
    pub quantity: i32,
    pub order_type: OrderType,
    pub limit_price: f64,
    pub stop_price: f64,
    pub order_id: u64,
}

impl PendingOrderInfo {
    pub fn from_order(order: &Order) -> Self {
        Self {
            symbol: order.symbol.clone(),
            side: order.side,
            quantity: order.quantity,
            order_type: order.order_type,
            limit_price: order.limit_price,
            stop_price: order.stop_price,
            order_id: order.order_id,
        }
    }
}

// ── OrderMatcher ────────────────────────────────────────────────────────────

pub struct OrderMatcher {
    pending: Vec<Order>,
    slippage_pct: f64,
    circuit_limits: HashMap<String, CircuitLimits>,
    max_volume_pct: f64,
    next_order_id: u64,
}

impl OrderMatcher {
    pub fn new(slippage_pct: f64, max_volume_pct: f64) -> Self {
        Self {
            pending: Vec::new(),
            slippage_pct,
            circuit_limits: HashMap::new(),
            max_volume_pct,
            next_order_id: 1,
        }
    }

    /// Remove and return all pending orders for the given symbol.
    pub fn cancel_orders_for_symbol(&mut self, symbol: &str) -> Vec<Order> {
        let mut cancelled = Vec::new();
        let mut remaining = Vec::new();
        for order in self.pending.drain(..) {
            if order.symbol == symbol {
                cancelled.push(order);
            } else {
                remaining.push(order);
            }
        }
        self.pending = remaining;
        cancelled
    }

    /// Remove and return a specific pending order by its order_id.
    pub fn cancel_order(&mut self, order_id: u64) -> Option<Order> {
        if let Some(pos) = self.pending.iter().position(|o| o.order_id == order_id) {
            Some(self.pending.remove(pos))
        } else {
            None
        }
    }

    /// Drain and return all pending orders.
    pub fn cancel_all_pending(&mut self) -> Vec<Order> {
        self.pending.drain(..).collect()
    }

    /// Cancel only DAY validity orders (used at 15:30 IST market close).
    /// GTC and other validity orders remain pending.
    pub fn cancel_day_orders(&mut self) -> Vec<Order> {
        let (day_orders, remaining): (Vec<Order>, Vec<Order>) = self.pending
            .drain(..)
            .partition(|o| o.validity == OrderValidity::Day);
        self.pending = remaining;
        day_orders
    }

    /// Read-only access to all pending orders.
    pub fn pending_orders(&self) -> &[Order] {
        &self.pending
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

    pub fn submit(&mut self, mut order: Order) {
        order.order_id = self.next_order_id;
        self.next_order_id += 1;
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
                            costs: 0.0,
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
                            costs: 0.0,
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

                // ── Limit orders (with gap handling) ──────────────────────────
                (OrderType::Limit, Side::Buy) => {
                    if bar.low <= order.limit_price {
                        // Gap: if bar opens at or below limit, fill at open (price improvement)
                        let price = if bar.open <= order.limit_price {
                            bar.open
                        } else {
                            order.limit_price
                        };
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Buy,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                                costs: 0.0,
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
                        // Gap: if bar opens at or above limit, fill at open (price improvement)
                        let price = if bar.open >= order.limit_price {
                            bar.open
                        } else {
                            order.limit_price
                        };
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Sell,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                                costs: 0.0,
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

                // ── Stop-loss orders (SL): two-price model with trigger + limit ──
                (OrderType::Sl, Side::Sell) => {
                    let trigger = if order.trigger_price > 0.0 { order.trigger_price } else { order.stop_price };
                    if bar.low <= trigger {
                        // Triggered! Now check limit fill
                        let limit = order.limit_price;
                        if limit > 0.0 && order.trigger_price > 0.0 {
                            // Two-price model: trigger_price triggers, limit_price determines fill
                            if bar.open <= limit {
                                // Gapped through limit -- fill at bar.open
                                let price = bar.open;
                                if self.within_circuit_limits(&order.symbol, price) {
                                    fills.push(Fill {
                                        symbol: order.symbol,
                                        side: Side::Sell,
                                        quantity: order.quantity,
                                        fill_price: price,
                                        timestamp_ms: bar.timestamp_ms,
                                        product_type: order.product_type,
                                        costs: 0.0,
                                    });
                                } else {
                                    rejections.push(OrderRejection {
                                        symbol: order.symbol,
                                        side: order.side,
                                        quantity: order.quantity,
                                        reason: "CIRCUIT_LIMIT".into(),
                                    });
                                }
                            } else if bar.low <= limit {
                                // Price reached limit -- fill at limit
                                let price = limit;
                                if self.within_circuit_limits(&order.symbol, price) {
                                    fills.push(Fill {
                                        symbol: order.symbol,
                                        side: Side::Sell,
                                        quantity: order.quantity,
                                        fill_price: price,
                                        timestamp_ms: bar.timestamp_ms,
                                        product_type: order.product_type,
                                        costs: 0.0,
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
                                // Triggered but price didn't reach limit -- convert to pending limit
                                let mut converted = order;
                                converted.order_type = OrderType::Limit;
                                converted.trigger_price = 0.0;
                                remaining.push(converted);
                            }
                        } else {
                            // No two-price: old behavior -- fill at trigger or bar.open on gap
                            let price = if bar.open <= trigger {
                                bar.open * (1.0 - self.slippage_pct)
                            } else {
                                trigger * (1.0 - self.slippage_pct)
                            };
                            if self.within_circuit_limits(&order.symbol, price) {
                                fills.push(Fill {
                                    symbol: order.symbol,
                                    side: Side::Sell,
                                    quantity: order.quantity,
                                    fill_price: price,
                                    timestamp_ms: bar.timestamp_ms,
                                    product_type: order.product_type,
                                    costs: 0.0,
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
                    } else {
                        remaining.push(order);
                    }
                }
                (OrderType::Sl, Side::Buy) => {
                    let trigger = if order.trigger_price > 0.0 { order.trigger_price } else { order.stop_price };
                    if bar.high >= trigger {
                        // Triggered! Now check limit fill
                        let limit = order.limit_price;
                        if limit > 0.0 && order.trigger_price > 0.0 {
                            // Two-price model: trigger_price triggers, limit_price determines fill
                            if bar.open >= limit {
                                // Gapped through limit -- fill at bar.open
                                let price = bar.open;
                                if self.within_circuit_limits(&order.symbol, price) {
                                    fills.push(Fill {
                                        symbol: order.symbol,
                                        side: Side::Buy,
                                        quantity: order.quantity,
                                        fill_price: price,
                                        timestamp_ms: bar.timestamp_ms,
                                        product_type: order.product_type,
                                        costs: 0.0,
                                    });
                                } else {
                                    rejections.push(OrderRejection {
                                        symbol: order.symbol,
                                        side: order.side,
                                        quantity: order.quantity,
                                        reason: "CIRCUIT_LIMIT".into(),
                                    });
                                }
                            } else if bar.high >= limit {
                                // Price reached limit -- fill at limit
                                let price = limit;
                                if self.within_circuit_limits(&order.symbol, price) {
                                    fills.push(Fill {
                                        symbol: order.symbol,
                                        side: Side::Buy,
                                        quantity: order.quantity,
                                        fill_price: price,
                                        timestamp_ms: bar.timestamp_ms,
                                        product_type: order.product_type,
                                        costs: 0.0,
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
                                // Triggered but price didn't reach limit -- convert to pending limit
                                let mut converted = order;
                                converted.order_type = OrderType::Limit;
                                converted.trigger_price = 0.0;
                                remaining.push(converted);
                            }
                        } else {
                            // No two-price: old behavior -- fill at trigger or bar.open on gap
                            let price = if bar.open >= trigger {
                                bar.open * (1.0 + self.slippage_pct)
                            } else {
                                trigger * (1.0 + self.slippage_pct)
                            };
                            if self.within_circuit_limits(&order.symbol, price) {
                                fills.push(Fill {
                                    symbol: order.symbol,
                                    side: Side::Buy,
                                    quantity: order.quantity,
                                    fill_price: price,
                                    timestamp_ms: bar.timestamp_ms,
                                    product_type: order.product_type,
                                    costs: 0.0,
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
                    } else {
                        remaining.push(order);
                    }
                }

                // ── Stop-loss market (SL-M): trigger like SL, fill at market (bar.open + slippage) ─
                (OrderType::SlM, Side::Sell) => {
                    if bar.low <= order.stop_price {
                        let price = bar.open * (1.0 - self.slippage_pct);
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Sell,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                                costs: 0.0,
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
                        let price = bar.open * (1.0 + self.slippage_pct);
                        if self.within_circuit_limits(&order.symbol, price) {
                            fills.push(Fill {
                                symbol: order.symbol,
                                side: Side::Buy,
                                quantity: order.quantity,
                                fill_price: price,
                                timestamp_ms: bar.timestamp_ms,
                                product_type: order.product_type,
                                costs: 0.0,
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

        // Apply volume constraints: clamp fill quantity to max_volume_pct of bar volume.
        // If clamped, create a remainder order that stays pending.
        if self.max_volume_pct < 1.0 {
            let max_qty = (bar.volume as f64 * self.max_volume_pct).max(1.0) as i32;
            let mut clamped_fills = Vec::new();
            for mut fill in fills {
                if fill.quantity > max_qty {
                    let remainder_qty = fill.quantity - max_qty;
                    fill.quantity = max_qty;
                    // Create remainder order that goes back to pending.
                    // The original trigger condition was met, so remainder fills
                    // at next bar open as a market order.
                    let remainder = Order {
                        symbol: fill.symbol.clone(),
                        side: fill.side,
                        quantity: remainder_qty,
                        order_type: OrderType::Market,
                        limit_price: 0.0,
                        stop_price: 0.0,
                        product_type: fill.product_type,
                        trigger_price: 0.0,
                        validity: OrderValidity::default(),
                        order_id: 0,
                    };
                    self.pending.push(remainder);
                }
                clamped_fills.push(fill);
            }
            fills = clamped_fills;
        }

        // Auto-cancel unfilled IOC orders
        self.pending.retain(|o| o.validity != OrderValidity::Ioc);

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
        let mut matcher = OrderMatcher::new(0.0, 1.0); // no slippage
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
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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
        let mut matcher = OrderMatcher::new(0.001, 1.0); // 0.1% slippage
        matcher.submit(Order::market_buy("RELIANCE", 10));

        let bar = make_bar("RELIANCE", 2500.0, 2520.0, 2480.0, 2510.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        // fill_price = 2500 * (1 + 0.001) = 2502.5
        assert!((fills[0].fill_price - 2502.5).abs() < 1e-10);
    }

    #[test]
    fn test_unfilled_orders_remain_pending() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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
        let mut matcher = OrderMatcher::new(0.0, 1.0);
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

    // ── SL-M market fill tests ─────────────────────────────────────────

    #[test]
    fn test_slm_sell_fills_at_market_price() {
        let slippage = 0.001; // 0.1%
        let mut matcher = OrderMatcher::new(slippage, 1.0);
        // SL-M sell with stop at 2450
        matcher.submit(Order::stop_loss_market_sell("RELIANCE", 10, 2450.0));

        // Bar where low=2440 triggers (low <= stop_price), open=2470
        let bar = make_bar("RELIANCE", 2470.0, 2480.0, 2440.0, 2460.0, 1_000);
        let (fills, rejections) = matcher.process_bar(&bar);

        assert!(rejections.is_empty());
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, Side::Sell);
        // SL-M should fill at bar.open * (1 - slippage) = 2470 * 0.999 = 2467.53
        let expected = 2470.0 * (1.0 - slippage);
        assert!(
            (fills[0].fill_price - expected).abs() < 1e-10,
            "SL-M sell should fill at bar.open*(1-slippage)={}, got {}",
            expected,
            fills[0].fill_price,
        );
        // Must NOT fill at stop_price
        assert!(
            (fills[0].fill_price - 2450.0).abs() > 1.0,
            "SL-M sell must not fill at stop_price"
        );
    }

    #[test]
    fn test_slm_buy_fills_at_market_price() {
        let slippage = 0.001; // 0.1%
        let mut matcher = OrderMatcher::new(slippage, 1.0);
        // SL-M buy with stop at 2550
        matcher.submit(Order::stop_loss_market_buy("RELIANCE", 10, 2550.0));

        // Bar where high=2560 triggers (high >= stop_price), open=2530
        let bar = make_bar("RELIANCE", 2530.0, 2560.0, 2520.0, 2540.0, 1_000);
        let (fills, rejections) = matcher.process_bar(&bar);

        assert!(rejections.is_empty());
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, Side::Buy);
        // SL-M should fill at bar.open * (1 + slippage) = 2530 * 1.001 = 2532.53
        let expected = 2530.0 * (1.0 + slippage);
        assert!(
            (fills[0].fill_price - expected).abs() < 1e-10,
            "SL-M buy should fill at bar.open*(1+slippage)={}, got {}",
            expected,
            fills[0].fill_price,
        );
        // Must NOT fill at stop_price
        assert!(
            (fills[0].fill_price - 2550.0).abs() > 1.0,
            "SL-M buy must not fill at stop_price"
        );
    }

    // ── Hold panic test ────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Hold signals should not be converted to orders")]
    fn test_from_signal_panics_on_hold() {
        let hold_signal = Signal {
            action: Action::Hold,
            symbol: "TEST".into(),
            quantity: 10,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
            product_type: ProductType::Cnc,
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            cancel_order_id: 0,
        };
        let _ = Order::from_signal(&hold_signal);
    }

    // ── Cancel panic test ─────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Cancel signals should not be converted to orders")]
    fn test_from_signal_panics_on_cancel() {
        let cancel_signal = Signal {
            action: Action::Cancel,
            symbol: "TEST".into(),
            quantity: 0,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
            product_type: ProductType::Cnc,
            trigger_price: 0.0,
            validity: OrderValidity::default(),
            cancel_order_id: 0,
        };
        let _ = Order::from_signal(&cancel_signal);
    }

    // ── Cancellation tests ────────────────────────────────────────────

    #[test]
    fn test_cancel_removes_pending_orders() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("RELIANCE", 10, 2400.0));

        assert_eq!(matcher.pending_orders().len(), 1);

        let cancelled = matcher.cancel_orders_for_symbol("RELIANCE");
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0].symbol, "RELIANCE");
        assert!(matcher.pending_orders().is_empty());
    }

    #[test]
    fn test_cancel_only_affects_target_symbol() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("RELIANCE", 10, 2400.0));
        matcher.submit(Order::limit_buy("INFY", 5, 1400.0));

        assert_eq!(matcher.pending_orders().len(), 2);

        let cancelled = matcher.cancel_orders_for_symbol("RELIANCE");
        assert_eq!(cancelled.len(), 1);
        assert_eq!(matcher.pending_orders().len(), 1);
        assert_eq!(matcher.pending_orders()[0].symbol, "INFY");
    }

    #[test]
    fn test_pending_orders_visible() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("RELIANCE", 10, 2400.0));

        let pending = matcher.pending_orders();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].symbol, "RELIANCE");
        assert_eq!(pending[0].side, Side::Buy);
        assert_eq!(pending[0].quantity, 10);
    }

    // ── Gap handling tests ────────────────────────────────────────────

    #[test]
    fn test_limit_buy_gap_fills_at_open() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        // Limit buy at 100
        matcher.submit(Order::limit_buy("TEST", 10, 100.0));

        // Bar opens at 95 (gapped below limit)
        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 95.0,
            high: 98.0,
            low: 93.0,
            close: 97.0,
            volume: 100_000,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price - 95.0).abs() < f64::EPSILON,
            "limit buy gap should fill at open 95, got {}", fills[0].fill_price);
    }

    #[test]
    fn test_limit_sell_gap_fills_at_open() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        // Limit sell at 100
        matcher.submit(Order::limit_sell("TEST", 10, 100.0));

        // Bar opens at 105 (gapped above limit)
        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 105.0,
            high: 108.0,
            low: 103.0,
            close: 106.0,
            volume: 100_000,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price - 105.0).abs() < f64::EPSILON,
            "limit sell gap should fill at open 105, got {}", fills[0].fill_price);
    }

    #[test]
    fn test_sl_sell_gap_fills_at_open() {
        let slippage = 0.001;
        let mut matcher = OrderMatcher::new(slippage, 1.0);
        // SL sell at 100
        matcher.submit(Order::stop_loss_sell("TEST", 10, 100.0));

        // Bar opens at 90 (gapped below stop)
        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 90.0,
            high: 92.0,
            low: 88.0,
            close: 91.0,
            volume: 100_000,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        let expected = 90.0 * (1.0 - slippage);
        assert!((fills[0].fill_price - expected).abs() < 1e-10,
            "SL sell gap should fill at open*(1-slippage)={}, got {}", expected, fills[0].fill_price);
    }

    #[test]
    fn test_sl_buy_gap_fills_at_open() {
        let slippage = 0.001;
        let mut matcher = OrderMatcher::new(slippage, 1.0);
        // SL buy at 100
        matcher.submit(Order::stop_loss_buy("TEST", 10, 100.0));

        // Bar opens at 110 (gapped above stop)
        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 110.0,
            high: 115.0,
            low: 108.0,
            close: 112.0,
            volume: 100_000,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        let expected = 110.0 * (1.0 + slippage);
        assert!((fills[0].fill_price - expected).abs() < 1e-10,
            "SL buy gap should fill at open*(1+slippage)={}, got {}", expected, fills[0].fill_price);
    }

    #[test]
    fn test_limit_buy_no_gap_fills_at_limit() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        // Limit buy at 100
        matcher.submit(Order::limit_buy("TEST", 10, 100.0));

        // Bar opens at 102 (above limit), low touches 100 -> fills at limit price
        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 102.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 100_000,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price - 100.0).abs() < f64::EPSILON,
            "limit buy no-gap should fill at limit 100, got {}", fills[0].fill_price);
    }

    // ── Volume constraint tests ───────────────────────────────────────

    #[test]
    fn test_volume_constraint_clamps_qty() {
        // max_pct = 0.1 (10%), bar volume = 100, so max_qty = 10
        let mut matcher = OrderMatcher::new(0.0, 0.1);
        matcher.submit(Order::market_buy("TEST", 1000));

        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 100,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 10,
            "fill qty should be clamped to 10% of volume 100 = 10, got {}", fills[0].quantity);
    }

    #[test]
    fn test_volume_remainder_stays_pending() {
        let mut matcher = OrderMatcher::new(0.0, 0.1);
        matcher.submit(Order::market_buy("TEST", 1000));

        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 100,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills[0].quantity, 10);
        // Remainder (1000 - 10 = 990) should be in pending
        let pending = matcher.pending_orders();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].quantity, 990);
        assert_eq!(pending[0].symbol, "TEST");
    }

    #[test]
    fn test_volume_no_constraint_default() {
        // max_pct = 1.0 means no constraint
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::market_buy("TEST", 1000));

        let bar = Bar {
            timestamp_ms: 1_000,
            symbol: "TEST".into(),
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 100,
            oi: 0,
        };
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 1000,
            "with max_pct=1.0, full order should fill regardless of volume");
        assert!(matcher.pending_orders().is_empty());
    }

    // ── IOC validity tests ────────────────────────────────────────────

    #[test]
    fn test_ioc_unfilled_cancelled() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        let mut order = Order::limit_buy("TEST", 10, 90.0);
        order.validity = OrderValidity::Ioc;
        matcher.submit(order);

        // Bar where low=95 does NOT touch limit=90 -> no fill, IOC should be auto-cancelled
        let bar = make_bar("TEST", 100.0, 105.0, 95.0, 102.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert!(fills.is_empty(), "IOC limit order should not fill");
        assert!(matcher.pending_orders().is_empty(),
            "unfilled IOC order should be auto-cancelled after process_bar");
    }

    #[test]
    fn test_ioc_filled_not_cancelled() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        let mut order = Order::limit_buy("TEST", 10, 100.0);
        order.validity = OrderValidity::Ioc;
        matcher.submit(order);

        // Bar where low=95 touches limit=100 -> fills
        let bar = make_bar("TEST", 102.0, 105.0, 95.0, 101.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1, "IOC limit order should fill when price touches");
        assert!((fills[0].fill_price - 100.0).abs() < f64::EPSILON);
    }

    // ── SL two-price model tests ──────────────────────────────────────

    #[test]
    fn test_sl_two_price_fill_at_limit() {
        // trigger=100, limit=99, bar touches 98 -> fill at 99
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        let mut order = Order::stop_loss_sell("TEST", 10, 100.0);
        order.trigger_price = 100.0;
        order.limit_price = 99.0;
        matcher.submit(order);

        // Bar: open=101, high=102, low=98, close=99
        // low=98 <= trigger=100, triggers. low=98 <= limit=99, fills at 99.
        let bar = make_bar("TEST", 101.0, 102.0, 98.0, 99.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price - 99.0).abs() < f64::EPSILON,
            "SL two-price should fill at limit=99, got {}", fills[0].fill_price);
    }

    #[test]
    fn test_sl_two_price_gap_through() {
        // trigger=100, limit=99, bar opens at 95 -> fill at 95 (gapped through)
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        let mut order = Order::stop_loss_sell("TEST", 10, 100.0);
        order.trigger_price = 100.0;
        order.limit_price = 99.0;
        matcher.submit(order);

        // Bar: open=95, high=96, low=94, close=95 — gapped through both trigger and limit
        let bar = make_bar("TEST", 95.0, 96.0, 94.0, 95.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price - 95.0).abs() < f64::EPSILON,
            "SL two-price gap should fill at open=95, got {}", fills[0].fill_price);
    }

    #[test]
    fn test_sl_two_price_trigger_no_fill() {
        // trigger=100, limit=99, bar.low=99.5 -> triggered but no fill, becomes pending limit
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        let mut order = Order::stop_loss_sell("TEST", 10, 100.0);
        order.trigger_price = 100.0;
        order.limit_price = 99.0;
        matcher.submit(order);

        // Bar: open=101, high=102, low=99.5, close=100 — triggers (low <= 100) but low > limit 99
        let bar = make_bar("TEST", 101.0, 102.0, 99.5, 100.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert!(fills.is_empty(), "SL two-price should not fill when price doesn't reach limit");
        let pending = matcher.pending_orders();
        assert_eq!(pending.len(), 1, "order should be converted to pending limit");
        assert_eq!(pending[0].order_type, OrderType::Limit,
            "triggered SL should convert to limit order");
    }

    #[test]
    fn test_sl_backward_compat() {
        // trigger_price=0, uses stop_price as trigger (existing behavior)
        // With 0 slippage, fill_price = trigger * (1.0 - 0.0) = trigger
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::stop_loss_sell("TEST", 10, 100.0));

        let bar = make_bar("TEST", 101.0, 102.0, 98.0, 99.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price - 100.0).abs() < f64::EPSILON,
            "backward compat: SL should fill at stop_price=100, got {}", fills[0].fill_price);
    }

    #[test]
    fn test_sl_single_price_sell_slippage_non_gap() {
        // SL sell at 100, bar does NOT gap (open=101 above trigger), low hits trigger.
        // Slippage should apply to trigger price: 100 * (1 - 0.001) = 99.9
        let slippage = 0.001;
        let mut matcher = OrderMatcher::new(slippage, 1.0);
        matcher.submit(Order::stop_loss_sell("TEST", 10, 100.0));

        let bar = make_bar("TEST", 101.0, 102.0, 98.0, 99.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        let expected = 100.0 * (1.0 - slippage);
        assert!((fills[0].fill_price - expected).abs() < 1e-10,
            "SL sell non-gap should fill at trigger*(1-slippage)={}, got {}", expected, fills[0].fill_price);
    }

    #[test]
    fn test_sl_single_price_buy_slippage_non_gap() {
        // SL buy at 100, bar does NOT gap (open=99 below trigger), high hits trigger.
        // Slippage should apply to trigger price: 100 * (1 + 0.001) = 100.1
        let slippage = 0.001;
        let mut matcher = OrderMatcher::new(slippage, 1.0);
        matcher.submit(Order::stop_loss_buy("TEST", 10, 100.0));

        let bar = make_bar("TEST", 99.0, 102.0, 98.0, 101.0, 1_000);
        let (fills, _) = matcher.process_bar(&bar);

        assert_eq!(fills.len(), 1);
        let expected = 100.0 * (1.0 + slippage);
        assert!((fills[0].fill_price - expected).abs() < 1e-10,
            "SL buy non-gap should fill at trigger*(1+slippage)={}, got {}", expected, fills[0].fill_price);
    }

    // ── Per-order cancellation tests ──────────────────────────────────

    #[test]
    fn test_cancel_specific_order_by_id() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("TEST", 10, 100.0));
        matcher.submit(Order::limit_buy("TEST", 5, 95.0));

        let id1 = matcher.pending_orders()[0].order_id;
        let _id2 = matcher.pending_orders()[1].order_id;

        // Cancel only the first order
        let cancelled = matcher.cancel_order(id1);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().quantity, 10);
        assert_eq!(matcher.pending_orders().len(), 1);
        assert_eq!(matcher.pending_orders()[0].quantity, 5);
    }

    #[test]
    fn test_cancel_all_still_works() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("TEST", 10, 100.0));
        matcher.submit(Order::limit_buy("TEST", 5, 95.0));

        let cancelled = matcher.cancel_orders_for_symbol("TEST");
        assert_eq!(cancelled.len(), 2);
        assert!(matcher.pending_orders().is_empty());
    }

    #[test]
    fn test_order_ids_sequential() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("A", 1, 100.0));
        matcher.submit(Order::limit_buy("B", 2, 200.0));
        matcher.submit(Order::limit_buy("C", 3, 300.0));

        let pending = matcher.pending_orders();
        assert_eq!(pending[0].order_id, 1);
        assert_eq!(pending[1].order_id, 2);
        assert_eq!(pending[2].order_id, 3);
    }

    #[test]
    fn test_pending_order_info_has_id() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);
        matcher.submit(Order::limit_buy("TEST", 10, 100.0));

        let info = PendingOrderInfo::from_order(&matcher.pending_orders()[0]);
        assert_eq!(info.order_id, 1);
        assert_eq!(info.symbol, "TEST");
        assert_eq!(info.quantity, 10);
    }

    // ── DAY expiry tests ──────────────────────────────────────────────

    #[test]
    fn test_cancel_day_orders_only_cancels_day_validity() {
        let mut matcher = OrderMatcher::new(0.0, 1.0);

        // Submit a DAY order
        let mut day_order = Order::limit_buy("TEST", 10, 100.0);
        day_order.validity = OrderValidity::Day;
        matcher.submit(day_order);

        // Submit an IOC order (not Day)
        let mut ioc_order = Order::limit_buy("TEST", 5, 95.0);
        ioc_order.validity = OrderValidity::Ioc;
        matcher.submit(ioc_order);

        assert_eq!(matcher.pending_orders().len(), 2);

        // Cancel only DAY orders
        let cancelled = matcher.cancel_day_orders();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0].quantity, 10);
        assert_eq!(cancelled[0].validity, OrderValidity::Day);

        // IOC order should remain
        assert_eq!(matcher.pending_orders().len(), 1);
        assert_eq!(matcher.pending_orders()[0].quantity, 5);
        assert_eq!(matcher.pending_orders()[0].validity, OrderValidity::Ioc);
    }
}
