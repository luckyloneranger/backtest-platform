use serde::{Deserialize, Serialize};

// ── Bar ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp_ms: i64,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub oi: i64,
}

// ── Action / OrderType ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Hold,
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Sl,
    SlM,
}

// ── Signal ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub action: Action,
    pub symbol: String,
    pub quantity: i32,
    pub order_type: OrderType,
    pub limit_price: f64,
    pub stop_price: f64,
}

impl Signal {
    /// Convenience constructor for a market buy order.
    pub fn market_buy(symbol: impl Into<String>, qty: i32) -> Self {
        Self {
            action: Action::Buy,
            symbol: symbol.into(),
            quantity: qty,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
        }
    }

    /// Convenience constructor for a market sell order.
    pub fn market_sell(symbol: impl Into<String>, qty: i32) -> Self {
        Self {
            action: Action::Sell,
            symbol: symbol.into(),
            quantity: qty,
            order_type: OrderType::Market,
            limit_price: 0.0,
            stop_price: 0.0,
        }
    }
}

// ── Position ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: i32,
    pub avg_price: f64,
    pub current_price: f64,
}

impl Position {
    /// Market value of the position: quantity * current_price.
    pub fn market_value(&self) -> f64 {
        self.quantity as f64 * self.current_price
    }

    /// Unrealized P&L: quantity * (current_price - avg_price).
    pub fn unrealized_pnl(&self) -> f64 {
        self.quantity as f64 * (self.current_price - self.avg_price)
    }
}

// ── Portfolio ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub cash: f64,
    pub positions: Vec<Position>,
}

impl Portfolio {
    /// Total equity = cash + sum of position market values.
    pub fn equity(&self) -> f64 {
        self.cash
            + self
                .positions
                .iter()
                .map(|p| p.market_value())
                .sum::<f64>()
    }
}

// ── Exchange / InstrumentType / Instrument ───────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Exchange {
    Nse,
    Bse,
    Mcx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstrumentType {
    Equity,
    FutureFO,
    OptionFO,
    Commodity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instrument {
    pub tradingsymbol: String,
    pub exchange: Exchange,
    pub instrument_type: InstrumentType,
    pub lot_size: i32,
    pub tick_size: f64,
    pub expiry: Option<String>,
    pub strike: Option<f64>,
    pub option_type: Option<String>,
}

// ── Interval ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    Minute,
    Minute3,
    Minute5,
    Minute10,
    Minute15,
    Minute30,
    Minute60,
    Day,
}

impl Interval {
    /// Returns the Kite Connect API string representation.
    pub fn as_kite_str(&self) -> &'static str {
        match self {
            Interval::Minute => "minute",
            Interval::Minute3 => "3minute",
            Interval::Minute5 => "5minute",
            Interval::Minute10 => "10minute",
            Interval::Minute15 => "15minute",
            Interval::Minute30 => "30minute",
            Interval::Minute60 => "60minute",
            Interval::Day => "day",
        }
    }

    /// Number of candle bars in a single Indian trading session (9:15 to 15:30 = 375 minutes).
    pub fn bars_per_day(&self) -> usize {
        match self {
            Interval::Minute => 375,
            Interval::Minute3 => 125,
            Interval::Minute5 => 75,
            Interval::Minute10 => 38,
            Interval::Minute15 => 25,
            Interval::Minute30 => 13,
            Interval::Minute60 => 7,
            Interval::Day => 1,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Bar tests --

    #[test]
    fn bar_creation_and_field_access() {
        let bar = Bar {
            timestamp_ms: 1_700_000_000_000,
            symbol: "RELIANCE".into(),
            open: 2450.0,
            high: 2475.0,
            low: 2440.0,
            close: 2465.0,
            volume: 1_000_000,
            oi: 0,
        };

        assert_eq!(bar.timestamp_ms, 1_700_000_000_000);
        assert_eq!(bar.symbol, "RELIANCE");
        assert!((bar.open - 2450.0).abs() < f64::EPSILON);
        assert!((bar.high - 2475.0).abs() < f64::EPSILON);
        assert!((bar.low - 2440.0).abs() < f64::EPSILON);
        assert!((bar.close - 2465.0).abs() < f64::EPSILON);
        assert_eq!(bar.volume, 1_000_000);
        assert_eq!(bar.oi, 0);
    }

    #[test]
    fn bar_clone_is_independent() {
        let bar = Bar {
            timestamp_ms: 100,
            symbol: "INFY".into(),
            open: 1.0,
            high: 2.0,
            low: 0.5,
            close: 1.5,
            volume: 10,
            oi: 0,
        };
        let bar2 = bar.clone();
        assert_eq!(bar.symbol, bar2.symbol);
        assert_eq!(bar.timestamp_ms, bar2.timestamp_ms);
    }

    // -- Signal tests --

    #[test]
    fn signal_market_buy() {
        let sig = Signal::market_buy("RELIANCE", 10);
        assert_eq!(sig.action, Action::Buy);
        assert_eq!(sig.symbol, "RELIANCE");
        assert_eq!(sig.quantity, 10);
        assert_eq!(sig.order_type, OrderType::Market);
        assert!((sig.limit_price - 0.0).abs() < f64::EPSILON);
        assert!((sig.stop_price - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn signal_market_sell() {
        let sig = Signal::market_sell("INFY", 5);
        assert_eq!(sig.action, Action::Sell);
        assert_eq!(sig.symbol, "INFY");
        assert_eq!(sig.quantity, 5);
        assert_eq!(sig.order_type, OrderType::Market);
    }

    // -- Position tests --

    #[test]
    fn position_market_value() {
        let pos = Position {
            symbol: "TCS".into(),
            quantity: 10,
            avg_price: 3400.0,
            current_price: 3500.0,
        };
        assert!((pos.market_value() - 35_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn position_unrealized_pnl_profit() {
        let pos = Position {
            symbol: "TCS".into(),
            quantity: 10,
            avg_price: 3400.0,
            current_price: 3500.0,
        };
        // pnl = 10 * (3500 - 3400) = 1000
        assert!((pos.unrealized_pnl() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn position_unrealized_pnl_loss() {
        let pos = Position {
            symbol: "TCS".into(),
            quantity: 10,
            avg_price: 3500.0,
            current_price: 3400.0,
        };
        // pnl = 10 * (3400 - 3500) = -1000
        assert!((pos.unrealized_pnl() - (-1000.0)).abs() < f64::EPSILON);
    }

    // -- Portfolio tests --

    #[test]
    fn portfolio_equity_cash_only() {
        let portfolio = Portfolio {
            cash: 100_000.0,
            positions: vec![],
        };
        assert!((portfolio.equity() - 100_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn portfolio_equity_with_positions() {
        let portfolio = Portfolio {
            cash: 50_000.0,
            positions: vec![
                Position {
                    symbol: "RELIANCE".into(),
                    quantity: 10,
                    avg_price: 2400.0,
                    current_price: 2500.0,
                },
                Position {
                    symbol: "INFY".into(),
                    quantity: 5,
                    avg_price: 1400.0,
                    current_price: 1500.0,
                },
            ],
        };
        // equity = 50_000 + (10*2500) + (5*1500) = 50_000 + 25_000 + 7_500 = 82_500
        assert!((portfolio.equity() - 82_500.0).abs() < f64::EPSILON);
    }

    // -- Interval tests --

    #[test]
    fn interval_as_kite_str() {
        assert_eq!(Interval::Minute.as_kite_str(), "minute");
        assert_eq!(Interval::Day.as_kite_str(), "day");
    }

    #[test]
    fn interval_all_kite_str_variants() {
        assert_eq!(Interval::Minute.as_kite_str(), "minute");
        assert_eq!(Interval::Minute3.as_kite_str(), "3minute");
        assert_eq!(Interval::Minute5.as_kite_str(), "5minute");
        assert_eq!(Interval::Minute10.as_kite_str(), "10minute");
        assert_eq!(Interval::Minute15.as_kite_str(), "15minute");
        assert_eq!(Interval::Minute30.as_kite_str(), "30minute");
        assert_eq!(Interval::Minute60.as_kite_str(), "60minute");
        assert_eq!(Interval::Day.as_kite_str(), "day");
    }

    #[test]
    fn interval_bars_per_day() {
        assert_eq!(Interval::Minute.bars_per_day(), 375);
        assert_eq!(Interval::Minute3.bars_per_day(), 125);
        assert_eq!(Interval::Minute5.bars_per_day(), 75);
        assert_eq!(Interval::Minute10.bars_per_day(), 38);
        assert_eq!(Interval::Minute15.bars_per_day(), 25);
        assert_eq!(Interval::Minute30.bars_per_day(), 13);
        assert_eq!(Interval::Minute60.bars_per_day(), 7);
        assert_eq!(Interval::Day.bars_per_day(), 1);
    }
}
