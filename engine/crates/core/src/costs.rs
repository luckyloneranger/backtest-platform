use crate::types::InstrumentType;

// ── Trade parameters ────────────────────────────────────────────────────────

pub struct TradeParams {
    pub instrument_type: InstrumentType,
    pub is_intraday: bool,
    pub buy_value: f64,  // quantity * buy_price
    pub sell_value: f64,  // quantity * sell_price
    pub quantity: i32,
}

// ── Computed costs ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct TradeCosts {
    pub total_brokerage: f64,
    pub stt: f64,
    pub transaction_charges: f64,
    pub gst: f64,
    pub sebi_fees: f64,
    pub stamp_duty: f64,
}

impl TradeCosts {
    pub fn total(&self) -> f64 {
        self.total_brokerage
            + self.stt
            + self.transaction_charges
            + self.gst
            + self.sebi_fees
            + self.stamp_duty
    }
}

// ── Zerodha cost model ─────────────────────────────────────────────────────

pub struct ZerodhaCostModel;

impl ZerodhaCostModel {
    pub fn calculate(&self, params: &TradeParams) -> TradeCosts {
        let turnover = params.buy_value + params.sell_value;

        // 1. Brokerage
        let total_brokerage = match params.instrument_type {
            InstrumentType::Equity => {
                // Equity (both delivery and intraday): zero brokerage
                // Zerodha removed all equity brokerage fees
                0.0
            }
            InstrumentType::FutureFO | InstrumentType::OptionFO | InstrumentType::Commodity => {
                // Flat Rs 20 per executed order (per side)
                20.0
            }
        };

        // 2. STT (Securities Transaction Tax)
        let stt = match params.instrument_type {
            InstrumentType::Equity if !params.is_intraday => {
                // Delivery: 0.1% on both buy and sell
                0.001 * (params.buy_value + params.sell_value)
            }
            InstrumentType::Equity => {
                // Intraday: 0.025% on sell side only
                0.00025 * params.sell_value
            }
            InstrumentType::FutureFO => {
                // Futures: 0.02% on sell side only
                0.0002 * params.sell_value
            }
            InstrumentType::OptionFO => {
                // Options: 0.1% on sell side (on premium value)
                0.001 * params.sell_value
            }
            InstrumentType::Commodity => {
                // Commodities on MCX: CTT of 0.01% on sell side
                // (No STT on commodities, but CTT applies — keeping simple)
                0.0001 * params.sell_value
            }
        };

        // 3. Transaction charges (exchange)
        // NSE: 0.00345% of turnover on both sides
        let transaction_charges = 0.0000345 * turnover;

        // 4. GST: 18% on (brokerage + transaction charges)
        let gst = 0.18 * (total_brokerage + transaction_charges);

        // 5. SEBI turnover fees: Rs 10 per crore = 10 / 10_000_000
        let sebi_fees = turnover * (10.0 / 10_000_000.0);

        // 6. Stamp duty: 0.015% on buy side only
        let stamp_duty = 0.00015 * params.buy_value;

        TradeCosts {
            total_brokerage,
            stt,
            transaction_charges,
            gst,
            sebi_fees,
            stamp_duty,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 0.01; // allow 1 paisa tolerance

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // Test 1: Equity delivery costs
    // Buy 100 shares at 2500, sell at 2525
    #[test]
    fn test_equity_delivery_costs() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::Equity,
            is_intraday: false,
            buy_value: 100.0 * 2500.0,  // 250_000
            sell_value: 100.0 * 2525.0,  // 252_500
            quantity: 100,
        };

        let costs = model.calculate(&params);

        // Brokerage: 0 (delivery)
        assert!(
            approx_eq(costs.total_brokerage, 0.0),
            "Expected brokerage 0, got {}",
            costs.total_brokerage
        );

        // STT: 0.1% of (250_000 + 252_500) = 0.001 * 502_500 = 502.50
        assert!(
            approx_eq(costs.stt, 502.50),
            "Expected STT 502.50, got {}",
            costs.stt
        );

        // Total must be > 0
        assert!(costs.total() > 0.0, "Total costs must be > 0");
    }

    // Test 2: Equity intraday costs
    // Buy 100 at 2500, sell at 2525 (intraday)
    #[test]
    fn test_equity_intraday_costs() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::Equity,
            is_intraday: true,
            buy_value: 100.0 * 2500.0,  // 250_000
            sell_value: 100.0 * 2525.0,  // 252_500
            quantity: 100,
        };

        let costs = model.calculate(&params);

        // Brokerage: 0 (Zerodha zero brokerage on all equity trades)
        assert!(
            approx_eq(costs.total_brokerage, 0.0),
            "Expected brokerage 0, got {}",
            costs.total_brokerage
        );

        // STT: 0.025% of 252_500 (sell side only) = 63.125
        assert!(
            approx_eq(costs.stt, 63.125),
            "Expected STT 63.125, got {}",
            costs.stt
        );

        // Total must be > 0
        assert!(costs.total() > 0.0, "Total costs must be > 0");
    }

    // Test 3: F&O futures costs
    // Futures trade with buy_value=500_000, sell_value=510_000
    #[test]
    fn test_fo_futures_costs() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::FutureFO,
            is_intraday: false,
            buy_value: 500_000.0,
            sell_value: 510_000.0,
            quantity: 25,
        };

        let costs = model.calculate(&params);

        // Brokerage: Rs 20 per side
        assert!(
            approx_eq(costs.total_brokerage, 20.0),
            "Expected brokerage 20, got {}",
            costs.total_brokerage
        );

        // STT: 0.02% of 510_000 (sell) = 102
        assert!(
            approx_eq(costs.stt, 102.0),
            "Expected STT 102, got {}",
            costs.stt
        );

        // Verify total > 0
        assert!(costs.total() > 0.0, "Total costs must be > 0");
    }

    // Test 4: Zero brokerage for equity delivery
    #[test]
    fn test_zero_brokerage_delivery() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::Equity,
            is_intraday: false,
            buy_value: 100_000.0,
            sell_value: 105_000.0,
            quantity: 50,
        };

        let costs = model.calculate(&params);

        assert!(
            approx_eq(costs.total_brokerage, 0.0),
            "Equity delivery brokerage must be 0, got {}",
            costs.total_brokerage
        );
    }

    // Test 5: Option costs
    #[test]
    fn test_option_costs() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::OptionFO,
            is_intraday: false,
            buy_value: 50_000.0,   // premium paid
            sell_value: 60_000.0,  // premium received
            quantity: 25,
        };

        let costs = model.calculate(&params);

        // Brokerage: flat Rs 20 per side
        assert!(
            approx_eq(costs.total_brokerage, 20.0),
            "Expected brokerage 20, got {}",
            costs.total_brokerage
        );

        // STT: 0.1% on sell side = 0.001 * 60_000 = 60
        assert!(
            approx_eq(costs.stt, 60.0),
            "Expected STT 60, got {}",
            costs.stt
        );
    }

    // Test 6: GST calculation
    #[test]
    fn test_gst_calculation() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::Equity,
            is_intraday: true,
            buy_value: 100_000.0,
            sell_value: 100_000.0,
            quantity: 100,
        };

        let costs = model.calculate(&params);

        // turnover = 200_000
        // Transaction charges: 0.0000345 * 200_000 = 6.90
        let expected_txn = 0.0000345 * 200_000.0;

        // Brokerage: 0 (zero brokerage on equity)
        let expected_brokerage = 0.0;

        // GST: 18% of (brokerage + txn charges)
        let expected_gst = 0.18 * (expected_brokerage + expected_txn);

        assert!(
            approx_eq(costs.gst, expected_gst),
            "Expected GST {}, got {}",
            expected_gst,
            costs.gst
        );
    }

    // Test 7: SEBI fees and stamp duty
    #[test]
    fn test_sebi_fees_and_stamp_duty() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::FutureFO,
            is_intraday: false,
            buy_value: 1_000_000.0,
            sell_value: 1_010_000.0,
            quantity: 50,
        };

        let costs = model.calculate(&params);

        let turnover = 1_000_000.0 + 1_010_000.0;

        // SEBI fees: Rs 10 per crore = turnover * 10 / 10_000_000
        let expected_sebi = turnover * 10.0 / 10_000_000.0;
        assert!(
            approx_eq(costs.sebi_fees, expected_sebi),
            "Expected SEBI fees {}, got {}",
            expected_sebi,
            costs.sebi_fees
        );

        // Stamp duty: 0.015% on buy side = 0.00015 * 1_000_000 = 150
        let expected_stamp = 0.00015 * 1_000_000.0;
        assert!(
            approx_eq(costs.stamp_duty, expected_stamp),
            "Expected stamp duty {}, got {}",
            expected_stamp,
            costs.stamp_duty
        );
    }

    // Test 8: TradeCosts::total() sums all components
    #[test]
    fn test_trade_costs_total() {
        let costs = TradeCosts {
            total_brokerage: 40.0,
            stt: 100.0,
            transaction_charges: 10.0,
            gst: 9.0,
            sebi_fees: 0.20,
            stamp_duty: 15.0,
        };

        let expected = 40.0 + 100.0 + 10.0 + 9.0 + 0.20 + 15.0;
        assert!(
            approx_eq(costs.total(), expected),
            "Expected total {}, got {}",
            expected,
            costs.total()
        );
    }

    // Test 9: Commodity costs
    #[test]
    fn test_commodity_costs() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::Commodity,
            is_intraday: false,
            buy_value: 200_000.0,
            sell_value: 205_000.0,
            quantity: 10,
        };

        let costs = model.calculate(&params);

        // Brokerage: flat Rs 20 per side
        assert!(
            approx_eq(costs.total_brokerage, 20.0),
            "Expected brokerage 20, got {}",
            costs.total_brokerage
        );

        // CTT: 0.01% on sell side = 0.0001 * 205_000 = 20.50
        assert!(
            approx_eq(costs.stt, 20.50),
            "Expected CTT 20.50, got {}",
            costs.stt
        );

        assert!(costs.total() > 0.0, "Total costs must be > 0");
    }

    // Test 10: Equity intraday also has zero brokerage
    #[test]
    fn test_equity_intraday_zero_brokerage() {
        let model = ZerodhaCostModel;
        let params = TradeParams {
            instrument_type: InstrumentType::Equity,
            is_intraday: true,
            buy_value: 10_000.0,
            sell_value: 10_050.0,
            quantity: 10,
        };

        let costs = model.calculate(&params);

        // Zero brokerage on all equity trades
        assert!(
            approx_eq(costs.total_brokerage, 0.0),
            "Expected brokerage 0, got {}",
            costs.total_brokerage
        );
    }
}
