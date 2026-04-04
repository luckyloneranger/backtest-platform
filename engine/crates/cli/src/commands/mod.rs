pub mod data;
pub mod results;
pub mod run;

/// Parse an interval string into the Interval enum. Supports all Kite Connect timeframes.
pub fn parse_interval(s: &str) -> anyhow::Result<backtest_core::types::Interval> {
    use backtest_core::types::Interval;
    match s {
        "minute" => Ok(Interval::Minute),
        "3minute" => Ok(Interval::Minute3),
        "5minute" => Ok(Interval::Minute5),
        "10minute" => Ok(Interval::Minute10),
        "15minute" => Ok(Interval::Minute15),
        "30minute" => Ok(Interval::Minute30),
        "60minute" => Ok(Interval::Minute60),
        "day" => Ok(Interval::Day),
        _ => anyhow::bail!(
            "unsupported interval '{}'. Use: minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day",
            s
        ),
    }
}
