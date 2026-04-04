use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::BacktestResult;
use crate::portfolio::{ClosedTrade, EquityPoint};

// ── Sharpe Ratio ────────────────────────────────────────────────────────────

/// Annualized Sharpe ratio.
///
/// sharpe = (mean_daily_return - daily_risk_free) / std_dev_daily_returns * sqrt(252)
///
/// Returns 0.0 if there are fewer than 2 returns or if std deviation is zero.
pub fn calculate_sharpe(daily_returns: &[f64], annual_risk_free_rate: f64) -> f64 {
    if daily_returns.len() < 2 {
        return 0.0;
    }
    let daily_rf = annual_risk_free_rate / 252.0;
    let mean_ret = mean(daily_returns);
    let sd = std_dev(daily_returns);
    if sd == 0.0 {
        return 0.0;
    }
    (mean_ret - daily_rf) / sd * (252.0_f64).sqrt()
}

// ── Sortino Ratio ───────────────────────────────────────────────────────────

/// Sortino ratio -- like Sharpe but only penalizes downside deviation.
///
/// sortino = (mean_daily_return - daily_risk_free) / downside_deviation * sqrt(252)
///
/// Returns 0.0 if there are fewer than 2 returns or if downside deviation is zero.
pub fn calculate_sortino(daily_returns: &[f64], annual_risk_free_rate: f64) -> f64 {
    if daily_returns.len() < 2 {
        return 0.0;
    }
    let daily_rf = annual_risk_free_rate / 252.0;
    let mean_ret = mean(daily_returns);

    // Downside deviation: sqrt(mean of squared negative excess returns)
    let downside_sq: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r < daily_rf)
        .map(|&r| (r - daily_rf).powi(2))
        .collect();

    if downside_sq.is_empty() {
        return 0.0;
    }

    let downside_dev = (downside_sq.iter().sum::<f64>() / downside_sq.len() as f64).sqrt();
    if downside_dev == 0.0 {
        return 0.0;
    }
    (mean_ret - daily_rf) / downside_dev * (252.0_f64).sqrt()
}

// ── Max Drawdown ────────────────────────────────────────────────────────────

/// Maximum drawdown percentage and the indices of the peak and trough.
///
/// Returns (max_dd_pct, peak_index, trough_index).  max_dd_pct is a positive
/// fraction (e.g. 0.1364 for 13.64%).  Returns (0.0, 0, 0) for empty input.
pub fn max_drawdown(equity_values: &[f64]) -> (f64, usize, usize) {
    if equity_values.is_empty() {
        return (0.0, 0, 0);
    }

    let mut peak = equity_values[0];
    let mut peak_idx = 0;
    let mut max_dd = 0.0;
    let mut max_dd_peak_idx = 0;
    let mut max_dd_trough_idx = 0;

    for (i, &val) in equity_values.iter().enumerate() {
        if val > peak {
            peak = val;
            peak_idx = i;
        }
        let dd = (peak - val) / peak;
        if dd > max_dd {
            max_dd = dd;
            max_dd_peak_idx = peak_idx;
            max_dd_trough_idx = i;
        }
    }

    (max_dd, max_dd_peak_idx, max_dd_trough_idx)
}

/// Duration of the longest drawdown period (in number of bars).
///
/// A drawdown period starts when equity drops below a previous peak and ends
/// when equity reaches a new high.  Returns 0 for empty input.
pub fn max_drawdown_duration(equity_values: &[f64]) -> usize {
    if equity_values.is_empty() {
        return 0;
    }

    let mut peak = equity_values[0];
    let mut dd_start: Option<usize> = None;
    let mut longest = 0_usize;

    for (i, &val) in equity_values.iter().enumerate() {
        if val >= peak {
            if let Some(start) = dd_start {
                let dur = i - start;
                if dur > longest {
                    longest = dur;
                }
                dd_start = None;
            }
            peak = val;
        } else if dd_start.is_none() {
            // Drawdown just started -- the peak was at the bar just before this one.
            dd_start = Some(i - 1);
        }
    }

    // If still in a drawdown at the end
    if let Some(start) = dd_start {
        let dur = equity_values.len() - 1 - start;
        if dur > longest {
            longest = dur;
        }
    }

    longest
}

// ── CAGR ────────────────────────────────────────────────────────────────────

/// Compound Annual Growth Rate.
///
/// cagr = (end_value / start_value)^(365/days) - 1
///
/// Returns 0.0 if days == 0 or start_value <= 0.
pub fn calculate_cagr(start_value: f64, end_value: f64, days: u32) -> f64 {
    if days == 0 || start_value <= 0.0 {
        return 0.0;
    }
    (end_value / start_value).powf(365.0 / days as f64) - 1.0
}

// ── Volatility ──────────────────────────────────────────────────────────────

/// Annualized volatility = std_dev(daily_returns) * sqrt(252).
///
/// Returns 0.0 if fewer than 2 returns.
pub fn calculate_volatility(daily_returns: &[f64]) -> f64 {
    if daily_returns.len() < 2 {
        return 0.0;
    }
    std_dev(daily_returns) * (252.0_f64).sqrt()
}

// ── Calmar Ratio ────────────────────────────────────────────────────────────

/// Calmar ratio = CAGR / abs(max_drawdown).
///
/// Returns 0.0 if max_dd is zero.
pub fn calculate_calmar(cagr: f64, max_dd: f64) -> f64 {
    if max_dd.abs() == 0.0 {
        return 0.0;
    }
    cagr / max_dd.abs()
}

// ── Trade Statistics ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeStatistics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64, // total_wins / total_losses
    pub avg_pnl: f64,
    pub total_pnl: f64,
    pub total_costs: f64,
}

pub fn trade_statistics(trades: &[ClosedTrade]) -> TradeStatistics {
    if trades.is_empty() {
        return TradeStatistics {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            avg_pnl: 0.0,
            total_pnl: 0.0,
            total_costs: 0.0,
        };
    }

    let total_trades = trades.len();
    let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
    let total_costs: f64 = trades.iter().map(|t| t.costs).sum();

    let winners: Vec<f64> = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).collect();
    let losers: Vec<f64> = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl).collect();

    let winning_trades = winners.len();
    let losing_trades = losers.len();

    let win_rate = winning_trades as f64 / total_trades as f64;

    let avg_win = if winning_trades > 0 {
        winners.iter().sum::<f64>() / winning_trades as f64
    } else {
        0.0
    };

    let avg_loss = if losing_trades > 0 {
        losers.iter().map(|l| l.abs()).sum::<f64>() / losing_trades as f64
    } else {
        0.0
    };

    let total_wins: f64 = winners.iter().sum();
    let total_losses: f64 = losers.iter().map(|l| l.abs()).sum();

    let profit_factor = if total_losses > 0.0 {
        total_wins / total_losses
    } else {
        0.0
    };

    let avg_pnl = total_pnl / total_trades as f64;

    TradeStatistics {
        total_trades,
        winning_trades,
        losing_trades,
        win_rate,
        avg_win,
        avg_loss,
        profit_factor,
        avg_pnl,
        total_pnl,
        total_costs,
    }
}

// ── Metrics Report ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub total_return_pct: f64,
    pub cagr: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown_pct: f64,
    pub max_drawdown_duration: usize,
    pub volatility: f64,
    pub trade_stats: TradeStatistics,
    pub final_equity: f64,
    pub initial_capital: f64,
}

impl MetricsReport {
    /// Compute all performance metrics from a completed backtest result.
    ///
    /// 1. Groups equity curve points by day (takes last equity per day).
    /// 2. Computes daily returns between consecutive days.
    /// 3. Calculates ratio-based metrics (Sharpe, Sortino, Calmar, etc.).
    /// 4. Computes trade statistics.
    /// 5. Assembles the full report.
    pub fn compute(result: &BacktestResult) -> Self {
        let initial_capital = result.initial_capital;
        let final_equity = result.final_equity;
        let total_return_pct = if initial_capital > 0.0 {
            (final_equity - initial_capital) / initial_capital
        } else {
            0.0
        };

        // 1. Extract daily equity values from the intraday equity curve
        let daily_eq = daily_equity_values(&result.equity_curve);

        // 2. Compute daily returns
        let daily_rets = daily_returns_from_equity(&daily_eq);

        // 3. Compute the number of calendar days from config
        let days = {
            let start =
                chrono::NaiveDate::parse_from_str(&result.config.start_date, "%Y-%m-%d");
            let end =
                chrono::NaiveDate::parse_from_str(&result.config.end_date, "%Y-%m-%d");
            match (start, end) {
                (Ok(s), Ok(e)) => (e - s).num_days().max(0) as u32,
                _ => 0,
            }
        };

        // 4. Compute ratio metrics (use 0% risk-free rate by default)
        let annual_rf = 0.0;
        let sharpe_ratio = calculate_sharpe(&daily_rets, annual_rf);
        let sortino_ratio = calculate_sortino(&daily_rets, annual_rf);
        let volatility = calculate_volatility(&daily_rets);

        let cagr = calculate_cagr(initial_capital, final_equity, days);

        let (max_dd_pct, _, _) = max_drawdown(&daily_eq);
        let max_dd_dur = max_drawdown_duration(&daily_eq);
        let calmar_ratio = calculate_calmar(cagr, max_dd_pct);

        // 5. Compute trade statistics
        let trade_stats = trade_statistics(&result.trades);

        Self {
            total_return_pct,
            cagr,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown_pct: max_dd_pct,
            max_drawdown_duration: max_dd_dur,
            volatility,
            trade_stats,
            final_equity,
            initial_capital,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Compute the mean of a slice of f64 values. Returns 0.0 for empty input.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute the population standard deviation of a slice. Returns 0.0 if fewer
/// than 2 values.
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Extract daily equity values from an equity curve by grouping points by
/// calendar day and keeping the last value per day.
fn daily_equity_values(curve: &[EquityPoint]) -> Vec<f64> {
    if curve.is_empty() {
        return Vec::new();
    }

    // Group by calendar day (using ms -> day number).
    // Use a BTreeMap so results are ordered by day.
    let mut day_map: BTreeMap<i64, f64> = BTreeMap::new();
    for pt in curve {
        let day = pt.timestamp_ms / 86_400_000; // ms per day
        // Always overwrite -- the last entry per day wins (curve is time-ordered)
        day_map.insert(day, pt.equity);
    }

    day_map.values().copied().collect()
}

/// Compute daily returns from a series of equity values.
/// return_i = (equity_i - equity_{i-1}) / equity_{i-1}
fn daily_returns_from_equity(equity_values: &[f64]) -> Vec<f64> {
    if equity_values.len() < 2 {
        return Vec::new();
    }
    equity_values
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_sharpe_ratio ───────────────────────────────────────────────

    #[test]
    fn test_sharpe_ratio() {
        // Known daily returns: positive on average
        let daily_returns = vec![0.01, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012];
        let sharpe = calculate_sharpe(&daily_returns, 0.06); // 6% annual risk-free
        assert!(sharpe > 0.0, "Sharpe should be positive for net positive returns");
        // With these returns the mean return is ~0.826%/day which is well above risk-free
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        assert_eq!(calculate_sharpe(&[], 0.06), 0.0);
    }

    #[test]
    fn test_sharpe_ratio_single_return() {
        assert_eq!(calculate_sharpe(&[0.01], 0.06), 0.0);
    }

    #[test]
    fn test_sharpe_ratio_constant_returns() {
        // All same returns => std_dev = 0 => sharpe = 0
        let daily_returns = vec![0.01, 0.01, 0.01, 0.01];
        assert_eq!(calculate_sharpe(&daily_returns, 0.0), 0.0);
    }

    // ── test_sortino_ratio ──────────────────────────────────────────────

    #[test]
    fn test_sortino_ratio() {
        // Mix of positive and negative returns
        let daily_returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003];
        let sortino = calculate_sortino(&daily_returns, 0.06);
        assert!(sortino > 0.0, "Sortino should be positive for net positive returns");
        assert!(sortino.is_finite());

        // Sortino should be >= Sharpe because it only penalizes downside
        let sharpe = calculate_sharpe(&daily_returns, 0.06);
        assert!(
            sortino >= sharpe,
            "Sortino ({sortino}) should be >= Sharpe ({sharpe}) for mixed returns"
        );
    }

    #[test]
    fn test_sortino_no_downside() {
        // All positive returns => downside deviation = 0 => sortino = 0
        let daily_returns = vec![0.01, 0.02, 0.03, 0.005];
        let sortino = calculate_sortino(&daily_returns, 0.0);
        // When risk-free is 0 and all returns are positive, no downside exists
        assert_eq!(sortino, 0.0);
    }

    // ── test_max_drawdown ───────────────────────────────────────────────

    #[test]
    fn test_max_drawdown() {
        // equity curve: [100, 110, 105, 95, 100, 115]
        // peak = 110 at index 1, trough = 95 at index 3
        // max DD = (110 - 95) / 110 = 15/110 = 0.13636...
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0, 115.0];
        let (dd, peak_idx, trough_idx) = max_drawdown(&equity);

        let expected_dd = (110.0 - 95.0) / 110.0;
        assert!(
            (dd - expected_dd).abs() < 1e-6,
            "Max DD should be ~13.64%, got {dd}"
        );
        assert_eq!(peak_idx, 1);
        assert_eq!(trough_idx, 3);
    }

    #[test]
    fn test_max_drawdown_empty() {
        let (dd, p, t) = max_drawdown(&[]);
        assert_eq!(dd, 0.0);
        assert_eq!(p, 0);
        assert_eq!(t, 0);
    }

    #[test]
    fn test_max_drawdown_monotonically_increasing() {
        // No drawdown at all
        let equity = vec![100.0, 110.0, 120.0, 130.0];
        let (dd, _, _) = max_drawdown(&equity);
        assert_eq!(dd, 0.0);
    }

    // ── test_max_drawdown_duration ──────────────────────────────────────

    #[test]
    fn test_max_drawdown_duration() {
        // equity: [100, 110, 105, 95, 100, 115]
        // drawdown starts after index 1 (peak=110), ends at index 5 (new high=115)
        // duration = 5 - 1 = 4 bars
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0, 115.0];
        let dur = max_drawdown_duration(&equity);
        assert_eq!(dur, 4);
    }

    #[test]
    fn test_max_drawdown_duration_no_drawdown() {
        let equity = vec![100.0, 110.0, 120.0, 130.0];
        assert_eq!(max_drawdown_duration(&equity), 0);
    }

    // ── test_cagr ───────────────────────────────────────────────────────

    #[test]
    fn test_cagr() {
        // 100K to 150K over 365 days = 50% CAGR
        let cagr = calculate_cagr(100_000.0, 150_000.0, 365);
        assert!(
            (cagr - 0.5).abs() < 1e-6,
            "CAGR should be 50%, got {cagr}"
        );
    }

    #[test]
    fn test_cagr_zero_days() {
        assert_eq!(calculate_cagr(100_000.0, 150_000.0, 0), 0.0);
    }

    #[test]
    fn test_cagr_zero_start() {
        assert_eq!(calculate_cagr(0.0, 150_000.0, 365), 0.0);
    }

    #[test]
    fn test_cagr_two_years() {
        // 100K to 121K over 730 days (2 years) => CAGR = sqrt(1.21) - 1 = 0.10
        let cagr = calculate_cagr(100_000.0, 121_000.0, 730);
        assert!(
            (cagr - 0.10).abs() < 1e-4,
            "CAGR should be ~10%, got {cagr}"
        );
    }

    // ── test_volatility ─────────────────────────────────────────────────

    #[test]
    fn test_volatility() {
        let daily_returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let vol = calculate_volatility(&daily_returns);
        assert!(vol > 0.0, "Volatility should be positive for varying returns");
        assert!(vol.is_finite());
    }

    #[test]
    fn test_volatility_constant() {
        // Constant returns => volatility = 0
        let daily_returns = vec![0.01, 0.01, 0.01, 0.01];
        let vol = calculate_volatility(&daily_returns);
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn test_volatility_empty() {
        assert_eq!(calculate_volatility(&[]), 0.0);
    }

    // ── test_calmar ─────────────────────────────────────────────────────

    #[test]
    fn test_calmar() {
        // CAGR = 20%, max DD = 10% => Calmar = 2.0
        let calmar = calculate_calmar(0.20, 0.10);
        assert!(
            (calmar - 2.0).abs() < 1e-6,
            "Calmar should be 2.0, got {calmar}"
        );
    }

    #[test]
    fn test_calmar_zero_dd() {
        assert_eq!(calculate_calmar(0.20, 0.0), 0.0);
    }

    // ── test_trade_statistics ───────────────────────────────────────────

    #[test]
    fn test_win_rate() {
        // 4 trades: 2 wins, 2 losses
        let trades = vec![
            ClosedTrade {
                pnl: 1000.0,
                costs: 50.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                pnl: -500.0,
                costs: 50.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                pnl: 800.0,
                costs: 30.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                pnl: -300.0,
                costs: 30.0,
                ..ClosedTrade::default()
            },
        ];

        let stats = trade_statistics(&trades);

        assert_eq!(stats.total_trades, 4);
        assert_eq!(stats.winning_trades, 2);
        assert_eq!(stats.losing_trades, 2);
        assert!((stats.win_rate - 0.5).abs() < 1e-6, "Win rate should be 0.5");

        // avg_win = (1000 + 800) / 2 = 900
        assert!((stats.avg_win - 900.0).abs() < 1e-6);
        // avg_loss = (500 + 300) / 2 = 400  (absolute)
        assert!((stats.avg_loss - 400.0).abs() < 1e-6);

        // profit_factor = total_wins / total_losses = 1800 / 800 = 2.25
        assert!(
            (stats.profit_factor - 2.25).abs() < 1e-6,
            "Profit factor should be 2.25, got {}",
            stats.profit_factor
        );

        // total_pnl = 1000 - 500 + 800 - 300 = 1000
        assert!((stats.total_pnl - 1000.0).abs() < 1e-6);

        // avg_pnl = 1000 / 4 = 250
        assert!((stats.avg_pnl - 250.0).abs() < 1e-6);

        // total_costs = 50 + 50 + 30 + 30 = 160
        assert!((stats.total_costs - 160.0).abs() < 1e-6);
    }

    #[test]
    fn test_trade_statistics_empty() {
        let stats = trade_statistics(&[]);
        assert_eq!(stats.total_trades, 0);
        assert_eq!(stats.winning_trades, 0);
        assert_eq!(stats.losing_trades, 0);
        assert_eq!(stats.win_rate, 0.0);
        assert_eq!(stats.avg_win, 0.0);
        assert_eq!(stats.avg_loss, 0.0);
        assert_eq!(stats.profit_factor, 0.0);
        assert_eq!(stats.avg_pnl, 0.0);
        assert_eq!(stats.total_pnl, 0.0);
        assert_eq!(stats.total_costs, 0.0);
    }

    // ── test helpers ────────────────────────────────────────────────────

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[]), 0.0);
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert!((mean(&[0.01, -0.01]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[5.0]), 0.0);
        // std_dev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&vals) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_daily_returns_from_equity() {
        let equity = vec![100.0, 110.0, 105.0, 115.0];
        let returns = daily_returns_from_equity(&equity);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.10).abs() < 1e-10); // (110-100)/100
        let expected_1 = (105.0 - 110.0) / 110.0;
        assert!((returns[1] - expected_1).abs() < 1e-10);
        let expected_2 = (115.0 - 105.0) / 105.0;
        assert!((returns[2] - expected_2).abs() < 1e-10);
    }

    #[test]
    fn test_daily_equity_values() {
        // Simulate intraday equity points across 2 days
        // Day 1: timestamps 0, 60_000, 120_000 (all within same day starting at epoch)
        // Day 2: timestamps 86_400_000, 86_460_000
        let curve = vec![
            EquityPoint { timestamp_ms: 0, equity: 100.0 },
            EquityPoint { timestamp_ms: 60_000, equity: 101.0 },
            EquityPoint { timestamp_ms: 120_000, equity: 102.0 },  // last of day 1
            EquityPoint { timestamp_ms: 86_400_000, equity: 103.0 },
            EquityPoint { timestamp_ms: 86_460_000, equity: 104.0 }, // last of day 2
        ];
        let daily = daily_equity_values(&curve);
        assert_eq!(daily.len(), 2);
        assert!((daily[0] - 102.0).abs() < 1e-10); // last of day 1
        assert!((daily[1] - 104.0).abs() < 1e-10); // last of day 2
    }

    #[test]
    fn test_daily_equity_values_empty() {
        assert!(daily_equity_values(&[]).is_empty());
    }

    // ── test MetricsReport::compute ─────────────────────────────────────

    #[test]
    fn test_metrics_report_compute() {
        use crate::config::BacktestConfig;
        use crate::types::Interval;

        // Build a minimal BacktestResult
        // 3 days of equity: 100K -> 105K -> 110K
        let equity_curve = vec![
            // Day 1 - multiple intraday points, last = 100_000
            EquityPoint { timestamp_ms: 0, equity: 99_500.0 },
            EquityPoint { timestamp_ms: 60_000, equity: 100_000.0 },
            // Day 2
            EquityPoint { timestamp_ms: 86_400_000, equity: 104_000.0 },
            EquityPoint { timestamp_ms: 86_460_000, equity: 105_000.0 },
            // Day 3
            EquityPoint { timestamp_ms: 172_800_000, equity: 109_000.0 },
            EquityPoint { timestamp_ms: 172_860_000, equity: 110_000.0 },
        ];

        let trades = vec![
            ClosedTrade {
                pnl: 5000.0,
                costs: 100.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                pnl: 5000.0,
                costs: 100.0,
                ..ClosedTrade::default()
            },
        ];

        let result = BacktestResult {
            trades,
            equity_curve,
            final_equity: 110_000.0,
            initial_capital: 100_000.0,
            config: BacktestConfig {
                strategy_name: "test".into(),
                symbols: vec!["TEST".into()],
                start_date: "2024-01-01".into(),
                end_date: "2024-01-03".into(),
                initial_capital: 100_000.0,
                interval: Interval::Minute,
                strategy_params: serde_json::json!({}),
                slippage_pct: 0.0,
                margin_available: None,
            },
            custom_metrics: serde_json::json!({}),
        };

        let report = MetricsReport::compute(&result);

        // total_return_pct = (110K - 100K) / 100K = 10%
        assert!(
            (report.total_return_pct - 0.10).abs() < 1e-4,
            "total_return_pct should be ~10%, got {}",
            report.total_return_pct
        );

        assert_eq!(report.final_equity, 110_000.0);
        assert_eq!(report.initial_capital, 100_000.0);
        assert!(report.sharpe_ratio.is_finite());
        assert!(report.volatility >= 0.0);
        assert_eq!(report.trade_stats.total_trades, 2);
        assert_eq!(report.trade_stats.winning_trades, 2);
    }
}
