use std::collections::BTreeMap;

use chrono::Datelike;
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

    let downside_dev = (downside_sq.iter().sum::<f64>() / daily_returns.len() as f64).sqrt();
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
    #[serde(default)]
    pub avg_duration_ms: f64,
    #[serde(default)]
    pub min_duration_ms: i64,
    #[serde(default)]
    pub max_duration_ms: i64,
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
            avg_duration_ms: 0.0,
            min_duration_ms: 0,
            max_duration_ms: 0,
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

    // Trade duration statistics
    let durations: Vec<i64> = trades
        .iter()
        .map(|t| t.exit_timestamp_ms - t.entry_timestamp_ms)
        .collect();
    let avg_duration_ms = durations.iter().sum::<i64>() as f64 / total_trades as f64;
    let min_duration_ms = *durations.iter().min().unwrap();
    let max_duration_ms = *durations.iter().max().unwrap();

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
        avg_duration_ms,
        min_duration_ms,
        max_duration_ms,
    }
}

// ── Per-Symbol Breakdown ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMetrics {
    pub symbol: String,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_pnl: f64,
}

// ── Monthly Returns ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyReturn {
    pub year: i32,
    pub month: u32,
    pub return_pct: f64,
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
    #[serde(default)]
    pub per_symbol: Vec<SymbolMetrics>,
    #[serde(default)]
    pub monthly_returns: Vec<MonthlyReturn>,
    #[serde(default)]
    pub benchmark_return_pct: f64,
    #[serde(default)]
    pub alpha_pct: f64,
}

impl MetricsReport {
    /// Compute all performance metrics from a completed backtest result.
    ///
    /// 1. Groups equity curve points by day (takes last equity per day).
    /// 2. Computes daily returns between consecutive days.
    /// 3. Calculates ratio-based metrics (Sharpe, Sortino, Calmar, etc.).
    /// 4. Computes trade statistics (including duration stats).
    /// 5. Computes per-symbol breakdown.
    /// 6. Computes monthly returns from equity curve.
    /// 7. Computes benchmark comparison (alpha).
    /// 8. Assembles the full report.
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

        // 3. Derive actual calendar days from equity curve timestamps
        let days = if result.equity_curve.len() >= 2 {
            let first_ts = result.equity_curve.first().unwrap().timestamp_ms;
            let last_ts = result.equity_curve.last().unwrap().timestamp_ms;
            ((last_ts - first_ts) / 86_400_000).max(0) as u32
        } else {
            0
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

        // 5. Compute trade statistics (includes duration stats)
        let trade_stats = trade_statistics(&result.trades);

        // 6. Compute per-symbol breakdown
        let per_symbol = compute_per_symbol(&result.trades);

        // 7. Compute monthly returns from equity curve
        let monthly_returns = compute_monthly_returns(&result.equity_curve);

        // 8. Compute benchmark comparison
        let benchmark_return_pct = result.benchmark_return_pct.unwrap_or(0.0);
        let alpha_pct = total_return_pct - benchmark_return_pct;

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
            per_symbol,
            monthly_returns,
            benchmark_return_pct,
            alpha_pct,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Compute per-symbol metrics by grouping closed trades by symbol.
fn compute_per_symbol(trades: &[ClosedTrade]) -> Vec<SymbolMetrics> {
    let mut grouped: BTreeMap<String, Vec<&ClosedTrade>> = BTreeMap::new();
    for trade in trades {
        grouped
            .entry(trade.symbol.clone())
            .or_default()
            .push(trade);
    }

    grouped
        .into_iter()
        .map(|(symbol, symbol_trades)| {
            let total_trades = symbol_trades.len();
            let winning_trades = symbol_trades.iter().filter(|t| t.pnl > 0.0).count();
            let losing_trades = symbol_trades.iter().filter(|t| t.pnl < 0.0).count();
            let win_rate = if total_trades > 0 {
                winning_trades as f64 / total_trades as f64
            } else {
                0.0
            };
            let total_pnl: f64 = symbol_trades.iter().map(|t| t.pnl).sum();
            let avg_pnl = if total_trades > 0 {
                total_pnl / total_trades as f64
            } else {
                0.0
            };
            SymbolMetrics {
                symbol,
                total_trades,
                winning_trades,
                losing_trades,
                win_rate,
                total_pnl,
                avg_pnl,
            }
        })
        .collect()
}

/// Compute monthly returns from the equity curve.
///
/// Groups equity curve points by YYYY-MM and computes the return for each
/// month as `(last_equity - first_equity) / first_equity`.
fn compute_monthly_returns(curve: &[EquityPoint]) -> Vec<MonthlyReturn> {
    if curve.is_empty() {
        return Vec::new();
    }

    // Group by (year, month). Use a BTreeMap for sorted output.
    let mut month_map: BTreeMap<(i32, u32), (f64, f64)> = BTreeMap::new();

    for pt in curve {
        // Convert timestamp_ms to a NaiveDateTime to extract year/month
        let secs = pt.timestamp_ms / 1000;
        let nsecs = ((pt.timestamp_ms % 1000) * 1_000_000) as u32;
        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nsecs) {
            let key = (dt.year(), dt.month());
            let entry = month_map.entry(key).or_insert((pt.equity, pt.equity));
            // Always update the last equity (curve is time-ordered)
            entry.1 = pt.equity;
        }
    }

    month_map
        .into_iter()
        .map(|((year, month), (first_eq, last_eq))| {
            let return_pct = if first_eq > 0.0 {
                (last_eq - first_eq) / first_eq
            } else {
                0.0
            };
            MonthlyReturn {
                year,
                month,
                return_pct,
            }
        })
        .collect()
}

/// Compute the mean of a slice of f64 values. Returns 0.0 for empty input.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute the sample standard deviation of a slice. Returns 0.0 if fewer
/// than 2 values. Uses Bessel's correction (N-1 denominator).
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance =
        values.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
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
        assert_eq!(stats.avg_duration_ms, 0.0);
        assert_eq!(stats.min_duration_ms, 0);
        assert_eq!(stats.max_duration_ms, 0);
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
        // sample std_dev of [2, 4, 4, 4, 5, 5, 7, 9]:
        // mean = 5, sum_sq_dev = 32, sample_var = 32/7, sample_sd = sqrt(32/7)
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let expected = (32.0_f64 / 7.0).sqrt();
        assert!((std_dev(&vals) - expected).abs() < 1e-10);
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
                lookback_window: 200,
                max_volume_pct: 1.0,
            },
            custom_metrics: serde_json::json!({}),
            benchmark_return_pct: None,
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

    // ── test per-symbol metrics ────────────────────────────────────────

    #[test]
    fn test_per_symbol_metrics() {
        // 4 trades across 2 symbols:
        // RELIANCE: 2 trades (+1000, -200) => 2 total, 1 win, 1 loss, win_rate=0.5, total_pnl=800, avg_pnl=400
        // INFY: 2 trades (+500, +300) => 2 total, 2 wins, 0 losses, win_rate=1.0, total_pnl=800, avg_pnl=400
        let trades = vec![
            ClosedTrade {
                symbol: "RELIANCE".into(),
                pnl: 1000.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                symbol: "RELIANCE".into(),
                pnl: -200.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                symbol: "INFY".into(),
                pnl: 500.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                symbol: "INFY".into(),
                pnl: 300.0,
                ..ClosedTrade::default()
            },
        ];

        let per_symbol = compute_per_symbol(&trades);
        assert_eq!(per_symbol.len(), 2);

        // BTreeMap ordering: INFY comes before RELIANCE
        let infy = &per_symbol[0];
        assert_eq!(infy.symbol, "INFY");
        assert_eq!(infy.total_trades, 2);
        assert_eq!(infy.winning_trades, 2);
        assert_eq!(infy.losing_trades, 0);
        assert!((infy.win_rate - 1.0).abs() < 1e-6);
        assert!((infy.total_pnl - 800.0).abs() < 1e-6);
        assert!((infy.avg_pnl - 400.0).abs() < 1e-6);

        let rel = &per_symbol[1];
        assert_eq!(rel.symbol, "RELIANCE");
        assert_eq!(rel.total_trades, 2);
        assert_eq!(rel.winning_trades, 1);
        assert_eq!(rel.losing_trades, 1);
        assert!((rel.win_rate - 0.5).abs() < 1e-6);
        assert!((rel.total_pnl - 800.0).abs() < 1e-6);
        assert!((rel.avg_pnl - 400.0).abs() < 1e-6);
    }

    // ── test monthly returns ───────────────────────────────────────────

    #[test]
    fn test_monthly_returns() {
        // Equity curve spanning 3 months:
        // Jan 2024: equity goes from 100_000 to 105_000 => return = 5%
        // Feb 2024: equity goes from 105_000 to 110_000 => return ~4.76%
        // Mar 2024: equity goes from 110_000 to 108_000 => return ~-1.82%

        // Jan 1 2024 00:00 UTC = 1704067200000 ms
        let jan_1 = 1_704_067_200_000_i64;
        let feb_1 = 1_706_745_600_000_i64; // Feb 1 2024 00:00 UTC
        let mar_1 = 1_709_251_200_000_i64; // Mar 1 2024 00:00 UTC

        let curve = vec![
            EquityPoint { timestamp_ms: jan_1, equity: 100_000.0 },
            EquityPoint { timestamp_ms: jan_1 + 86_400_000, equity: 102_000.0 },
            EquityPoint { timestamp_ms: jan_1 + 86_400_000 * 15, equity: 105_000.0 },
            EquityPoint { timestamp_ms: feb_1, equity: 105_000.0 },
            EquityPoint { timestamp_ms: feb_1 + 86_400_000 * 14, equity: 110_000.0 },
            EquityPoint { timestamp_ms: mar_1, equity: 110_000.0 },
            EquityPoint { timestamp_ms: mar_1 + 86_400_000 * 10, equity: 108_000.0 },
        ];

        let monthly = compute_monthly_returns(&curve);
        assert_eq!(monthly.len(), 3);

        // January 2024
        assert_eq!(monthly[0].year, 2024);
        assert_eq!(monthly[0].month, 1);
        // return = (105_000 - 100_000) / 100_000 = 0.05
        assert!(
            (monthly[0].return_pct - 0.05).abs() < 1e-6,
            "Jan return should be 5%, got {}",
            monthly[0].return_pct
        );

        // February 2024
        assert_eq!(monthly[1].year, 2024);
        assert_eq!(monthly[1].month, 2);
        // return = (110_000 - 105_000) / 105_000 = 0.047619...
        let expected_feb = (110_000.0 - 105_000.0) / 105_000.0;
        assert!(
            (monthly[1].return_pct - expected_feb).abs() < 1e-6,
            "Feb return should be ~4.76%, got {}",
            monthly[1].return_pct
        );

        // March 2024
        assert_eq!(monthly[2].year, 2024);
        assert_eq!(monthly[2].month, 3);
        // return = (108_000 - 110_000) / 110_000 = -0.01818...
        let expected_mar = (108_000.0 - 110_000.0) / 110_000.0;
        assert!(
            (monthly[2].return_pct - expected_mar).abs() < 1e-6,
            "Mar return should be ~-1.82%, got {}",
            monthly[2].return_pct
        );
    }

    // ── test trade duration statistics ─────────────────────────────────

    #[test]
    fn test_trade_duration_stats() {
        // 3 trades with known timestamps:
        // Trade 1: entry=1000, exit=5000  => duration=4000ms
        // Trade 2: entry=2000, exit=12000 => duration=10000ms
        // Trade 3: entry=3000, exit=6000  => duration=3000ms
        let trades = vec![
            ClosedTrade {
                entry_timestamp_ms: 1000,
                exit_timestamp_ms: 5000,
                pnl: 100.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                entry_timestamp_ms: 2000,
                exit_timestamp_ms: 12000,
                pnl: -50.0,
                ..ClosedTrade::default()
            },
            ClosedTrade {
                entry_timestamp_ms: 3000,
                exit_timestamp_ms: 6000,
                pnl: 200.0,
                ..ClosedTrade::default()
            },
        ];

        let stats = trade_statistics(&trades);

        // avg_duration = (4000 + 10000 + 3000) / 3 = 17000/3 = 5666.666...
        let expected_avg = 17_000.0 / 3.0;
        assert!(
            (stats.avg_duration_ms - expected_avg).abs() < 1e-6,
            "avg_duration_ms should be ~5666.67, got {}",
            stats.avg_duration_ms
        );
        assert_eq!(stats.min_duration_ms, 3000);
        assert_eq!(stats.max_duration_ms, 10000);
    }

    #[test]
    fn test_empty_trades_duration() {
        let stats = trade_statistics(&[]);
        assert_eq!(stats.avg_duration_ms, 0.0);
        assert_eq!(stats.min_duration_ms, 0);
        assert_eq!(stats.max_duration_ms, 0);
    }
}
