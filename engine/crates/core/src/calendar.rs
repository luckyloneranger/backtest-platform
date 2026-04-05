use std::collections::HashSet;

use chrono::{Datelike, NaiveDate};

/// NSE (National Stock Exchange of India) trading calendar.
///
/// Contains hardcoded market holidays from 2020 to 2027. Weekends (Saturday
/// and Sunday) are always non-trading days.
pub struct TradingCalendar {
    holidays: HashSet<NaiveDate>,
}

/// Helper to build a `NaiveDate` concisely; panics on invalid input (fine for
/// compile-time-known constants).
fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).expect("invalid date constant in calendar")
}

impl TradingCalendar {
    /// Create a calendar pre-loaded with all NSE holidays from 2020 through 2027.
    pub fn nse() -> Self {
        let mut holidays = HashSet::new();

        // ── 2020 ──────────────────────────────────────────────────────────
        for date in [
            d(2020, 2, 21),  // Maha Shivaratri
            d(2020, 3, 10),  // Holi
            d(2020, 4, 2),   // Ram Navami
            d(2020, 4, 6),   // Mahavir Jayanti
            d(2020, 4, 10),  // Good Friday
            d(2020, 4, 14),  // Dr Ambedkar Jayanti
            d(2020, 5, 1),   // May Day
            d(2020, 5, 25),  // Eid ul-Fitr
            d(2020, 8, 12),  // Janmashtami (Ashtami)
            d(2020, 8, 15),  // Independence Day (Saturday – observed, won't matter)
            d(2020, 10, 2),  // Mahatma Gandhi Jayanti
            d(2020, 10, 29), // Diwali / Laxmi Puja (Balipratipada)
            d(2020, 11, 16), // Guru Nanak Jayanti
            d(2020, 11, 30), // Guru Nanak Jayanti (observed)
            d(2020, 12, 25), // Christmas
        ] {
            holidays.insert(date);
        }

        // ── 2021 ──────────────────────────────────────────────────────────
        for date in [
            d(2021, 1, 26),  // Republic Day
            d(2021, 3, 11),  // Maha Shivaratri
            d(2021, 3, 29),  // Holi
            d(2021, 4, 2),   // Good Friday
            d(2021, 4, 14),  // Dr Ambedkar Jayanti
            d(2021, 4, 21),  // Ram Navami
            d(2021, 4, 25),  // Mahavir Jayanti
            d(2021, 5, 1),   // May Day (Saturday)
            d(2021, 5, 13),  // Eid ul-Fitr
            d(2021, 7, 21),  // Eid ul-Adha (Bakri Id)
            d(2021, 8, 19),  // Muharram
            d(2021, 8, 30),  // Janmashtami (Ashtami)
            d(2021, 10, 15), // Dussehra
            d(2021, 11, 4),  // Diwali / Laxmi Puja
            d(2021, 11, 5),  // Diwali (Balipratipada)
            d(2021, 11, 19), // Guru Nanak Jayanti
        ] {
            holidays.insert(date);
        }

        // ── 2022 ──────────────────────────────────────────────────────────
        for date in [
            d(2022, 1, 26),  // Republic Day
            d(2022, 3, 1),   // Maha Shivaratri
            d(2022, 3, 18),  // Holi
            d(2022, 4, 14),  // Dr Ambedkar Jayanti / Mahavir Jayanti
            d(2022, 4, 15),  // Good Friday
            d(2022, 5, 3),   // Eid ul-Fitr
            d(2022, 8, 9),   // Muharram
            d(2022, 8, 15),  // Independence Day
            d(2022, 8, 31),  // Ganesh Chaturthi
            d(2022, 10, 5),  // Dussehra
            d(2022, 10, 24), // Diwali / Laxmi Puja
            d(2022, 10, 26), // Diwali (Balipratipada)
            d(2022, 11, 8),  // Guru Nanak Jayanti
        ] {
            holidays.insert(date);
        }

        // ── 2023 ──────────────────────────────────────────────────────────
        for date in [
            d(2023, 1, 26),  // Republic Day
            d(2023, 3, 7),   // Holi
            d(2023, 3, 30),  // Ram Navami
            d(2023, 4, 4),   // Mahavir Jayanti
            d(2023, 4, 7),   // Good Friday
            d(2023, 4, 14),  // Dr Ambedkar Jayanti
            d(2023, 4, 22),  // Eid ul-Fitr (Saturday – included for completeness)
            d(2023, 5, 1),   // May Day
            d(2023, 6, 28),  // Eid ul-Adha (Bakri Id) (Wednesday)
            d(2023, 6, 29),  // Eid ul-Adha (observed)
            d(2023, 8, 15),  // Independence Day
            d(2023, 9, 19),  // Ganesh Chaturthi (Tuesday)
            d(2023, 10, 2),  // Mahatma Gandhi Jayanti
            d(2023, 10, 24), // Dussehra
            d(2023, 11, 14), // Diwali / Laxmi Puja (Tuesday)
            d(2023, 11, 13), // Diwali (Balipratipada) (Monday)
            d(2023, 11, 27), // Guru Nanak Jayanti
            d(2023, 12, 25), // Christmas
        ] {
            holidays.insert(date);
        }

        // ── 2024 ──────────────────────────────────────────────────────────
        // (from specification + official NSE list)
        for date in [
            d(2024, 1, 26),  // Republic Day
            d(2024, 3, 8),   // Maha Shivaratri
            d(2024, 3, 25),  // Holi
            d(2024, 3, 29),  // Good Friday
            d(2024, 4, 11),  // Eid ul-Fitr (Ramzan)
            d(2024, 4, 14),  // Dr Ambedkar Jayanti
            d(2024, 4, 17),  // Ram Navami
            d(2024, 4, 21),  // Mahavir Jayanti (Sunday – included for completeness)
            d(2024, 5, 1),   // May Day
            d(2024, 5, 23),  // Buddha Purnima
            d(2024, 6, 17),  // Eid ul-Adha (Bakri Id)
            d(2024, 7, 17),  // Muharram
            d(2024, 8, 15),  // Independence Day
            d(2024, 9, 16),  // Milad-un-Nabi (Prophet's Birthday)
            d(2024, 10, 2),  // Mahatma Gandhi Jayanti
            d(2024, 11, 1),  // Diwali / Laxmi Puja
            d(2024, 11, 15), // Guru Nanak Jayanti
            d(2024, 12, 25), // Christmas
        ] {
            holidays.insert(date);
        }

        // ── 2025 ──────────────────────────────────────────────────────────
        // (from specification + official NSE list)
        for date in [
            d(2025, 1, 26),  // Republic Day (Sunday – included for completeness)
            d(2025, 2, 26),  // Maha Shivaratri
            d(2025, 3, 14),  // Holi
            d(2025, 3, 31),  // Eid ul-Fitr (Ramzan)
            d(2025, 4, 10),  // Mahavir Jayanti
            d(2025, 4, 14),  // Dr Ambedkar Jayanti
            d(2025, 4, 18),  // Good Friday
            d(2025, 5, 1),   // May Day
            d(2025, 5, 12),  // Buddha Purnima
            d(2025, 6, 7),   // Eid ul-Adha (Bakri Id) (Saturday – included)
            d(2025, 8, 15),  // Independence Day
            d(2025, 8, 27),  // Janmashtami (Ashtami)
            d(2025, 9, 5),   // Milad-un-Nabi / Prophet's Birthday
            d(2025, 10, 2),  // Mahatma Gandhi Jayanti / Dussehra
            d(2025, 10, 21), // Diwali / Laxmi Puja
            d(2025, 10, 22), // Diwali (Balipratipada)
            d(2025, 11, 5),  // Guru Nanak Jayanti
            d(2025, 12, 25), // Christmas
        ] {
            holidays.insert(date);
        }

        // ── 2026 ──────────────────────────────────────────────────────────
        for date in [
            d(2026, 1, 26),  // Republic Day
            d(2026, 2, 17),  // Maha Shivaratri
            d(2026, 3, 4),   // Holi (approx)
            d(2026, 3, 20),  // Eid ul-Fitr (approx)
            d(2026, 4, 3),   // Good Friday
            d(2026, 4, 14),  // Dr Ambedkar Jayanti
            d(2026, 5, 1),   // May Day
            d(2026, 5, 28),  // Eid ul-Adha (approx)
            d(2026, 8, 15),  // Independence Day
            d(2026, 8, 17),  // Janmashtami (approx)
            d(2026, 10, 2),  // Mahatma Gandhi Jayanti
            d(2026, 10, 10), // Diwali / Laxmi Puja (approx)
            d(2026, 10, 26), // Guru Nanak Jayanti (approx)
            d(2026, 12, 25), // Christmas
        ] {
            holidays.insert(date);
        }

        // ── 2027 ──────────────────────────────────────────────────────────
        for date in [
            d(2027, 1, 26),  // Republic Day
            d(2027, 2, 7),   // Maha Shivaratri (approx)
            d(2027, 3, 11),  // Eid ul-Fitr (approx)
            d(2027, 3, 22),  // Holi (approx)
            d(2027, 3, 26),  // Good Friday
            d(2027, 4, 14),  // Dr Ambedkar Jayanti
            d(2027, 5, 1),   // May Day
            d(2027, 5, 18),  // Eid ul-Adha (approx)
            d(2027, 8, 7),   // Janmashtami (approx)
            d(2027, 8, 15),  // Independence Day (Sunday – included)
            d(2027, 10, 2),  // Mahatma Gandhi Jayanti (Saturday – included)
            d(2027, 10, 29), // Diwali / Laxmi Puja (approx)
            d(2027, 11, 14), // Guru Nanak Jayanti (approx)
            d(2027, 12, 25), // Christmas (Saturday – included)
        ] {
            holidays.insert(date);
        }

        Self { holidays }
    }

    /// Returns `true` if `date` is a trading day (weekday and not a gazetted holiday).
    pub fn is_trading_day(&self, date: NaiveDate) -> bool {
        let weekday = date.weekday();
        weekday != chrono::Weekday::Sat
            && weekday != chrono::Weekday::Sun
            && !self.holidays.contains(&date)
    }

    /// Returns the next trading day **strictly after** `date`.
    pub fn next_trading_day(&self, date: NaiveDate) -> NaiveDate {
        let mut d = date + chrono::Duration::days(1);
        while !self.is_trading_day(d) {
            d += chrono::Duration::days(1);
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_republic_day_is_holiday() {
        let cal = TradingCalendar::nse();
        // Jan 26 2024 is Republic Day (Friday) — not a trading day
        let date = NaiveDate::from_ymd_opt(2024, 1, 26).unwrap();
        assert!(
            !cal.is_trading_day(date),
            "Republic Day 2024 should not be a trading day"
        );
    }

    #[test]
    fn test_weekend_not_trading_day() {
        let cal = TradingCalendar::nse();
        // 2024-01-27 is a Saturday
        let saturday = NaiveDate::from_ymd_opt(2024, 1, 27).unwrap();
        assert!(
            !cal.is_trading_day(saturday),
            "Saturday should not be a trading day"
        );
    }

    #[test]
    fn test_normal_weekday_is_trading_day() {
        let cal = TradingCalendar::nse();
        // 2024-01-22 is a Monday with no holiday
        let monday = NaiveDate::from_ymd_opt(2024, 1, 22).unwrap();
        assert!(
            cal.is_trading_day(monday),
            "A regular Monday should be a trading day"
        );
    }

    #[test]
    fn test_next_trading_day_skips_weekend() {
        let cal = TradingCalendar::nse();
        // 2024-01-19 is a Friday
        let friday = NaiveDate::from_ymd_opt(2024, 1, 19).unwrap();
        let next = cal.next_trading_day(friday);
        let expected_monday = NaiveDate::from_ymd_opt(2024, 1, 22).unwrap();
        assert_eq!(
            next, expected_monday,
            "next_trading_day after Friday should be Monday"
        );
    }

    #[test]
    fn test_next_trading_day_skips_holiday() {
        let cal = TradingCalendar::nse();
        // 2024-01-25 is a Thursday; Jan 26 (Friday) is Republic Day
        let thursday = NaiveDate::from_ymd_opt(2024, 1, 25).unwrap();
        let next = cal.next_trading_day(thursday);
        // Jan 26 = holiday, Jan 27 = Sat, Jan 28 = Sun → next trading day = Jan 29 (Mon)
        let expected = NaiveDate::from_ymd_opt(2024, 1, 29).unwrap();
        assert_eq!(
            next, expected,
            "next_trading_day should skip the holiday + weekend"
        );
    }

    #[test]
    fn test_diwali_2024_is_holiday() {
        let cal = TradingCalendar::nse();
        // Nov 1 2024 is Diwali / Laxmi Puja (Friday) — not a trading day
        let date = NaiveDate::from_ymd_opt(2024, 11, 1).unwrap();
        assert!(
            !cal.is_trading_day(date),
            "Diwali 2024 (Nov 1) should not be a trading day"
        );
    }

    #[test]
    fn test_milad_un_nabi_2025_is_holiday() {
        let cal = TradingCalendar::nse();
        // Sep 5 2025 is Milad-un-Nabi (Friday) — not a trading day
        let date = NaiveDate::from_ymd_opt(2025, 9, 5).unwrap();
        assert!(
            !cal.is_trading_day(date),
            "Milad-un-Nabi 2025 (Sep 5) should not be a trading day"
        );
    }
}
