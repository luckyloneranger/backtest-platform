use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;

use backtest_core::types::{Exchange, Instrument, InstrumentType};

// ── CorporateAction ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CorporateAction {
    pub symbol: String,
    pub exchange: String,
    pub date: String,           // YYYY-MM-DD
    pub action_type: String,    // "SPLIT", "BONUS", "DIVIDEND"
    pub ratio: f64,             // 2.0 for 1:2 split, 5.0 for ₹5 dividend
}

// ── String conversions for SQLite storage ────────────────────────────────────

fn exchange_to_str(e: Exchange) -> &'static str {
    match e {
        Exchange::Nse => "NSE",
        Exchange::Bse => "BSE",
        Exchange::Mcx => "MCX",
    }
}

fn str_to_exchange(s: &str) -> Result<Exchange> {
    match s {
        "NSE" => Ok(Exchange::Nse),
        "BSE" => Ok(Exchange::Bse),
        "MCX" => Ok(Exchange::Mcx),
        other => anyhow::bail!("unknown exchange: {other}"),
    }
}

fn instrument_type_to_str(it: InstrumentType) -> &'static str {
    match it {
        InstrumentType::Equity => "EQ",
        InstrumentType::FutureFO => "FUT",
        InstrumentType::OptionFO => "OPT",
        InstrumentType::Commodity => "COM",
    }
}

fn str_to_instrument_type(s: &str) -> Result<InstrumentType> {
    match s {
        "EQ" => Ok(InstrumentType::Equity),
        "FUT" => Ok(InstrumentType::FutureFO),
        "OPT" => Ok(InstrumentType::OptionFO),
        "COM" => Ok(InstrumentType::Commodity),
        other => anyhow::bail!("unknown instrument type: {other}"),
    }
}

// ── SQL ──────────────────────────────────────────────────────────────────────

const DROP_TABLE: &str = "DROP TABLE IF EXISTS instruments;";

const CREATE_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_token TEXT,
        tradingsymbol    TEXT    NOT NULL,
        name             TEXT,
        exchange         TEXT    NOT NULL,
        instrument_type  TEXT    NOT NULL,
        lot_size         INTEGER NOT NULL,
        tick_size        REAL    NOT NULL,
        expiry           TEXT,
        strike           REAL,
        option_type      TEXT,
        segment          TEXT,
        PRIMARY KEY (tradingsymbol, exchange)
    );
";

const UPSERT: &str = "
    INSERT OR REPLACE INTO instruments
        (instrument_token, tradingsymbol, name, exchange, instrument_type,
         lot_size, tick_size, expiry, strike, option_type, segment)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11);
";

const FIND: &str = "
    SELECT instrument_token, tradingsymbol, name, exchange, instrument_type,
           lot_size, tick_size, expiry, strike, option_type, segment
    FROM instruments
    WHERE tradingsymbol = ?1 AND exchange = ?2;
";

const LIST_BY_EXCHANGE: &str = "
    SELECT instrument_token, tradingsymbol, name, exchange, instrument_type,
           lot_size, tick_size, expiry, strike, option_type, segment
    FROM instruments
    WHERE exchange = ?1;
";

const CREATE_CORPORATE_ACTIONS: &str = "
    CREATE TABLE IF NOT EXISTS corporate_actions (
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        date TEXT NOT NULL,
        action_type TEXT NOT NULL,
        ratio REAL NOT NULL,
        PRIMARY KEY (symbol, exchange, date, action_type)
    );
";

const UPSERT_CORPORATE_ACTION: &str = "
    INSERT OR REPLACE INTO corporate_actions
        (symbol, exchange, date, action_type, ratio)
    VALUES (?1, ?2, ?3, ?4, ?5);
";

const GET_CORPORATE_ACTIONS: &str = "
    SELECT symbol, exchange, date, action_type, ratio
    FROM corporate_actions
    WHERE symbol = ?1 AND exchange = ?2
    ORDER BY date ASC;
";

// ── InstrumentStore ─────────────────────────────────────────────────────────

/// Wraps a `rusqlite::Connection` to store and query instrument metadata.
pub struct InstrumentStore {
    conn: Connection,
}

impl InstrumentStore {
    /// Opens (or creates) a SQLite file at `path` and ensures the
    /// `instruments` table exists with the latest schema.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::ensure_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Creates an in-memory SQLite database (useful for tests).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::ensure_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Drop and recreate the instruments table to ensure the schema is current.
    fn ensure_schema(conn: &Connection) -> Result<()> {
        // Check if the table already has the new columns; if not, recreate.
        let has_new_schema = conn
            .prepare("SELECT instrument_token FROM instruments LIMIT 0")
            .is_ok();
        if !has_new_schema {
            conn.execute_batch(DROP_TABLE)?;
        }
        conn.execute_batch(CREATE_TABLE)?;
        conn.execute_batch(CREATE_CORPORATE_ACTIONS)?;
        Ok(())
    }

    /// INSERT OR REPLACE a batch of instruments.
    pub fn upsert(&self, instruments: &[Instrument]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached(UPSERT)?;
            for inst in instruments {
                stmt.execute(params![
                    inst.instrument_token,
                    inst.tradingsymbol,
                    inst.name,
                    exchange_to_str(inst.exchange),
                    instrument_type_to_str(inst.instrument_type),
                    inst.lot_size,
                    inst.tick_size,
                    inst.expiry,
                    inst.strike,
                    inst.option_type,
                    inst.segment,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Look up a single instrument by trading symbol and exchange.
    pub fn find(&self, symbol: &str, exchange: Exchange) -> Result<Option<Instrument>> {
        let mut stmt = self.conn.prepare_cached(FIND)?;
        let mut rows = stmt.query(params![symbol, exchange_to_str(exchange)])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_instrument(row)?)),
            None => Ok(None),
        }
    }

    /// Look up the instrument_token for a given symbol and exchange.
    pub fn find_token(&self, symbol: &str, exchange: Exchange) -> Result<Option<String>> {
        let inst = self.find(symbol, exchange)?;
        Ok(inst.and_then(|i| i.instrument_token))
    }

    /// Return all instruments on a given exchange.
    pub fn list_by_exchange(&self, exchange: Exchange) -> Result<Vec<Instrument>> {
        let mut stmt = self.conn.prepare_cached(LIST_BY_EXCHANGE)?;
        let rows = stmt.query_map(params![exchange_to_str(exchange)], |row| {
            Ok(row_to_instrument_unchecked(row))
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row??);
        }
        Ok(result)
    }

    /// Insert a corporate action record (split, bonus, or dividend).
    pub fn insert_corporate_action(&self, action: &CorporateAction) -> Result<()> {
        let mut stmt = self.conn.prepare_cached(UPSERT_CORPORATE_ACTION)?;
        stmt.execute(params![
            action.symbol,
            action.exchange,
            action.date,
            action.action_type,
            action.ratio,
        ])?;
        Ok(())
    }

    /// Get all corporate actions for a given symbol and exchange, ordered by date.
    pub fn get_corporate_actions(&self, symbol: &str, exchange: &str) -> Result<Vec<CorporateAction>> {
        let mut stmt = self.conn.prepare_cached(GET_CORPORATE_ACTIONS)?;
        let rows = stmt.query_map(params![symbol, exchange], |row| {
            Ok(CorporateAction {
                symbol: row.get(0)?,
                exchange: row.get(1)?,
                date: row.get(2)?,
                action_type: row.get(3)?,
                ratio: row.get(4)?,
            })
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }
}

/// Converts a SQLite row into an `Instrument`. Used inside `query_map` where
/// only `rusqlite::Error` is available, so conversion errors are mapped.
fn row_to_instrument_unchecked(row: &rusqlite::Row) -> Result<Instrument> {
    row_to_instrument(row)
}

fn row_to_instrument(row: &rusqlite::Row) -> Result<Instrument> {
    let exchange_str: String = row.get(3)?;
    let itype_str: String = row.get(4)?;

    Ok(Instrument {
        instrument_token: row.get(0)?,
        tradingsymbol: row.get(1)?,
        name: row.get(2)?,
        exchange: str_to_exchange(&exchange_str)?,
        instrument_type: str_to_instrument_type(&itype_str)?,
        lot_size: row.get(5)?,
        tick_size: row.get(6)?,
        expiry: row.get(7)?,
        strike: row.get(8)?,
        option_type: row.get(9)?,
        segment: row.get(10)?,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use backtest_core::types::{Exchange, Instrument, InstrumentType};

    /// Helper: build a simple equity instrument for testing.
    fn make_equity(symbol: &str, exchange: Exchange) -> Instrument {
        Instrument {
            instrument_token: None,
            tradingsymbol: symbol.to_string(),
            name: None,
            exchange,
            instrument_type: InstrumentType::Equity,
            lot_size: 1,
            tick_size: 0.05,
            expiry: None,
            strike: None,
            option_type: None,
            segment: None,
        }
    }

    #[test]
    fn test_insert_and_query_instrument() {
        let store = InstrumentStore::in_memory().expect("failed to create in-memory store");

        let inst = Instrument {
            instrument_token: Some("11536386".to_string()),
            tradingsymbol: "NIFTY24APRFUT".to_string(),
            name: Some("NIFTY".to_string()),
            exchange: Exchange::Nse,
            instrument_type: InstrumentType::FutureFO,
            lot_size: 50,
            tick_size: 0.05,
            expiry: Some("2024-04-25".to_string()),
            strike: None,
            option_type: None,
            segment: Some("NFO-FUT".to_string()),
        };

        store.upsert(&[inst.clone()]).expect("upsert failed");

        let found = store
            .find("NIFTY24APRFUT", Exchange::Nse)
            .expect("find failed")
            .expect("instrument not found");

        assert_eq!(found.tradingsymbol, "NIFTY24APRFUT");
        assert_eq!(found.exchange, Exchange::Nse);
        assert_eq!(found.instrument_type, InstrumentType::FutureFO);
        assert_eq!(found.lot_size, 50);
        assert!((found.tick_size - 0.05).abs() < f64::EPSILON);
        assert_eq!(found.expiry, Some("2024-04-25".to_string()));
        assert_eq!(found.strike, None);
        assert_eq!(found.option_type, None);
    }

    #[test]
    fn test_list_by_exchange() {
        let store = InstrumentStore::in_memory().expect("failed to create in-memory store");

        let reliance = make_equity("RELIANCE", Exchange::Nse);
        let infy = make_equity("INFY", Exchange::Nse);

        store
            .upsert(&[reliance, infy])
            .expect("upsert failed");

        let nse_instruments = store
            .list_by_exchange(Exchange::Nse)
            .expect("list_by_exchange failed");

        assert_eq!(nse_instruments.len(), 2);

        let symbols: Vec<&str> = nse_instruments
            .iter()
            .map(|i| i.tradingsymbol.as_str())
            .collect();
        assert!(symbols.contains(&"RELIANCE"));
        assert!(symbols.contains(&"INFY"));
    }

    #[test]
    fn test_corporate_actions_db_roundtrip() {
        let store = InstrumentStore::in_memory().expect("failed to create in-memory store");

        let split = CorporateAction {
            symbol: "RELIANCE".to_string(),
            exchange: "NSE".to_string(),
            date: "2024-10-28".to_string(),
            action_type: "SPLIT".to_string(),
            ratio: 2.0,
        };
        let dividend = CorporateAction {
            symbol: "RELIANCE".to_string(),
            exchange: "NSE".to_string(),
            date: "2024-09-15".to_string(),
            action_type: "DIVIDEND".to_string(),
            ratio: 10.0,
        };

        store.insert_corporate_action(&split).expect("insert split failed");
        store.insert_corporate_action(&dividend).expect("insert dividend failed");

        let actions = store
            .get_corporate_actions("RELIANCE", "NSE")
            .expect("get_corporate_actions failed");

        assert_eq!(actions.len(), 2);
        // Results are ordered by date ASC, so dividend (2024-09-15) comes first
        assert_eq!(actions[0].action_type, "DIVIDEND");
        assert_eq!(actions[0].date, "2024-09-15");
        assert!((actions[0].ratio - 10.0).abs() < f64::EPSILON);
        assert_eq!(actions[1].action_type, "SPLIT");
        assert_eq!(actions[1].date, "2024-10-28");
        assert!((actions[1].ratio - 2.0).abs() < f64::EPSILON);

        // Verify no results for a different symbol
        let empty = store
            .get_corporate_actions("INFY", "NSE")
            .expect("get_corporate_actions failed");
        assert!(empty.is_empty());
    }
}
