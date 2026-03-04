pub mod photos;

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

pub struct DbState(pub Mutex<Connection>);

// ---------------------------------------------------------------------------
// Serde types for frontend JSON
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct PhotoRow {
    pub path: String,
    pub species_idx: i64,
    pub scientific_name: String, // effective = COALESCE(user_species, model_species)
    pub model_species: String,
    pub confidence: f64,
    pub exif_date: Option<String>,
    pub exif_lat: Option<f64>,
    pub exif_lon: Option<f64>,
    pub top_k: Option<String>, // raw JSON string
    pub thumb_path: Option<String>,
    pub user_species: Option<String>,
}

// Types for deserializing Python's scan_results.json
#[derive(Deserialize)]
pub struct ScanResultsJson {
    pub folders: Vec<String>,
    pub photos: std::collections::HashMap<String, PhotoEntryJson>,
}

#[derive(Deserialize)]
pub struct PhotoEntryJson {
    pub species_idx: i64,
    #[serde(rename = "scientificName")]
    pub scientific_name: String,
    pub confidence: f64,
    pub exif_date: Option<String>,
    pub exif_lat: Option<f64>,
    pub exif_lon: Option<f64>,
    pub file_mtime: Option<f64>,
    pub file_size: Option<i64>,
    pub top_k: Option<Vec<TopKEntryJson>>,
}

#[derive(Deserialize, Serialize)]
pub struct TopKEntryJson {
    #[serde(rename = "scientificName")]
    pub scientific_name: String,
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
    path              TEXT PRIMARY KEY,
    folder            TEXT NOT NULL,
    model_species     TEXT NOT NULL,
    model_species_idx INTEGER NOT NULL,
    model_confidence  REAL NOT NULL,
    user_species      TEXT,
    exif_date         TEXT,
    exif_lat          REAL,
    exif_lon          REAL,
    file_mtime        REAL,
    file_size         INTEGER,
    top_k             TEXT,
    thumb_path        TEXT
);

CREATE INDEX IF NOT EXISTS idx_photos_folder ON photos(folder);
";

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

pub fn init_db(db_path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    conn.execute_batch(SCHEMA_SQL)?;
    Ok(conn)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub fn get_config_folders(conn: &Connection) -> Vec<String> {
    let result: rusqlite::Result<String> = conn.query_row(
        "SELECT value FROM config WHERE key = 'folders'",
        [],
        |row| row.get(0),
    );
    match result {
        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
        Err(_) => vec![],
    }
}

pub fn set_config_folders(conn: &Connection, folders: &[String]) -> rusqlite::Result<()> {
    let json = serde_json::to_string(folders).unwrap_or_else(|_| "[]".to_string());
    conn.execute(
        "INSERT INTO config(key, value) VALUES('folders', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![json],
    )?;
    Ok(())
}
