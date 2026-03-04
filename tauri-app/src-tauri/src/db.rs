use std::collections::HashMap;
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
    pub photos: HashMap<String, PhotoEntryJson>,
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

// ---------------------------------------------------------------------------
// Photo upsert from JSON
// ---------------------------------------------------------------------------

/// Normalize path separators to forward slashes for consistent matching.
fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

/// Find which configured folder a photo path belongs to.
fn find_folder_for_path(photo_path: &str, folders: &[String]) -> String {
    let norm = normalize_path(photo_path);
    let mut best = String::new();
    for f in folders {
        let nf = normalize_path(f);
        if norm.starts_with(&nf) && nf.len() > best.len() {
            best = f.clone();
        }
    }
    if best.is_empty() && !folders.is_empty() {
        folders[0].clone()
    } else {
        best
    }
}

pub fn upsert_photos_from_json(conn: &Connection, json_path: &Path) -> Result<usize, String> {
    let content =
        std::fs::read_to_string(json_path).map_err(|e| format!("Read JSON failed: {e}"))?;
    let data: ScanResultsJson =
        serde_json::from_str(&content).map_err(|e| format!("Parse JSON failed: {e}"))?;

    let tx = conn
        .unchecked_transaction()
        .map_err(|e| format!("Transaction failed: {e}"))?;

    // Collect current photo paths from the scanned folders to detect deletions
    let scanned_paths: std::collections::HashSet<&str> =
        data.photos.keys().map(|s| s.as_str()).collect();

    // Delete photos that are in the same folders but no longer in the scan results
    // (i.e., files that were deleted from disk)
    for folder in &data.folders {
        let mut stmt = tx
            .prepare("SELECT path FROM photos WHERE folder = ?1")
            .map_err(|e| e.to_string())?;
        let existing: Vec<String> = stmt
            .query_map(params![folder], |row| row.get(0))
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        for existing_path in &existing {
            if !scanned_paths.contains(existing_path.as_str()) {
                tx.execute("DELETE FROM photos WHERE path = ?1", params![existing_path])
                    .map_err(|e| e.to_string())?;
            }
        }
    }

    let mut count = 0usize;
    for (path, entry) in &data.photos {
        let folder = find_folder_for_path(path, &data.folders);
        let top_k_json = entry
            .top_k
            .as_ref()
            .map(|tk| serde_json::to_string(tk).unwrap_or_default());

        tx.execute(
            "INSERT INTO photos(path, folder, model_species, model_species_idx, model_confidence,
                                exif_date, exif_lat, exif_lon, file_mtime, file_size, top_k)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(path) DO UPDATE SET
                 folder = excluded.folder,
                 model_species = excluded.model_species,
                 model_species_idx = excluded.model_species_idx,
                 model_confidence = excluded.model_confidence,
                 exif_date = excluded.exif_date,
                 exif_lat = excluded.exif_lat,
                 exif_lon = excluded.exif_lon,
                 file_mtime = excluded.file_mtime,
                 file_size = excluded.file_size,
                 top_k = excluded.top_k",
            params![
                path,
                folder,
                entry.scientific_name,
                entry.species_idx,
                entry.confidence,
                entry.exif_date,
                entry.exif_lat,
                entry.exif_lon,
                entry.file_mtime,
                entry.file_size,
                top_k_json,
            ],
        )
        .map_err(|e| format!("Upsert failed for {path}: {e}"))?;
        count += 1;
    }

    tx.commit().map_err(|e| format!("Commit failed: {e}"))?;
    Ok(count)
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

pub fn get_all_photos(conn: &Connection) -> Vec<PhotoRow> {
    let mut stmt = conn
        .prepare(
            "SELECT path, model_species_idx, model_species,
                    COALESCE(user_species, model_species) AS effective_species,
                    model_confidence, exif_date, exif_lat, exif_lon,
                    top_k, thumb_path, user_species
             FROM photos
             WHERE model_species != '__skipped__'
             ORDER BY path",
        )
        .unwrap();

    stmt.query_map([], |row| {
        Ok(PhotoRow {
            path: row.get(0)?,
            species_idx: row.get(1)?,
            scientific_name: row.get(3)?, // effective
            model_species: row.get(2)?,
            confidence: row.get(4)?,
            exif_date: row.get(5)?,
            exif_lat: row.get(6)?,
            exif_lon: row.get(7)?,
            top_k: row.get(8)?,
            thumb_path: row.get(9)?,
            user_species: row.get(10)?,
        })
    })
    .unwrap()
    .filter_map(|r| r.ok())
    .collect()
}

pub fn delete_photos_by_folder(conn: &Connection, folder: &str) -> rusqlite::Result<usize> {
    conn.execute("DELETE FROM photos WHERE folder = ?1", params![folder])
}

pub fn set_user_species(
    conn: &Connection,
    path: &str,
    species: Option<&str>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE photos SET user_species = ?1 WHERE path = ?2",
        params![species, path],
    )?;
    Ok(())
}

pub fn get_photos_needing_thumbnails(conn: &Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare("SELECT path FROM photos WHERE thumb_path IS NULL AND model_species != '__skipped__'")
        .unwrap();
    stmt.query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
}

pub fn set_thumb_path(conn: &Connection, path: &str, thumb_path: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE photos SET thumb_path = ?1 WHERE path = ?2",
        params![thumb_path, path],
    )?;
    Ok(())
}
