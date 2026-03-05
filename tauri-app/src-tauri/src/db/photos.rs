use std::path::Path;

use rusqlite::{params, Connection};

use super::{PhotoRow, ScanResultsJson};

// ---------------------------------------------------------------------------
// Import from JSON (kept for backward compat)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
// Direct upsert (native inference, no JSON intermediate)
// ---------------------------------------------------------------------------

/// Check if a photo is unchanged (same mtime + size) and can be skipped.
pub fn is_photo_unchanged(conn: &Connection, path: &str, mtime: f64, size: i64) -> bool {
    conn.query_row(
        "SELECT 1 FROM photos WHERE path = ?1 AND file_mtime = ?2 AND file_size = ?3",
        params![path, mtime, size],
        |_| Ok(()),
    )
    .is_ok()
}

/// Upsert a single photo row directly from Rust inference results.
pub fn upsert_single_photo(
    conn: &Connection,
    path: &str,
    folder: &str,
    species: &str,
    species_idx: i64,
    confidence: f64,
    exif_date: Option<&str>,
    exif_lat: Option<f64>,
    exif_lon: Option<f64>,
    file_mtime: f64,
    file_size: i64,
    top_k_json: Option<&str>,
) -> Result<(), String> {
    conn.execute(
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
            species,
            species_idx,
            confidence,
            exif_date,
            exif_lat,
            exif_lon,
            file_mtime,
            file_size,
            top_k_json,
        ],
    )
    .map_err(|e| format!("Upsert failed for {path}: {e}"))?;
    Ok(())
}

/// Get all photo paths in given folders (for stale detection).
pub fn get_photo_paths_in_folders(conn: &Connection, folders: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    for folder in folders {
        let mut stmt = conn
            .prepare("SELECT path FROM photos WHERE folder = ?1")
            .unwrap();
        let rows: Vec<String> = stmt
            .query_map(params![folder], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        paths.extend(rows);
    }
    paths
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

pub fn reassign_folder_single(conn: &Connection, path: &str, new_folder: &str) -> rusqlite::Result<usize> {
    conn.execute(
        "UPDATE photos SET folder = ?1 WHERE path = ?2",
        params![new_folder, path],
    )
}

pub fn get_thumb_path(conn: &Connection, path: &str) -> Option<String> {
    conn.query_row(
        "SELECT thumb_path FROM photos WHERE path = ?1 AND thumb_path IS NOT NULL",
        params![path],
        |row| row.get(0),
    )
    .ok()
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
        .prepare("SELECT path FROM photos WHERE thumb_path IS NULL")
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
