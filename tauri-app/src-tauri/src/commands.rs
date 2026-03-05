use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::Serialize;
use tauri::AppHandle;
use tauri_plugin_dialog::DialogExt;

use crate::db::{self, DbState};
use crate::thumbs;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FullRescanInfo {
    pub purged_count: u64,
    pub total_remaining: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelConflict {
    pub path: String,
    pub model_species: String,
    pub model_confidence: f64,
    pub user_species: String,
    pub thumb_path: Option<String>,
}

/// Absolute path to the bird-classification project root (resolved at compile time).
const PROJECT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

pub(crate) fn project_dir() -> PathBuf {
    let p = PathBuf::from(PROJECT_DIR);
    p.canonicalize().unwrap_or(p)
}

#[tauri::command]
pub fn load_species_db() -> Result<serde_json::Value, String> {
    let path = project_dir().join("data/species_db.json");
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Could not read species_db.json: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("Invalid species_db.json: {e}"))
}

#[tauri::command]
pub fn load_scan_results(db: tauri::State<'_, DbState>) -> Result<Option<serde_json::Value>, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let photos = db::photos::get_all_photos(&conn);
    if photos.is_empty() {
        return Ok(None);
    }
    let data_dir = project_dir().join("data");

    let mut photos_map = serde_json::Map::new();
    for photo in &photos {
        let mut entry = serde_json::Map::new();
        entry.insert(
            "scientificName".into(),
            photo.scientific_name.clone().into(),
        );
        entry.insert("confidence".into(), photo.confidence.into());
        if let Some(ref d) = photo.exif_date {
            entry.insert("exif_date".into(), d.clone().into());
        }
        if let Some(lat) = photo.exif_lat {
            entry.insert("exif_lat".into(), lat.into());
        }
        if let Some(lon) = photo.exif_lon {
            entry.insert("exif_lon".into(), lon.into());
        }
        if let Some(ref tk) = photo.top_k {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tk) {
                entry.insert("top_k".into(), parsed);
            }
        }
        if let Some(ref tp) = photo.thumb_path {
            let abs_thumb = data_dir.join("thumbs").join(tp);
            entry.insert(
                "thumbPath".into(),
                abs_thumb.to_string_lossy().into_owned().into(),
            );
        }
        if let Some(ref us) = photo.user_species {
            entry.insert("userSpecies".into(), us.clone().into());
        }
        entry.insert("modelSpecies".into(), photo.model_species.clone().into());

        photos_map.insert(photo.path.clone(), serde_json::Value::Object(entry));
    }

    Ok(Some(serde_json::Value::Object(photos_map)))
}

#[tauri::command]
pub fn get_data_dir() -> String {
    project_dir().join("data").to_string_lossy().into_owned()
}

/// Open a native folder-picker dialog and return the selected path.
#[tauri::command]
pub async fn open_folder_dialog(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .map(|p| p.to_string())
}

#[tauri::command]
pub fn load_config(db: tauri::State<'_, DbState>) -> Result<serde_json::Value, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "folders": db::get_config_folders(&conn) }))
}

#[tauri::command]
pub fn save_config(db: tauri::State<'_, DbState>, folders: Vec<String>) -> Result<(), String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    db::set_config_folders(&conn, &folders).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_user_species(
    db: tauri::State<'_, DbState>,
    path: String,
    species: Option<String>,
) -> Result<(), String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    db::photos::set_user_species(&conn, &path, species.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn remove_folder_photos(
    db: tauri::State<'_, DbState>,
    folder: String,
    remaining_folders: Vec<String>,
) -> Result<(), String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;

    // For each photo in the removed folder, check if another folder still covers it
    let photo_paths = db::photos::get_photo_paths_in_folders(&conn, &[folder.clone()]);
    let thumbs_dir = project_dir().join("data/thumbs");

    for path in &photo_paths {
        let norm = path.replace('\\', "/");
        let new_folder = remaining_folders.iter().find(|f| {
            let nf = f.replace('\\', "/");
            norm.starts_with(&nf)
        });

        if let Some(nf) = new_folder {
            // Reassign to the covering folder
            let _ = db::photos::reassign_folder_single(&conn, path, nf);
        } else {
            // No covering folder — delete thumbnail + DB row
            if let Some(thumb) = db::photos::get_thumb_path(&conn, path) {
                let _ = std::fs::remove_file(thumbs_dir.join(&thumb));
            }
            let _ = conn.execute("DELETE FROM photos WHERE path = ?1", rusqlite::params![path]);
        }
    }

    Ok(())
}

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

// ---------------------------------------------------------------------------
// Missing photo detection & recovery
// ---------------------------------------------------------------------------

/// Check all photos in the DB and return paths that no longer exist on disk.
#[tauri::command]
pub fn check_missing_photos(db: tauri::State<'_, DbState>) -> Result<Vec<String>, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT path FROM photos")
        .map_err(|e| e.to_string())?;
    let all_paths: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    let missing: Vec<String> = all_paths
        .into_iter()
        .filter(|p| !Path::new(p).exists())
        .collect();

    Ok(missing)
}

/// Recursively scan the given folders, try to match missing photos by filename,
/// and update DB paths + thumbnails for any matches found.
/// Returns the list of paths that are still missing after relocation.
#[tauri::command]
pub fn relocate_missing_photos(
    db: tauri::State<'_, DbState>,
    missing_paths: Vec<String>,
    search_folders: Vec<String>,
) -> Result<Vec<String>, String> {
    if missing_paths.is_empty() || search_folders.is_empty() {
        return Ok(missing_paths);
    }

    // Build a filename → [(path, size)] index from the search folders
    let mut file_index: HashMap<String, Vec<(String, i64)>> = HashMap::new();
    for folder in &search_folders {
        index_folder_recursive(Path::new(folder), &mut file_index);
    }

    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let thumbs_dir = project_dir().join("data/thumbs");
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;

    let mut still_missing: Vec<String> = Vec::new();

    for old_path in &missing_paths {
        // Extract filename from old path
        let filename = Path::new(old_path)
            .file_name()
            .map(|f| f.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        if filename.is_empty() {
            still_missing.push(old_path.clone());
            continue;
        }

        let candidates = match file_index.get(&filename) {
            Some(c) => c,
            None => {
                still_missing.push(old_path.clone());
                continue;
            }
        };

        // Get the expected file_size from the DB to match against
        let db_size: Option<i64> = tx
            .query_row(
                "SELECT file_size FROM photos WHERE path = ?1",
                params![old_path],
                |row| row.get(0),
            )
            .ok()
            .flatten();

        // Find a candidate matching by file size (or take the only candidate if size is unknown)
        let new_path = if let Some(expected_size) = db_size {
            candidates
                .iter()
                .find(|(_, size)| *size == expected_size)
                .map(|(p, _)| normalize_path(p))
        } else if candidates.len() == 1 {
            Some(normalize_path(&candidates[0].0))
        } else {
            None
        };

        let new_path = match new_path {
            Some(p) => p,
            None => {
                still_missing.push(old_path.clone());
                continue;
            }
        };

        // Check if the new path already exists in DB (e.g., already re-scanned)
        let already_exists: bool = tx
            .query_row(
                "SELECT 1 FROM photos WHERE path = ?1",
                params![new_path],
                |_| Ok(()),
            )
            .is_ok();

        if already_exists {
            // New location already scanned — just delete the old (missing) row + its thumbnail
            let old_thumb: Option<String> = tx
                .query_row(
                    "SELECT thumb_path FROM photos WHERE path = ?1",
                    params![old_path],
                    |row| row.get(0),
                )
                .ok()
                .flatten();
            if let Some(ref ot) = old_thumb {
                let _ = std::fs::remove_file(thumbs_dir.join(ot));
            }
            tx.execute("DELETE FROM photos WHERE path = ?1", params![old_path])
                .map_err(|e| format!("Failed to delete {}: {}", old_path, e))?;
            continue;
        }

        // Get current thumb_path
        let old_thumb: Option<String> = tx
            .query_row(
                "SELECT thumb_path FROM photos WHERE path = ?1",
                params![old_path],
                |row| row.get(0),
            )
            .ok()
            .flatten();

        // Rename thumbnail on disk
        let new_thumb_name = thumbs::thumb_name(&new_path);
        if let Some(ref ot) = old_thumb {
            let old_file = thumbs_dir.join(ot);
            let new_file = thumbs_dir.join(&new_thumb_name);
            if old_file.exists() {
                let _ = std::fs::rename(&old_file, &new_file);
            }
        }

        // Find the best matching folder for the new path
        let new_folder = search_folders
            .iter()
            .filter(|f| normalize_path(&new_path).starts_with(&normalize_path(f)))
            .max_by_key(|f| f.len())
            .cloned()
            .unwrap_or_default();

        let thumb_val = if old_thumb.is_some() {
            Some(new_thumb_name)
        } else {
            None
        };

        tx.execute(
            "UPDATE photos SET path = ?1, folder = ?2, thumb_path = ?3 WHERE path = ?4",
            params![new_path, new_folder, thumb_val, old_path],
        )
        .map_err(|e| format!("Failed to relocate {}: {}", old_path, e))?;
    }

    tx.commit().map_err(|e| e.to_string())?;
    Ok(still_missing)
}

/// Delete photos from DB + their thumbnails from disk for all given paths.
#[tauri::command]
pub fn purge_missing_photos(
    db: tauri::State<'_, DbState>,
    paths: Vec<String>,
) -> Result<u64, String> {
    if paths.is_empty() {
        return Ok(0);
    }

    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let thumbs_dir = project_dir().join("data/thumbs");
    let mut count = 0u64;

    for path in &paths {
        if let Some(thumb) = db::photos::get_thumb_path(&conn, path) {
            let _ = std::fs::remove_file(thumbs_dir.join(&thumb));
        }
        let deleted = conn
            .execute("DELETE FROM photos WHERE path = ?1", params![path])
            .unwrap_or(0);
        count += deleted as u64;
    }

    Ok(count)
}

// ---------------------------------------------------------------------------
// Full rescan
// ---------------------------------------------------------------------------

/// Prepare for a complete rescan: purge missing photos, delete all thumbnails,
/// and reset mtime so every photo gets re-processed by the normal scan pipeline.
#[tauri::command]
pub fn prepare_full_rescan(
    db: tauri::State<'_, DbState>,
) -> Result<FullRescanInfo, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let thumbs_dir = project_dir().join("data/thumbs");

    // 1. Purge photos whose files no longer exist on disk
    let mut stmt = conn
        .prepare("SELECT path, thumb_path FROM photos")
        .map_err(|e| e.to_string())?;
    let all: Vec<(String, Option<String>)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    let mut purged_count = 0u64;
    for (path, thumb) in &all {
        if !Path::new(path).exists() {
            if let Some(ref t) = thumb {
                let _ = std::fs::remove_file(thumbs_dir.join(t));
            }
            let _ = conn.execute("DELETE FROM photos WHERE path = ?1", params![path]);
            purged_count += 1;
        }
    }

    // 2. Delete ALL thumbnail files from disk
    if thumbs_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&thumbs_dir) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    // 3. Reset thumb_path and file_mtime to force full re-processing
    conn.execute("UPDATE photos SET thumb_path = NULL, file_mtime = 0", [])
        .map_err(|e| e.to_string())?;

    let total_remaining: u64 = conn
        .query_row("SELECT COUNT(*) FROM photos", [], |row| row.get(0))
        .unwrap_or(0);

    Ok(FullRescanInfo {
        purged_count,
        total_remaining,
    })
}

/// Return photos where the user's manual label disagrees with the model prediction.
#[tauri::command]
pub fn get_label_conflicts(
    db: tauri::State<'_, DbState>,
) -> Result<Vec<LabelConflict>, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let data_dir = project_dir().join("data");

    let mut stmt = conn
        .prepare(
            "SELECT path, model_species, model_confidence, user_species, thumb_path
             FROM photos
             WHERE user_species IS NOT NULL AND user_species != model_species",
        )
        .map_err(|e| e.to_string())?;

    let conflicts: Vec<LabelConflict> = stmt
        .query_map([], |row| {
            let thumb: Option<String> = row.get(4)?;
            Ok(LabelConflict {
                path: row.get(0)?,
                model_species: row.get(1)?,
                model_confidence: row.get(2)?,
                user_species: row.get(3)?,
                thumb_path: thumb.map(|t| {
                    data_dir
                        .join("thumbs")
                        .join(&t)
                        .to_string_lossy()
                        .into_owned()
                }),
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(conflicts)
}

/// Accept model predictions for selected photos by clearing their user_species.
#[tauri::command]
pub fn resolve_label_conflicts(
    db: tauri::State<'_, DbState>,
    accept_model_paths: Vec<String>,
) -> Result<(), String> {
    if accept_model_paths.is_empty() {
        return Ok(());
    }
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    for path in &accept_model_paths {
        conn.execute(
            "UPDATE photos SET user_species = NULL WHERE path = ?1",
            params![path],
        )
        .map_err(|e| format!("Failed to resolve conflict for {}: {}", path, e))?;
    }
    Ok(())
}

/// Recursively index all image files in a directory by lowercase filename.
/// Stores (normalized_path, file_size) for each file so we can match by size.
fn index_folder_recursive(dir: &Path, index: &mut HashMap<String, Vec<(String, i64)>>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            index_folder_recursive(&path, index);
        } else if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if matches!(
                ext_lower.as_str(),
                "jpg" | "jpeg" | "png" | "tiff" | "tif" | "webp" | "bmp"
            ) {
                if let Some(name) = path.file_name() {
                    let key = name.to_string_lossy().to_lowercase();
                    let size = path.metadata().map(|m| m.len() as i64).unwrap_or(0);
                    index
                        .entry(key)
                        .or_default()
                        .push((normalize_path(&path.to_string_lossy()), size));
                }
            }
        }
    }
}
