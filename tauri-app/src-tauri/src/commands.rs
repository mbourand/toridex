use std::path::PathBuf;

use tauri::AppHandle;
use tauri_plugin_dialog::DialogExt;

use crate::db::{self, DbState};

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
