use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use tauri::{AppHandle, Emitter};
use tauri_plugin_dialog::DialogExt;

use crate::db::{self, DbState};
use crate::thumbs;

/// Absolute path to the bird-classification project root (resolved at compile time).
const PROJECT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

fn project_dir() -> PathBuf {
    PathBuf::from(PROJECT_DIR)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(PROJECT_DIR))
}

/// Python executable: prefer the project's venv, fall back to PATH.
fn python_exe() -> PathBuf {
    let venv_python = project_dir().join(".venv/Scripts/python.exe");
    if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python")
    }
}

#[tauri::command]
pub fn load_species_db() -> Result<String, String> {
    let path = project_dir().join("data/species_db.json");
    std::fs::read_to_string(&path)
        .map_err(|e| format!("Could not read species_db.json: {e}"))
}

#[tauri::command]
pub fn load_scan_results(db: tauri::State<'_, DbState>) -> Result<Option<String>, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let photos = db::get_all_photos(&conn);
    if photos.is_empty() {
        return Ok(None);
    }
    let folders = db::get_config_folders(&conn);
    let data_dir = project_dir().join("data");

    let mut photos_map = serde_json::Map::new();
    for photo in &photos {
        let mut entry = serde_json::Map::new();
        entry.insert("species_idx".into(), photo.species_idx.into());
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

    let result = serde_json::json!({
        "folders": folders,
        "scanned_at": "",
        "photos": photos_map,
    });

    Ok(Some(
        serde_json::to_string(&result).map_err(|e| e.to_string())?,
    ))
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
pub fn load_config(db: tauri::State<'_, DbState>) -> Result<String, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let folders = db::get_config_folders(&conn);
    serde_json::to_string(&serde_json::json!({ "folders": folders })).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn save_config(db: tauri::State<'_, DbState>, config: String) -> Result<(), String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    let parsed: serde_json::Value =
        serde_json::from_str(&config).map_err(|e| e.to_string())?;
    let folders: Vec<String> = parsed["folders"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    db::set_config_folders(&conn, &folders).map_err(|e| e.to_string())
}

/// Scan events emitted to the frontend:
///   "scan-progress"  →  { current: u32, total: u32 }
///   "scan-complete"  →  null
///   "scan-error"     →  string
#[tauri::command]
pub async fn scan_photos_folder(
    app: AppHandle,
    db: tauri::State<'_, DbState>,
    folders: Vec<String>,
) -> Result<(), String> {
    if folders.is_empty() {
        return Err("No folders configured".to_string());
    }

    let python = python_exe();
    let script = project_dir().join("analyze_photos.py");
    let output = project_dir().join("data/scan_results.json");
    let proj = project_dir();

    // Phase 1: Run Python
    let mut cmd = Command::new(&python);
    cmd.arg(&script);
    for f in &folders {
        cmd.arg("--folder").arg(f);
    }
    cmd.arg("--output")
        .arg(&output)
        .arg("--top")
        .arg("5")
        .arg("--incremental")
        .current_dir(&proj)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start Python: {e}"))?;

    let stdout = child.stdout.take().expect("stdout piped");
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if let Some(rest) = line.strip_prefix("PROGRESS:") {
            let parts: Vec<&str> = rest.splitn(2, ':').collect();
            if parts.len() == 2 {
                let current: u32 = parts[0].parse().unwrap_or(0);
                let total: u32 = parts[1].parse().unwrap_or(0);
                let _ = app.emit(
                    "scan-progress",
                    serde_json::json!({ "current": current, "total": total }),
                );
            }
        }
        // Don't emit scan-complete on DONE — we do it after import + thumbs
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        let _ = app.emit("scan-error", "Python process exited with error");
        return Err("Python scan failed".into());
    }

    // Phase 2: Import JSON into SQLite
    {
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        db::upsert_photos_from_json(&conn, &output)?;
    }

    // Phase 3: Generate thumbnails
    let thumbs_dir = project_dir().join("data/thumbs");
    let needing_thumbs = {
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        db::get_photos_needing_thumbnails(&conn)
    };

    if !needing_thumbs.is_empty() {
        let generated = thumbs::generate_thumbnails(&needing_thumbs, &thumbs_dir);
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        for (orig, thumb_name) in &generated {
            let _ = db::set_thumb_path(&conn, orig, thumb_name);
        }
    }

    // Phase 4: Emit completion
    let _ = app.emit("scan-complete", serde_json::json!(null));

    Ok(())
}

#[tauri::command]
pub fn set_user_species(
    db: tauri::State<'_, DbState>,
    path: String,
    species: Option<String>,
) -> Result<(), String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    db::set_user_species(&conn, &path, species.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_photos_by_folder(
    db: tauri::State<'_, DbState>,
    folder: String,
) -> Result<usize, String> {
    let conn = db.0.lock().map_err(|e| e.to_string())?;
    db::delete_photos_by_folder(&conn, &folder).map_err(|e| e.to_string())
}
