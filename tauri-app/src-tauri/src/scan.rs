use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::commands::project_dir;
use crate::db::{self, DbState};
use crate::exif;
use crate::thumbs;

const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "tiff", "tif", "webp", "bmp"];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileToProcess {
    pub path: String,
    pub folder: String,
    pub file_mtime: f64,
    pub file_size: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedScan {
    pub to_process: Vec<FileToProcess>,
    pub skipped_count: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPaths {
    pub detector: String,
    pub classifier: String,
    pub label_map: String,
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Collect images from folders, filter out unchanged files, return list to process.
#[tauri::command]
pub fn prepare_scan(
    db: tauri::State<'_, DbState>,
    folders: Vec<String>,
) -> Result<PreparedScan, String> {
    if folders.is_empty() {
        return Err("No folders configured".to_string());
    }

    let image_paths = collect_images(&folders);
    let conn = db.0.lock().map_err(|e| e.to_string())?;

    let mut to_process = Vec::new();
    let mut skipped_count = 0usize;

    for path in &image_paths {
        let path_str = normalize_path(&path.to_string_lossy());
        let folder = find_folder_for_path(&path_str, &folders);

        match path.metadata() {
            Ok(meta) => {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                let size = meta.len() as i64;

                if db::photos::is_photo_unchanged(&conn, &path_str, mtime, size) {
                    skipped_count += 1;
                } else {
                    to_process.push(FileToProcess {
                        path: path_str,
                        folder,
                        file_mtime: mtime,
                        file_size: size,
                    });
                }
            }
            Err(_) => {
                to_process.push(FileToProcess {
                    path: path_str,
                    folder,
                    file_mtime: 0.0,
                    file_size: 0,
                });
            }
        }
    }

    Ok(PreparedScan {
        to_process,
        skipped_count,
    })
}

/// Store a single photo result in the database. EXIF is extracted server-side.
#[tauri::command]
pub fn store_photo_result(
    db: tauri::State<'_, DbState>,
    path: String,
    folder: String,
    species: String,
    species_idx: i64,
    confidence: f64,
    file_mtime: f64,
    file_size: i64,
    top_k_json: Option<String>,
) -> Result<(), String> {
    let exif_data = exif::extract_exif(Path::new(&path));

    let conn = db.0.lock().map_err(|e| e.to_string())?;
    db::photos::upsert_single_photo(
        &conn,
        &path,
        &folder,
        &species,
        species_idx,
        confidence,
        exif_data.date.as_deref(),
        exif_data.lat,
        exif_data.lon,
        file_mtime,
        file_size,
        top_k_json.as_deref(),
    )
}

/// Return absolute paths to the model files so the frontend can load them via asset://.
#[tauri::command]
pub fn get_model_paths() -> ModelPaths {
    let models_dir = project_dir().join("data/models");
    ModelPaths {
        detector: models_dir
            .join("bird_detector.onnx")
            .to_string_lossy()
            .into_owned(),
        classifier: models_dir
            .join("bird_classifier.onnx")
            .to_string_lossy()
            .into_owned(),
        label_map: models_dir
            .join("label_map.json")
            .to_string_lossy()
            .into_owned(),
    }
}

/// Clean up stale DB entries and generate missing thumbnails.
#[tauri::command]
pub fn finalize_scan(
    app: AppHandle,
    db: tauri::State<'_, DbState>,
    folders: Vec<String>,
) -> Result<(), String> {
    let image_paths = collect_images(&folders);

    // Remove stale entries
    {
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        let db_paths = db::photos::get_photo_paths_in_folders(&conn, &folders);
        let current_set: HashSet<String> = image_paths
            .iter()
            .map(|p| normalize_path(&p.to_string_lossy()))
            .collect();
        for db_path in &db_paths {
            if !current_set.contains(&normalize_path(db_path)) {
                let _ = conn.execute(
                    "DELETE FROM photos WHERE path = ?1",
                    rusqlite::params![db_path],
                );
            }
        }
    }

    // Generate thumbnails
    let thumbs_dir = project_dir().join("data/thumbs");
    let needing_thumbs = {
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        db::photos::get_photos_needing_thumbnails(&conn)
    };
    if !needing_thumbs.is_empty() {
        let generated = thumbs::generate_thumbnails(&needing_thumbs, &thumbs_dir, |current, total| {
            let _ = app.emit("thumb-progress", serde_json::json!({ "current": current, "total": total }));
        });
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        for (orig, thumb_name) in &generated {
            let _ = db::photos::set_thumb_path(&conn, orig, thumb_name);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_images(folders: &[String]) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for folder in folders {
        let dir = Path::new(folder);
        if !dir.is_dir() {
            continue;
        }
        collect_images_recursive(dir, &mut paths, &mut seen);
    }
    paths.sort();
    paths
}

fn collect_images_recursive(dir: &Path, out: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_images_recursive(&path, out, seen);
        } else if let Some(ext) = path.extension() {
            if SUPPORTED_EXTENSIONS.contains(&ext.to_string_lossy().to_lowercase().as_str())
                && seen.insert(path.clone())
            {
                out.push(path);
            }
        }
    }
}

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

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
