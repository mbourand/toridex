use std::path::PathBuf;

use tauri::AppHandle;
use tauri::Manager;

/// Root data directory for runtime state (DB, thumbs, models).
pub fn data_dir(app: &AppHandle) -> PathBuf {
    if cfg!(debug_assertions) {
        dev_project_dir().join("data")
    } else {
        let dir = app
            .path()
            .app_data_dir()
            .expect("Failed to resolve app_data_dir");
        std::fs::create_dir_all(&dir).ok();
        dir
    }
}

/// Directory for downloaded/local models.
pub fn models_dir(app: &AppHandle) -> PathBuf {
    data_dir(app).join("models")
}

/// Directory for generated thumbnails.
pub fn thumbs_dir(app: &AppHandle) -> PathBuf {
    data_dir(app).join("thumbs")
}

/// Path to the SQLite database.
pub fn db_path(app: &AppHandle) -> PathBuf {
    data_dir(app).join("birds.db")
}

/// Path to species_db.json.
/// In dev: lives in the project data/ folder.
/// In release: bundled as a Tauri resource.
pub fn species_db_path(app: &AppHandle) -> PathBuf {
    if cfg!(debug_assertions) {
        dev_project_dir().join("data/species_db.json")
    } else {
        app.path()
            .resource_dir()
            .expect("Failed to resolve resource_dir")
            .join("resources/species_db.json")
    }
}

/// Strip Windows `\\?\` prefix from canonicalized paths.
fn strip_unc(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        p
    }
}

/// Dev-only: project root from CARGO_MANIFEST_DIR.
fn dev_project_dir() -> PathBuf {
    let p = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    strip_unc(p.canonicalize().unwrap_or(p))
}
