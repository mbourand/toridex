use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use tauri::{AppHandle, Emitter};
use tauri_plugin_dialog::DialogExt;

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
pub fn load_scan_results() -> Result<Option<String>, String> {
    let path = project_dir().join("data/scan_results.json");
    if path.exists() {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Could not read scan_results.json: {e}"))?;
        Ok(Some(content))
    } else {
        Ok(None)
    }
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

/// Scan events emitted to the frontend:
///   "scan-progress"  →  { current: u32, total: u32 }
///   "scan-complete"  →  null
///   "scan-error"     →  string
#[tauri::command]
pub async fn scan_photos_folder(app: AppHandle, folder: String) -> Result<(), String> {
    let python = python_exe();
    let script = project_dir().join("analyze_photos.py");
    let output = project_dir().join("data/scan_results.json");
    let proj = project_dir();

    let mut child = Command::new(&python)
        .arg(&script)
        .arg("--folder")
        .arg(&folder)
        .arg("--output")
        .arg(&output)
        .arg("--top")
        .arg("5")
        .current_dir(&proj)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
                let _ = app.emit("scan-progress", serde_json::json!({ "current": current, "total": total }));
            }
        } else if line.starts_with("DONE:") {
            let _ = app.emit("scan-complete", serde_json::json!(null));
        }
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        let _ = app.emit("scan-error", "Python process exited with error");
    }

    Ok(())
}
