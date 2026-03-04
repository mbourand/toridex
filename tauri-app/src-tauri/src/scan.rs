use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use tauri::{AppHandle, Emitter};

use crate::commands::project_dir;
use crate::db::{self, DbState};
use crate::thumbs;

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
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("Python scan failed".into());
    }

    // Phase 2: Import JSON into SQLite
    {
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        db::photos::upsert_photos_from_json(&conn, &output)?;
    }

    // Phase 3: Generate thumbnails
    let thumbs_dir = project_dir().join("data/thumbs");
    let needing_thumbs = {
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        db::photos::get_photos_needing_thumbnails(&conn)
    };

    if !needing_thumbs.is_empty() {
        let generated = thumbs::generate_thumbnails(&needing_thumbs, &thumbs_dir);
        let conn = db.0.lock().map_err(|e| e.to_string())?;
        for (orig, thumb_name) in &generated {
            let _ = db::photos::set_thumb_path(&conn, orig, thumb_name);
        }
    }

    Ok(())
}
