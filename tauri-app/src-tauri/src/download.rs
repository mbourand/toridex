use futures_util::StreamExt;
use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

/// Base URL for model downloads from GitHub Releases.
/// Update this to your actual GitHub release URL before building.
const GITHUB_RELEASE_BASE: &str =
    "https://github.com/OWNER/REPO/releases/download/models-v1";

/// Files to download from the release.
const MODEL_FILES: &[&str] = &[
    "bird_detector.onnx",
    "bird_classifier.onnx",
    "label_map.json",
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub detector_ready: bool,
    pub classifier_ready: bool,
    pub label_map_ready: bool,
}

/// Check which models are present on disk.
#[tauri::command]
pub fn check_models(app: AppHandle) -> ModelStatus {
    let dir = crate::paths::models_dir(&app);
    ModelStatus {
        detector_ready: dir.join("bird_detector.onnx").exists(),
        classifier_ready: dir.join("bird_classifier.onnx").exists(),
        label_map_ready: dir.join("label_map.json").exists(),
    }
}

/// Download missing models from GitHub Releases.
/// Emits `download-progress` events: `{ file, downloaded, total }`.
#[tauri::command]
pub async fn download_models(app: AppHandle) -> Result<(), String> {
    let dir = crate::paths::models_dir(&app);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let client = reqwest::Client::new();

    for filename in MODEL_FILES {
        let dest = dir.join(filename);
        if dest.exists() {
            continue;
        }

        let url = format!("{GITHUB_RELEASE_BASE}/{filename}");
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to download {filename}: {e}"))?;

        if !response.status().is_success() {
            return Err(format!(
                "Download of {filename} failed: HTTP {}",
                response.status()
            ));
        }

        let total = response.content_length().unwrap_or(0);
        let tmp_path = dir.join(format!("{filename}.tmp"));
        let mut file = tokio::fs::File::create(&tmp_path)
            .await
            .map_err(|e| format!("Failed to create {filename}.tmp: {e}"))?;

        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        let mut last_emitted: u64 = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download error for {filename}: {e}"))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("Write error for {filename}: {e}"))?;
            downloaded += chunk.len() as u64;

            // Emit progress every ~256 KB
            if downloaded - last_emitted >= 256 * 1024 || downloaded == total {
                last_emitted = downloaded;
                let _ = app.emit(
                    "download-progress",
                    serde_json::json!({
                        "file": filename,
                        "downloaded": downloaded,
                        "total": total,
                    }),
                );
            }
        }

        file.flush().await.map_err(|e| e.to_string())?;
        drop(file);

        // Atomic-ish rename: .tmp → final
        std::fs::rename(&tmp_path, &dest)
            .map_err(|e| format!("Failed to finalize {filename}: {e}"))?;
    }

    Ok(())
}
