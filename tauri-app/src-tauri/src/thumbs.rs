use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::ImageReader;
use rayon::prelude::*;

const THUMB_WIDTH: u32 = 800;
const JPEG_QUALITY: u8 = 90;

/// Deterministic thumbnail filename from original path.
pub fn thumb_name(path: &str) -> String {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("{:016x}.jpg", hasher.finish())
}

/// Generate thumbnails for photos that don't have one yet.
/// Returns `(original_path, thumb_filename)` pairs for DB update.
/// Calls `on_progress(current, total)` after each thumbnail.
pub fn generate_thumbnails(
    paths: &[String],
    thumbs_dir: &Path,
    on_progress: impl Fn(usize, usize) + Sync,
) -> Vec<(String, String)> {
    log::info!("generate_thumbnails: {} photos, dir={}", paths.len(), thumbs_dir.display());
    fs::create_dir_all(thumbs_dir).ok();
    let total = paths.len();
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);

    let results: Vec<_> = paths
        .par_iter()
        .filter_map(|original| {
            let name = thumb_name(original);
            let dest = thumbs_dir.join(&name);

            let ok = if dest.exists() {
                true
            } else {
                match generate_one(original, &dest) {
                    Ok(()) => true,
                    Err(e) => {
                        log::warn!("Thumbnail failed for {original}: {e}");
                        failed.fetch_add(1, Ordering::Relaxed);
                        false
                    }
                }
            };

            let current = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(current, total);

            ok.then(|| (original.clone(), name))
        })
        .collect();

    let fail_count = failed.load(Ordering::Relaxed);
    log::info!("generate_thumbnails: done — {} succeeded, {} failed", results.len(), fail_count);
    results
}

fn generate_one(src: &str, dest: &Path) -> Result<(), String> {
    let img = ImageReader::open(src)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;

    let thumb = img.resize(THUMB_WIDTH, u32::MAX, FilterType::Lanczos3);

    let file = fs::File::create(dest).map_err(|e| e.to_string())?;
    let encoder = JpegEncoder::new_with_quality(file, JPEG_QUALITY);
    thumb.write_with_encoder(encoder).map_err(|e| e.to_string())?;
    Ok(())
}
