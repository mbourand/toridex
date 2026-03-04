use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

use image::imageops::FilterType;
use image::ImageReader;

const THUMB_WIDTH: u32 = 300;

/// Deterministic thumbnail filename from original path.
fn thumb_name(path: &str) -> String {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("{:016x}.jpg", hasher.finish())
}

/// Generate thumbnails for photos that don't have one yet.
/// Returns `(original_path, thumb_filename)` pairs for DB update.
pub fn generate_thumbnails(paths: &[String], thumbs_dir: &Path) -> Vec<(String, String)> {
    fs::create_dir_all(thumbs_dir).ok();
    let mut results = Vec::new();

    for original in paths {
        let name = thumb_name(original);
        let dest = thumbs_dir.join(&name);

        // Skip if already exists on disk
        if dest.exists() {
            results.push((original.clone(), name));
            continue;
        }

        match generate_one(original, &dest) {
            Ok(()) => results.push((original.clone(), name)),
            Err(e) => eprintln!("Thumbnail failed for {original}: {e}"),
        }
    }
    results
}

fn generate_one(src: &str, dest: &Path) -> Result<(), String> {
    let img = ImageReader::open(src)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;

    let thumb = img.resize(THUMB_WIDTH, u32::MAX, FilterType::Lanczos3);
    thumb.save(dest).map_err(|e| e.to_string())?;
    Ok(())
}
