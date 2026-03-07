mod commands;
mod db;
mod exif;
mod paths;
mod scan;
mod thumbs;

use db::DbState;
use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_log::{Target, TargetKind};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    Target::new(TargetKind::LogDir { file_name: Some("app".into()) }),
                    Target::new(TargetKind::Stdout),
                ])
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();

            // Ensure data directories exist
            let data = paths::data_dir(&handle);
            std::fs::create_dir_all(&data).ok();
            std::fs::create_dir_all(data.join("thumbs")).ok();

            log::info!("=== App started ===");
            log::info!("Data dir: {}", data.display());
            log::info!("Thumbs dir: {}", paths::thumbs_dir(&handle).display());
            log::info!("Models dir: {}", paths::models_dir(&handle).display());
            log::info!("Species DB: {}", paths::species_db_path(&handle).display());

            // Initialize SQLite
            let db_path = paths::db_path(&handle);
            log::info!("DB path: {}", db_path.display());
            let conn = db::init_db(&db_path).expect("Failed to initialize SQLite database");
            app.manage(DbState(Mutex::new(conn)));

            log::info!("Setup complete");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_species_db,
            commands::load_scan_results,
            commands::get_data_dir,
            commands::open_folder_dialog,
            commands::load_config,
            commands::save_config,
            commands::set_user_species,
            commands::set_front_photo,
            commands::get_front_photos,
            commands::remove_folder_photos,
            commands::check_missing_photos,
            commands::relocate_missing_photos,
            commands::purge_missing_photos,
            commands::prepare_full_rescan,
            commands::get_label_conflicts,
            commands::resolve_label_conflicts,
            scan::prepare_scan,
            scan::store_photo_result,
            scan::read_file_bytes,
            scan::get_model_paths,
            scan::finalize_scan,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
