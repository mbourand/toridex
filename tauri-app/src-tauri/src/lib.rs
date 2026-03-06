mod commands;
mod db;
mod download;
mod exif;
mod paths;
mod scan;
mod thumbs;

use db::DbState;
use std::sync::Mutex;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();

            // Ensure data directories exist
            let data = paths::data_dir(&handle);
            std::fs::create_dir_all(&data).ok();
            std::fs::create_dir_all(data.join("models")).ok();
            std::fs::create_dir_all(data.join("thumbs")).ok();

            // Initialize SQLite
            let db_path = paths::db_path(&handle);
            let conn = db::init_db(&db_path).expect("Failed to initialize SQLite database");
            app.manage(DbState(Mutex::new(conn)));

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
            commands::remove_folder_photos,
            commands::check_missing_photos,
            commands::relocate_missing_photos,
            commands::purge_missing_photos,
            commands::prepare_full_rescan,
            commands::get_label_conflicts,
            commands::resolve_label_conflicts,
            scan::prepare_scan,
            scan::store_photo_result,
            scan::get_model_paths,
            scan::finalize_scan,
            download::check_models,
            download::download_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
