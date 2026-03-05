mod commands;
mod db;
mod exif;
mod scan;
mod thumbs;

use db::DbState;
use std::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let project_dir = commands::project_dir();

    // Ensure data directory exists
    std::fs::create_dir_all(project_dir.join("data")).ok();

    let db_path = project_dir.join("data/birds.db");
    let conn = db::init_db(&db_path).expect("Failed to initialize SQLite database");

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(DbState(Mutex::new(conn)))
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
            scan::prepare_scan,
            scan::store_photo_result,
            scan::get_model_paths,
            scan::finalize_scan,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
