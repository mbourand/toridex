mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::load_species_db,
            commands::load_scan_results,
            commands::get_data_dir,
            commands::scan_photos_folder,
            commands::open_folder_dialog,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
