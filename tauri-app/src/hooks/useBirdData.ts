import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import { Species, PhotoResult, AppConfig, UserPhoto, FilterMode, SortMode } from "../types";

export default function useBirdData() {
  const [species, setSpecies] = useState<Species[]>([]);
  const [scanResults, setScanResults] = useState<Record<string, PhotoResult> | null>(null);
  const [dataDir, setDataDir] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [sort, setSort] = useState<SortMode>("name");

  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);
  const [config, setConfig] = useState<AppConfig>({ folders: [] });

  const [selected, setSelected] = useState<Species | null>(null);

  // Load data on mount
  useEffect(() => {
    async function load() {
      try {
        const dir = await invoke<string>("get_data_dir");
        setDataDir(dir);

        setSpecies(await invoke<Species[]>("load_species_db"));
        setScanResults(await invoke<Record<string, PhotoResult> | null>("load_scan_results"));
        setConfig(await invoke<AppConfig>("load_config"));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Listen for real-time scan progress
  useEffect(() => {
    const unlisten = listen<{ current: number; total: number }>(
      "scan-progress",
      (e) => setProgress(e.payload),
    );
    return () => { unlisten.then((f) => f()); };
  }, []);

  // scientificName → display name lookup (used by DetailModal for top-k labels)
  const speciesDisplay = useMemo(() => {
    const map = new Map<string, string>();
    for (const s of species) {
      map.set(s.scientificName, s.frenchName || s.scientificName);
    }
    return (sciName: string) => map.get(sciName) ?? sciName;
  }, [species]);

  // All photos grouped by scientificName (including "__unknown__").
  // Each group is sorted by confidence descending so the best shot comes first.
  const photosBySpecies = useMemo<Map<string, UserPhoto[]>>(() => {
    const map = new Map<string, UserPhoto[]>();
    if (!scanResults) return map;
    for (const [path, result] of Object.entries(scanResults)) {
      const key = result.scientificName;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push({ path, result });
    }
    for (const arr of map.values()) {
      arr.sort((a, b) => b.result.confidence - a.result.confidence);
    }
    return map;
  }, [scanResults]);

  const unknownPhotos = photosBySpecies.get("__unknown__") ?? [];
  const foundCount = [...photosBySpecies.keys()].filter(
    (k) => k !== "__unknown__",
  ).length;

  // Filter + search + sort
  const visible = useMemo(() => {
    let list = species;

    if (filter === "found")
      list = list.filter((s) => photosBySpecies.has(s.scientificName));
    else if (filter === "not-found")
      list = list.filter((s) => !photosBySpecies.has(s.scientificName));

    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (s) =>
          s.frenchName.toLowerCase().includes(q) ||
          s.scientificName.toLowerCase().includes(q) ||
          s.epithet.toLowerCase().includes(q),
      );
    }

    list = [...list].sort((a, b) => {
      if (sort === "name") {
        return (a.frenchName || a.scientificName).localeCompare(
          b.frenchName || b.scientificName,
          "fr",
        );
      }
      if (sort === "rarity") {
        return a.occurrenceCount - b.occurrenceCount;
      }
      // date: found with a date first (most recent first), unfound last
      const da =
        photosBySpecies.get(a.scientificName)?.[0]?.result.exif_date ?? "";
      const db =
        photosBySpecies.get(b.scientificName)?.[0]?.result.exif_date ?? "";
      if (da && db) return db.localeCompare(da);
      if (da) return -1;
      if (db) return 1;
      return 0;
    });

    return list;
  }, [species, filter, search, sort, photosBySpecies]);

  // Auto-scan on launch if folders are configured
  const autoScannedRef = useRef(false);
  useEffect(() => {
    if (!loading && config.folders.length > 0 && !scanning && !autoScannedRef.current) {
      autoScannedRef.current = true;
      handleScan(config.folders);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, config.folders]);

  async function handleAddFolder() {
    const picked = await invoke<string | null>("open_folder_dialog");
    if (picked && !config.folders.includes(picked)) {
      const updated = { folders: [...config.folders, picked] };
      setConfig(updated);
      await invoke("save_config", { folders: updated.folders });
    }
  }

  async function handleRemoveFolder(folder: string) {
    const updated = { folders: config.folders.filter(f => f !== folder) };
    setConfig(updated);
    await invoke("save_config", { folders: updated.folders });
    await invoke("delete_photos_by_folder", { folder });
    setScanResults(await invoke<Record<string, PhotoResult> | null>("load_scan_results"));
  }

  async function handleScan(folders?: string[]) {
    const foldersToScan = folders ?? config.folders;
    if (foldersToScan.length === 0 || scanning) return;
    setScanning(true);
    setProgress({ current: 0, total: 0 });
    try {
      await invoke("scan_photos_folder", { folders: foldersToScan });
    } catch (e) {
      console.error(e);
    } finally {
      setScanning(false);
      setProgress(null);
      setScanResults(await invoke<Record<string, PhotoResult> | null>("load_scan_results"));
    }
  }

  return {
    // Data
    species, dataDir, loading, error,
    // Search/filter
    search, setSearch, filter, setFilter, sort, setSort,
    // Scan state
    scanning, progress, config,
    // Derived
    speciesDisplay, photosBySpecies, unknownPhotos, foundCount, visible,
    // Selection
    selected, setSelected,
    // Actions
    handleAddFolder, handleRemoveFolder, handleScan,
  };
}
