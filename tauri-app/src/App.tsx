import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { convertFileSrc } from "@tauri-apps/api/core";
import "./App.css";

import { Species, PhotoResult, AppConfig, UserPhoto, FilterMode, SortMode } from "./types";
import SearchFilterBar from "./components/SearchFilterBar";
import ScanPanel from "./components/ScanPanel";
import SpeciesCard from "./components/SpeciesCard";
import DetailModal from "./components/DetailModal";
import UnknownPanel from "./components/UnknownPanel";

export default function App() {
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

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-950 text-white">
        Chargement...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-950 text-red-400 text-sm p-8 text-center">
        <div>
          <p className="font-bold mb-2">Erreur de chargement</p>
          <p className="font-mono text-xs">{error}</p>
          <p className="text-gray-500 mt-2 text-xs">
            Lancez d'abord : <code>python generate_species_db.py</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-900 border-b border-gray-800 flex items-center gap-3">
        <h1 className="text-lg font-bold tracking-tight">🐦 Bird Pokédex</h1>
        <span className="text-sm text-gray-400">
          {foundCount}/{species.length} espèces observées
        </span>
      </div>

      {/* Scan panel */}
      <ScanPanel
        scanning={scanning}
        progress={progress}
        folders={config.folders}
        onAddFolder={handleAddFolder}
        onRemoveFolder={handleRemoveFolder}
        onScan={() => handleScan()}
      />

      {/* Search/filter bar */}
      <SearchFilterBar
        search={search}
        filter={filter}
        sort={sort}
        foundCount={foundCount}
        totalCount={species.length}
        onSearch={setSearch}
        onFilter={setFilter}
        onSort={setSort}
      />

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">
        {/* Unknown photos banner */}
        <UnknownPanel photos={unknownPhotos} speciesDisplay={speciesDisplay} />

        {/* Species grid */}
        <div className="p-4">
          {visible.length === 0 ? (
            <div className="text-center text-gray-500 mt-20 text-sm">
              Aucune espèce trouvée.
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {visible.map((s) => (
                <SpeciesCard
                  key={s.idx}
                  species={s}
                  photos={photosBySpecies.get(s.scientificName) ?? []}
                  dataDir={dataDir}
                  onClick={() => setSelected(s)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Detail modal for a known species */}
      {selected && (
        <DetailModal
          title={selected.frenchName || selected.scientificName}
          subtitle={selected.frenchName ? selected.scientificName : undefined}
          photos={photosBySpecies.get(selected.scientificName) ?? []}
          referenceImgSrc={
            selected.referencePhotoId !== null
              ? convertFileSrc(
                  `${dataDir}/images/${selected.referencePhotoId}.jpg`,
                )
              : undefined
          }
          occurrenceCount={selected.occurrenceCount}
          speciesDisplay={speciesDisplay}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
