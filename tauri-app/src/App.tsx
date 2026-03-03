import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { convertFileSrc } from "@tauri-apps/api/core";
import "./App.css";

import { Species, ScanResults, UserPhoto, FilterMode, SortMode, AmbiguousAlternative } from "./types";
import SearchFilterBar from "./components/SearchFilterBar";
import ScanPanel from "./components/ScanPanel";
import SpeciesCard from "./components/SpeciesCard";
import DetailModal from "./components/DetailModal";
import UnknownPanel from "./components/UnknownPanel";

export default function App() {
  const [species, setSpecies] = useState<Species[]>([]);
  const [scanResults, setScanResults] = useState<ScanResults | null>(null);
  const [dataDir, setDataDir] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [sort, setSort] = useState<SortMode>("name");

  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [folder, setFolder] = useState("");

  const [selected, setSelected] = useState<Species | null>(null);

  // Load data on mount
  useEffect(() => {
    async function load() {
      try {
        const dir = await invoke<string>("get_data_dir");
        setDataDir(dir);

        const dbJson = await invoke<string>("load_species_db");
        setSpecies(JSON.parse(dbJson));

        const resultsJson = await invoke<string | null>("load_scan_results");
        if (resultsJson) setScanResults(JSON.parse(resultsJson));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Listen for scan events
  useEffect(() => {
    const unlistenProgress = listen<{ current: number; total: number }>(
      "scan-progress",
      e => setProgress(e.payload)
    );
    const unlistenComplete = listen("scan-complete", async () => {
      setScanning(false);
      setProgress(null);
      const resultsJson = await invoke<string | null>("load_scan_results");
      if (resultsJson) setScanResults(JSON.parse(resultsJson));
    });
    const unlistenError = listen("scan-error", () => {
      setScanning(false);
      setProgress(null);
    });

    return () => {
      unlistenProgress.then(f => f());
      unlistenComplete.then(f => f());
      unlistenError.then(f => f());
    };
  }, []);

  // epithet → display name lookup (used by DetailModal for top-k labels)
  const epithetToName = useMemo(() => {
    const map = new Map<string, string>();
    for (const s of species) {
      map.set(s.epithet, s.frenchName || s.scientificName);
    }
    return (epithet: string) => map.get(epithet) ?? epithet;
  }, [species]);

  // epithet → alternative species that share the same epithet (ambiguous classes)
  const epithetToAmbiguous = useMemo(() => {
    const map = new Map<string, AmbiguousAlternative[]>();
    for (const s of species) {
      if (s.ambiguousAlternatives?.length > 0) {
        map.set(s.epithet, s.ambiguousAlternatives);
      }
    }
    return (epithet: string) => map.get(epithet);
  }, [species]);

  // All photos grouped by epithet (including "__unknown__").
  // Each group is sorted by confidence descending so the best shot comes first.
  const photosByEpithet = useMemo<Map<string, UserPhoto[]>>(() => {
    const map = new Map<string, UserPhoto[]>();
    if (!scanResults) return map;
    for (const [path, result] of Object.entries(scanResults.photos)) {
      const key = result.epithet;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push({ path, result });
    }
    for (const arr of map.values()) {
      arr.sort((a, b) => b.result.confidence - a.result.confidence);
    }
    return map;
  }, [scanResults]);

  const unknownPhotos = photosByEpithet.get("__unknown__") ?? [];
  const foundCount = [...photosByEpithet.keys()].filter(k => k !== "__unknown__").length;

  // Filter + search + sort
  const visible = useMemo(() => {
    let list = species;

    if (filter === "found") list = list.filter(s => photosByEpithet.has(s.epithet));
    else if (filter === "not-found") list = list.filter(s => !photosByEpithet.has(s.epithet));

    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        s =>
          s.frenchName.toLowerCase().includes(q) ||
          s.scientificName.toLowerCase().includes(q) ||
          s.epithet.toLowerCase().includes(q)
      );
    }

    list = [...list].sort((a, b) => {
      if (sort === "name") {
        return (a.frenchName || a.scientificName).localeCompare(
          b.frenchName || b.scientificName, "fr"
        );
      }
      if (sort === "rarity") {
        return a.occurrenceCount - b.occurrenceCount;
      }
      // date: found with a date first (most recent first), unfound last
      const da = photosByEpithet.get(a.epithet)?.[0]?.result.exif_date ?? "";
      const db = photosByEpithet.get(b.epithet)?.[0]?.result.exif_date ?? "";
      if (da && db) return db.localeCompare(da);
      if (da) return -1;
      if (db) return 1;
      return 0;
    });

    return list;
  }, [species, filter, search, sort, photosByEpithet]);

  async function handlePickFolder() {
    const picked = await invoke<string | null>("open_folder_dialog");
    if (picked) setFolder(picked);
  }

  async function handleScan() {
    if (!folder || scanning) return;
    setScanning(true);
    setProgress({ current: 0, total: 0 });
    try {
      await invoke("scan_photos_folder", { folder });
    } catch (e) {
      setScanning(false);
      setProgress(null);
      console.error(e);
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
        lastFolder={folder || scanResults?.folder || ""}
        onPickFolder={handlePickFolder}
        onScan={handleScan}
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
        <UnknownPanel photos={unknownPhotos} epithetToName={epithetToName} />

        {/* Species grid */}
        <div className="p-4">
          {visible.length === 0 ? (
            <div className="text-center text-gray-500 mt-20 text-sm">
              Aucune espèce trouvée.
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {visible.map(s => (
                <SpeciesCard
                  key={s.idx}
                  species={s}
                  photos={photosByEpithet.get(s.epithet) ?? []}
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
          photos={photosByEpithet.get(selected.epithet) ?? []}
          referenceImgSrc={
            selected.referencePhotoId !== null
              ? convertFileSrc(`${dataDir}/images/${selected.referencePhotoId}.jpg`)
              : undefined
          }
          occurrenceCount={selected.occurrenceCount}
          epithetToName={epithetToName}
          epithetToAmbiguous={epithetToAmbiguous}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
