import { useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import {
  Species,
  PhotoResult,
  AppConfig,
  UserPhoto,
  FilterMode,
  SortMode,
  PreparedScan,
  ModelPaths,
  ModelStatus,
  FullRescanInfo,
  LabelConflict,
} from "../types";
import { initInferenceWorker, processImage } from "../inference";

const DETECT_THRESHOLD = 0.3;
const MIN_CONFIDENCE = 0.5;
const TOP_K = 5;

export default function useBirdData() {
  const [species, setSpecies] = useState<Species[]>([]);
  const [scanResults, setScanResults] = useState<Record<
    string,
    PhotoResult
  > | null>(null);
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
  const [thumbProgress, setThumbProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);
  const [modelStatus, setModelStatus] = useState("");
  const [config, setConfig] = useState<AppConfig>({ folders: [] });

  // Missing photos state
  const [missingPhotos, setMissingPhotos] = useState<string[]>([]);
  const [showMissingPhotosModal, setShowMissingPhotosModal] = useState(false);
  const [missingPhotosStatus, setMissingPhotosStatus] = useState<"pending" | "searching" | "done">("pending");
  const [relocatedPhotosCount, setRelocatedPhotosCount] = useState(0);
  const [purgedPhotosCount, setPurgedPhotosCount] = useState(0);

  // Label conflict review (after full rescan)
  const [labelConflicts, setLabelConflicts] = useState<LabelConflict[]>([]);
  const [showLabelConflictModal, setShowLabelConflictModal] = useState(false);

  const [selected, setSelected] = useState<Species | null>(null);

  // Abort controller for cancelling scans
  const abortRef = useRef<AbortController | null>(null);

  // Load data on mount
  useEffect(() => {
    async function load() {
      try {
        setSpecies(await invoke<Species[]>("load_species_db"));
        setScanResults(
          await invoke<Record<string, PhotoResult> | null>("load_scan_results"),
        );
        const cfg = await invoke<AppConfig>("load_config");
        setConfig(cfg);

        // Check for missing photos on disk
        const missingPaths = await invoke<string[]>("check_missing_photos");
        if (missingPaths.length > 0) {
          setMissingPhotos(missingPaths);
          setShowMissingPhotosModal(true);
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // scientificName → display name lookup (used by DetailModal for top-k labels)
  const speciesDisplay = useMemo(() => {
    const map = new Map<string, string>();
    for (const s of species) {
      map.set(s.scientificName, s.frenchName || s.scientificName);
    }
    // Pseudo-species display names
    map.set("__unknown__", "Incertain");
    map.set("__skipped__", "Pas d'oiseau");
    map.set("__no_bird__", "Pas d'oiseau");
    map.set("__unlisted__", "Espèce non répertoriée");
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

  const PSEUDO_SPECIES = new Set(["__unknown__", "__skipped__", "__no_bird__", "__unlisted__"]);

  const unknownPhotos = photosBySpecies.get("__unknown__") ?? [];
  const noBirdPhotos = useMemo(() => [
    ...(photosBySpecies.get("__skipped__") ?? []),
    ...(photosBySpecies.get("__no_bird__") ?? []),
  ], [photosBySpecies]);
  const unlistedPhotos = photosBySpecies.get("__unlisted__") ?? [];
  const foundCount = [...photosBySpecies.keys()].filter(
    (k) => !PSEUDO_SPECIES.has(k),
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
    if (
      !loading &&
      config.folders.length > 0 &&
      !scanning &&
      !autoScannedRef.current &&
      !showMissingPhotosModal
    ) {
      autoScannedRef.current = true;
      handleScan(config.folders);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, config.folders, showMissingPhotosModal]);

  async function handleAddFolder() {
    const picked = await invoke<string | null>("open_folder_dialog");
    if (!picked || config.folders.includes(picked)) return;

    const updated = { folders: [...config.folders, picked] };
    setConfig(updated);
    await invoke("save_config", { folders: updated.folders });
  }

  async function handleRemoveFolder(folder: string) {
    const remainingFolders = config.folders.filter((f) => f !== folder);
    setConfig({ folders: remainingFolders });
    await invoke("save_config", { folders: remainingFolders });
    await invoke("remove_folder_photos", { folder, remainingFolders });
    setScanResults(
      await invoke<Record<string, PhotoResult> | null>("load_scan_results"),
    );
  }

  // --- Missing photos handlers ---

  async function handleAddSearchFolder() {
    const picked = await invoke<string | null>("open_folder_dialog");
    if (!picked) return;

    setMissingPhotosStatus("searching");

    try {
      // Try to relocate missing photos by searching in this folder
      const stillMissing = await invoke<string[]>("relocate_missing_photos", {
        missingPaths: missingPhotos,
        searchFolders: [picked],
      });

      const found = missingPhotos.length - stillMissing.length;
      setRelocatedPhotosCount((prev) => prev + found);
      setMissingPhotos(stillMissing);

      // Add the folder to config if it's not already there
      if (!config.folders.includes(picked)) {
        const updated = { folders: [...config.folders, picked] };
        setConfig(updated);
        await invoke("save_config", { folders: updated.folders });
      }
    } finally {
      setMissingPhotosStatus("pending");
    }
  }

  async function handleMissingPhotosDone() {
    // Purge any remaining missing photos
    if (missingPhotos.length > 0) {
      const purged = await invoke<number>("purge_missing_photos", {
        paths: missingPhotos,
      });
      setPurgedPhotosCount(purged);
    }

    setMissingPhotosStatus("done");
    setShowMissingPhotosModal(false);

    // Refresh scan results after cleanup
    setScanResults(
      await invoke<Record<string, PhotoResult> | null>("load_scan_results"),
    );
  }

  function handleSkipMissingPhotos() {
    // Close modal without purging — photos stay in DB for next launch
    setShowMissingPhotosModal(false);
  }

  async function handleScan(folders?: string[]) {
    const foldersToScan = folders ?? config.folders;
    if (foldersToScan.length === 0 || scanning) return;

    const abort = new AbortController();
    abortRef.current = abort;

    setScanning(true);
    setProgress(null);
    setModelStatus("");

    try {
      // Step 1: Prepare — collect image paths, filter unchanged
      const prepared = await invoke<PreparedScan>("prepare_scan", {
        folders: foldersToScan,
      });

      if (prepared.toProcess.length === 0) {
        await invoke("finalize_scan", { folders: foldersToScan });
        return;
      }

      // Step 2: Download models if needed, then load in Web Worker
      const status = await invoke<ModelStatus>("check_models");
      if (!status.detectorReady || !status.classifierReady || !status.labelMapReady) {
        setModelStatus("Téléchargement des modèles...");
        let unlistenDl: UnlistenFn | undefined;
        try {
          unlistenDl = await listen<{ file: string; downloaded: number; total: number }>(
            "download-progress",
            (e) => {
              const { file, downloaded, total } = e.payload;
              const pct = total > 0 ? Math.round((downloaded / total) * 100) : 0;
              const mb = (downloaded / 1_048_576).toFixed(0);
              const totalMb = (total / 1_048_576).toFixed(0);
              setModelStatus(`Téléchargement ${file}... ${mb}/${totalMb} Mo (${pct}%)`);
            },
          );
          await invoke("download_models");
        } finally {
          unlistenDl?.();
        }
      }

      const modelPaths = await invoke<ModelPaths>("get_model_paths");
      await initInferenceWorker(
        convertFileSrc(modelPaths.detector),
        convertFileSrc(modelPaths.classifier),
        convertFileSrc(modelPaths.labelMap),
        setModelStatus,
      );

      // Step 3: Process each image (inference runs in worker, UI stays responsive)
      const total = prepared.toProcess.length;

      for (let i = 0; i < total; i++) {
        if (abort.signal.aborted) break;

        const file = prepared.toProcess[i];
        setProgress({ current: i + 1, total });

        try {
          const result = await processImage({
            url: convertFileSrc(file.path),
            path: file.path,
            folder: file.folder,
            fileMtime: file.fileMtime,
            fileSize: file.fileSize,
            detectThreshold: DETECT_THRESHOLD,
            minConfidence: MIN_CONFIDENCE,
            topK: TOP_K,
          });

          await invoke("store_photo_result", {
            path: result.path,
            folder: result.folder,
            species: result.species,
            speciesIdx: result.speciesIdx,
            confidence: result.confidence,
            fileMtime: result.fileMtime,
            fileSize: result.fileSize,
            topKJson: result.topKJson,
          });
        } catch (err) {
          console.error(`Failed to process ${file.path}:`, err);
        }
      }

      // Step 4: Cleanup + thumbnails
      let unlistenThumbs: UnlistenFn | undefined;
      try {
        unlistenThumbs = await listen<{ current: number; total: number }>(
          "thumb-progress",
          (e) => setThumbProgress(e.payload),
        );
        await invoke("finalize_scan", { folders: foldersToScan });
      } finally {
        unlistenThumbs?.();
        setThumbProgress(null);
      }
    } catch (e) {
      console.error("Scan error:", e);
    } finally {
      setScanning(false);
      setProgress(null);
      setModelStatus("");
      abortRef.current = null;
      setScanResults(
        await invoke<Record<string, PhotoResult> | null>("load_scan_results"),
      );
    }
  }

  function cancelScan() {
    abortRef.current?.abort();
  }

  // --- Full rescan ---

  async function handleFullRescan() {
    if (scanning || config.folders.length === 0) return;

    // Step 1: Reset DB state (purge missing, delete thumbs, reset mtime)
    await invoke<FullRescanInfo>("prepare_full_rescan");

    // Step 2: Run normal scan (all photos will be re-processed since mtime=0)
    await handleScan(config.folders);

    // Step 3: Check for label conflicts
    const conflicts = await invoke<LabelConflict[]>("get_label_conflicts");
    if (conflicts.length > 0) {
      setLabelConflicts(conflicts);
      setShowLabelConflictModal(true);
    }
  }

  async function handleSetUserSpecies(path: string, species: string | null) {
    await invoke("set_user_species", { path, species });
    setScanResults(
      await invoke<Record<string, PhotoResult> | null>("load_scan_results"),
    );
  }

  async function handleResolveConflicts(acceptModelPaths: string[]) {
    if (acceptModelPaths.length > 0) {
      await invoke("resolve_label_conflicts", { acceptModelPaths });
    }
    setShowLabelConflictModal(false);
    setLabelConflicts([]);

    // Refresh scan results to reflect resolved conflicts
    setScanResults(
      await invoke<Record<string, PhotoResult> | null>("load_scan_results"),
    );
  }

  return {
    // Data
    species,
    loading,
    error,
    // Search/filter
    search,
    setSearch,
    filter,
    setFilter,
    sort,
    setSort,
    // Scan state
    scanning,
    progress,
    thumbProgress,
    modelStatus,
    config,
    // Derived
    speciesDisplay,
    photosBySpecies,
    unknownPhotos,
    noBirdPhotos,
    unlistedPhotos,
    foundCount,
    visible,
    // Selection
    selected,
    setSelected,
    // Missing photos
    showMissingPhotosModal,
    missingPhotosCount: missingPhotos.length,
    missingPhotosStatus,
    relocatedPhotosCount,
    purgedPhotosCount,
    handleAddSearchFolder,
    handleMissingPhotosDone,
    handleSkipMissingPhotos,
    // Actions
    handleAddFolder,
    handleRemoveFolder,
    handleScan,
    handleFullRescan,
    cancelScan,
    // Manual categorization
    handleSetUserSpecies,
    // Label conflicts
    labelConflicts,
    showLabelConflictModal,
    handleResolveConflicts,
  };
}
