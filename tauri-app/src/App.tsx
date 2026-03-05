import { convertFileSrc } from "@tauri-apps/api/core";
import "./App.css";

import SearchFilterBar from "./components/SearchFilterBar";
import ScanPanel from "./components/ScanPanel";
import SpeciesCard from "./components/SpeciesCard";
import DetailModal from "./components/DetailModal";
import UnknownPanel from "./components/UnknownPanel";
import useBirdData from "./hooks/useBirdData";

export default function App() {
  const {
    species, dataDir, loading, error,
    search, setSearch, filter, setFilter, sort, setSort,
    scanning, progress, thumbProgress, modelStatus, config,
    speciesDisplay, photosBySpecies, unknownPhotos, foundCount, visible,
    selected, setSelected,
    handleAddFolder, handleRemoveFolder, handleScan, cancelScan,
  } = useBirdData();

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
        thumbProgress={thumbProgress}
        modelStatus={modelStatus}
        folders={config.folders}
        onAddFolder={handleAddFolder}
        onRemoveFolder={handleRemoveFolder}
        onScan={() => handleScan()}
        onCancel={cancelScan}
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
