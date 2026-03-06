import "./App.css";

import { Routes, Route, NavLink } from "react-router-dom";

import ScanPanel from "./components/ScanPanel";
import MissingPhotosModal from "./components/MissingPhotosModal";
import LabelConflictModal from "./components/LabelConflictModal";
import PokedexTab from "./components/PokedexTab";
import PseudoSpeciesTab from "./components/PseudoSpeciesTab";
import useBirdData from "./hooks/useBirdData";

function NavTab({
  to,
  label,
  count,
}: {
  to: string;
  label: string;
  count?: number;
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-1.5 text-sm rounded-lg transition-colors flex items-center gap-1.5 ${
          isActive
            ? "bg-gray-700 text-white"
            : "text-gray-400 hover:text-white hover:bg-gray-800"
        }`
      }
    >
      {label}
      {count !== undefined && count > 0 && (
        <span className="bg-gray-600 text-gray-200 text-xs px-1.5 py-0.5 rounded-full leading-none">
          {count}
        </span>
      )}
    </NavLink>
  );
}

export default function App() {
  const {
    species,
    loading,
    error,
    search,
    setSearch,
    filter,
    setFilter,
    sort,
    setSort,
    scanning,
    progress,
    thumbProgress,
    modelStatus,
    config,
    speciesDisplay,
    photosBySpecies,
    unknownPhotos,
    noBirdPhotos,
    unlistedPhotos,
    foundCount,
    visible,
    selected,
    setSelected,
    showMissingPhotosModal,
    missingPhotosCount,
    missingPhotosStatus,
    relocatedPhotosCount,
    purgedPhotosCount,
    handleAddSearchFolder,
    handleMissingPhotosDone,
    handleSkipMissingPhotos,
    handleAddFolder,
    handleRemoveFolder,
    handleScan,
    handleFullRescan,
    cancelScan,
    handleSetUserSpecies,
    frontPhotos,
    handleSetFrontPhoto,
    labelConflicts,
    showLabelConflictModal,
    handleResolveConflicts,
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
        <h1 className="text-lg font-bold tracking-tight">🐦 ToriDex</h1>
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
        onFullRescan={handleFullRescan}
        onCancel={cancelScan}
      />

      {/* Navigation tabs */}
      <div className="px-4 py-2 bg-gray-900/50 border-b border-gray-800 flex items-center gap-2 overflow-x-auto">
        <NavTab to="/" label="Mon Dex" count={foundCount} />
        <NavTab
          to="/uncertain"
          label="Incertains"
          count={unknownPhotos.length}
        />
        <NavTab
          to="/unlisted"
          label="Non répertoriées"
          count={unlistedPhotos.length}
        />
        <NavTab
          to="/no-bird"
          label="Pas d'oiseau"
          count={noBirdPhotos.length}
        />
      </div>

      {/* Route content */}
      <Routes>
        <Route
          path="/"
          element={
            <PokedexTab
              species={species}
              visible={visible}
              photosBySpecies={photosBySpecies}
              foundCount={foundCount}
              search={search}
              setSearch={setSearch}
              filter={filter}
              setFilter={setFilter}
              sort={sort}
              setSort={setSort}
              selected={selected}
              setSelected={setSelected}
              speciesDisplay={speciesDisplay}
              handleSetUserSpecies={handleSetUserSpecies}
              frontPhotos={frontPhotos}
              handleSetFrontPhoto={handleSetFrontPhoto}
            />
          }
        />
        <Route
          path="/uncertain"
          element={
            <PseudoSpeciesTab
              title="Incertains"
              photos={unknownPhotos}
              speciesDisplay={speciesDisplay}
              allSpecies={species}
              onSetSpecies={handleSetUserSpecies}
            />
          }
        />
        <Route
          path="/unlisted"
          element={
            <PseudoSpeciesTab
              title="Espèce non répertoriée"
              photos={unlistedPhotos}
              speciesDisplay={speciesDisplay}
              allSpecies={species}
              onSetSpecies={handleSetUserSpecies}
            />
          }
        />
        <Route
          path="/no-bird"
          element={
            <PseudoSpeciesTab
              title="Pas d'oiseau"
              photos={noBirdPhotos}
              speciesDisplay={speciesDisplay}
              allSpecies={species}
              onSetSpecies={handleSetUserSpecies}
            />
          }
        />
      </Routes>

      {/* Missing photos modal */}
      {showMissingPhotosModal && (
        <MissingPhotosModal
          missingCount={missingPhotosCount}
          status={missingPhotosStatus}
          relocatedCount={relocatedPhotosCount}
          purgedCount={purgedPhotosCount}
          onAddSearchFolder={handleAddSearchFolder}
          onDone={handleMissingPhotosDone}
          onSkip={handleSkipMissingPhotos}
        />
      )}

      {/* Label conflict review modal (after full rescan) */}
      {showLabelConflictModal && labelConflicts.length > 0 && (
        <LabelConflictModal
          conflicts={labelConflicts}
          speciesDisplay={speciesDisplay}
          onResolve={handleResolveConflicts}
        />
      )}
    </div>
  );
}
