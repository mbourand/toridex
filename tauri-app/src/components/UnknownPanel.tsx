import { useState } from "react";
import { Species, UserPhoto } from "../types";
import DetailModal from "./DetailModal";

interface Props {
  photos: UserPhoto[];
  speciesDisplay: (sciName: string) => string;
  allSpecies: Species[];
  onSetSpecies: (path: string, species: string | null) => void;
}

export default function UnknownPanel({ photos, speciesDisplay, allSpecies, onSetSpecies }: Props) {
  const [open, setOpen] = useState(false);

  if (photos.length === 0) return null;

  function handleSetSpecies(path: string, species: string | null) {
    onSetSpecies(path, species);
    // Close if no more unknown photos will remain after this action
    // (the parent will re-render with updated photos list)
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="mx-4 mt-3 flex items-center gap-2 w-fit px-3 py-2 rounded-lg border border-yellow-500/40 bg-yellow-500/10 hover:bg-yellow-500/20 transition-colors text-sm"
      >
        <span className="text-yellow-400">⚠</span>
        <span className="text-yellow-300 font-semibold">
          {photos.length} photo{photos.length > 1 ? "s" : ""} incertaine
          {photos.length > 1 ? "s" : ""}
        </span>
        <span className="text-gray-500 text-xs">— cliquez pour voir</span>
      </button>

      {open && (
        <DetailModal
          title="Photos incertaines"
          subtitle="Confiance insuffisante pour identifier l'espece"
          photos={photos}
          speciesDisplay={speciesDisplay}
          allSpecies={allSpecies}
          onSetSpecies={handleSetSpecies}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
