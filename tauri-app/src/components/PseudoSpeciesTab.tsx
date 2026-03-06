import { useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import DetailModal from "./DetailModal";
import { Species, UserPhoto } from "../types";

interface Props {
  title: string;
  photos: UserPhoto[];
  speciesDisplay: (sciName: string) => string;
  allSpecies: Species[];
  onSetSpecies: (path: string, species: string | null) => Promise<void>;
}

export default function PseudoSpeciesTab({
  title,
  photos,
  speciesDisplay,
  allSpecies,
  onSetSpecies,
}: Props) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  // Sort by date descending (most recent first): EXIF date preferred, file mtime as fallback
  const sorted = useMemo(() => {
    const dateKey = (p: UserPhoto): number => {
      if (p.result.exif_date) return new Date(p.result.exif_date).getTime() || 0;
      if (p.result.fileMtime) return p.result.fileMtime * 1000;
      return 0;
    };
    return [...photos].sort((a, b) => dateKey(b) - dateKey(a));
  }, [photos]);

  if (photos.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-gray-500 text-sm">Aucune photo dans cette catégorie.</p>
      </div>
    );
  }

  return (
    <>
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <p className="text-gray-400 text-sm mb-3">
            {sorted.length} photo{sorted.length > 1 ? "s" : ""}
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {sorted.map((p, i) => (
              <button
                key={p.path}
                onClick={() => setSelectedIdx(i)}
                className="group relative rounded-xl overflow-hidden bg-gray-800 aspect-square"
              >
                <img
                  src={convertFileSrc(p.result.thumbPath ?? p.path)}
                  className="w-full h-full object-cover transition-transform group-hover:scale-105"
                />
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                  <p className="text-white text-xs truncate">
                    {speciesDisplay(p.result.modelSpecies ?? p.result.scientificName)}
                  </p>
                  <p className="text-gray-400 text-[10px]">
                    {(p.result.confidence * 100).toFixed(0)}%
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {selectedIdx !== null && (
        <DetailModal
          title={title}
          photos={sorted}
          initialIndex={selectedIdx}
          speciesDisplay={speciesDisplay}
          allSpecies={allSpecies}
          onSetSpecies={async (path, sp) => {
            await onSetSpecies(path, sp);
            setSelectedIdx(null);
          }}
          onClose={() => setSelectedIdx(null)}
        />
      )}
    </>
  );
}
