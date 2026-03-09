import { useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import DetailModal from "./DetailModal";
import { Species, UserPhoto } from "../types";
import { useVirtualizedGrid } from "../hooks/useVirtualizedGrid";

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

  const { scrollRef, virtualizer, columns } = useVirtualizedGrid(sorted.length, 220);

  if (photos.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-gray-500 text-sm">Aucune photo dans cette catégorie.</p>
      </div>
    );
  }

  return (
    <>
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="p-4">
          <p className="text-gray-400 text-sm mb-3">
            {sorted.length} photo{sorted.length > 1 ? "s" : ""}
          </p>
          <div
            style={{
              height: virtualizer.getTotalSize(),
              position: "relative",
              width: "100%",
            }}
          >
            {virtualizer.getVirtualItems().map((vRow) => {
              const startIdx = vRow.index * columns;
              const rowItems = sorted.slice(startIdx, startIdx + columns);
              return (
                <div
                  key={vRow.key}
                  data-index={vRow.index}
                  ref={virtualizer.measureElement}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    transform: `translateY(${vRow.start}px)`,
                    display: "grid",
                    gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                    columnGap: "0.75rem",
                    paddingBottom: "0.75rem",
                  }}
                >
                  {rowItems.map((p, colIdx) => (
                    <button
                      key={p.path}
                      onClick={() => setSelectedIdx(startIdx + colIdx)}
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
              );
            })}
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
