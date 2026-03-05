import { useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { Species, UserPhoto } from "../types";
import DetailModal from "./DetailModal";

interface Props {
  label: string;
  icon: string;
  borderClass: string;
  badgeClass: string;
  photos: UserPhoto[];
  speciesDisplay: (sciName: string) => string;
  allSpecies: Species[];
  onSetSpecies: (path: string, species: string | null) => void;
}

export default function PseudoSpeciesCard({
  label,
  icon,
  borderClass,
  badgeClass,
  photos,
  speciesDisplay,
  allSpecies,
  onSetSpecies,
}: Props) {
  const [open, setOpen] = useState(false);

  if (photos.length === 0) return null;

  const best = photos[0];
  const imgSrc = best.result.thumbPath
    ? convertFileSrc(best.result.thumbPath)
    : convertFileSrc(best.path);

  return (
    <>
      <div
        onClick={() => setOpen(true)}
        className={`relative rounded-xl overflow-hidden cursor-pointer bg-gray-900 shadow-lg hover:scale-105 hover:shadow-2xl transition-transform duration-200 group ring-2 ${borderClass}`}
      >
        {/* Photo */}
        <div className="aspect-square w-full overflow-hidden bg-gray-800">
          <img
            src={imgSrc}
            alt={label}
            className="w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity"
            loading="lazy"
          />

          {/* Center icon overlay */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-4xl opacity-80">{icon}</span>
          </div>

          {/* Photo count badge */}
          <div className="absolute top-2 right-2 bg-black/60 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
            {photos.length}
          </div>
        </div>

        {/* Info */}
        <div className="p-2">
          <div className="flex items-start justify-between gap-1">
            <p className="text-white text-sm font-semibold truncate leading-tight">
              {label}
            </p>
            <span
              className={`shrink-0 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full ${badgeClass}`}
            >
              {photos.length} photo{photos.length > 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </div>

      {open && (
        <DetailModal
          title={label}
          subtitle={`${photos.length} photo${photos.length > 1 ? "s" : ""}`}
          photos={photos}
          speciesDisplay={speciesDisplay}
          allSpecies={allSpecies}
          onSetSpecies={onSetSpecies}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
