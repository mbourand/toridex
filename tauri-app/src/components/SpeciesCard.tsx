import { convertFileSrc } from "@tauri-apps/api/core";
import { Species, UserPhoto } from "../types";

interface Props {
  species: Species;
  photos: UserPhoto[];
  dataDir: string;
  onClick: () => void;
}

function rarityLabel(count: number): { label: string; className: string } {
  if (count < 1_000) return { label: "Rare", className: "bg-red-500" };
  if (count < 10_000) return { label: "Peu commun", className: "bg-orange-400" };
  if (count < 100_000) return { label: "Commun", className: "bg-yellow-400 text-gray-800" };
  return { label: "Très commun", className: "bg-green-500" };
}

export default function SpeciesCard({ species, photos, dataDir, onClick }: Props) {
  const found = photos.length > 0;
  const best = found ? photos[0] : null;
  const rarity = rarityLabel(species.occurrenceCount);

  let imgSrc: string | null = null;
  if (best) {
    imgSrc = convertFileSrc(best.result.thumbPath ?? best.path);
  } else if (species.referencePhotoId !== null) {
    imgSrc = convertFileSrc(`${dataDir}/images/${species.referencePhotoId}.jpg`);
  }

  return (
    <div
      onClick={onClick}
      className="relative rounded-xl overflow-hidden cursor-pointer bg-gray-900 shadow-lg hover:scale-105 hover:shadow-2xl transition-transform duration-200 group"
    >
      {/* Photo */}
      <div className="aspect-square w-full overflow-hidden bg-gray-800">
        {imgSrc ? (
          <img
            src={imgSrc}
            alt={species.frenchName || species.scientificName}
            className={`w-full h-full object-cover transition-all duration-300 ${!found ? "grayscale opacity-40 group-hover:opacity-50" : ""}`}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-600">
            <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
        )}

        {/* Not found overlay */}
        {!found && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="bg-black/60 text-white text-xs font-semibold px-2 py-1 rounded-full">
              Non observé
            </span>
          </div>
        )}

        {/* Multiple photos badge */}
        {photos.length > 1 && (
          <div className="absolute top-2 right-2 bg-black/60 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
            {photos.length} 📷
          </div>
        )}
      </div>

      {/* Info */}
      <div className="p-2">
        <div className="flex items-start justify-between gap-1">
          <div className="min-w-0">
            <p className="text-white text-sm font-semibold truncate leading-tight">
              {species.frenchName || species.scientificName}
            </p>
            <p className="text-gray-400 text-xs italic truncate">
              {species.scientificName}
            </p>
          </div>
          <span className={`shrink-0 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full ${rarity.className}`}>
            {rarity.label}
          </span>
        </div>

        {best?.result.exif_date && (
          <p className="text-gray-400 text-xs mt-1">📅 {best.result.exif_date}</p>
        )}
      </div>
    </div>
  );
}
