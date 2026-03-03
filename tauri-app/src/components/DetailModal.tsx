import { useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { UserPhoto, AmbiguousAlternative } from "../types";

interface Props {
  title: string;
  subtitle?: string;
  /** User's own photos (empty = species not observed yet) */
  photos: UserPhoto[];
  /** Shown greyscale when photos is empty */
  referenceImgSrc?: string;
  occurrenceCount?: number;
  /** Resolve an epithet to a display name (French or scientific) */
  epithetToName: (epithet: string) => string;
  /** Returns other species sharing the same epithet, if any */
  epithetToAmbiguous?: (epithet: string) => AmbiguousAlternative[] | undefined;
  onClose: () => void;
}

export default function DetailModal({
  title,
  subtitle,
  photos,
  referenceImgSrc,
  occurrenceCount,
  epithetToName,
  epithetToAmbiguous,
  onClose,
}: Props) {
  const [idx, setIdx] = useState(0);

  const hasPhotos = photos.length > 0;
  const current = hasPhotos ? photos[idx] : null;

  const imgSrc = current
    ? convertFileSrc(current.path)
    : (referenceImgSrc ?? null);

  const topK = current?.result.top_k ?? [];
  const alternatives =
    current && epithetToAmbiguous
      ? (epithetToAmbiguous(current.result.epithet) ?? [])
      : [];

  console.log({ current, topK, alternatives });

  return (
    <div
      className="fixed inset-0 bg-black/75 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Image + nav */}
        <div className="relative bg-gray-800 flex-shrink-0">
          <div className="aspect-video">
            {imgSrc ? (
              <img
                src={imgSrc}
                alt={title}
                className={`w-full h-full object-cover ${!hasPhotos ? "grayscale opacity-40" : ""}`}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-600">
                <svg
                  className="w-16 h-16"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
            )}
          </div>

          {/* Prev / next */}
          {photos.length > 1 && (
            <>
              <button
                onClick={() => setIdx((i) => Math.max(0, i - 1))}
                disabled={idx === 0}
                className="absolute left-2 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 disabled:opacity-20 text-white rounded-full w-9 h-9 flex items-center justify-center transition-colors"
              >
                ‹
              </button>
              <button
                onClick={() =>
                  setIdx((i) => Math.min(photos.length - 1, i + 1))
                }
                disabled={idx === photos.length - 1}
                className="absolute right-2 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 disabled:opacity-20 text-white rounded-full w-9 h-9 flex items-center justify-center transition-colors"
              >
                ›
              </button>
              <div className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-black/50 text-white text-xs px-2 py-0.5 rounded-full">
                {idx + 1} / {photos.length}
              </div>
            </>
          )}

          {/* Close */}
          <button
            onClick={onClose}
            className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white rounded-full w-8 h-8 flex items-center justify-center transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Details */}
        <div className="p-5 overflow-y-auto">
          <h2 className="text-white text-xl font-bold leading-tight">
            {title}
          </h2>
          {subtitle && (
            <p className="text-gray-400 italic text-sm mb-3">{subtitle}</p>
          )}

          {/* Thumbnail strip */}
          {photos.length > 1 && (
            <div className="flex gap-1.5 mb-4 overflow-x-auto pb-1">
              {photos.map((p, i) => (
                <button
                  key={p.path}
                  onClick={() => setIdx(i)}
                  className={`shrink-0 w-14 h-14 rounded-lg overflow-hidden border-2 transition-colors ${
                    i === idx
                      ? "border-blue-500"
                      : "border-transparent opacity-60 hover:opacity-100"
                  }`}
                >
                  <img
                    src={convertFileSrc(p.path)}
                    className="w-full h-full object-cover"
                  />
                </button>
              ))}
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-2 gap-2 text-sm mb-4">
            {occurrenceCount !== undefined && (
              <div className="bg-gray-800 rounded-lg p-2">
                <span className="text-gray-500 text-xs block">
                  Observations GBIF
                </span>
                <span className="text-white font-semibold">
                  {occurrenceCount.toLocaleString("fr-FR")}
                </span>
              </div>
            )}
            {current?.result.exif_date && (
              <div className="bg-gray-800 rounded-lg p-2">
                <span className="text-gray-500 text-xs block">Date</span>
                <span className="text-white font-semibold">
                  {current.result.exif_date}
                </span>
              </div>
            )}
            {current?.result.exif_lat != null && (
              <div className="bg-gray-800 rounded-lg p-2 col-span-2">
                <span className="text-gray-500 text-xs block">
                  Localisation
                </span>
                <span className="text-white font-semibold text-xs">
                  {current.result.exif_lat!.toFixed(4)}°,{" "}
                  {current.result.exif_lon!.toFixed(4)}°
                </span>
              </div>
            )}
          </div>

          {/* Ambiguity warning */}
          {alternatives.length > 0 && (
            <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
              <p className="text-amber-400 text-xs font-semibold mb-1">
                ⚠ Épithète partagée :{" "}
                <span className="font-mono">'{current!.result.epithet}'</span>{" "}
                peut aussi désigner
              </p>
              <ul className="space-y-0.5">
                {alternatives.map((alt) => (
                  <li
                    key={alt.scientificName}
                    className="text-amber-200/70 text-xs"
                  >
                    •{" "}
                    {alt.frenchName ? (
                      <>
                        <span className="font-medium">{alt.frenchName}</span>{" "}
                        <span className="italic text-amber-200/40">
                          ({alt.scientificName})
                        </span>
                      </>
                    ) : (
                      <span className="italic">{alt.scientificName}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Top-k */}
          {current && topK.length > 0 && (
            <div>
              <p className="text-gray-500 text-xs mb-2">
                Confiance : {(current.result.confidence * 100).toFixed(1)}%
              </p>
              <div className="space-y-1.5">
                <p className="text-gray-500 text-xs">Top prédictions :</p>
                {topK.map((k, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full"
                        style={{ width: `${k.confidence * 100}%` }}
                      />
                    </div>
                    <div className="w-36 min-w-0">
                      <p className="text-gray-300 text-xs truncate leading-tight">
                        {epithetToName(k.epithet)}
                      </p>
                      <p className="text-gray-600 text-[10px] truncate italic leading-tight">
                        {k.epithet}
                      </p>
                    </div>
                    <span className="text-gray-500 text-xs w-10 text-right">
                      {(k.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!hasPhotos && (
            <p className="text-gray-500 text-sm text-center mt-2">
              Espèce non encore observée
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
