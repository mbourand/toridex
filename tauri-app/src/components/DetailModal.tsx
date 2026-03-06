import { useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { Species, UserPhoto } from "../types";
import SpeciesPicker from "./SpeciesPicker";

interface Props {
  title: string;
  subtitle?: string;
  /** User's own photos (empty = species not observed yet) */
  photos: UserPhoto[];
  /** Shown greyscale when photos is empty */
  referenceImgSrc?: string;
  occurrenceCount?: number;
  /** Resolve a scientificName to a display name (French or scientific) */
  speciesDisplay: (sciName: string) => string;
  /** Full species list for the picker */
  allSpecies?: Species[];
  /** Called to set or clear user species override */
  onSetSpecies?: (path: string, species: string | null) => void;
  onClose: () => void;
}

export default function DetailModal({
  title,
  subtitle,
  photos,
  referenceImgSrc,
  occurrenceCount,
  speciesDisplay,
  allSpecies,
  onSetSpecies,
  onClose,
}: Props) {
  const [idx, setIdx] = useState(0);
  const [showPicker, setShowPicker] = useState(false);

  const hasPhotos = photos.length > 0;
  const current = hasPhotos ? photos[idx] : null;

  const imgSrc = current
    ? convertFileSrc(current.path)
    : (referenceImgSrc ?? null);

  const topK = current?.result.top_k ?? [];
  const hasUserOverride = !!current?.result.userSpecies;
  const canEdit = !!onSetSpecies && !!current;
  const effectiveSpecies = current?.result.scientificName ?? "";

  function handleSelectSpecies(scientificName: string) {
    if (!current || !onSetSpecies) return;
    onSetSpecies(current.path, scientificName);
  }

  function handleReset() {
    if (!current || !onSetSpecies) return;
    onSetSpecies(current.path, null);
  }

  function handleMarkNoBird() {
    if (!current || !onSetSpecies) return;
    onSetSpecies(current.path, "__no_bird__");
  }

  function handleMarkUncertain() {
    if (!current || !onSetSpecies) return;
    onSetSpecies(current.path, "__unknown__");
  }

  function handleMarkUnlisted() {
    if (!current || !onSetSpecies) return;
    onSetSpecies(current.path, "__unlisted__");
  }

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
                className={`w-full h-full object-cover`}
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
                  onClick={() => {
                    setIdx(i);
                    setShowPicker(false);
                  }}
                  className={`shrink-0 w-14 h-14 rounded-lg overflow-hidden border-2 transition-colors ${
                    i === idx
                      ? "border-blue-500"
                      : "border-transparent opacity-60 hover:opacity-100"
                  }`}
                >
                  <img
                    src={convertFileSrc(p.result.thumbPath ?? p.path)}
                    className="w-full h-full object-cover"
                  />
                </button>
              ))}
            </div>
          )}

          {/* User override badge */}
          {canEdit && hasUserOverride && (
            <div className="flex items-center gap-2 mb-3 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
              <span className="text-yellow-400 text-xs font-medium">
                Corrigé manuellement
              </span>
              <span className="text-gray-500 text-xs">
                (modèle : {speciesDisplay(current!.result.modelSpecies ?? "")})
              </span>
              <button
                onClick={handleReset}
                className="ml-auto text-xs text-gray-400 hover:text-white transition-colors"
              >
                Réinitialiser
              </button>
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

          {/* Top-k predictions / Species picker */}
          {current && !showPicker && topK.length > 0 && (
            <div>
              <p className="text-gray-500 text-xs mb-2">
                Confiance : {(current.result.confidence * 100).toFixed(1)}%
              </p>
              <div className="space-y-1.5">
                <p className="text-gray-500 text-xs">
                  Top prédictions :
                  {canEdit && (
                    <span className="text-gray-600 ml-1">
                      (cliquez pour corriger)
                    </span>
                  )}
                </p>
                {topK.map((k, i) => {
                  const row = (
                    <div
                      key={i}
                      className={`flex items-center gap-2 ${canEdit ? "cursor-pointer rounded-lg px-1 py-0.5 -mx-1 hover:bg-gray-800 transition-colors" : ""}`}
                      onClick={
                        canEdit
                          ? () => handleSelectSpecies(k.scientificName)
                          : undefined
                      }
                    >
                      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${k.confidence * 100}%` }}
                        />
                      </div>
                      <div className="w-36 min-w-0">
                        <p className="text-gray-300 text-xs truncate leading-tight">
                          {speciesDisplay(k.scientificName)}
                        </p>
                        <p className="text-gray-600 text-[10px] truncate italic leading-tight">
                          {k.scientificName}
                        </p>
                      </div>
                      <span className="text-gray-500 text-xs w-10 text-right">
                        {(k.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                  return row;
                })}
              </div>

              {/* Action buttons */}
              {canEdit && (
                <div className="flex items-center gap-3 mt-3 flex-wrap">
                  {allSpecies && (
                    <button
                      onClick={() => setShowPicker(true)}
                      className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                    >
                      Autre espece...
                    </button>
                  )}
                  {effectiveSpecies !== "__unknown__" && (
                    <button
                      onClick={handleMarkUncertain}
                      className="text-xs text-gray-400 hover:text-yellow-400 transition-colors"
                    >
                      Incertain
                    </button>
                  )}
                  {effectiveSpecies !== "__unlisted__" && (
                    <button
                      onClick={handleMarkUnlisted}
                      className="text-xs text-gray-400 hover:text-orange-400 transition-colors"
                    >
                      Espèce non répertoriée
                    </button>
                  )}
                  {effectiveSpecies !== "__no_bird__" &&
                    effectiveSpecies !== "__skipped__" && (
                      <button
                        onClick={handleMarkNoBird}
                        className="text-xs text-gray-400 hover:text-red-400 transition-colors"
                      >
                        Pas d'oiseau
                      </button>
                    )}
                </div>
              )}
            </div>
          )}

          {/* Species picker (replaces top-k when open) */}
          {current && showPicker && allSpecies && (
            <SpeciesPicker
              species={allSpecies}
              onSelect={handleSelectSpecies}
              onCancel={() => setShowPicker(false)}
            />
          )}

          {/* For photos with no top-k (unknown/skipped), show picker button directly */}
          {current && !showPicker && topK.length === 0 && canEdit && (
            <div className="mt-2">
              <p className="text-gray-500 text-xs mb-2">
                Aucune prediction disponible.
              </p>
              <div className="flex items-center gap-3 flex-wrap">
                {allSpecies && (
                  <button
                    onClick={() => setShowPicker(true)}
                    className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
                  >
                    Assigner une espece...
                  </button>
                )}
                {effectiveSpecies !== "__unknown__" && (
                  <button
                    onClick={handleMarkUncertain}
                    className="text-sm text-gray-400 hover:text-yellow-400 transition-colors"
                  >
                    Incertain
                  </button>
                )}
                {effectiveSpecies !== "__unlisted__" && (
                  <button
                    onClick={handleMarkUnlisted}
                    className="text-sm text-gray-400 hover:text-orange-400 transition-colors"
                  >
                    Espèce non répertoriée
                  </button>
                )}
                {effectiveSpecies !== "__no_bird__" &&
                  effectiveSpecies !== "__skipped__" && (
                    <button
                      onClick={handleMarkNoBird}
                      className="text-sm text-gray-400 hover:text-red-400 transition-colors"
                    >
                      Pas d'oiseau
                    </button>
                  )}
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
