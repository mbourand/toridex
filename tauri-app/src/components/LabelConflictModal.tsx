import { useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { LabelConflict } from "../types";

interface Props {
  conflicts: LabelConflict[];
  speciesDisplay: (sciName: string) => string;
  onResolve: (acceptModelPaths: string[]) => void;
}

export default function LabelConflictModal({
  conflicts,
  speciesDisplay,
  onResolve,
}: Props) {
  // Track which conflicts the user wants to accept the model's prediction for
  const [acceptSet, setAcceptSet] = useState<Set<string>>(new Set());

  function toggle(path: string) {
    setAcceptSet((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function acceptAll() {
    setAcceptSet(new Set(conflicts.map((c) => c.path)));
  }

  function keepAll() {
    setAcceptSet(new Set());
  }

  return (
    <div className="fixed inset-0 bg-black/75 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-5 border-b border-gray-700">
          <h2 className="text-white text-lg font-semibold">
            Labels manuels en conflit
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            {conflicts.length} photo{conflicts.length !== 1 ? "s" : ""} avec un
            label manuel qui diffère de la nouvelle prédiction du modèle.
            Cochez celles pour lesquelles vous souhaitez accepter la suggestion
            du modèle.
          </p>
        </div>

        {/* Conflict list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {conflicts.map((c) => {
            const accepted = acceptSet.has(c.path);
            const thumbSrc = c.thumbPath
              ? convertFileSrc(c.thumbPath)
              : undefined;
            const filename = c.path.split(/[/\\]/).pop() || c.path;

            return (
              <label
                key={c.path}
                className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                  accepted
                    ? "bg-blue-900/30 border border-blue-600"
                    : "bg-gray-800 border border-gray-700 hover:border-gray-600"
                }`}
              >
                <input
                  type="checkbox"
                  checked={accepted}
                  onChange={() => toggle(c.path)}
                  className="shrink-0 w-4 h-4 accent-blue-500"
                />

                {/* Thumbnail */}
                {thumbSrc ? (
                  <img
                    src={thumbSrc}
                    alt=""
                    className="w-12 h-12 rounded object-cover shrink-0"
                  />
                ) : (
                  <div className="w-12 h-12 rounded bg-gray-700 shrink-0" />
                )}

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-500 truncate" title={c.path}>
                    {filename}
                  </p>
                  <p className="text-sm text-white">
                    <span className="text-yellow-400">Votre label :</span>{" "}
                    {speciesDisplay(c.userSpecies)}
                  </p>
                  <p className="text-sm text-white">
                    <span className="text-blue-400">Modèle :</span>{" "}
                    {speciesDisplay(c.modelSpecies)}{" "}
                    <span className="text-gray-500">
                      ({Math.round(c.modelConfidence * 100)}%)
                    </span>
                  </p>
                </div>
              </label>
            );
          })}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex items-center gap-2">
          <button
            onClick={acceptAll}
            className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded-lg transition-colors"
          >
            Tout accepter
          </button>
          <button
            onClick={keepAll}
            className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded-lg transition-colors"
          >
            Tout garder
          </button>

          <div className="flex-1" />

          <span className="text-xs text-gray-500">
            {acceptSet.size}/{conflicts.length} acceptée
            {acceptSet.size !== 1 ? "s" : ""}
          </span>

          <button
            onClick={() => onResolve([...acceptSet])}
            className="text-sm font-semibold text-white bg-blue-600 hover:bg-blue-500 px-4 py-2.5 rounded-lg transition-colors"
          >
            Valider
          </button>
        </div>
      </div>
    </div>
  );
}
