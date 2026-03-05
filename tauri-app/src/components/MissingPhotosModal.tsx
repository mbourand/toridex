interface Props {
  missingCount: number;
  status: "pending" | "searching" | "done";
  relocatedCount: number;
  purgedCount: number;
  onAddSearchFolder: () => void;
  onDone: () => void;
  onSkip: () => void;
}

export default function MissingPhotosModal({
  missingCount,
  status,
  relocatedCount,
  purgedCount,
  onAddSearchFolder,
  onDone,
  onSkip,
}: Props) {
  const remaining = missingCount - relocatedCount - purgedCount;

  return (
    <div className="fixed inset-0 bg-black/75 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-5 border-b border-gray-700">
          <h2 className="text-white text-lg font-semibold">
            Photos introuvables
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            {missingCount} photo{missingCount !== 1 ? "s" : ""}{" "}
            {missingCount !== 1
              ? "ne sont plus accessibles"
              : "n'est plus accessible"}{" "}
            sur le disque.
          </p>
          <p className="text-gray-400 text-sm mt-1">
            Ajoutez un ou plusieurs dossiers pour tenter de retrouver les
            fichiers déplacés. Les photos introuvables seront supprimées.
          </p>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Add folder button */}
          <button
            onClick={onAddSearchFolder}
            disabled={status === "searching"}
            className="w-full text-sm font-medium text-white bg-gray-700 hover:bg-gray-600 disabled:opacity-50 px-4 py-2.5 rounded-lg transition-colors"
          >
            + Ajouter un dossier de recherche
          </button>

          {/* Progress */}
          {status === "searching" && (
            <div className="flex items-center gap-2 text-sm text-yellow-400">
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Recherche en cours...
            </div>
          )}

          {/* Results summary */}
          {(relocatedCount > 0 || purgedCount > 0) && (
            <div className="space-y-1 text-sm">
              {relocatedCount > 0 && (
                <p className="text-green-400">
                  {relocatedCount} photo{relocatedCount !== 1 ? "s" : ""}{" "}
                  retrouvée{relocatedCount !== 1 ? "s" : ""}
                </p>
              )}
              {purgedCount > 0 && (
                <p className="text-red-400">
                  {purgedCount} photo{purgedCount !== 1 ? "s" : ""} supprimée
                  {purgedCount !== 1 ? "s" : ""}
                </p>
              )}
              {remaining > 0 && (
                <p className="text-gray-400">
                  {remaining} photo{remaining !== 1 ? "s" : ""} encore manquante
                  {remaining !== 1 ? "s" : ""}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex gap-2">
          {remaining > 0 && (
            <button
              onClick={onSkip}
              disabled={status === "searching"}
              className="flex-1 text-sm font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 px-4 py-2.5 rounded-lg transition-colors"
            >
              Passer
            </button>
          )}
          <button
            onClick={remaining <= 0 ? onDone : onDone}
            disabled={status === "searching"}
            className="flex-1 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2.5 rounded-lg transition-colors"
          >
            {remaining <= 0 ? "Continuer" : "Supprimer les manquantes"}
          </button>
        </div>
      </div>
    </div>
  );
}
