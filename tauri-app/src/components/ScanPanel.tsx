interface Props {
  scanning: boolean;
  progress: { current: number; total: number } | null;
  thumbProgress: { current: number; total: number } | null;
  modelStatus: string;
  folders: string[];
  onAddFolder: () => void;
  onRemoveFolder: (folder: string) => void;
  onScan: () => void;
  onCancel: () => void;
}

export default function ScanPanel({ scanning, progress, thumbProgress, modelStatus, folders, onAddFolder, onRemoveFolder, onScan, onCancel }: Props) {
  const activeProgress = thumbProgress ?? progress;
  const pct = activeProgress && activeProgress.total > 0
    ? Math.round((activeProgress.current / activeProgress.total) * 100)
    : 0;
  const progressLabel = thumbProgress ? "Miniatures" : "Analyse";

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-gray-800 border-b border-gray-700 flex-wrap">
      <button
        onClick={onAddFolder}
        disabled={scanning}
        className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 px-3 py-2 rounded-lg transition-colors shrink-0"
      >
        + Ajouter dossier
      </button>

      {folders.map(f => {
        const short = f.split(/[/\\]/).pop() || f;
        return (
          <span key={f} title={f}
            className="flex items-center gap-1 bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded-full max-w-[200px]">
            <span className="truncate">{short}</span>
            <button onClick={() => onRemoveFolder(f)} disabled={scanning}
              className="text-gray-500 hover:text-red-400 disabled:opacity-50 ml-0.5">
              &times;
            </button>
          </span>
        );
      })}

      <button
        onClick={onScan}
        disabled={scanning || folders.length === 0}
        className="text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors shrink-0"
      >
        {scanning ? "Analyse en cours..." : "Rafraichir"}
      </button>

      {scanning && (
        <button
          onClick={onCancel}
          className="text-xs text-gray-300 bg-red-700 hover:bg-red-600 px-3 py-2 rounded-lg transition-colors shrink-0"
        >
          Annuler
        </button>
      )}

      {scanning && modelStatus && (
        <span className="text-xs text-yellow-400 shrink-0">{modelStatus}</span>
      )}

      {scanning && activeProgress && (
        <div className="flex-1 flex items-center gap-2 min-w-[200px]">
          <span className="text-xs text-gray-400 shrink-0">{progressLabel}</span>
          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-200 ${thumbProgress ? "bg-green-500" : "bg-blue-500"}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-gray-400 shrink-0">
            {activeProgress.current}/{activeProgress.total} — {pct}%
          </span>
        </div>
      )}
    </div>
  );
}
