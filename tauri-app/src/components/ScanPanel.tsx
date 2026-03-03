interface Props {
  scanning: boolean;
  progress: { current: number; total: number } | null;
  lastFolder: string;
  onPickFolder: () => void;
  onScan: () => void;
}

export default function ScanPanel({ scanning, progress, lastFolder, onPickFolder, onScan }: Props) {
  const pct = progress && progress.total > 0
    ? Math.round((progress.current / progress.total) * 100)
    : 0;

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-gray-800 border-b border-gray-700">
      <button
        onClick={onPickFolder}
        disabled={scanning}
        className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 px-3 py-2 rounded-lg transition-colors"
      >
        📁 Choisir dossier
      </button>

      {lastFolder && (
        <span className="text-xs text-gray-500 truncate max-w-xs" title={lastFolder}>
          {lastFolder}
        </span>
      )}

      <button
        onClick={onScan}
        disabled={scanning || !lastFolder}
        className="text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors"
      >
        {scanning ? "Analyse en cours..." : "Analyser"}
      </button>

      {scanning && progress && (
        <div className="flex-1 flex items-center gap-2">
          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-200"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-gray-400 shrink-0">
            {progress.current}/{progress.total} — {pct}%
          </span>
        </div>
      )}
    </div>
  );
}
