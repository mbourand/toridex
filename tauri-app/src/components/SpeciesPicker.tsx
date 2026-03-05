import { useEffect, useRef, useState } from "react";
import { Species } from "../types";

interface Props {
  species: Species[];
  onSelect: (scientificName: string) => void;
  onCancel: () => void;
}

const MAX_RESULTS = 20;

export default function SpeciesPicker({ species, onSelect, onCancel }: Props) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const filtered = query.trim()
    ? (() => {
        const q = query.toLowerCase();
        return species
          .filter(
            (s) =>
              s.frenchName.toLowerCase().includes(q) ||
              s.scientificName.toLowerCase().includes(q) ||
              s.epithet.toLowerCase().includes(q),
          )
          .slice(0, MAX_RESULTS);
      })()
    : [];

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.stopPropagation();
      onCancel();
    } else if (e.key === "Enter" && filtered.length === 1) {
      onSelect(filtered[0].scientificName);
    }
  }

  return (
    <div className="mt-2">
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Rechercher une espece..."
        className="w-full bg-gray-800 text-white text-sm px-3 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
      />

      {filtered.length > 0 && (
        <div className="mt-1 max-h-48 overflow-y-auto rounded-lg border border-gray-700 bg-gray-800">
          {filtered.map((s) => (
            <button
              key={s.idx}
              onClick={() => onSelect(s.scientificName)}
              className="w-full text-left px-3 py-2 hover:bg-gray-700 transition-colors"
            >
              <span className="text-white text-sm">{s.frenchName || s.scientificName}</span>
              {s.frenchName && (
                <span className="text-gray-500 text-xs italic ml-2">
                  {s.scientificName}
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      <button
        onClick={onCancel}
        className="mt-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
      >
        Annuler
      </button>
    </div>
  );
}
