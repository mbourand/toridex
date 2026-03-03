import { FilterMode, SortMode } from "../types";

interface Props {
  search: string;
  filter: FilterMode;
  sort: SortMode;
  foundCount: number;
  totalCount: number;
  onSearch: (q: string) => void;
  onFilter: (f: FilterMode) => void;
  onSort: (s: SortMode) => void;
}

export default function SearchFilterBar({
  search, filter, sort, foundCount, totalCount,
  onSearch, onFilter, onSort,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3 p-4 bg-gray-900 border-b border-gray-800">
      {/* Search */}
      <input
        type="text"
        placeholder="Rechercher une espèce..."
        value={search}
        onChange={e => onSearch(e.target.value)}
        className="flex-1 min-w-48 bg-gray-800 text-white placeholder-gray-500 rounded-lg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500"
      />

      {/* Filter */}
      <div className="flex rounded-lg overflow-hidden border border-gray-700">
        {(["all", "found", "not-found"] as FilterMode[]).map(f => (
          <button
            key={f}
            onClick={() => onFilter(f)}
            className={`px-3 py-2 text-xs font-medium transition-colors ${
              filter === f
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {f === "all" ? `Tous (${totalCount})` : f === "found" ? `Observés (${foundCount})` : `Non observés (${totalCount - foundCount})`}
          </button>
        ))}
      </div>

      {/* Sort */}
      <select
        value={sort}
        onChange={e => onSort(e.target.value as SortMode)}
        className="bg-gray-800 text-gray-300 text-xs rounded-lg px-3 py-2 outline-none border border-gray-700"
      >
        <option value="name">Trier : Nom</option>
        <option value="rarity">Trier : Rareté ↑</option>
        <option value="date">Trier : Date observée</option>
      </select>
    </div>
  );
}
