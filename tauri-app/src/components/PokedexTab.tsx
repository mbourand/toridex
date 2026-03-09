import { useEffect } from "react";
import SearchFilterBar from "./SearchFilterBar";
import SpeciesCard from "./SpeciesCard";
import DetailModal from "./DetailModal";
import { useVirtualizedGrid } from "../hooks/useVirtualizedGrid";

import { Species, UserPhoto, FilterMode, SortMode } from "../types";

interface Props {
  species: Species[];
  visible: Species[];
  photosBySpecies: Map<string, UserPhoto[]>;
  foundCount: number;
  search: string;
  setSearch: (s: string) => void;
  filter: FilterMode;
  setFilter: (f: FilterMode) => void;
  sort: SortMode;
  setSort: (s: SortMode) => void;
  selected: Species | null;
  setSelected: (s: Species | null) => void;
  speciesDisplay: (sciName: string) => string;
  handleSetUserSpecies: (path: string, species: string | null) => Promise<void>;
  frontPhotos: Record<string, string>;
  handleSetFrontPhoto: (scientificName: string, photoPath: string | null) => Promise<void>;
}

export default function PokedexTab({
  species,
  visible,
  photosBySpecies,
  foundCount,
  search, setSearch,
  filter, setFilter,
  sort, setSort,
  selected, setSelected,
  speciesDisplay,
  handleSetUserSpecies,
  frontPhotos,
  handleSetFrontPhoto,
}: Props) {
  const { scrollRef, virtualizer, columns } = useVirtualizedGrid(visible.length, 260);

  // Scroll to top when filter/search/sort changes
  useEffect(() => {
    scrollRef.current?.scrollTo(0, 0);
  }, [search, filter, sort]);

  return (
    <>
      <SearchFilterBar
        search={search}
        filter={filter}
        sort={sort}
        foundCount={foundCount}
        totalCount={species.length}
        onSearch={setSearch}
        onFilter={setFilter}
        onSort={setSort}
      />

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="p-4">
          {visible.length === 0 ? (
            <div className="text-center text-gray-500 mt-20 text-sm">
              Aucune espèce trouvée.
            </div>
          ) : (
            <div
              style={{
                height: virtualizer.getTotalSize(),
                position: "relative",
                width: "100%",
              }}
            >
              {virtualizer.getVirtualItems().map((vRow) => {
                const startIdx = vRow.index * columns;
                const rowItems = visible.slice(startIdx, startIdx + columns);
                return (
                  <div
                    key={vRow.key}
                    data-index={vRow.index}
                    ref={virtualizer.measureElement}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      transform: `translateY(${vRow.start}px)`,
                      display: "grid",
                      gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                      columnGap: "0.75rem",
                      paddingBottom: "0.75rem",
                    }}
                  >
                    {rowItems.map((s) => (
                      <SpeciesCard
                        key={s.idx}
                        species={s}
                        photos={photosBySpecies.get(s.scientificName) ?? []}
                        onClick={() => setSelected(s)}
                      />
                    ))}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {selected && (
        <DetailModal
          title={selected.frenchName || selected.scientificName}
          subtitle={selected.frenchName ? selected.scientificName : undefined}
          photos={photosBySpecies.get(selected.scientificName) ?? []}
          referenceImgSrc={selected.referencePhotoUrl ?? undefined}
          occurrenceCount={selected.occurrenceCount}
          speciesDisplay={speciesDisplay}
          allSpecies={species}
          onSetSpecies={async (path, sp) => {
            await handleSetUserSpecies(path, sp);
            setSelected(null);
          }}
          frontPhotoPath={frontPhotos[selected.scientificName] ?? null}
          onSetFrontPhoto={(path) => handleSetFrontPhoto(selected.scientificName, path)}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  );
}
