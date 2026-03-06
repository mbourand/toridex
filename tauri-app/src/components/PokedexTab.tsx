import SearchFilterBar from "./SearchFilterBar";
import SpeciesCard from "./SpeciesCard";
import DetailModal from "./DetailModal";

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

      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          {visible.length === 0 ? (
            <div className="text-center text-gray-500 mt-20 text-sm">
              Aucune espèce trouvée.
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {visible.map((s) => (
                <SpeciesCard
                  key={s.idx}
                  species={s}
                  photos={photosBySpecies.get(s.scientificName) ?? []}
                  onClick={() => setSelected(s)}
                />
              ))}
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
