"""
Fetch the list of bird species (Aves) observed in France from GBIF.

Outputs: data/french_species.csv
Columns: species (canonical name), gbif_species_key, occurrence_count
"""

import csv
import time
from pathlib import Path

from pygbif import occurrences, species as gbif_species

OUTPUT_PATH = Path("data/french_species.csv")

# Minimum number of GBIF observations to include a species.
# Filters out data-entry errors and extreme accidentals.
MIN_OCCURRENCES = 5

# GBIF caps facetLimit at 1500 per request — we paginate to be safe.
FACET_PAGE_SIZE = 1500


def get_aves_class_key() -> int:
    result = gbif_species.name_backbone(scientificName="Aves", taxonRank="CLASS")
    # New pygbif response: key is nested under result['usage']['key'] (str)
    # Old pygbif response: key was a top-level int at result['classKey'] or result['usageKey']
    key = (
        (result.get("usage") or {}).get("key")
        or result.get("classKey")
        or result.get("usageKey")
    )
    if key is None:
        raise RuntimeError(f"Could not resolve Aves class key. GBIF response: {result}")
    print(f"Aves classKey: {key}")
    return int(key)


def fetch_species_page(aves_key: int, offset: int) -> list[dict]:
    """Return one page of species facets for France."""
    res = occurrences.search(
        classKey=aves_key,
        country="FR",
        facet="speciesKey",
        facetLimit=FACET_PAGE_SIZE,
        facetOffset=offset,
        facetMincount=MIN_OCCURRENCES,
        limit=0,  # we only want facet counts, not raw records
    )
    facets = res.get("facets", [])
    if not facets:
        return []
    return facets[0].get("counts", [])


def resolve_species_names(species_keys: list[int]) -> dict[int, str]:
    """
    Map GBIF speciesKeys → canonical scientific names.
    Uses the species/match endpoint in small batches to stay within rate limits.
    """
    names: dict[int, str] = {}
    batch_size = 50
    for i in range(0, len(species_keys), batch_size):
        batch = species_keys[i : i + batch_size]
        for key in batch:
            try:
                info = gbif_species.name_usage(key=key, data="name")
                names[key] = info.get("canonicalName") or info.get("scientificName", "")
            except Exception as exc:
                print(f"  Warning: could not resolve key {key}: {exc}")
                names[key] = ""
        print(f"  Resolved {min(i + batch_size, len(species_keys))}/{len(species_keys)} names...")
        time.sleep(0.2)  # be polite to GBIF
    return names


def fetch_french_species(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Resolving Aves class key from GBIF...")
    aves_key = get_aves_class_key()

    print("Fetching French bird species from GBIF occurrence facets...")
    all_counts: list[dict] = []
    offset = 0
    while True:
        page = fetch_species_page(aves_key, offset)
        if not page:
            break
        all_counts.extend(page)
        print(f"  Fetched {len(all_counts)} species so far (offset={offset})...")
        if len(page) < FACET_PAGE_SIZE:
            break
        offset += FACET_PAGE_SIZE
        time.sleep(0.5)

    print(f"Found {len(all_counts)} species with ≥{MIN_OCCURRENCES} occurrences in France.")

    # Resolve numeric keys to canonical scientific names
    species_keys = [int(c["name"]) for c in all_counts]
    key_to_count = {int(c["name"]): int(c["count"]) for c in all_counts}

    print("Resolving scientific names for each species key...")
    key_to_name = resolve_species_names(species_keys)

    # Write output CSV
    rows = []
    for key in species_keys:
        name = key_to_name.get(key, "")
        if name:
            rows.append({
                "species": name,
                "gbif_species_key": key,
                "occurrence_count": key_to_count[key],
            })

    rows.sort(key=lambda r: r["occurrence_count"], reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "gbif_species_key", "occurrence_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} species to {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_french_species()
