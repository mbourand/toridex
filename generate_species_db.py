"""
Generate data/species_db.json — the species metadata used by the Tauri app.

Merges:
  - data/splits/label_map.json     {epithet: int}
  - data/french_species.csv        species (binomial), gbif_species_key, occurrence_count
  - data/french_names.json         {binomial_lower: french_name}
  - data/image_index.parquet       picks one reference photo_id per species

Output: data/species_db.json
  [{"idx": 0, "epithet": "aalge", "scientificName": "Uria aalge",
    "frenchName": "Guillemot marmette", "occurrenceCount": 12345,
    "referencePhotoId": 87654321}, ...]

Usage:
    python generate_species_db.py
"""

import json
from pathlib import Path

import pandas as pd

SPLITS_DIR = Path("data/splits")
FRENCH_SPECIES_PATH = Path("data/french_species.csv")
FRENCH_NAMES_PATH = Path("data/french_names.json")
IMAGE_INDEX_PATH = Path("data/image_index.parquet")
OUTPUT_PATH = Path("data/species_db.json")


def main() -> None:
    # 1. Label map: epithet → int index
    with open(SPLITS_DIR / "label_map.json") as f:
        label_map: dict[str, int] = json.load(f)

    # 2. French species CSV: binomial → occurrence_count
    df_species = pd.read_csv(FRENCH_SPECIES_PATH)
    binomial_to_count = dict(zip(df_species["species"], df_species["occurrence_count"]))

    # 3. French names: binomial_lower → french_name
    with open(FRENCH_NAMES_PATH, encoding="utf-8") as f:
        french_names: dict[str, str] = json.load(f)

    # 4. Image index: derive ground-truth scientificName per epithet from actual training data.
    # Using mode() (most frequent) avoids hybrids or mis-labelled outliers dominating.
    print("Loading image index...")
    df_images = pd.read_parquet(IMAGE_INDEX_PATH, columns=["photo_id", "species", "scientificName"])
    epithet_to_sciname = (
        df_images.groupby("species")["scientificName"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    reference_photos = df_images.groupby("species")["photo_id"].first().to_dict()

    # For each epithet, collect ALL scientificNames with their image counts (sorted desc).
    # Used to flag ambiguous epithets shared by multiple species.
    epithet_all_names: dict[str, list[str]] = (
        df_images.groupby(["species", "scientificName"])
        .size()
        .reset_index(name="count")
        .sort_values(["species", "count"], ascending=[True, False])
        .groupby("species")["scientificName"]
        .apply(list)
        .to_dict()
    )

    # 5. Build species list sorted by idx
    species_list = []
    missing_binomial = []
    missing_french = []
    missing_photo = []

    for epithet, idx in sorted(label_map.items(), key=lambda x: x[1]):
        # Ground-truth binomial comes from the training images, not from the CSV.
        # This ensures "caeruleus" → "Cyanistes caeruleus" (Mésange bleue),
        # not whatever happens to share the epithet in the GBIF species list.
        binomial = epithet_to_sciname.get(epithet)
        if binomial is None:
            missing_binomial.append(epithet)
            binomial = epithet.capitalize()

        french_name = french_names.get(binomial.lower(), "")
        if not french_name:
            missing_french.append(epithet)

        occurrence_count = binomial_to_count.get(binomial, 0)
        reference_photo_id = reference_photos.get(epithet)
        if reference_photo_id is None:
            missing_photo.append(epithet)

        # Build list of OTHER species that share this epithet (ambiguous class).
        ambiguous_alternatives = []
        for alt_sciname in epithet_all_names.get(epithet, []):
            if alt_sciname != binomial:
                ambiguous_alternatives.append({
                    "scientificName": alt_sciname,
                    "frenchName": french_names.get(alt_sciname.lower(), ""),
                })

        species_list.append({
            "idx": idx,
            "epithet": epithet,
            "scientificName": binomial,
            "frenchName": french_name,
            "occurrenceCount": int(occurrence_count),
            "referencePhotoId": int(reference_photo_id) if reference_photo_id is not None else None,
            "ambiguousAlternatives": ambiguous_alternatives,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(species_list, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(species_list)} species to {OUTPUT_PATH}")
    if missing_binomial:
        print(f"  Warning: {len(missing_binomial)} epithets had no matching binomial")
    if missing_french:
        print(f"  Warning: {len(missing_french)} species had no French name")
    if missing_photo:
        print(f"  Warning: {len(missing_photo)} species had no reference photo")


if __name__ == "__main__":
    main()
