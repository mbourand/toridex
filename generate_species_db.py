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
    # 1. Label map: binomial → int index  (keys are now full scientific names)
    with open(SPLITS_DIR / "label_map.json") as f:
        label_map: dict[str, int] = json.load(f)

    # 2. French species CSV: binomial → occurrence_count
    df_species = pd.read_csv(FRENCH_SPECIES_PATH)
    binomial_to_count = dict(zip(df_species["species"], df_species["occurrence_count"]))

    # 3. French names: binomial_lower → french_name
    with open(FRENCH_NAMES_PATH, encoding="utf-8") as f:
        french_names: dict[str, str] = json.load(f)

    # 4. Image index: pick one reference photo URL per species.
    print("Loading image index...")
    df_images = pd.read_parquet(IMAGE_INDEX_PATH, columns=["photo_id", "photo_url", "scientificName"])
    ref_rows = df_images.groupby("scientificName").first()
    reference_photos = ref_rows["photo_id"].to_dict()
    reference_urls = ref_rows["photo_url"].to_dict()

    # 5. Build species list sorted by idx
    species_list = []
    missing_french = []
    missing_photo = []

    for binomial, idx in sorted(label_map.items(), key=lambda x: x[1]):
        # Derive the specific epithet from the binomial (e.g. "Parus major" → "major")
        epithet = binomial.split()[-1] if " " in binomial else binomial

        french_name = french_names.get(binomial.lower(), "")
        if not french_name:
            missing_french.append(binomial)

        occurrence_count = binomial_to_count.get(binomial, 0)
        reference_photo_id = reference_photos.get(binomial)
        reference_url = reference_urls.get(binomial)
        if reference_photo_id is None:
            missing_photo.append(binomial)

        species_list.append({
            "idx": idx,
            "epithet": epithet,
            "scientificName": binomial,
            "frenchName": french_name,
            "occurrenceCount": int(occurrence_count),
            "referencePhotoId": int(reference_photo_id) if reference_photo_id is not None else None,
            "referencePhotoUrl": reference_url,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(species_list, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(species_list)} species to {OUTPUT_PATH}")
    if missing_french:
        print(f"  Warning: {len(missing_french)} species had no French name")
    if missing_photo:
        print(f"  Warning: {len(missing_photo)} species had no reference photo")


if __name__ == "__main__":
    main()
