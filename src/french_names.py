"""
Fetch and cache French common names for bird species from GBIF vernacular names API.

Outputs: data/french_names.json  — {canonical_name_lower: french_common_name}

Usage:
    from src.french_names import load_french_names
    names = load_french_names()  # fetches + caches on first call, instant after
    names["parus major"]  # → "Mésange charbonnière"
"""

import json
import time
from pathlib import Path

import pandas as pd
from pygbif import species as gbif_species
from tqdm import tqdm

FRENCH_SPECIES_PATH = Path("data/french_species.csv")
CACHE_PATH = Path("data/french_names.json")


def load_french_names(cache_path: Path = CACHE_PATH) -> dict[str, str]:
    """
    Return {canonical_name_lower: french_name} for all species.
    Builds and caches the result from GBIF on first call (~2 min), then loads instantly.
    """
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return _build_cache(cache_path)


def _pick_french_name(vernaculars: list[dict]) -> str | None:
    """Pick the best French name from a list of GBIF vernacular name records."""
    # GBIF uses ISO 639-2 "fra" for French; some sources use "fre" or "fr"
    french_codes = {"fra", "fre", "fr"}
    candidates = [v for v in vernaculars if v.get("language") in french_codes]
    if not candidates:
        return None
    # Prefer "preferred" entries, otherwise take the first
    preferred = [c for c in candidates if c.get("preferred")]
    return (preferred or candidates)[0]["vernacularName"]


def _build_cache(cache_path: Path) -> dict[str, str]:
    if not FRENCH_SPECIES_PATH.exists():
        raise FileNotFoundError(
            f"{FRENCH_SPECIES_PATH} not found. Run pipeline step 1 first."
        )

    df = pd.read_csv(FRENCH_SPECIES_PATH)
    names: dict[str, str] = {}
    missing: list[str] = []

    print(f"Fetching French names for {len(df)} species from GBIF (one-time, ~2 min)...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="GBIF vernacular names"):
        canonical: str = row["species"]          # e.g. "Parus major"
        key = int(row["gbif_species_key"])

        try:
            result = gbif_species.name_usage(key=key, data="vernacularNames")
            vernaculars = result.get("results", [])
            french = _pick_french_name(vernaculars)
            if french:
                names[canonical.lower()] = french
            else:
                missing.append(canonical)
        except Exception:
            missing.append(canonical)

        time.sleep(0.1)  # stay within GBIF rate limits

    if missing:
        print(f"  No French name found for {len(missing)} species "
              f"(e.g. {missing[:3]}). Scientific name will be shown instead.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Saved {len(names)} French names to {cache_path}")
    return names


if __name__ == "__main__":
    names = load_french_names()
    # Print a sample
    for canonical, french in list(names.items())[:10]:
        print(f"  {canonical:<35} → {french}")
