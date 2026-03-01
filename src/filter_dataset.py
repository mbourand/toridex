"""
Filter the BioTrove-Train metadata to keep only bird species observed in France.

Inputs:
  - data/biotrove_metadata/  (parquet chunks from download_metadata.py)
  - data/french_species.csv  (species list from fetch_french_species.py)

Output:
  - data/biotrove_france_aves.parquet
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

METADATA_DIR = Path("data/biotrove_metadata")
FRENCH_SPECIES_PATH = Path("data/french_species.csv")
OUTPUT_PATH = Path("data/biotrove_france_aves.parquet")


def normalize_name(name: str) -> str:
    """Lowercase and strip whitespace for robust matching."""
    return str(name).lower().strip()


def load_french_species(path: Path) -> set[str]:
    df = pd.read_csv(path)
    # Normalize for case-insensitive matching
    return {normalize_name(s) for s in df["species"].dropna()}


def load_biotrove_chunks(metadata_dir: Path) -> pd.DataFrame:
    """
    Load all parquet chunks from the metadata directory into a single DataFrame.
    Processes them one by one to keep peak memory usage low.
    """
    parquet_files = sorted(metadata_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {metadata_dir}")

    print(f"Found {len(parquet_files)} parquet chunk(s) to process.")
    chunks = []
    for pf in tqdm(parquet_files, desc="Loading parquet chunks"):
        chunks.append(pd.read_parquet(pf))

    return pd.concat(chunks, ignore_index=True)


def filter_to_french_aves(
    metadata_dir: Path = METADATA_DIR,
    french_species_path: Path = FRENCH_SPECIES_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    # 1. Load the French species set
    print(f"Loading French species list from {french_species_path}...")
    french_species = load_french_species(french_species_path)
    print(f"  {len(french_species)} French bird species loaded.")

    # 2. Load all BioTrove metadata chunks
    print(f"\nLoading BioTrove metadata from {metadata_dir}...")
    df = load_biotrove_chunks(metadata_dir)
    print(f"  Total rows loaded: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # 3. Keep only Aves
    # The 'class' column in BioTrove holds the taxonomic class name.
    aves_mask = df["class"].str.lower().str.strip() == "aves"
    df_aves = df[aves_mask].copy()
    print(f"\nRows after filtering to Aves: {len(df_aves):,}")

    # 4. Match against French species list.
    # In BioTrove, 'species' holds only the specific epithet (e.g. "major"),
    # not the full binomial. We reconstruct the binomial from genus + species
    # and normalize to lowercase for matching.
    df_aves["_binomial"] = (df_aves["genus"] + " " + df_aves["species"]).apply(normalize_name)
    france_mask = df_aves["_binomial"].isin(french_species)
    df_france = df_aves[france_mask].drop(columns=["_binomial"])
    print(f"Rows after filtering to French species: {len(df_france):,}")

    # 5. Report unmatched French species (useful for debugging name mismatches)
    matched_binomials = set(
        (df_france["genus"] + " " + df_france["species"]).apply(normalize_name).unique()
    )
    matched_species = matched_binomials
    unmatched = french_species - matched_species
    if unmatched:
        print(f"\n  {len(unmatched)} French species had no match in BioTrove:")
        for name in sorted(unmatched)[:20]:
            print(f"    - {name}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more.")

    # 6. Summary stats
    n_species = df_france["species"].nunique()
    n_images = len(df_france)
    print(f"\nFinal dataset: {n_species} species, {n_images:,} images")

    # 7. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_france.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return output_path


if __name__ == "__main__":
    filter_to_french_aves()
