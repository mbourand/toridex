"""
BioTrove France-Aves pipeline.

Runs the three preparation steps in order:
  1. Fetch French bird species from GBIF  → data/french_species.csv
  2. Download BioTrove metadata from HF   → data/biotrove_metadata/
  3. Filter metadata to French Aves       → data/biotrove_france_aves.parquet

Each step is skipped if its output already exists (re-run safe).

Usage:
    python pipeline.py              # run all steps
    python pipeline.py --step 1     # run a single step
    python pipeline.py --force      # ignore existing outputs and re-run all
"""

import argparse
from pathlib import Path

from src.fetch_french_species import fetch_french_species, OUTPUT_PATH as SPECIES_PATH
from src.download_metadata import download_biotrove_metadata, OUTPUT_DIR as METADATA_DIR
from src.filter_dataset import filter_to_french_aves, OUTPUT_PATH as FILTERED_PATH


STEPS = {
    1: {
        "name": "Fetch French bird species from GBIF",
        "output": SPECIES_PATH,
        "fn": fetch_french_species,
    },
    2: {
        "name": "Download BioTrove metadata from HuggingFace",
        "output": METADATA_DIR,
        "fn": download_biotrove_metadata,
    },
    3: {
        "name": "Filter BioTrove metadata to French Aves",
        "output": FILTERED_PATH,
        "fn": filter_to_french_aves,
    },
}


def output_exists(path: Path) -> bool:
    return path.exists() and (path.is_file() or any(path.iterdir()))


def run_step(step_id: int, force: bool) -> None:
    step = STEPS[step_id]
    print(f"\n{'=' * 60}")
    print(f"Step {step_id}: {step['name']}")
    print(f"{'=' * 60}")

    if not force and output_exists(step["output"]):
        print(f"Output already exists ({step['output']}), skipping.")
        print("Pass --force to re-run.")
        return

    step["fn"]()


def main() -> None:
    parser = argparse.ArgumentParser(description="BioTrove France-Aves preparation pipeline")
    parser.add_argument(
        "--step",
        type=int,
        choices=STEPS.keys(),
        help="Run a single step (1, 2, or 3). Default: run all steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if their outputs already exist.",
    )
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else list(STEPS.keys())
    for step_id in steps_to_run:
        run_step(step_id, force=args.force)

    print("\nPipeline complete.")
    print(f"  Species list : {SPECIES_PATH}")
    print(f"  Filtered index: {FILTERED_PATH}")
    print("\nNext step: bulk-download images using biotrove-process.")


if __name__ == "__main__":
    main()
