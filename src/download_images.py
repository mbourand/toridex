"""
Download bird images from iNaturalist S3 URLs.

Reads data/biotrove_france_aves.parquet, samples up to MAX_PER_SPECIES images
per species, then downloads them concurrently to data/images/{photo_id}.jpg.

After downloading, writes data/image_index.parquet containing only rows for
which an image file was successfully saved — this is the authoritative index
used by the training pipeline.

Usage:
    python -m src.download_images
    python -m src.download_images --max-per-species 500 --workers 16
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

METADATA_PATH = Path("data/biotrove_france_aves.parquet")
IMAGE_DIR = Path("data/images")
INDEX_PATH = Path("data/image_index.parquet")

MAX_PER_SPECIES = 1000
NUM_WORKERS = 32
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


def download_image(photo_id: int, url: str, image_dir: Path) -> bool:
    """Download a single image. Returns True on success, False on failure."""
    dest = image_dir / f"{photo_id}.jpg"
    if dest.exists():
        return True

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15, stream=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return False


def sample_metadata(metadata_path: Path, max_per_species: int) -> pd.DataFrame:
    """Load parquet and sample up to max_per_species rows per species (binomial)."""
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)
    df = df.dropna(subset=["photo_url"])

    print(f"  {len(df):,} total rows, {df['scientificName'].nunique()} species.")
    print(f"  Sampling up to {max_per_species} images per species...")

    sampled = (
        df.groupby("scientificName", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_per_species), random_state=42))
        .reset_index(drop=True)
    )
    print(f"  Sampled {len(sampled):,} images across {sampled['scientificName'].nunique()} species.")
    return sampled


def download_all(df: pd.DataFrame, image_dir: Path, num_workers: int) -> pd.DataFrame:
    """Download all images concurrently. Returns DataFrame of successfully downloaded rows."""
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = df[["photo_id", "photo_url"]].to_dict("records")
    success_ids: set[int] = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_image, r["photo_id"], r["photo_url"], image_dir): r["photo_id"]
            for r in rows
        }
        with tqdm(total=len(futures), desc="Downloading images", unit="img") as pbar:
            for future in as_completed(futures):
                photo_id = futures[future]
                try:
                    if future.result():
                        success_ids.add(photo_id)
                except Exception:
                    pass
                pbar.update(1)

    print(f"\nDownloaded {len(success_ids):,} / {len(rows):,} images successfully.")
    return df[df["photo_id"].isin(success_ids)].reset_index(drop=True)


def download_images(
    metadata_path: Path = METADATA_PATH,
    image_dir: Path = IMAGE_DIR,
    index_path: Path = INDEX_PATH,
    max_per_species: int = MAX_PER_SPECIES,
    num_workers: int = NUM_WORKERS,
) -> Path:
    df_sampled = sample_metadata(metadata_path, max_per_species)
    df_downloaded = download_all(df_sampled, image_dir, num_workers)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    df_downloaded.to_parquet(index_path, index=False)
    print(f"Image index saved to {index_path}")
    return index_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-species", type=int, default=MAX_PER_SPECIES)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    download_images(max_per_species=args.max_per_species, num_workers=args.workers)
