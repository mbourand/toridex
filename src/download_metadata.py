"""
Download BioTrove-Train metadata (parquet + CSV files) from HuggingFace.

Only the metadata is downloaded here (~3.7 GB) — not the images.
Images are fetched later via the photo_url column.

Outputs: data/biotrove_metadata/  (parquet chunks + benchmark CSVs)
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "BGLab/BioTrove-Train"
OUTPUT_DIR = Path("data/biotrove_metadata")

# We only want metadata files, not any large model weights or extras.
ALLOW_PATTERNS = ["*.parquet", "*.csv"]


def download_biotrove_metadata(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BioTrove-Train metadata from HuggingFace ({REPO_ID})...")
    print("This will download ~3.7 GB of parquet/CSV files. Please wait.")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=ALLOW_PATTERNS,
    )

    downloaded = list(output_dir.rglob("*.parquet")) + list(output_dir.rglob("*.csv"))
    print(f"Download complete. {len(downloaded)} files in {output_dir}")
    return output_dir


if __name__ == "__main__":
    download_biotrove_metadata()
