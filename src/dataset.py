"""
PyTorch Dataset for French bird classification + stratified split creation.

Usage:
    # Create splits once (after download_images.py):
    from src.dataset import create_splits
    create_splits()

    # Use in training:
    from src.dataset import BirdDataset, get_transforms
    train_ds = BirdDataset("data/splits/train.parquet", "data/images", get_transforms("train"))
"""

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

INDEX_PATH = Path("data/image_index.parquet")
SPLITS_DIR = Path("data/splits")
IMAGE_DIR = Path("data/images")

# BioCLIP / OpenAI CLIP normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transforms(split: str) -> transforms.Compose:
    """Return the appropriate transform pipeline for train / val / test."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])


def create_splits(
    index_path: Path = INDEX_PATH,
    splits_dir: Path = SPLITS_DIR,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """
    Create stratified train/val/test splits from the image index.

    Outputs:
      data/splits/train.parquet
      data/splits/val.parquet
      data/splits/test.parquet
      data/splits/label_map.json   ← {species_name: int_label}
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image index from {index_path}...")
    df = pd.read_parquet(index_path)
    print(f"  {len(df):,} images, {df['species'].nunique()} species.")

    # Build sorted label map for reproducibility
    species_sorted = sorted(df["species"].unique())
    label_map = {sp: i for i, sp in enumerate(species_sorted)}
    with open(splits_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # First split off test set, then split remainder into train/val
    test_size = test_ratio
    val_size = val_ratio / (1.0 - test_ratio)  # relative to train+val remainder

    df_train_val, df_test = train_test_split(
        df, test_size=test_size, stratify=df["species"], random_state=seed
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size, stratify=df_train_val["species"], random_state=seed
    )

    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = splits_dir / f"{name}.parquet"
        split_df.reset_index(drop=True).to_parquet(path, index=False)
        print(f"  {name}: {len(split_df):,} images")

    print(f"Splits saved to {splits_dir}/")
    return splits_dir


class BirdDataset(Dataset):
    """
    Loads bird images and returns (image_tensor, label_int) pairs.

    Args:
        split_parquet: Path to one of the split parquet files.
        image_dir:     Directory containing {photo_id}.jpg files.
        transform:     torchvision transform pipeline (from get_transforms).
        label_map:     Optional dict {species: int}. If None, built from split.
    """

    def __init__(
        self,
        split_parquet: Path,
        image_dir: Path = IMAGE_DIR,
        transform: transforms.Compose | None = None,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.df = pd.read_parquet(split_parquet).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or get_transforms("val")

        if label_map is None:
            label_map_path = Path(split_parquet).parent / "label_map.json"
            with open(label_map_path) as f:
                label_map = json.load(f)
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['photo_id']}.jpg"

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = self.label_map[row["species"]]
        return img, label
