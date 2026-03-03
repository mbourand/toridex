"""
BioCLIP-based bird species classifier.

Architecture:
  - Encoder: BioCLIP ViT-B/16 visual backbone (512-dim output)
  - Head:    Linear(512, num_classes)

Training strategy:
  - Phase 1: encoder frozen, only head trained
  - Phase 2: last N transformer blocks + head fine-tuned with separate LRs
"""

from pathlib import Path

import torch
import torch.nn as nn

BIOCLIP_MODEL_ID = "hf-hub:imageomics/bioclip"
ENCODER_DIM = 512  # ViT-B/16 output dimension


class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_encoder: bool = True) -> None:
        super().__init__()

        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL_ID)
        self.encoder = clip_model.visual
        self.head = nn.Linear(ENCODER_DIM, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        """Unfreeze the last n transformer blocks of the ViT for phase-2 fine-tuning."""
        blocks = list(self.encoder.transformer.resblocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Also unfreeze the final layer norm
        for p in self.encoder.ln_post.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list[dict]:
        """
        Return optimizer param groups with separate LRs for the head and encoder.
        Only includes encoder parameters that are actually trainable.
        """
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        groups = [{"params": head_params, "lr": head_lr}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        return groups

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, num_classes: int) -> "BirdClassifier":
        model = cls(num_classes=num_classes, freeze_encoder=False)
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        return model
