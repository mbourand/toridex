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
import torch.nn.functional as F

from src.dataset import INPUT_SIZE

BIOCLIP_MODEL_ID = "hf-hub:imageomics/bioclip"
ENCODER_DIM = 512  # ViT-B/16 output dimension
_PATCH_SIZE = 16


class BirdClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_encoder: bool = True,
        num_genera: int = 0,
        num_families: int = 0,
    ) -> None:
        super().__init__()

        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL_ID)
        self.encoder = clip_model.visual
        self._interpolate_pos_embed(INPUT_SIZE)
        self.head = nn.Linear(ENCODER_DIM, num_classes)

        # Taxonomy auxiliary heads (training-time regularizers)
        self.genus_head = nn.Linear(ENCODER_DIM, num_genera) if num_genera > 0 else None
        self.family_head = nn.Linear(ENCODER_DIM, num_families) if num_families > 0 else None

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _interpolate_pos_embed(self, input_size: int) -> None:
        """Interpolate positional embeddings for input sizes != 224."""
        pos_embed = self.encoder.positional_embedding  # (num_pos, dim)
        orig_grid = int((pos_embed.shape[0] - 1) ** 0.5)  # 14 for 224
        new_grid = input_size // _PATCH_SIZE  # 21 for 336

        if orig_grid == new_grid:
            return

        cls_embed = pos_embed[:1]  # (1, dim)
        patch_embed = pos_embed[1:]  # (orig_grid^2, dim)
        dim = patch_embed.shape[-1]

        patch_embed = patch_embed.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(
            patch_embed.float(), size=(new_grid, new_grid),
            mode="bicubic", align_corners=False,
        ).to(pos_embed.dtype)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(-1, dim)

        self.encoder.positional_embedding = nn.Parameter(
            torch.cat([cls_embed, patch_embed], dim=0)
        )

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        """Unfreeze the last n transformer blocks of the ViT for phase-2 fine-tuning."""
        blocks = list(self.encoder.transformer.resblocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Also unfreeze the final layer norm
        for p in self.encoder.ln_post.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        features = self.encoder(x)
        species_logits = self.head(features)

        if self.training and self.genus_head is not None and self.family_head is not None:
            return species_logits, self.genus_head(features), self.family_head(features)

        return species_logits

    def get_param_groups(
        self, head_lr: float, encoder_lr: float, layer_decay: float = 1.0,
    ) -> list[dict]:
        """
        Return optimizer param groups with separate LRs.

        With layer_decay < 1, each unfrozen ViT block gets a progressively
        lower LR the further it is from the head:
            block[-1] → encoder_lr
            block[-2] → encoder_lr * layer_decay
            block[-3] → encoder_lr * layer_decay^2  ...
        """
        head_params = list(self.head.parameters())
        groups = [{"params": head_params, "lr": head_lr}]

        # Auxiliary taxonomy heads get head_lr too
        for aux_head in (self.genus_head, self.family_head):
            if aux_head is not None:
                groups.append({"params": list(aux_head.parameters()), "lr": head_lr})

        # Encoder: per-block groups with layer-wise decay
        blocks = list(self.encoder.transformer.resblocks)
        for depth_from_top, block in enumerate(reversed(blocks)):
            trainable = [p for p in block.parameters() if p.requires_grad]
            if trainable:
                lr = encoder_lr * (layer_decay ** depth_from_top)
                groups.append({"params": trainable, "lr": lr})

        # ln_post gets full encoder_lr
        ln_post_params = [p for p in self.encoder.ln_post.parameters() if p.requires_grad]
        if ln_post_params:
            groups.append({"params": ln_post_params, "lr": encoder_lr})

        return groups

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(
        cls, path: Path, num_classes: int,
        num_genera: int = 0, num_families: int = 0,
    ) -> "BirdClassifier":
        model = cls(
            num_classes=num_classes, freeze_encoder=False,
            num_genera=num_genera, num_families=num_families,
        )
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # Handle checkpoint from different input size (pos embed shape mismatch)
        pos_key = "encoder.positional_embedding"
        if pos_key in state_dict:
            ckpt_pos = state_dict[pos_key]
            model_pos = model.encoder.positional_embedding
            if ckpt_pos.shape != model_pos.shape:
                orig_grid = int((ckpt_pos.shape[0] - 1) ** 0.5)
                new_grid = int((model_pos.shape[0] - 1) ** 0.5)
                cls_emb = ckpt_pos[:1]
                patch_emb = ckpt_pos[1:].reshape(1, orig_grid, orig_grid, -1).permute(0, 3, 1, 2)
                patch_emb = F.interpolate(
                    patch_emb.float(), size=(new_grid, new_grid),
                    mode="bicubic", align_corners=False,
                ).to(ckpt_pos.dtype)
                patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(-1, patch_emb.shape[1])
                state_dict[pos_key] = torch.cat([cls_emb, patch_emb], dim=0)

        # strict=False so checkpoints without taxonomy heads still load fine
        model.load_state_dict(state_dict, strict=False)
        return model
