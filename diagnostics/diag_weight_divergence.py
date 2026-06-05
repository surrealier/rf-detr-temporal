# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Diagnostic: did the 9ch patch-embed conv learn to differentiate the 3 frames?

Root-cause hypothesis: the patch embedding `Conv2d(9, 768, 14, 14)` sums over all
9 input channels, and `_expand_patch_embed_channels` initialised the three
frame-groups (ch 0:3, 3:6, 6:9) to IDENTICAL copies of the pretrained RGB weight
(each scaled by 1/3). At init the conv therefore computes `W * mean(f0, f1, f2)`
- a temporal average / motion blur. If, after training, the three frame-groups
are still nearly identical (high cosine similarity, low relative L2), the model
never learned to treat the frames differently => it is still averaging => the
temporal/change signal is destroyed before attention ever sees it.

This script loads each temporal checkpoint, slices the patch-embed weight into
the three frame-groups, and measures their divergence. CPU-only (torch.load).

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/diag_weight_divergence.py
"""

import torch
import torch.nn.functional as F  # noqa: N812

CKPTS = {
    "exp1_lr2e4": "runs/260501_0358_exp1_lr2e4/best_map.pth",
    "exp2_lr5e5": "runs/260502_0445_exp2_lr5e5/best_map.pth",
    "exp3_stronghsv": "runs/260502_1449_exp3_stronghsv/best_map.pth",
}
PROJ_SUFFIX = "patch_embeddings.projection.weight"


def load_proj_weight(path: str) -> tuple[str, torch.Tensor]:
    """Load the patch-embed projection weight from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt)))
    key = next(k for k in state if k.endswith(PROJ_SUFFIX))
    return key, state[key].float()


def analyse(tag: str, w: torch.Tensor) -> None:
    """Report per-frame-group divergence for a (hidden, 9, ps, ps) weight."""
    n_ch = w.shape[1]
    assert n_ch % 3 == 0, f"expected multiple of 3 channels, got {n_ch}"
    n_frames = n_ch // 3
    groups = [w[:, i * 3 : (i + 1) * 3] for i in range(n_frames)]
    flat = [g.reshape(-1) for g in groups]
    print(f"\n[{tag}]  weight shape={tuple(w.shape)}  ({n_frames} frame-groups)")
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            cos = F.cosine_similarity(flat[i], flat[j], dim=0).item()
            norm_ratio = (flat[j].norm() / flat[i].norm()).item()
            rel_l2 = ((groups[i] - groups[j]).norm() / groups[i].norm()).item()
            print(
                f"  g{i} vs g{j}:  cosine={cos:.4f}  "
                f"norm_ratio={norm_ratio:.4f}  rel_L2_diff={rel_l2:.4f}"
            )
    mean_w = sum(groups) / n_frames
    dev = [((g - mean_w).norm() / mean_w.norm()).item() for g in groups]
    print("  per-group ||g - mean|| / ||mean||: " + ", ".join(f"{d:.4f}" for d in dev))
    # Verdict
    cos_min = min(
        F.cosine_similarity(flat[i], flat[j], dim=0).item()
        for i in range(n_frames)
        for j in range(i + 1, n_frames)
    )
    verdict = "STILL AVERAGING (groups ~identical)" if cos_min > 0.98 else "differentiated"
    print(f"  => min pairwise cosine={cos_min:.4f}  =>  {verdict}")


def main() -> None:
    print("=" * 70)
    print("Patch-embed frame-group divergence (temporal RF-DETR checkpoints)")
    print("=" * 70)
    for tag, path in CKPTS.items():
        try:
            key, w = load_proj_weight(path)
            analyse(tag, w)
        except FileNotFoundError:
            print(f"\n[{tag}]  MISSING: {path}")
    print("\nReference: at init, all groups are identical (cosine=1.0000) because")
    print("_expand_patch_embed_channels copies the same RGB weight to every frame.")
    print("If trained checkpoints stay near cosine=1.0, training never broke the")
    print("symmetry => the model is averaging the 3 frames => motion blur confirmed.")


if __name__ == "__main__":
    main()
