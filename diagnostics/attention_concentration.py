# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Quantify whether a temporal model attends to change regions (before/after).

Uses input-gradient saliency: backprop the top detection's score to the input
and measure where the model "looks". Three metrics, averaged over val samples:

  1. gt_concentration  - fraction of saliency energy inside GT boxes
                         (higher = attention focused on the actual objects).
  2. past_frame_share  - fraction of saliency energy on the past-frame channels
                         (ch 0:6) vs the current frame (ch 6:9). A genuine
                         temporal model draws information from the past frames;
                         an averaging / single-frame model does not.
  3. motion_alignment  - cosine similarity between the spatial saliency map and
                         the inter-frame difference magnitude |f2-f0|+|f2-f1|
                         (higher = attention lands on moving/changing regions).

Supports both architectures so it can be run before and after the fix:
  --mode old   (temporal_fusion="none", 9ch patch embed; loads exp2 checkpoint)
  --mode new   (temporal_fusion="preembed"; loads a TemporalPreEmbed checkpoint)

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/attention_concentration.py \
        --mode old --ckpt runs/260502_0445_exp2_lr5e5/best_map.pth --n 24
"""

import argparse
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRTemporalBaseConfig, TrainConfig
from rfdetr.models.lwdetr import build_model
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from train_temporal_base_v4 import TemporalManifestDataset

RES = 952
MANIFEST = "data_manifests/valid.txt"


def build(mode: str):
    fusion = "preembed" if mode == "new" else "none"
    config = RFDETRTemporalBaseConfig(
        pretrain_weights=None, num_classes=3, resolution=RES, temporal_fusion=fusion
    )
    ns = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="."))
    return build_model(ns).eval()


def load_ckpt(model, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    nonpre_missing = [k for k in missing if "temporal_pre" not in k]
    if nonpre_missing:
        print(f"  WARN: {len(nonpre_missing)} unexpected missing keys (e.g. {nonpre_missing[:2]})")


def boxes_to_mask(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Binary (H,W) mask of the union of normalised cxcywh GT boxes."""
    mask = torch.zeros(h, w)
    for cx, cy, bw, bh in boxes.tolist():
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


@torch.enable_grad()
def saliency_for_sample(model, stacked: torch.Tensor):
    """Return (spatial_saliency HxW, per-channel-group energy [past,current])."""
    # Build the NestedTensor first (its inner tensor is a fresh leaf), then turn
    # on grad so the inplace pad-copy inside nested_tensor_from_tensor_list does
    # not run in grad mode.
    samples = nested_tensor_from_tensor_list([stacked])
    samples.tensors.requires_grad_(True)
    out = model(samples)
    scores = out["pred_logits"].sigmoid().max(dim=-1).values  # (1, nq)
    top = scores.max()
    model.zero_grad(set_to_none=True)
    top.backward()
    g = samples.tensors.grad[0].abs()  # (9, H, W)
    spatial = g.sum(0)  # (H, W)
    past = g[:6].sum().item()
    current = g[6:9].sum().item()
    return spatial.detach(), past, current


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build(args.mode)
    load_ckpt(model, args.ckpt)

    ds = TemporalManifestDataset(MANIFEST, RES, augment=False)
    # Sample evenly across the val set.
    idxs = list(range(0, len(ds), max(1, len(ds) // args.n)))[: args.n]

    gt_conc, past_share, motion_align, n_used = 0.0, 0.0, 0.0, 0
    for i in idxs:
        stacked, target = ds[i]
        if target["boxes"].numel() == 0:
            continue
        spatial, past, current = saliency_for_sample(model, stacked)
        h, w = spatial.shape
        # 1. GT concentration
        gtm = boxes_to_mask(target["boxes"], h, w)
        total = spatial.sum().item() + 1e-9
        gt_conc += (spatial * gtm).sum().item() / total
        # 2. Past-frame share
        past_share += past / (past + current + 1e-9)
        # 3. Motion alignment vs inter-frame difference magnitude
        f0, f1, f2 = stacked[0:3], stacked[3:6], stacked[6:9]
        motion = (f2 - f0).abs().sum(0) + (f2 - f1).abs().sum(0)  # (H, W)
        sflat, mflat = spatial.flatten(), motion.flatten()
        motion_align += F.cosine_similarity(sflat, mflat, dim=0).item()
        n_used += 1

    print(f"\n=== attention concentration [{args.mode}] {args.ckpt} ===")
    print(f"samples used: {n_used}")
    print(f"  gt_concentration  = {gt_conc / n_used:.4f}   (saliency energy inside GT boxes)")
    print(f"  past_frame_share  = {past_share / n_used:.4f}   (0.667 = uniform; <0.667 => past underused)")
    print(f"  motion_alignment  = {motion_align / n_used:.4f}   (saliency vs inter-frame motion)")


if __name__ == "__main__":
    main()
