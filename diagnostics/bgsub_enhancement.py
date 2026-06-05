# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Measure WHERE the temporal pre-embed enhancement acts and HOW MUCH it adds.

The TemporalPreEmbed output is ``out = current + enhancement`` where the
enhancement is the temporal contribution fed to the patch embed:

  - ``preembed`` mode: enhancement = learned zero-init residual.
  - ``bgsub`` mode:    enhancement = ``alpha * (current - mean(past)) + residual``
    (an explicit background-subtraction add-weight, active from step one).

Over val samples this reports, for whichever checkpoint/mode is given:

  - enh_strength    = mean ||enhancement|| / ||current||  (how much is injected).
  - enh_gt_conc     = fraction of enhancement energy inside GT boxes
                      (does the change signal land on the objects?).
  - enh_motion_align= cosine(enhancement map, inter-frame difference map)
                      (does it activate on actual motion regions?).
  - gt_area_frac    = GT-box area fraction (random baseline for enh_gt_conc).

Comparing ``bgsub`` vs ``preembed`` on the same val set quantifies the goal:
"raise attention on the moving regions". Higher enh_gt_conc / enh_motion_align /
enh_strength at equal or better mAP => the change signal is being used, not
averaged away.

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/bgsub_enhancement.py \
        --ckpt runs/260602_0206_bgsub/best_map.pth --mode bgsub --n 32
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


def boxes_to_mask(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Rasterise normalised cxcywh boxes into a binary (H, W) mask."""
    mask = torch.zeros(h, w)
    for cx, cy, bw, bh in boxes.tolist():
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["preembed", "bgsub"], default="bgsub")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(0)

    config = RFDETRTemporalBaseConfig(
        pretrain_weights=None, num_classes=3, resolution=RES, temporal_fusion=args.mode
    )
    ns = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="."))
    model = build_model(ns).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)

    pre = model.backbone[0].encoder.temporal_pre
    if pre.alpha is not None:
        print(f"alpha (per-channel add-weight gain) = {pre.alpha.detach().tolist()}")

    cap = {}

    def hook(_m, inp, out):
        # Capture both the module input (stacked frames) and output (out=cur+enh).
        cap["x"] = inp[0].detach()
        cap["out"] = out.detach()

    pre.register_forward_hook(hook)

    ds = TemporalManifestDataset(MANIFEST, RES, augment=False)
    idxs = list(range(0, len(ds), max(1, len(ds) // args.n)))[: args.n]

    strength = gt_conc = motion_align = area_frac = 0.0
    n = 0
    with torch.no_grad():
        for i in idxs:
            stacked, target = ds[i]
            if target["boxes"].numel() == 0:
                continue
            model(nested_tensor_from_tensor_list([stacked]))
            x = cap["x"][0]  # (9, H, W)
            cur = x[6:9]
            enh = cap["out"][0] - cur  # (3, H, W) temporal enhancement
            enh_map = enh.abs().sum(0)  # (H, W)
            h, w = enh_map.shape

            strength += (enh.norm() / (cur.norm() + 1e-9)).item()
            gtm = boxes_to_mask(target["boxes"], h, w)
            gt_conc += (enh_map * gtm).sum().item() / (enh_map.sum().item() + 1e-9)
            area_frac += gtm.mean().item()
            f0, f1, f2 = x[0:3], x[3:6], x[6:9]
            motion = (f2 - f0).abs().sum(0) + (f2 - f1).abs().sum(0)
            motion_align += F.cosine_similarity(
                enh_map.flatten(), motion.flatten(), dim=0
            ).item()
            n += 1

    print(f"\n=== enhancement activation [{args.mode}] {args.ckpt} ===")
    print(f"samples used: {n}")
    print(f"  enh_strength      = {strength / n:.4f}   (||enh||/||current||)")
    print(f"  enh_gt_conc       = {gt_conc / n:.4f}   (enh energy inside GT boxes)")
    print(f"  gt_area_frac      = {area_frac / n:.4f}   (random baseline for enh_gt_conc)")
    print(f"  enh_motion_align  = {motion_align / n:.4f}   (enh vs inter-frame motion)")
    lift = (gt_conc / n) / (area_frac / n + 1e-9)
    print(f"  gt_conc / area    = {lift:.2f}x   (>1 => enhancement concentrates on objects)")


if __name__ == "__main__":
    main()
