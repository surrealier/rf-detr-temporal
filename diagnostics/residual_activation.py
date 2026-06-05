# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Probe the TemporalPreEmbed motion residual: WHERE does the learned temporal
contribution act, and HOW MUCH does it add?

For the preembed model, output = current_frame + residual, where the residual is
the learned temporal/motion contribution (=0 at init). This script hooks the
residual (conv_out output) and, over val samples, reports:

  - residual_strength    = mean ||residual|| / ||current||  (how much motion adds;
                           0 at init, grows as the module learns).
  - residual_gt_conc     = fraction of residual energy inside GT boxes
                           (does the motion pathway activate on the objects?).
  - residual_motion_align= cosine(residual map, inter-frame difference map)
                           (does it activate on actual change/motion regions?).

This is the direct "change-region attention" measure the temporal design targets,
unlike input-gradient saliency which is dominated by the current-frame identity
path. Only meaningful for --mode new (preembed).

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/residual_activation.py \
        --ckpt runs/260601_0520_preembed/best_map.pth --n 24
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
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    torch.manual_seed(0)

    config = RFDETRTemporalBaseConfig(
        pretrain_weights=None, num_classes=3, resolution=RES, temporal_fusion="preembed"
    )
    ns = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="."))
    model = build_model(ns).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)

    pre = model.backbone[0].encoder.temporal_pre
    cap = {}

    def hook(_m, inp, _out):
        # residual = conv_out(act2(conv_dw(act1(conv_in(motion))))); recompute from input
        cap["residual"] = _out.detach()

    pre.conv_out.register_forward_hook(hook)

    ds = TemporalManifestDataset(MANIFEST, RES, augment=False)
    idxs = list(range(0, len(ds), max(1, len(ds) // args.n)))[: args.n]

    strength, gt_conc, motion_align, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for i in idxs:
            stacked, target = ds[i]
            if target["boxes"].numel() == 0:
                continue
            model(nested_tensor_from_tensor_list([stacked]))
            res = cap["residual"][0]  # (3, H, W)
            cur = stacked[6:9]
            strength += (res.norm() / (cur.norm() + 1e-9)).item()
            res_map = res.abs().sum(0)  # (H, W)
            h, w = res_map.shape
            gtm = boxes_to_mask(target["boxes"], h, w)
            gt_conc += (res_map * gtm).sum().item() / (res_map.sum().item() + 1e-9)
            f0, f1, f2 = stacked[0:3], stacked[3:6], stacked[6:9]
            motion = (f2 - f0).abs().sum(0) + (f2 - f1).abs().sum(0)
            motion_align += F.cosine_similarity(res_map.flatten(), motion.flatten(), dim=0).item()
            n += 1

    print(f"\n=== residual activation [preembed] {args.ckpt} ===")
    print(f"samples used: {n}")
    print(f"  residual_strength      = {strength / n:.4f}   (||residual||/||current||; 0 at init)")
    print(f"  residual_gt_conc       = {gt_conc / n:.4f}   (residual energy inside GT boxes)")
    print(f"  residual_motion_align  = {motion_align / n:.4f}   (residual vs inter-frame motion)")
    # GT-box area fraction for reference (random baseline for gt_conc)
    print("  (compare residual_gt_conc to the GT-box area fraction ~0.05-0.15: higher => the")
    print("   learned temporal pathway concentrates on the objects, not the background.)")


if __name__ == "__main__":
    main()
