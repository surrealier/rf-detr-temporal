# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Motion-stratified per-class mAP: the decisive test for the temporal goal.

Aggregate val mAP is dominated by near-static / large objects, so a temporal
enhancement that only helps *moving* targets (the distant, small, translucent
small/distant objects we care about) is invisible in the headline number. This script
splits the val set by inter-frame motion into low / high strata and reports
per-class AP@0.5 for each checkpoint, so a temporal model's gain on the
high-motion stratum can be seen directly.

It compares any number of (checkpoint, mode) pairs on the SAME stratified split.

Run (after training frees the GPUs):
    uv run --no-sync python diagnostics/motion_stratified_eval.py \
        --pair runs/260601_0520_preembed/best_map.pth:preembed \
        --pair runs/260602_0206_bgsub/best_map.pth:bgsub \
        --subsample 5
"""

import argparse
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader, Subset

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRTemporalBaseConfig, TrainConfig
from rfdetr.models.lwdetr import build_model
from train_temporal_base_v4 import (
    CLASS_NAMES,
    NUM_CLASSES,
    TemporalManifestDataset,
    collate_fn,
    compute_map,
)

RES = 952
MANIFEST = "data_manifests/valid.txt"


def build_eval_model(ckpt: str, mode: str, device: str) -> torch.nn.Module:
    """Build a temporal model in the given fusion mode and load a checkpoint."""
    config = RFDETRTemporalBaseConfig(
        pretrain_weights=None, num_classes=NUM_CLASSES, resolution=RES, temporal_fusion=mode
    )
    ns = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="."))
    model = build_model(ns).to(device).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pair", action="append", required=True,
        help="checkpoint:mode, e.g. runs/x/best_map.pth:bgsub (repeatable)",
    )
    ap.add_argument("--subsample", type=int, default=5, help="use every k-th val sample")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    ds = TemporalManifestDataset(MANIFEST, RES, augment=False)
    idxs = list(range(0, len(ds), args.subsample))

    # Motion score per sample = mean |f_cur - f_past| over past frames (in the
    # normalised space; monotonic for ranking). Decode each sample once.
    print(f"scoring motion for {len(idxs)} val samples (every {args.subsample}-th)...")
    scored = []
    for j, i in enumerate(idxs):
        stacked, target = ds[i]
        if target["boxes"].numel() == 0:
            continue
        f0, f1, f2 = stacked[0:3], stacked[3:6], stacked[6:9]
        motion = (f2 - f0).abs().mean().item() + (f2 - f1).abs().mean().item()
        scored.append((i, motion))
        if (j + 1) % 200 == 0:
            print(f"  scored {j + 1}/{len(idxs)}")

    scored.sort(key=lambda t: t[1])
    half = len(scored) // 2
    low_idx = [i for i, _ in scored[:half]]
    high_idx = [i for i, _ in scored[half:]]
    thr = scored[half][1]
    print(f"\nsplit: {len(low_idx)} low-motion / {len(high_idx)} high-motion (threshold={thr:.4f})\n")

    strata = {"low-motion": low_idx, "high-motion": high_idx}

    def loader(indices: list[int]) -> DataLoader:
        return DataLoader(
            Subset(ds, indices), batch_size=8, shuffle=False,
            collate_fn=collate_fn, num_workers=4,
        )

    results = {}
    for pair in args.pair:
        ckpt, mode = pair.rsplit(":", 1)
        tag = f"{mode}:{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]}"
        model = build_eval_model(ckpt, mode, args.device)
        for sname, sidx in strata.items():
            map_val, class_aps = compute_map(model, loader(sidx), args.device)
            results[(tag, sname)] = (map_val, class_aps)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Report
    print("\n=== motion-stratified mAP@0.5 ===")
    cls_hdr = "  ".join(f"{c:>8}" for c in CLASS_NAMES)
    for sname in strata:
        print(f"\n[{sname}]")
        print(f"  {'model':<32} {'mAP':>6}   {cls_hdr}")
        for pair in args.pair:
            ckpt, mode = pair.rsplit(":", 1)
            tag = f"{mode}:{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]}"
            map_val, caps = results[(tag, sname)]
            cls_str = "  ".join(f"{caps[c]:>8.3f}" for c in range(NUM_CLASSES))
            print(f"  {tag:<32} {map_val:>6.3f}   {cls_str}")


if __name__ == "__main__":
    main()
