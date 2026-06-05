# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Measure small-object detection on a SYNTHETIC small val set.

Real small/distant objects are absent from the data (size diagnostic: target
small=0), so it cannot be measured directly. This builds a deterministic
small-object-pasted val (same pastes for every checkpoint, seeded per index) and
reports per-class AP@0.5 in the COCO ``small`` bin - i.e. how well a model
detects the manufactured small objects. Compare a baseline checkpoint vs one
trained with AUG_SMALLOBJ to see whether the augmentation helps.

Run:
    uv run --no-sync python diagnostics/synthetic_small_eval.py \
        --pair runs/260601_0520_preembed/best_map.pth:preembed \
        --pair runs/260605_0017_smallobj_preembed/best_map.pth:preembed \
        --subsample 4
"""

import argparse
import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from diagnostics.size_stratified_eval import BINS, build_eval_model, evaluate
from train_temporal_base_v4 import (
    CLASS_NAMES,
    MEANS,
    NUM_CLASSES,
    NUM_FRAMES,
    STDS,
    TemporalManifestDataset,
    augment_small_object_paste,
    collate_fn,
)

RES = 952
MANIFEST = "data_manifests/valid.txt"


class PastedVal(Dataset):
    """Val dataset with deterministic small-object paste (seeded per index)."""

    def __init__(self, base: TemporalManifestDataset, size_px, alpha, n_max: int, motion_px: float = 0.0) -> None:
        self.base = base
        self.size_px = size_px
        self.alpha = alpha
        self.n_max = n_max
        self.motion_px = motion_px

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        stacked, target = self.base._load_raw(i)
        random.seed(100000 + i)  # deterministic, identical for every checkpoint
        aug, boxes, labels = augment_small_object_paste(
            stacked, target["boxes"], target["labels"], self.n_max, self.size_px, self.alpha,
            self.motion_px,
        )
        for fi in range(NUM_FRAMES):
            for c in range(3):
                aug[fi * 3 + c] = (aug[fi * 3 + c] - MEANS[c]) / STDS[c]
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([i]),
            "orig_size": target["orig_size"],
        }
        return aug, target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="ckpt:mode (repeatable)")
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--motion", type=float, default=0.0,
                    help="per-frame displacement (px) of pasted objects; >0 = moving small targets")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    base = TemporalManifestDataset(MANIFEST, RES, augment=False)
    pasted = PastedVal(base, (16, 64), (0.6, 0.95), 3, args.motion)
    idxs = list(range(0, len(base), args.subsample))
    loader = DataLoader(
        Subset(pasted, idxs), batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=5
    )

    rows, n_gt_ref = [], None
    for pair in args.pair:
        ckpt, mode = pair.rsplit(":", 1)
        tag = f"{mode}:{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]}"
        model = build_eval_model(ckpt, mode, args.device)
        aps, n_gt = evaluate(model, loader, args.device)
        rows.append((tag, aps))
        n_gt_ref = n_gt
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n=== SYNTHETIC small-object AP@0.5  ({len(idxs)} val imgs, every {args.subsample}-th) ===")
    print("GT instance counts per (class, size) on pasted val:")
    for c in range(NUM_CLASSES):
        cnts = "  ".join(f"{b[0]}={n_gt_ref[(c, b[0])]}" for b in BINS)
        print(f"  {CLASS_NAMES[c]:<10} {cnts}")
    for bname, _, _ in BINS:
        print(f"\n[{bname}]")
        hdr = "  ".join(f"{c:>9}" for c in CLASS_NAMES)
        print(f"  {'model':<44} {hdr}")
        for tag, aps in rows:
            cells = "  ".join(f"{aps[(c, bname)]:>9.3f}" for c in range(NUM_CLASSES))
            print(f"  {tag:<44} {cells}")


if __name__ == "__main__":
    main()
