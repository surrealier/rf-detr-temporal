# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Size-stratified per-class AP@0.5 (COCO-style small/medium/large).

To decide whether *small, distant* objects are the detection bottleneck we
need the AP as a function of object size, not the aggregate. This bins both GT
and detections by box area (in the model's input-resolution pixel space) and
runs an independent greedy match per size bin, reporting per-class AP@0.5 for
each (checkpoint, size) cell.

Bins (COCO convention, area = w_px * h_px at RES x RES):
    small  : area <  32^2   (1024)
    medium : 32^2 .. 96^2   (1024 .. 9216)
    large  : area >= 96^2   (9216)

Run:
    uv run --no-sync python diagnostics/size_stratified_eval.py \
        --pair runs/260601_0520_preembed/best_map.pth:preembed --subsample 4
"""

import argparse
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRTemporalBaseConfig, TrainConfig
from rfdetr.models.lwdetr import build_model
from train_temporal_base_v4 import (
    CLASS_NAMES,
    NUM_CLASSES,
    TemporalManifestDataset,
    _box_iou_single,
    collate_fn,
)

RES = 952
MANIFEST = "data_manifests/valid.txt"
BINS = [("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, float("inf"))]


def size_bin(box: list[float]) -> str:
    """Return the COCO size bin name for a normalised cxcywh box at RES."""
    area = (box[2] * RES) * (box[3] * RES)
    for name, lo, hi in BINS:
        if lo <= area < hi:
            return name
    return "large"


def ap_from(preds: list[tuple[float, bool]], n_gt: int) -> float:
    """11-point interpolated AP from sorted-by-score (score, is_tp) and GT count."""
    if n_gt == 0:
        return float("nan")
    preds = sorted(preds, key=lambda x: -x[0])
    tp = fp = 0
    precisions, recalls = [], []
    for _, is_tp in preds:
        tp, fp = (tp + 1, fp) if is_tp else (tp, fp + 1)
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        ap += max([p for p, r in zip(precisions, recalls) if r >= t], default=0) / 11
    return ap


def build_eval_model(ckpt: str, mode: str, device: str) -> torch.nn.Module:
    config = RFDETRTemporalBaseConfig(
        pretrain_weights=None, num_classes=NUM_CLASSES, resolution=RES, temporal_fusion=mode
    )
    ns = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="."))
    model = build_model(ns).to(device).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict:
    """Per (class, size-bin) AP@0.5 with independent per-bin greedy matching."""
    preds = {(c, b[0]): [] for c in range(NUM_CLASSES) for b in BINS}
    n_gt = {(c, b[0]): 0 for c in range(NUM_CLASSES) for b in BINS}
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            out = model(samples)
            logits = out["pred_logits"].sigmoid()
            boxes = out["pred_boxes"]
            for bi in range(logits.shape[0]):
                gt_b = targets[bi]["boxes"]
                gt_l = targets[bi]["labels"]
                scores, cls = logits[bi].max(dim=-1)
                keep = scores > 0.01
                sc, cl, bx = scores[keep], cls[keep], boxes[bi][keep]
                order = sc.argsort(descending=True)
                sc, cl, bx = sc[order], cl[order], bx[order]
                for c in range(NUM_CLASSES):
                    gmask = (gt_l == c).nonzero(as_tuple=True)[0]
                    for bname, lo, hi in BINS:
                        gidx = [g.item() for g in gmask if size_bin(gt_b[g].tolist()) == bname]
                        n_gt[(c, bname)] += len(gidx)
                        matched = set()
                        pidx = [
                            p for p in range(len(sc))
                            if cl[p].item() == c and size_bin(bx[p].tolist()) == bname
                        ]
                        for p in pidx:
                            best_iou, best_g = 0.0, -1
                            for g in gidx:
                                if g in matched:
                                    continue
                                iou = _box_iou_single(bx[p], gt_b[g])
                                if iou > best_iou:
                                    best_iou, best_g = iou, g
                            is_tp = best_iou >= 0.5 and best_g >= 0
                            if is_tp:
                                matched.add(best_g)
                            preds[(c, bname)].append((sc[p].item(), is_tp))
    return {k: ap_from(preds[k], n_gt[k]) for k in preds}, n_gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="ckpt:mode (repeatable)")
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    ds = TemporalManifestDataset(MANIFEST, RES, augment=False)
    idxs = list(range(0, len(ds), args.subsample))
    loader = DataLoader(
        Subset(ds, idxs), batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=5
    )

    n_gt_ref = None
    rows = []
    for pair in args.pair:
        ckpt, mode = pair.rsplit(":", 1)
        tag = f"{mode}:{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]}"
        model = build_eval_model(ckpt, mode, args.device)
        aps, n_gt = evaluate(model, loader, args.device)
        n_gt_ref = n_gt
        rows.append((tag, aps))
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n=== size-stratified AP@0.5  ({len(idxs)} val imgs, every {args.subsample}-th) ===")
    print("GT instance counts per (class, size):")
    for c in range(NUM_CLASSES):
        cnts = "  ".join(f"{b[0]}={n_gt_ref[(c, b[0])]}" for b in BINS)
        print(f"  {CLASS_NAMES[c]:<10} {cnts}")
    for bname, _, _ in BINS:
        print(f"\n[{bname}]")
        hdr = "  ".join(f"{c:>9}" for c in CLASS_NAMES)
        print(f"  {'model':<40} {hdr}")
        for tag, aps in rows:
            cells = "  ".join(f"{aps[(c, bname)]:>9.3f}" for c in range(NUM_CLASSES))
            print(f"  {tag:<40} {cells}")


if __name__ == "__main__":
    main()
