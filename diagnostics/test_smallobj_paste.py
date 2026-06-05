# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Quick test of the small-object copy-paste augmentation and render boxes.

Loads a few val sequences, applies augment_small_object_paste, draws GT boxes on
the current frame and writes PNGs so the pasted small objects and their boxes can
be visually verified to align.

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/test_smallobj_paste.py
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_temporal_base_v4 import TemporalManifestDataset, augment_small_object_paste

RES = 952
OUT = "diagnostics/smallobj_samples"
COLORS = {0: (255, 80, 80), 1: (80, 160, 255), 2: (80, 255, 120)}


def render(stacked, boxes, labels, path, orig_n):
    cur = stacked[6:9].clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(cur)
    d = ImageDraw.Draw(img)
    _, h, w = stacked.shape
    for i, (cx, cy, bw, bh) in enumerate(boxes.tolist()):
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        c = COLORS.get(int(labels[i]), (255, 255, 0))
        # Pasted boxes (index >= orig_n) drawn thicker.
        d.rectangle([x1, y1, x2, y2], outline=c, width=4 if i >= orig_n else 2)
    img.save(path)


def main() -> None:
    import os

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    import random

    random.seed(0)

    ds = TemporalManifestDataset("data_manifests/valid.txt", RES, augment=False)
    # pick samples that have GT (large objects are common)
    picked = 0
    for i in range(0, len(ds), 137):
        stacked, target = ds._load_raw(i)
        if target["boxes"].numel() == 0:
            continue
        n0 = len(target["boxes"])
        aug, boxes, labels = augment_small_object_paste(
            stacked, target["boxes"], target["labels"], 3, (16, 64), (0.6, 0.95)
        )
        render(aug, boxes, labels, f"{OUT}/sample_{picked}.png", n0)
        areas = [(b[2] * RES) * (b[3] * RES) for b in boxes[n0:].tolist()]
        print(
            f"sample {picked}: orig boxes={n0}, pasted={len(boxes) - n0}, "
            f"pasted areas(px^2)={[int(a) for a in areas]} "
            f"(small<1024, med<9216)"
        )
        picked += 1
        if picked >= 4:
            break
    print(f"\nwrote {picked} PNGs to {OUT}/ (thick boxes = pasted small objects)")


if __name__ == "__main__":
    main()
