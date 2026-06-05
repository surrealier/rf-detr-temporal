# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Diagnostic: does real 3-frame motion HELP or HURT vs a static (duplicated) frame?

If the patch embed averages the 3 frames (motion blur), then feeding the SAME
current frame x3 (static, zero motion -> average == sharp current frame) should
give HIGHER confidence / more detections than the genuine 3-frame motion stack
(average == blurred). We also ablate the past frames (zero / shuffle) to test
whether they contribute at all.

Uses the exported ONNX (no torch / GPU driver needed). The ONNX takes a
pre-normalised (1, 9, 952, 952) tensor and returns boxes (1,300,4) and
post-sigmoid scores (1,300,num_classes).

Run (ONNX CPU venv or full env):
    python diagnostics/diag_dup_vs_real.py --video flame-scene001.mp4
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort

RES = 952
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ONNX_PATH = "onnx_exports/temporal_exp2_952.onnx"


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> (3, RES, RES) float32, ImageNet-normalised (matches infer_video.py)."""
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (RES, RES)), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def run(sess: ort.InferenceSession, stack9: np.ndarray) -> tuple[float, int]:
    """Return (mean of per-query max score, #detections>0.3) for a (9,RES,RES) input."""
    inp = stack9[None].astype(np.float32)
    boxes, scores = sess.run(["boxes", "scores"], {"images": inp})
    per_query_max = scores[0].max(axis=-1)  # (300,)
    return float(per_query_max.max()), int((per_query_max > 0.3).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="flame-scene001.mp4")
    ap.add_argument("--onnx", default=ONNX_PATH)
    ap.add_argument("--stride", type=int, default=5, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(args.video)

    buf: list[np.ndarray] = []
    rows = {"real": [], "dup": [], "zero_past": [], "shuffle": []}
    idx = 0
    used = 0
    while used < args.max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        idx += 1
        buf.append(preprocess(fr))
        buf = buf[-3:]
        if len(buf) < 3:
            continue
        used += 1
        f0, f1, f2 = buf
        zero = np.zeros_like(f2)
        variants = {
            "real": np.concatenate([f0, f1, f2], 0),  # genuine motion
            "dup": np.concatenate([f2, f2, f2], 0),  # static: avg == sharp current
            "zero_past": np.concatenate([zero, zero, f2], 0),  # only current frame
            "shuffle": np.concatenate([f2, f0, f1], 0),  # permuted temporal order
        }
        for name, stack in variants.items():
            conf, ndet = run(sess, stack)
            rows[name].append((conf, ndet))

    print("=" * 64)
    print(f"dup-vs-real diagnostic  ({used} sampled frames from {args.video})")
    print("=" * 64)
    print(f"{'variant':<12}{'mean max-conf':>16}{'mean #det>0.3':>16}")
    for name, vals in rows.items():
        if not vals:
            continue
        mc = np.mean([v[0] for v in vals])
        nd = np.mean([v[1] for v in vals])
        print(f"{name:<12}{mc:>16.4f}{nd:>16.2f}")
    print("\nInterpretation:")
    print("  dup >= real            => motion blur from averaging HURTS detection")
    print("  zero_past ~= real      => past frames contribute ~nothing (frames unused)")
    print("  shuffle ~= real        => order-invariant => no temporal reasoning")


if __name__ == "__main__":
    main()
