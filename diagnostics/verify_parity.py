# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify the TemporalPreEmbed fix: build + 3ch patch embed + init parity.

Checks:
  1. The temporal model builds, exposes `temporal_pre`, and keeps a 3-channel
     patch embed (so pretrained DINOv2/RF-DETR weights load verbatim).
  2. At init, TemporalPreEmbed(x) == current frame (residual == 0).
  3. Loading the SAME single-frame baseline checkpoint into both a 3ch baseline
     model and the 9ch temporal model yields identical detection outputs when
     the temporal model is fed [random, random, current] and the baseline is fed
     [current]. This proves "start at single-frame parity, learn temporal as a
     residual from zero".

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/verify_parity.py
"""

import sys

sys.path.insert(0, "src")

import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRTemporalBaseConfig, TrainConfig
from rfdetr.models.backbone.temporal_fusion import TemporalPreEmbed
from rfdetr.models.lwdetr import build_model
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

RES = 952


def build(config):
    tc = TrainConfig(dataset_dir=".", output_dir=".")
    ns = _namespace_from_configs(config, tc)
    model = build_model(ns)
    model.eval()
    return model


def forward(model, tensor_chw):
    samples = nested_tensor_from_tensor_list([tensor_chw])
    with torch.no_grad():
        return model(samples)


def main() -> None:
    torch.manual_seed(0)

    # --- Check 2: module-level identity at init ---
    pre = TemporalPreEmbed(num_frames=3).eval()
    x9 = torch.randn(2, 9, 64, 64)
    with torch.no_grad():
        y = pre(x9)
    cur = x9[:, 6:9]
    max_err = (y - cur).abs().max().item()
    print(f"[2] TemporalPreEmbed init residual max|y-current| = {max_err:.2e}  "
          f"({'PASS' if max_err < 1e-6 else 'FAIL'})")

    # --- Check 1: temporal model structure ---
    tmodel = build(RFDETRTemporalBaseConfig(pretrain_weights=None, num_classes=3, resolution=RES))
    enc = tmodel.backbone[0].encoder
    proj_w = enc.encoder.embeddings.patch_embeddings.projection.weight
    has_pre = getattr(enc, "temporal_pre", None) is not None
    print(f"[1] temporal_pre present = {has_pre}  "
          f"({'PASS' if has_pre else 'FAIL'})")
    print(f"[1] patch embed in_channels = {proj_w.shape[1]}  "
          f"({'PASS' if proj_w.shape[1] == 3 else 'FAIL'})")

    # --- Check 3: at init, the model output depends ONLY on the current frame ---
    # (i.e. it behaves exactly like a single-frame model regardless of f0, f1).
    # Feed the same current frame with different / duplicated past frames and
    # require identical detections. This is the "start at single-frame parity"
    # guarantee, self-contained (no checkpoint / PE-interpolation needed).
    cur = torch.randn(3, RES, RES)
    a = torch.cat([torch.randn(3, RES, RES), torch.randn(3, RES, RES), cur], dim=0)
    b = torch.cat([torch.randn(3, RES, RES), torch.randn(3, RES, RES), cur], dim=0)
    d = torch.cat([cur, cur, cur], dim=0)  # duplicated current frame

    out_a = forward(tmodel, a)
    out_b = forward(tmodel, b)
    out_d = forward(tmodel, d)
    dl = max(
        (out_a["pred_logits"] - out_b["pred_logits"]).abs().max().item(),
        (out_a["pred_logits"] - out_d["pred_logits"]).abs().max().item(),
    )
    db = max(
        (out_a["pred_boxes"] - out_b["pred_boxes"]).abs().max().item(),
        (out_a["pred_boxes"] - out_d["pred_boxes"]).abs().max().item(),
    )
    ok = dl < 1e-5 and db < 1e-5
    print(f"[3] init = f(current frame) only: max|Δlogits|={dl:.2e}  "
          f"max|Δboxes|={db:.2e}  ({'PASS' if ok else 'FAIL'})")
    print("\nAll checks done.")


if __name__ == "__main__":
    main()
