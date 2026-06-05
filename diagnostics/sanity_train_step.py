# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU sanity check of one temporal training step with TemporalPreEmbed.

Validates the full path end-to-end without a GPU: remapped manifest -> dataset
(9-channel stack) -> TemporalPreEmbed -> backbone -> decoder -> criterion ->
backward. Confirms the loss is finite and gradients reach the new module.

Run:
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python diagnostics/sanity_train_step.py
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRTemporalBaseConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_and_postprocessors, build_model
from train_temporal_base_v4 import TemporalManifestDataset, collate_fn

RES = 952
MANIFEST = "data_manifests/valid.txt"


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    config = RFDETRTemporalBaseConfig(pretrain_weights=None, num_classes=3, resolution=RES)
    tc = TrainConfig(dataset_dir=".", output_dir=".")
    ns = _namespace_from_configs(config, tc)
    model = build_model(ns).to(device).train()
    criterion, _ = build_criterion_and_postprocessors(ns)
    criterion = criterion.to(device)

    ds = TemporalManifestDataset(MANIFEST, RES, augment=True)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    print(f"batch tensors: {tuple(samples.tensors.shape)}  (expect (2, 9, {RES}, {RES}))")

    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    print(f"total loss = {loss.item():.4f}  finite={torch.isfinite(loss).item()}")

    loss.backward()
    pre = model.backbone[0].encoder.temporal_pre
    g_out = pre.conv_out.weight.grad
    g_in = pre.conv_in.weight.grad
    print(f"temporal_pre.conv_out.weight grad norm = "
          f"{(g_out.norm().item() if g_out is not None else float('nan')):.4e}  "
          f"(nonzero => module is learning)")
    print(f"temporal_pre.conv_in.weight  grad norm = "
          f"{(g_in.norm().item() if g_in is not None else float('nan')):.4e}  "
          f"(0 at first step is expected: zero-conv bootstraps conv_out first)")
    ok = torch.isfinite(loss).item() and g_out is not None and g_out.norm().item() > 0
    print("SANITY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
