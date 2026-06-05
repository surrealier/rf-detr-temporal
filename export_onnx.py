"""Export Temporal RF-DETR (9ch) to ONNX - 3 variants: 672, 952, dynamic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Monkey-patch: replace bicubic (unsupported in ONNX) with bilinear
_orig_interpolate = F.interpolate

def _patched_interpolate(*args, **kwargs):
    if kwargs.get("mode") == "bicubic":
        kwargs["mode"] = "bilinear"
        kwargs.pop("antialias", None)
    return _orig_interpolate(*args, **kwargs)

F.interpolate = _patched_interpolate

from rfdetr.config import RFDETRTemporalBaseConfig, TrainConfig
from rfdetr._namespace import _namespace_from_configs
from rfdetr.models.lwdetr import build_model

# Overridable via env so the same script exports old (averaging) and new
# (preembed) checkpoints. temporal_fusion is read from the config default
# (preembed); set TEMPORAL_FUSION=none to export an old 9ch-expand checkpoint.
CHECKPOINT = os.environ.get("CHECKPOINT", "runs/260502_0445_exp2_lr5e5/best_map.pth")
TEMPORAL_FUSION = os.environ.get("TEMPORAL_FUSION", "preembed")
NUM_CLASSES = 3
OUTPUT_DIR = "onnx_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TemporalDetector(nn.Module):
    """Wrapper that outputs boxes + scores directly."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # pred_logits: [B, num_queries, num_classes+1] (last dim is bg)
        # pred_boxes: [B, num_queries, 4] (cx, cy, w, h normalized)
        logits = out["pred_logits"][:, :300]  # top 300 queries only
        boxes = out["pred_boxes"][:, :300]
        scores = logits.sigmoid()
        return boxes, scores


def build_and_load(resolution):
    # Always build with training resolution (952) for correct PE size
    config = RFDETRTemporalBaseConfig(
        num_classes=NUM_CLASSES, resolution=952, temporal_fusion=TEMPORAL_FUSION
    )
    tc = TrainConfig(dataset_dir=".", output_dir=".")
    ns = _namespace_from_configs(config, tc)
    model = build_model(ns)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt)))
    model.load_state_dict(state, strict=False)
    model.eval()
    return TemporalDetector(model)


def export_fixed(resolution, name):
    print(f"\n=== Exporting {name} (resolution={resolution}) ===")
    model = build_and_load(resolution)
    dummy = torch.randn(1, 9, resolution, resolution)
    path = f"{OUTPUT_DIR}/{name}.onnx"
    torch.onnx.export(
        model, dummy, path,
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return path


def export_dynamic(resolution, name):
    print(f"\n=== Exporting {name} (dynamic input) ===")
    model = build_and_load(resolution)
    dummy = torch.randn(1, 9, resolution, resolution)
    path = f"{OUTPUT_DIR}/{name}.onnx"
    torch.onnx.export(
        model, dummy, path,
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"},
        },
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return path


if __name__ == "__main__":
    name = os.environ.get("EXPORT_NAME", "temporal_exp2")
    export_fixed(952, f"{name}_952")
    export_dynamic(952, f"{name}_dynamic")
    print("\nDone! All ONNX files in:", OUTPUT_DIR)
