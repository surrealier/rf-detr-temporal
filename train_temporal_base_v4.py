"""Train temporal RF-DETR (9-channel) with 2-GPU DDP, augmentation, mAP eval, and full logging."""
import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, "src")

import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRTemporalBaseConfig as RFDETRTemporalConfig
from rfdetr.config import TrainConfig
from rfdetr.models.lwdetr import build_criterion_and_postprocessors, build_model
from rfdetr.utilities.tensors import NestedTensor

# ─── Config (overridable via env vars) ───
DATA_DIR = os.environ.get("DATA_DIR", "/home/data/hdd1_bongkj/data_ready_v4")
from datetime import datetime

_run_tag = os.environ.get("RUN_TAG", "")
_run_name = datetime.now().strftime("%y%m%d_%H%M") + (f"_{_run_tag}" if _run_tag else "")
OUTPUT_DIR = f"runs/{_run_name}"
RESOLUTION = int(os.environ.get("RESOLUTION", "952"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LR = float(os.environ.get("LR", "1e-4"))
NUM_CLASSES = 3
NUM_FRAMES = 3
# Temporal fusion mode: "bgsub" (background-subtraction add-weight, default),
# "preembed" (zero-init motion residual), or "none" (naive 9ch averaging).
TEMPORAL_FUSION = os.environ.get("TEMPORAL_FUSION", "bgsub")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))
# Frames are 1080x1920 webp (~60ms/frame to decode); with 3 frames/sequence the
# input pipeline is CPU-decode bound (the box has 64 cores). Use enough workers
# to keep the GPUs fed, but bounded by Docker /dev/shm (8 GB here): each batch is
# ~260 MB and shm holds ~ num_gpus * NUM_WORKERS * PREFETCH batches at once, so
# keep num_gpus * NUM_WORKERS * PREFETCH * 260MB under ~7 GB.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "5"))
PREFETCH = int(os.environ.get("PREFETCH", "1"))
CLASS_NAMES = ["class0", "class1", "class2"]

# Augmentation params (overridable via env vars)
AUG_FLIPLR = float(os.environ.get("AUG_FLIPLR", "0.5"))
AUG_HSV_H = float(os.environ.get("AUG_HSV_H", "0.015"))
AUG_HSV_S = float(os.environ.get("AUG_HSV_S", "0.7"))
AUG_HSV_V = float(os.environ.get("AUG_HSV_V", "0.4"))
AUG_MIXUP_P = float(os.environ.get("AUG_MIXUP_P", "0.0"))
AUG_CUTMIX_P = float(os.environ.get("AUG_CUTMIX_P", "0.0"))
AUG_MOSAIC_P = float(os.environ.get("AUG_MOSAIC_P", "0.0"))
AUG_SCALE_P = float(os.environ.get("AUG_SCALE_P", "0.0"))
AUG_SCALE_MIN = float(os.environ.get("AUG_SCALE_MIN", "0.5"))
AUG_SCALE_MAX = float(os.environ.get("AUG_SCALE_MAX", "1.5"))
# Small-object copy-paste: manufacture small objects (small-size regime absent from data)
# by shrinking GT instances and alpha-blending them back. See
# augment_small_object_paste. Off by default.
AUG_SMALLOBJ_P = float(os.environ.get("AUG_SMALLOBJ_P", "0.0"))
AUG_SMALLOBJ_N = int(os.environ.get("AUG_SMALLOBJ_N", "3"))
# Target longer-side size (px @ RES) for pasted objects: small..medium regime.
AUG_SMALLOBJ_SMIN = float(os.environ.get("AUG_SMALLOBJ_SMIN", "16"))
AUG_SMALLOBJ_SMAX = float(os.environ.get("AUG_SMALLOBJ_SMAX", "64"))
AUG_SMALLOBJ_AMIN = float(os.environ.get("AUG_SMALLOBJ_AMIN", "0.6"))
AUG_SMALLOBJ_AMAX = float(os.environ.get("AUG_SMALLOBJ_AMAX", "0.95"))
# Max per-frame displacement (px @ RES) for pasted small objects. >0 manufactures
# small MOVING objects (real inter-frame foreground for the motion add-weight to
# amplify); 0 = static paste.
AUG_SMALLOBJ_MOTION = float(os.environ.get("AUG_SMALLOBJ_MOTION", "0.0"))

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]

# ─── Augmentation ───
def augment_hsv(img_tensor, gains):
    """Apply HSV augmentation to a (3,H,W) float RGB tensor [0,1] with pre-computed gains."""
    r_h, r_s, r_v = gains
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + (r_h - 1.0) * 180) % 180  # Hue: additive shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r_s, 0, 255)        # Saturation: multiplicative
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r_v, 0, 255)        # Value: multiplicative
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return torch.from_numpy(img / 255.0).permute(2, 0, 1).float()


def augment_fliplr(stacked, boxes):
    """Horizontal flip (9,H,W) tensor and boxes (N,4) in cx,cy,w,h format."""
    if random.random() > AUG_FLIPLR:
        return stacked, boxes
    stacked = stacked.flip(-1)  # flip W dimension
    if len(boxes) > 0:
        boxes = boxes.clone()
        boxes[:, 0] = 1.0 - boxes[:, 0]  # flip cx
    return stacked, boxes


def augment_scale(stacked, boxes, labels, scale_min, scale_max):
    """Random scale augmentation for 9ch temporal tensor. Preserves output resolution."""
    scale = random.uniform(scale_min, scale_max)
    _, h, w = stacked.shape
    new_h, new_w = int(h * scale), int(w * scale)
    # Resize all 9 channels together
    scaled = F.interpolate(stacked.unsqueeze(0), size=(new_h, new_w),
                           mode="bilinear", align_corners=False).squeeze(0)
    out = torch.zeros_like(stacked)
    if scale >= 1.0:
        # Crop center
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        out = scaled[:, y0:y0+h, x0:x0+w]
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, 0] = (boxes[:, 0] * scale - x0 / w) / scale  # adjust cx
            boxes[:, 1] = (boxes[:, 1] * scale - y0 / h) / scale  # adjust cy
            # Clamp centers and filter out-of-bounds
            valid = (boxes[:, 0] > 0) & (boxes[:, 0] < 1) & (boxes[:, 1] > 0) & (boxes[:, 1] < 1)
            boxes = boxes[valid]
            labels = labels[valid]
    else:
        # Pad with zeros, place scaled image in center
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        out[:, y0:y0+new_h, x0:x0+new_w] = scaled
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, 0] = boxes[:, 0] * scale + x0 / w
            boxes[:, 1] = boxes[:, 1] * scale + y0 / h
            boxes[:, 2] *= scale
            boxes[:, 3] *= scale
    return out, boxes, labels


def _feather_mask(h, w):
    """2D raised-cosine alpha mask: ~1 in the centre, fading to 0 at the border.

    Used to blend a pasted patch into the background so small synthetic
    objects read as a soft, semi-transparent blob rather than a hard
    rectangle.
    """
    wy = torch.hann_window(h, periodic=False) if h > 1 else torch.ones(1)
    wx = torch.hann_window(w, periodic=False) if w > 1 else torch.ones(1)
    return (wy[:, None] * wx[None, :]).clamp(min=0.0)  # (h, w)


def augment_small_object_paste(stacked, boxes, labels, n_max, size_px_range, alpha_range, motion_px=0.0):
    """Manufacture small objects by copy-pasting downscaled GT instances.

    The val/train data has essentially no *small* objects (small=0), so the model
    never learns the distant/small regime. This takes existing GT instances,
    shrinks them so their longer side falls in ``size_px_range`` pixels (the
    small/medium regime), and alpha-blends them back, adding the corresponding
    boxes/labels for the LAST (annotated) frame.

    Motion: with ``motion_px > 0`` the pasted object is placed at a different
    position in each frame (a random per-sequence velocity, arriving at the GT
    location on the last frame), manufacturing a small *moving* object with real
    inter-frame foreground — the target the background-subtraction add-weight is
    meant to amplify. With ``motion_px == 0`` it is a static paste (identical
    location across frames, no fake motion).

    Operates on the unnormalised 9ch tensor in [0, 1].
    """
    if len(boxes) == 0:
        return stacked, boxes, labels
    c, h, w = stacked.shape
    nf = c // 3
    out = stacked.clone()
    new_boxes, new_labels = [], []
    for _ in range(random.randint(1, n_max)):
        si = random.randrange(len(boxes))
        cx, cy, bw, bh = boxes[si].tolist()
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
        ph0, pw0 = y2 - y1, x2 - x1
        if ph0 < 8 or pw0 < 8:
            continue
        # Downscale so the longer side hits a target absolute size (px). Skip if
        # the source isn't large enough to shrink into the target range.
        target = random.uniform(*size_px_range)
        scale = target / max(ph0, pw0)
        if scale >= 1.0:
            continue
        patch = out[:, y1:y2, x1:x2].unsqueeze(0)
        nph, npw = max(2, int(ph0 * scale)), max(2, int(pw0 * scale))
        patch_s = F.interpolate(patch, size=(nph, npw), mode="bilinear", align_corners=False).squeeze(0)
        # Last-frame (GT) position, plus a per-frame velocity for the moving case.
        px, py = random.randint(0, w - npw), random.randint(0, h - nph)
        vx = random.uniform(-motion_px, motion_px) if motion_px > 0 else 0.0
        vy = random.uniform(-motion_px, motion_px) if motion_px > 0 else 0.0
        mask2d = _feather_mask(nph, npw) * random.uniform(*alpha_range)  # (nph,npw)
        for t in range(nf):
            # Offset so frame t = base - (last - t) * v  =>  arrives at base on the last frame.
            off = nf - 1 - t
            tx = min(max(0, int(round(px - off * vx))), w - npw)
            ty = min(max(0, int(round(py - off * vy))), h - nph)
            sub = patch_s[t * 3:(t + 1) * 3]  # this frame's 3ch appearance
            region = out[t * 3:(t + 1) * 3, ty:ty + nph, tx:tx + npw]
            out[t * 3:(t + 1) * 3, ty:ty + nph, tx:tx + npw] = region * (1 - mask2d) + sub * mask2d
        new_boxes.append([(px + npw / 2) / w, (py + nph / 2) / h, npw / w, nph / h])
        new_labels.append(int(labels[si]))
    if new_boxes:
        boxes = torch.cat([boxes, torch.tensor(new_boxes, dtype=boxes.dtype)], dim=0)
        labels = torch.cat([labels, torch.tensor(new_labels, dtype=labels.dtype)], dim=0)
    return out, boxes, labels


def mixup_data(stacked1, target1, stacked2, target2):
    """MixUp two samples. Returns mixed tensor and merged targets."""
    r = np.random.beta(32.0, 32.0)
    mixed = stacked1 * r + stacked2 * (1 - r)
    boxes = torch.cat([target1["boxes"], target2["boxes"]], dim=0)
    labels = torch.cat([target1["labels"], target2["labels"]], dim=0)
    target = {
        "boxes": boxes,
        "labels": labels,
        "image_id": target1["image_id"],
        "orig_size": target1["orig_size"],
    }
    return mixed, target


def cutmix_data(stacked1, target1, stacked2, target2):
    """CutMix two samples."""
    _, h, w = stacked1.shape
    lam = np.random.beta(1.0, 1.0)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx, cy = random.randint(0, w), random.randint(0, h)
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, w), min(cy + cut_h // 2, h)

    mixed = stacked1.clone()
    mixed[:, y1:y2, x1:x2] = stacked2[:, y1:y2, x1:x2]

    # Keep boxes from img2 that overlap with cut region
    if len(target2["boxes"]) > 0:
        b2 = target2["boxes"]  # cx,cy,w,h normalized
        # Convert cut region to normalized coords
        nx1, ny1, nx2, ny2 = x1/w, y1/h, x2/w, y2/h
        # Check overlap: box center inside cut region
        inside = (b2[:,0] >= nx1) & (b2[:,0] <= nx2) & (b2[:,1] >= ny1) & (b2[:,1] <= ny2)
        boxes = torch.cat([target1["boxes"], b2[inside]], dim=0)
        labels = torch.cat([target1["labels"], target2["labels"][inside]], dim=0)
    else:
        boxes, labels = target1["boxes"], target1["labels"]

    target = {"boxes": boxes, "labels": labels, "image_id": target1["image_id"], "orig_size": target1["orig_size"]}
    return mixed, target


# ─── Dataset ───
class TemporalManifestDataset(Dataset):
    """Load 3-frame sequences from a manifest file with optional augmentation."""
    def __init__(self, manifest_path, resolution, augment=False):
        self.resolution = resolution
        self.augment = augment
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                frames = parts[:NUM_FRAMES]
                label_str = parts[NUM_FRAMES] if len(parts) > NUM_FRAMES else ""
                source_tag = parts[NUM_FRAMES + 1] if len(parts) > NUM_FRAMES + 1 else ""
                self.samples.append((frames, label_str, source_tag))

    def __len__(self):
        return len(self.samples)

    def _load_raw(self, idx):
        """Load frames and labels without normalization (for mixup/cutmix)."""
        frame_paths, label_str, source_tag = self.samples[idx]
        resized = []
        h_orig, w_orig = 0, 0
        for fi, fp in enumerate(frame_paths):
            img = Image.open(fp).convert("RGB")
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            if fi == 0:
                h_orig, w_orig = t.shape[1], t.shape[2]
            t = F.interpolate(t.unsqueeze(0), size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False).squeeze(0)
            resized.append(t)

        # Pre-augmentation: subtle sensor noise for flagged sequences
        # Simulates camera read noise (σ=1-2 DN on 8-bit scale) to break identical frames
        if self.augment and source_tag == "v4":
            noise_indices = random.sample(range(NUM_FRAMES), 2)  # 2 of 3 frames
            for ni in noise_indices:
                sigma = random.uniform(1.0, 2.0) / 255.0  # 1-2 DN in [0,1] scale
                # Per-channel independent noise
                noise = torch.randn_like(resized[ni]) * sigma
                resized[ni] = (resized[ni] + noise).clamp(0.0, 1.0)

        stacked = torch.cat(resized, dim=0)

        boxes, labels = [], []
        if label_str:
            for entry in label_str.split(";"):
                parts = entry.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < NUM_CLASSES:
                        labels.append(cls_id)
                        boxes.append([float(x) for x in parts[1:5]])

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t,
                  "image_id": torch.tensor([idx]), "orig_size": torch.tensor([h_orig, w_orig])}
        return stacked, target

    def _mosaic(self, idx):
        """Mosaic: combine 4 sequences into 2x2 grid, per-frame."""
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        res = self.resolution
        half = res // 2
        stacked = torch.zeros(9, res, res)
        all_boxes, all_labels = [], []
        offsets = [(0, 0), (half, 0), (0, half), (half, half)]
        for i, (oi, (ox, oy)) in enumerate(zip(indices, offsets)):
            s, t = self._load_raw(oi)
            for fi in range(NUM_FRAMES):
                ch = fi * 3
                frame = s[ch:ch+3].unsqueeze(0)
                frame = F.interpolate(frame, size=(half, half), mode="bilinear", align_corners=False).squeeze(0)
                stacked[ch:ch+3, oy:oy+half, ox:ox+half] = frame
            if len(t["boxes"]) > 0:
                b = t["boxes"].clone()
                b[:, 0] = (b[:, 0] * 0.5) + (ox / res)
                b[:, 1] = (b[:, 1] * 0.5) + (oy / res)
                b[:, 2] *= 0.5
                b[:, 3] *= 0.5
                all_boxes.append(b)
                all_labels.append(t["labels"])
        boxes = torch.cat(all_boxes) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.cat(all_labels) if all_labels else torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor([idx]), "orig_size": torch.tensor([res, res])}
        return stacked, target

    def __getitem__(self, idx):
        if self.augment and random.random() < AUG_MOSAIC_P:
            stacked, target = self._mosaic(idx)
        else:
            stacked, target = self._load_raw(idx)

        if self.augment:
            # Small-object copy-paste: manufacture small objects (operate on
            # unnormalised [0,1] frames, before HSV/flip/scale).
            if random.random() < AUG_SMALLOBJ_P:
                stacked, target["boxes"], target["labels"] = augment_small_object_paste(
                    stacked, target["boxes"], target["labels"],
                    AUG_SMALLOBJ_N, (AUG_SMALLOBJ_SMIN, AUG_SMALLOBJ_SMAX),
                    (AUG_SMALLOBJ_AMIN, AUG_SMALLOBJ_AMAX), AUG_SMALLOBJ_MOTION,
                )
            # HSV augmentation - same gains for all frames in sequence
            if random.random() < 0.5:
                gains = np.random.uniform(-1, 1, 3) * [AUG_HSV_H, AUG_HSV_S, AUG_HSV_V] + 1
                for i in range(NUM_FRAMES):
                    stacked[i*3:(i+1)*3] = augment_hsv(stacked[i*3:(i+1)*3], gains)
            # Horizontal flip - applied to entire stacked tensor
            stacked, target["boxes"] = augment_fliplr(stacked, target["boxes"])
            # Scale augmentation
            if random.random() < AUG_SCALE_P:
                stacked, target["boxes"], target["labels"] = augment_scale(stacked, target["boxes"], target["labels"], AUG_SCALE_MIN, AUG_SCALE_MAX)

        # Normalize
        for i in range(NUM_FRAMES):
            for c in range(3):
                ch = i * 3 + c
                stacked[ch] = (stacked[ch] - MEANS[c]) / STDS[c]

        return stacked, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    tensors = torch.stack(imgs)
    masks = torch.zeros(tensors.shape[0], tensors.shape[2], tensors.shape[3], dtype=torch.bool)
    return NestedTensor(tensors, masks), list(targets)


# ─── mAP Evaluation ───
def compute_map(model, dataloader, device, iou_thresh=0.5):
    """Compute per-class AP and mAP@50."""
    model.eval()
    # Accumulate per-class TP/FP/scores
    all_preds = {c: [] for c in range(NUM_CLASSES)}  # list of (score, is_tp)
    all_n_gt = {c: 0 for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            logits = outputs["pred_logits"].sigmoid()
            pred_boxes = outputs["pred_boxes"]

            for bi in range(logits.shape[0]):
                gt_boxes = targets[bi]["boxes"]
                gt_labels = targets[bi]["labels"]
                scores, cls_ids = logits[bi].max(dim=-1)

                # Filter by threshold
                keep = scores > 0.01
                scores_k = scores[keep]
                cls_k = cls_ids[keep]
                boxes_k = pred_boxes[bi][keep]

                # Sort by score
                order = scores_k.argsort(descending=True)
                scores_k, cls_k, boxes_k = scores_k[order], cls_k[order], boxes_k[order]

                # Count GT per class
                for c in range(NUM_CLASSES):
                    all_n_gt[c] += (gt_labels == c).sum().item()

                # Match predictions to GT
                matched_gt = set()
                for si in range(len(scores_k)):
                    c = cls_k[si].item()
                    if c >= NUM_CLASSES:
                        continue
                    score = scores_k[si].item()
                    box = boxes_k[si]

                    # Find best IoU with GT of same class
                    best_iou, best_gi = 0, -1
                    gt_mask = gt_labels == c
                    gt_idx = gt_mask.nonzero(as_tuple=True)[0]
                    for gi in gt_idx:
                        if gi.item() in matched_gt:
                            continue
                        iou = _box_iou_single(box, gt_boxes[gi])
                        if iou > best_iou:
                            best_iou, best_gi = iou, gi.item()

                    is_tp = best_iou >= iou_thresh and best_gi >= 0
                    if is_tp:
                        matched_gt.add(best_gi)
                    all_preds[c].append((score, is_tp))

    # Compute AP per class
    aps = {}
    for c in range(NUM_CLASSES):
        preds = sorted(all_preds[c], key=lambda x: -x[0])
        tp_cum, fp_cum = 0, 0
        precisions, recalls = [], []
        n_gt = all_n_gt[c]
        if n_gt == 0:
            aps[c] = 0.0
            continue
        for score, is_tp in preds:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)
        # AP via 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p_at_r = max([p for p, r in zip(precisions, recalls) if r >= t], default=0)
            ap += p_at_r / 11
        aps[c] = ap

    mAP = np.mean(list(aps.values())) if aps else 0.0
    return mAP, aps


def _box_iou_single(box1, box2):
    """IoU between two boxes in cx,cy,w,h format."""
    x1 = max(box1[0] - box1[2]/2, box2[0] - box2[2]/2)
    y1 = max(box1[1] - box1[3]/2, box2[1] - box2[3]/2)
    x2 = min(box1[0] + box1[2]/2, box2[0] + box2[2]/2)
    y2 = min(box1[1] + box1[3]/2, box2[1] + box2[3]/2)
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = box1[2] * box1[3]
    a2 = box2[2] * box2[3]
    return inter / (a1 + a2 - inter + 1e-7)


# ─── Visualization ───
def save_overlay(model, dataset, epoch, device, output_dir, num_samples=10):
    model.eval()
    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    # Pick samples from different videos (by parent directory of first frame)
    seen_dirs = {}
    for i, (frames, *_rest) in enumerate(dataset.samples):
        parent = os.path.dirname(frames[0])
        if parent not in seen_dirs:
            seen_dirs[parent] = i
        if len(seen_dirs) >= num_samples:
            break
    sample_indices = list(seen_dirs.values())[:num_samples]

    for si, idx in enumerate(sample_indices):
        stacked, target = dataset._load_raw(idx)
        # Normalize for inference
        inp = stacked.clone()
        for i in range(NUM_FRAMES):
            for c in range(3):
                inp[i*3+c] = (inp[i*3+c] - MEANS[c]) / STDS[c]

        frame_paths = dataset.samples[idx][0]
        last_frame = np.array(Image.open(frame_paths[-1]).convert("RGB"))
        h_orig, w_orig = last_frame.shape[:2]

        inp_b = inp.unsqueeze(0).to(device)
        mask = torch.zeros(1, inp.shape[1], inp.shape[2], dtype=torch.bool, device=device)
        with torch.no_grad():
            out = model(NestedTensor(inp_b, mask))

        logits = out["pred_logits"].sigmoid()
        boxes = out["pred_boxes"]
        max_scores, _ = logits[0].max(dim=-1)
        keep = max_scores > 0.3
        det_scores, det_classes = max_scores[keep], logits[0][keep].argmax(dim=-1)
        det_boxes = boxes[0][keep]

        img = Image.fromarray(last_frame)
        draw = ImageDraw.Draw(img)
        COLORS = [(0,0,255), (255,0,0)]
        for score, cls_id, box in zip(det_scores, det_classes, det_boxes):
            cx, cy, bw, bh = box.cpu().numpy()
            x1, y1 = (cx-bw/2)*w_orig, (cy-bh/2)*h_orig
            x2, y2 = (cx+bw/2)*w_orig, (cy+bh/2)*h_orig
            c = cls_id.item()
            color = COLORS[c % len(COLORS)]
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.text((x1, y1-12), f"{name} {score:.2f}", fill=color)
        for box in target["boxes"]:
            cx, cy, bw, bh = box.numpy()
            x1, y1 = (cx-bw/2)*w_orig, (cy-bh/2)*h_orig
            x2, y2 = (cx+bw/2)*w_orig, (cy+bh/2)*h_orig
            draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
        img.save(overlay_dir / f"epoch{epoch:03d}_sample{si}.jpg")
    model.train()


def save_preprocessed_examples(dataset, output_dir, n=4):
    """Save preprocessed input tensor as image for verification (no augmentation)."""
    ex_dir = Path(output_dir) / "preprocessed_examples"
    ex_dir.mkdir(exist_ok=True)
    for idx in range(min(n, len(dataset))):
        stacked, _ = dataset._load_raw(idx)
        # Save all 3 frames side by side
        frames = []
        for i in range(NUM_FRAMES):
            frame = stacked[i*3:(i+1)*3].clone()
            frame = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            frames.append(frame)
        combined = np.concatenate(frames, axis=1)  # side by side
        Image.fromarray(combined).save(ex_dir / f"sample{idx}_frames.jpg")


# ─── Training ───
def train_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/overlays", exist_ok=True)

    # Build model
    config = RFDETRTemporalConfig(
        num_classes=NUM_CLASSES, resolution=RESOLUTION, temporal_fusion=TEMPORAL_FUSION
    )
    tc = TrainConfig(dataset_dir=".", output_dir=OUTPUT_DIR)
    ns = _namespace_from_configs(config, tc)
    model = build_model(ns).to(device)
    from rfdetr.models.weights import load_pretrain_weights
    load_pretrain_weights(model, config)
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    criterion, postprocess = build_criterion_and_postprocessors(ns)
    criterion = criterion.to(device)

    # Datasets
    train_ds = TemporalManifestDataset(f"{DATA_DIR}/train.txt", RESOLUTION, augment=True)
    val_ds = TemporalManifestDataset(f"{DATA_DIR}/valid.txt", RESOLUTION, augment=False)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
                              prefetch_factor=PREFETCH)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
                            collate_fn=collate_fn, num_workers=max(2, NUM_WORKERS // 2),
                            pin_memory=True, prefetch_factor=PREFETCH)

    if rank == 0:
        print(f"=== Temporal RF-DETR Training (DDP {world_size} GPUs) ===")
        print(f"Resolution: {RESOLUTION}, Epochs: {EPOCHS}, Batch/GPU: {BATCH_SIZE}")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        # Save config yaml
        cfg = {"resolution": RESOLUTION, "epochs": EPOCHS, "batch_size_per_gpu": BATCH_SIZE,
               "lr": LR, "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
               "num_gpus": NUM_GPUS, "num_frames": NUM_FRAMES,
               "augmentation": {"fliplr": AUG_FLIPLR, "hsv_h": AUG_HSV_H, "hsv_s": AUG_HSV_S,
                                "hsv_v": AUG_HSV_V, "mixup_p": AUG_MIXUP_P, "cutmix_p": AUG_CUTMIX_P},
               "model": "RFDETRTemporalBase (Base backbone ps=14, 9ch, 3cls)", "optimizer": "AdamW",
               "scheduler": "ConstantLR (StepLR drop=100)", "train_samples": len(train_ds),
               "val_samples": len(val_ds)}
        with open(f"{OUTPUT_DIR}/train_config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        # Save preprocessed examples
        save_preprocessed_examples(train_ds, OUTPUT_DIR)

    # Optimizer — layer-wise LR decay for backbone (RF-DETR/LW-DETR style)
    LR_ENCODER = 1.5e-4
    LR_LAYER_DECAY = 0.8
    LR_COMPONENT_DECAY = 0.7
    NUM_BACKBONE_LAYERS = 13  # out_feature_indexes[-1]+1 = 12+1

    def _get_backbone_lr(param_name):
        """Layer-wise LR: deeper layers get higher LR, with component decay."""
        import re
        match = re.search(r'encoder\.layer\.(\d+)', param_name)
        if match:
            layer_id = int(match.group(1)) + 1  # 1-indexed
        else:
            layer_id = 0  # embeddings, layernorm
        scale = LR_LAYER_DECAY ** (NUM_BACKBONE_LAYERS + 1 - layer_id)
        return LR_ENCODER * scale * (LR_COMPONENT_DECAY ** 2)

    def _no_weight_decay(param_name):
        """Params that should have weight_decay=0."""
        for kw in ('bias', 'norm', 'gamma', 'pos_embed', 'rel_pos', 'cls_token', 'mask_token', 'position_embeddings', 'embeddings', 'alpha'):
            if kw in param_name:
                return True
        return False

    # The TemporalPreEmbed module is new (zero-init) and must learn a temporal
    # residual from scratch while the rest of the network is pretrained. The
    # generic 'backbone' branch would assign it the layer-0 decayed LR
    # (~3e-6) — far too low for it to ever leave its identity init, which would
    # leave the model behaving as a single-frame detector. Give it a healthy LR.
    LR_TEMPORAL = float(os.environ.get("TEMPORAL_LR", "5e-4"))

    param_groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if _no_weight_decay(n) else 1e-4
        if 'temporal_pre' in n:
            lr = LR_TEMPORAL  # new motion-fusion module: learn from zero-init
        elif 'backbone' in n:
            lr = _get_backbone_lr(n)
        elif 'transformer.decoder' in n:
            lr = LR * LR_COMPONENT_DECAY  # decoder: lr × 0.7 = 7e-5
        else:
            lr = LR  # head/projector params
        param_groups.append({"params": [p], "lr": lr, "weight_decay": wd})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS, gamma=0.1)  # constant LR

    # EMA with tau warm-up (RF-DETR style)
    ema_decay = 0.993
    ema_tau = 100
    ema_updates = 0
    ema_model = None
    if rank == 0:
        import copy
        ema_model = copy.deepcopy(model.module)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # Metrics log
    history = []
    best_val_loss = float("inf")
    best_map = 0.0
    patience = 2
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        # Per-loss accumulators
        loss_accum = {}
        n_batches = 0

        for samples, targets in train_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # MixUp / CutMix (batch-level, sample pairs)
            if AUG_MIXUP_P > 0 or AUG_CUTMIX_P > 0:
                B = samples.tensors.shape[0]
                if B >= 2:
                    perm = torch.randperm(B, device=samples.tensors.device)
                    for bi in range(B):
                        bj = perm[bi].item()
                        if bi == bj:
                            continue
                        r = random.random()
                        if r < AUG_MIXUP_P:
                            mixed, tgt = mixup_data(samples.tensors[bi], targets[bi],
                                                    samples.tensors[bj], targets[bj])
                            samples.tensors[bi] = mixed
                            targets[bi] = tgt
                        elif r < AUG_MIXUP_P + AUG_CUTMIX_P:
                            mixed, tgt = cutmix_data(samples.tensors[bi], targets[bi],
                                                    samples.tensors[bj], targets[bj])
                            samples.tensors[bi] = mixed
                            targets[bi] = tgt
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # EMA update with tau warm-up + buffer sync
            if ema_model is not None:
                ema_updates += 1
                d = ema_decay * (1 - math.exp(-ema_updates / ema_tau))
                with torch.no_grad():
                    for ema_v, model_v in zip(ema_model.state_dict().values(),
                                              model.module.state_dict().values()):
                        if ema_v.is_floating_point():
                            ema_v.mul_(d).add_(model_v, alpha=1 - d)
                        else:
                            ema_v.copy_(model_v)

            # Accumulate individual losses
            for k in loss_dict:
                if k in weight_dict:
                    loss_accum[k] = loss_accum.get(k, 0.0) + loss_dict[k].item()
            n_batches += 1

        scheduler.step()

        # Validation loss
        model.eval()
        val_loss_accum = {}
        val_batches = 0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)
                for k in loss_dict:
                    if k in weight_dict:
                        val_loss_accum[k] = val_loss_accum.get(k, 0.0) + loss_dict[k].item()
                val_batches += 1

        # Reduce across GPUs
        def _reduce_dict(d, n):
            t = torch.tensor([d.get(k, 0.0) for k in sorted(d.keys())] + [n], device=device)
            dist.all_reduce(t)
            keys = sorted(d.keys())
            total_n = t[-1].item()
            return {k: t[i].item() / max(total_n, 1) for i, k in enumerate(keys)}, total_n

        train_losses, _ = _reduce_dict(loss_accum, n_batches)
        val_losses, _ = _reduce_dict(val_loss_accum, val_batches)

        if rank == 0:
            train_total = sum(train_losses[k] * weight_dict[k] for k in train_losses if k in weight_dict)
            val_total = sum(val_losses[k] * weight_dict[k] for k in val_losses if k in weight_dict)

            # mAP (on rank 0 only, using EMA model if available)
            eval_model = ema_model if ema_model is not None else model.module
            if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
                # Use non-distributed loader for full val evaluation
                full_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                             collate_fn=collate_fn, num_workers=NUM_WORKERS,
                                             pin_memory=True, prefetch_factor=PREFETCH)
                mAP, class_aps = compute_map(eval_model, full_val_loader, device)
            else:
                mAP, class_aps = -1, {c: -1 for c in range(NUM_CLASSES)}

            # Log
            epoch_log = {"epoch": epoch, "train_loss": train_total,
                         "val_loss": val_total, "mAP50": mAP,
                         "train_losses": train_losses, "val_losses": val_losses,
                         "class_AP": {CLASS_NAMES[c]: class_aps[c] for c in range(NUM_CLASSES)}}
            history.append(epoch_log)

            # Print
            ap_str = " | ".join(f"{CLASS_NAMES[c]}={class_aps[c]:.3f}" for c in range(NUM_CLASSES)) if mAP >= 0 else ""
            print(f"Epoch {epoch:3d}/{EPOCHS} | train={train_total:.4f} | val={val_total:.4f}"
                  f" | mAP50={mAP:.3f}" + (f" [{ap_str}]" if ap_str else ""))
            for k in sorted(train_losses.keys()):
                print(f"    {k}: train={train_losses[k]:.4f} val={val_losses.get(k,0):.4f}")

            # Save overlay every epoch (using EMA model)
            save_overlay(eval_model, val_ds, epoch, device, OUTPUT_DIR)

            # Save best (EMA weights)
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save({"model": eval_model.state_dict(), "epoch": epoch}, f"{OUTPUT_DIR}/best.pth")
            if mAP > best_map:
                best_map = mAP
                epochs_no_improve = 0
                torch.save({"model": eval_model.state_dict(), "epoch": epoch, "mAP": mAP}, f"{OUTPUT_DIR}/best_map.pth")
            elif mAP >= 0:
                epochs_no_improve += 1

            # Save last (every epoch)
            torch.save({"model": eval_model.state_dict(), "epoch": epoch}, f"{OUTPUT_DIR}/last.pth")

            # Save history
            with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
                json.dump(history, f, indent=2)

            # Early stopping (check only on mAP epochs)
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no mAP improvement for {patience} eval cycles)")
                break

    # Final save and plot
    if rank == 0:
        final_model = ema_model if ema_model is not None else model.module
        torch.save({"model": final_model.state_dict(), "epoch": EPOCHS}, f"{OUTPUT_DIR}/final.pth")
        print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}, Best mAP={best_map:.3f}")
        _plot_metrics(history)

    dist.destroy_process_group()


def _plot_metrics(history):
    """Generate training curves plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    mAPs = [h["mAP50"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0,0].plot(epochs, train_loss, label="train")
    axes[0,0].plot(epochs, val_loss, label="val")
    axes[0,0].set_title("Total Loss"); axes[0,0].legend(); axes[0,0].set_xlabel("Epoch")

    # mAP
    valid_maps = [(e, m) for e, m in zip(epochs, mAPs) if m >= 0]
    if valid_maps:
        axes[0,1].plot(*zip(*valid_maps), marker="o")
        # Per-class AP
        for c in range(NUM_CLASSES):
            class_ap = [(h["epoch"], h["class_AP"][CLASS_NAMES[c]]) for h in history if h["class_AP"][CLASS_NAMES[c]] >= 0]
            if class_ap:
                axes[0,1].plot(*zip(*class_ap), "--", label=CLASS_NAMES[c])
    axes[0,1].set_title("mAP@50"); axes[0,1].legend(); axes[0,1].set_xlabel("Epoch")

    # Individual losses (train)
    loss_keys = sorted(history[0]["train_losses"].keys())
    for k in loss_keys[:6]:
        axes[1,0].plot(epochs, [h["train_losses"].get(k, 0) for h in history], label=k)
    axes[1,0].set_title("Train Losses (components)"); axes[1,0].legend(fontsize=7); axes[1,0].set_xlabel("Epoch")

    # Individual losses (val)
    for k in loss_keys[:6]:
        axes[1,1].plot(epochs, [h["val_losses"].get(k, 0) for h in history], label=k)
    axes[1,1].set_title("Val Losses (components)"); axes[1,1].legend(fontsize=7); axes[1,1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.multiprocessing.spawn(train_ddp, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
