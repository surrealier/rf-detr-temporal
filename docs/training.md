# Temporal Training, Export, and Inference

> Part of RF-DETR-Temporal — see the [README](../README.md).

This document describes the end-to-end workflow for training the 9-channel temporal
detector on CCTV sequences: the manifest data format, the dataset and augmentation
pipeline, the distributed training loop, ONNX export, and the inference helpers. The
canonical training entry point is
[`train_temporal_base_v4.py`](../train_temporal_base_v4.py).

The detector ingests a short **3-frame** sliding window (9 input channels = 3 RGB
frames concatenated channel-wise) and predicts boxes for the **last (current)
frame**. Past frames provide motion context; supervision is causal (current-frame
only).

---

## 1. Manifest data format

Training reads plain-text manifests (one sequence per line). Each line is
pipe-delimited:

```
frame0|frame1|frame2|labels[|source_tag]
```

- `frame0`, `frame1`, `frame2` — absolute paths to the three frames in temporal
  order (`frame2` is the current/annotated frame).
- `labels` — semicolon-separated detections for the **last frame only**, each entry
  in normalized YOLO form `class_id cx cy w h` (e.g. `0 0.51 0.33 0.08 0.12;1 ...`).
  `class_id` is a generic integer in `[0, NUM_CLASSES)`; box coordinates are
  normalized to `[0, 1]`. Sequences whose label is empty after filtering are dropped
  at build time.
- `source_tag` (optional) — a provenance tag for the sequence. When set to `v4`, the
  loader applies a small per-frame sensor-noise pre-augmentation (see §3) to break up
  near-identical frames.

Labels always describe the **current frame**; the past two frames are unlabeled
context. This is a causal-detection convention: predict the present given recent history.

### Bring your own data

The pipeline is dataset-agnostic — it consumes the manifest format above and nothing
else. Produce `data_manifests/train.txt` and `data_manifests/valid.txt` with one clip
per line, where:

- each line is `frame0|frame1|frame2|labels[|source_tag]`,
- `frame*` are absolute paths to three temporally-ordered frames (`frame2` = current),
- `labels` are space-separated YOLO boxes `class_id cx cy w h` (normalized) for the
  **last** frame, multiple boxes separated by `;`,
- the optional `source_tag` opts a clip into the per-frame sensor-noise pre-augmentation.

Any sliding-window export of your own clips that emits this format works; point
training at the directory with `DATA_DIR=...`.

---

## 2. `TemporalManifestDataset`

`TemporalManifestDataset` (in `train_temporal_base_v4.py`) parses a manifest, then
per item:

1. Loads the 3 frames, converts to RGB float `[0, 1]`, and bilinearly resizes each to
   `RESOLUTION x RESOLUTION`.
2. Concatenates them channel-wise into a `(9, H, W)` tensor (`frame0` = channels 0-2,
   `frame1` = 3-5, `frame2` = 6-8).
3. Parses the last-frame labels into `boxes` (cx, cy, w, h) and integer `class_id`
   `labels`, keeping only ids in `[0, NUM_CLASSES)`.
4. Applies augmentation (training split only — see §3).
5. **Per-frame ImageNet normalization:** each 3-channel frame is normalized
   independently with `MEANS = [0.485, 0.456, 0.406]` and
   `STDS = [0.229, 0.224, 0.225]`. Normalization is the last step, after all spatial
   and photometric augmentation, so augmentations operate on raw `[0, 1]` pixels.

`collate_fn` stacks samples into a `NestedTensor` (with an all-false padding mask,
since all inputs are square and same-size) plus a list of per-sample target dicts.

---

## 3. Augmentation

All augmentations are applied **only on the training split** (`augment=True`) and are
**frame-consistent** within a sequence — the same geometric/photometric transform is
applied to all three frames so temporal correspondence is preserved. Probabilities
and ranges are env-configurable (§7).

**Pre-augmentation sensor noise** — for sequences tagged `v4`, Gaussian noise
(σ ≈ 1–2 DN on the 8-bit scale) is added to 2 of the 3 frames, simulating camera read
noise so otherwise-identical context frames are not bit-exact.

**HSV jitter** (`augment_hsv`, gated at 0.5 internally) — additive hue shift,
multiplicative saturation/value gains drawn once per sequence and applied to every
frame, controlled by `AUG_HSV_H/S/V`.

**Horizontal flip** (`augment_fliplr`, prob `AUG_FLIPLR`) — flips the full `(9, H, W)`
tensor along width and mirrors box `cx`.

**Random scale** (`augment_scale`, prob `AUG_SCALE_P`, range
`AUG_SCALE_MIN..AUG_SCALE_MAX`) — resizes all 9 channels by a common factor and
center-crops (scale ≥ 1) or zero-pads (scale < 1) back to `RESOLUTION`, adjusting box
centers/sizes and dropping out-of-bounds boxes.

**Mosaic** (`_mosaic`, prob `AUG_MOSAIC_P`) — builds a 2×2 grid from 4 sequences,
tiling each frame into its quadrant per-frame (preserving the 3-frame structure) and
remapping all boxes into the composite.

**MixUp** (`mixup_data`, prob `AUG_MIXUP_P`) — batch-level; blends two stacked tensors
with a `Beta(32, 32)` weight (near-0.5) and concatenates both targets.

**CutMix** (`cutmix_data`, prob `AUG_CUTMIX_P`) — batch-level; pastes a rectangular
region from a second sample and keeps the second sample's boxes whose centers fall
inside the cut.

MixUp/CutMix are applied in the training loop over batch pairs (a random permutation);
each pair is mixed with prob `AUG_MIXUP_P`, else cut-mixed with prob `AUG_CUTMIX_P`.

### Small-object copy-paste (`augment_small_object_paste`)

This is the key augmentation for distant/small targets. A size census of the data
showed the **small** COCO size bin was essentially empty for the primary target
category — the data is dominated by large instances — so the detector never sees the
small regime. This augmentation manufactures it.

Operating on the **unnormalized** `[0, 1]` 9-channel tensor (gated at prob
`AUG_SMALLOBJ_P`, up to `AUG_SMALLOBJ_N` pastes per sequence):

1. Pick an existing GT instance and crop its current-frame patch.
2. **Downscale** it so its longer side lands in `AUG_SMALLOBJ_SMIN..AUG_SMALLOBJ_SMAX`
   pixels (the small/medium regime). Sources too small to shrink into range are
   skipped.
3. **Alpha-blend** the shrunk patch back at a random location using a 2D
   raised-cosine feather mask (`_feather_mask`, a Hann-window outer product) scaled by
   a random alpha in `AUG_SMALLOBJ_AMIN..AUG_SMALLOBJ_AMAX`. The feather makes the
   paste read as a soft, semi-transparent blob rather than a hard rectangle.
4. The paste is **frame-consistent**: the same shrunk appearance is blended into all
   three frames. A new box/label (same `class_id` as the source) is added for the
   **last** frame.

`_feather_mask(h, w)` returns `hann(h) ⊗ hann(w)`, clamped ≥ 0 — ~1 in the center,
fading to 0 at the border.

**Motion (`AUG_SMALLOBJ_MOTION`)** — with a value > 0, the paste is given a random
per-sequence velocity and placed at a **different position in each frame**, arriving
at the GT location on the last frame. This manufactures small **moving** targets with
real inter-frame foreground — the signal the background-subtraction / motion add-weight
fusion is designed to amplify. With motion = 0 it is a static paste (identical
location across frames).

> On a deterministic synthetic small validation set, enabling this augmentation lifted
> class-averaged AP@0.5 in the small regime from ≈0.06 to ≈0.80 (medium ≈0.36 →
> ≈0.86; large stayed ≈0.95). Caveat: that synthetic set is in-distribution to the
> augmentation — it proves the detector *learns* small objects once they exist, not
> that this generalizes to real distant targets, which the sweep (§8) is measuring.

---

## 4. Distributed training loop

Training uses single-node multi-GPU **DDP** via `torch.multiprocessing.spawn`
(`NUM_GPUS` processes, NCCL backend, `find_unused_parameters=True`). Per process:

- Builds `RFDETRTemporalBaseConfig(num_classes, resolution, temporal_fusion)` and the
  LW-DETR model via `build_model`, then loads pretrained weights with
  `load_pretrain_weights`.
- Wraps in `DistributedDataParallel`; builds criterion + postprocessors.
- Uses `DistributedSampler` for both train and val loaders;
  `train_sampler.set_epoch(epoch)` each epoch for correct shuffling.
- Per step: optional MixUp/CutMix, forward, weighted-sum loss, `zero_grad`,
  `backward`, **grad clip to 0.1**, `optimizer.step()`.

**EMA.** Rank 0 maintains an exponential-moving-average copy of the model. The decay
ramps in with a τ warm-up: `d = ema_decay * (1 - exp(-updates / ema_tau))` with
`ema_decay = 0.993`, `ema_tau = 100`. Floating-point tensors are EMA-blended;
non-float buffers are copied. **All checkpoints and all evaluation use the EMA
weights.**

**Layer-wise LR decay (LLRD).** Per-parameter learning rates are assigned by name:

- `temporal_pre*` → `LR_TEMPORAL` (env `TEMPORAL_LR`, default `5e-4`). This is the
  **decisive special case**: the temporal fusion module is new and zero-initialized
  (an identity/zero-residual init) and must learn a motion residual from scratch. If
  it fell into the generic `backbone` branch it would get the layer-0 decayed LR
  (~3e-6) and never leave its identity init — collapsing the model back to a
  single-frame detector. It is therefore given a healthy dedicated LR.
- `backbone*` → `_get_backbone_lr`: `LR_ENCODER (1.5e-4) * 0.8^(depth-position) *
  0.7^2`, i.e. deeper encoder layers get higher LR (embeddings/early layers lowest),
  with an extra component decay.
- `transformer.decoder*` → `LR * 0.7`.
- everything else (head/projector) → `LR`.

Weight decay is `1e-4`, set to `0` for biases, norms, positional/relative embeddings,
class/mask tokens, and gain (`alpha`/`gamma`) params. Optimizer is **AdamW**; the
scheduler is `StepLR(step_size=EPOCHS, gamma=0.1)`, i.e. effectively **constant LR**
over the run.

**Per-epoch evaluation + early stopping.** Each epoch logs reduced train/val loss
components (all-reduced across GPUs). On rank 0, mAP@50 is computed with the EMA model
on epochs `1`, every 5th epoch, and the final epoch (a manual 11-point-interpolation
AP averaged over classes; IoU threshold 0.5; reported as class-averaged mAP). Each
epoch writes `last.pth`, `best.pth` (lowest val loss), and `best_map.pth` (highest
mAP), plus `metrics.json`, qualitative `overlays/`, and at the end
`training_curves.png` and `final.pth`. Early stopping triggers after `patience = 2`
mAP-eval cycles with no mAP improvement.

> Class-free result context: a single-frame baseline reaches mAP@50 ≈ 0.875; naive
> 9-channel averaging ≈ 0.892; the zero-init motion-residual fusion (`preembed`)
> reaches ≈ 0.9074. A plain add-weight fusion (`bgsub`) raised motion alignment ~3×
> but did **not** improve detection (best ≈ 0.8913, declining with training, and
> regressing on the low-motion subset by amplifying sensor noise). A
> coherence-gated add-weight (`bgsubcoh`) fixes the noise amplification; its A/B vs
> `preembed` (both with the moving-small augmentation) is the subject of the sweep
> (§8).

---

## 5. Example launch

Run from the repo root (training reads `${DATA_DIR}/train.txt` and
`${DATA_DIR}/valid.txt`):

```bash
DATA_DIR=data_manifests \
NUM_GPUS=4 NUM_WORKERS=5 PREFETCH=1 \
RESOLUTION=952 EPOCHS=12 BATCH_SIZE=8 LR=1e-4 \
TEMPORAL_FUSION=bgsubcoh TEMPORAL_LR=5e-4 \
AUG_SMALLOBJ_P=0.5 AUG_SMALLOBJ_MOTION=14 \
RUN_TAG=bgsubcoh_move \
python train_temporal_base_v4.py
```

Outputs land in `runs/<YYMMDD_HHMM>[_<RUN_TAG>]/`. The input pipeline is CPU-decode
bound (large frames, 3 per sequence); size `NUM_WORKERS`/`PREFETCH` to keep the GPUs
fed while staying within Docker `/dev/shm` (each batch is ~260 MB, and roughly
`NUM_GPUS * NUM_WORKERS * PREFETCH` batches are resident at once).

---

## 6. Environment variable reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `/home/data/hdd1_bongkj/data_ready_v4` | Directory containing `train.txt` / `valid.txt` |
| `RUN_TAG` | `""` | Suffix appended to the timestamped run dir under `runs/` |
| `NUM_GPUS` | `2` | DDP world size (processes spawned) |
| `NUM_WORKERS` | `5` | DataLoader workers per process |
| `PREFETCH` | `1` | DataLoader `prefetch_factor` |
| `RESOLUTION` | `952` | Square input resolution |
| `EPOCHS` | `30` | Training epochs (also the `StepLR` step size = constant LR) |
| `BATCH_SIZE` | `8` | Per-GPU batch size |
| `LR` | `1e-4` | Base LR (head/projector; decoder gets `LR × 0.7`) |
| `TEMPORAL_FUSION` | `bgsub` | Fusion mode: `bgsub`, `bgsubcoh`, `preembed`, or `none` (naive 9ch averaging) |
| `TEMPORAL_LR` | `5e-4` | Dedicated LR for the zero-init `temporal_pre*` fusion params |
| `AUG_FLIPLR` | `0.5` | Horizontal-flip probability |
| `AUG_HSV_H` | `0.015` | HSV hue jitter gain |
| `AUG_HSV_S` | `0.7` | HSV saturation jitter gain |
| `AUG_HSV_V` | `0.4` | HSV value jitter gain |
| `AUG_MIXUP_P` | `0.0` | MixUp probability (batch-level) |
| `AUG_CUTMIX_P` | `0.0` | CutMix probability (batch-level) |
| `AUG_MOSAIC_P` | `0.0` | 2×2 mosaic probability |
| `AUG_SCALE_P` | `0.0` | Random-scale probability |
| `AUG_SCALE_MIN` | `0.5` | Min scale factor |
| `AUG_SCALE_MAX` | `1.5` | Max scale factor |
| `AUG_SMALLOBJ_P` | `0.0` | Small-object copy-paste probability |
| `AUG_SMALLOBJ_N` | `3` | Max pastes per sequence |
| `AUG_SMALLOBJ_SMIN` | `16` | Min target longer-side (px @ RES) for pastes |
| `AUG_SMALLOBJ_SMAX` | `64` | Max target longer-side (px @ RES) for pastes |
| `AUG_SMALLOBJ_AMIN` | `0.6` | Min blend alpha |
| `AUG_SMALLOBJ_AMAX` | `0.95` | Max blend alpha |
| `AUG_SMALLOBJ_MOTION` | `0.0` | Max per-frame displacement (px @ RES); > 0 = moving small targets, 0 = static |

`MASTER_ADDR` / `MASTER_PORT` default to `localhost` / `29500` for the DDP rendezvous.

---

## 7. ONNX export

[`export_onnx.py`](../export_onnx.py) is **environment-configured (no CLI flags)**. It
monkey-patches `F.interpolate` to swap unsupported `bicubic` for `bilinear`, builds the
model at the training resolution (952, for correct positional-embedding size), loads
the checkpoint (`strict=False`), and wraps it so the graph outputs `boxes` and
`scores` (top-300 queries). It exports both a fixed-resolution 952 graph and a
dynamic-axes graph (batch/height/width) at opset 17 into `onnx_exports/`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT` | `runs/260502_0445_exp2_lr5e5/best_map.pth` | Checkpoint to export |
| `TEMPORAL_FUSION` | `preembed` | Must match how the checkpoint was trained (`none` for an old 9ch-expand checkpoint) |
| `EXPORT_NAME` | `temporal_exp2` | Base name; produces `<name>_952.onnx` and `<name>_dynamic.onnx` |

```bash
CHECKPOINT=runs/<run>/best_map.pth TEMPORAL_FUSION=bgsubcoh EXPORT_NAME=temporal_bgsubcoh \
  python export_onnx.py
```

---

## 8. Export & inference

Export a trained checkpoint to ONNX with [`export_onnx.py`](../export_onnx.py), which is
configured by **environment variables** (not CLI flags):

```bash
CHECKPOINT=runs/<run>/best_map.pth TEMPORAL_FUSION=bgsubcoh EXPORT_NAME=temporal \
uv run --no-sync python export_onnx.py     # -> onnx_exports/<EXPORT_NAME>_<res>.onnx
```

The exported graph takes the same `(1, 9, RES, RES)` input — the `TemporalPreEmbed`
fusion runs inside the graph. To run on video, maintain a 3-frame sliding buffer
(duplicate the first frames to pad), preprocess each frame exactly as in training
(resize to `RESOLUTION` + per-frame ImageNet normalization), stack into a 9-channel
tensor with the current frame last, and feed it to the model / ONNX session.
