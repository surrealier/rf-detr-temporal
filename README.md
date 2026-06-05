# RF-DETR-Temporal

**RF-DETR-Temporal** extends [Roboflow's RF-DETR](https://github.com/roboflow/rf-detr) from a
single-image detector into a **multi-frame, motion-aware** detector. It stacks three consecutive
frames and fuses them with a small **temporal pre-embedding** module placed *in front of the
unmodified pretrained DINOv2 patch embed* — so existing RF-DETR weights load verbatim and
training starts at exact single-frame parity, then learns temporal cues as a residual.

Its whole purpose is **moving objects** — e.g. **smoke or fire** — that are typically small,
distant, and semi-transparent in surveillance video: precisely the regime where a single frame
is weakest and **inter-frame motion is the discriminative cue** a still-image detector throws
away. The 9-channel design exists to exploit that motion.

> Derived from `roboflow/rf-detr` (Apache-2.0); significant changes were made — see
> [Attribution & license](#attribution--license).

## Pipeline

```
   frame t-2 ┐
   frame t-1 ┤  stack on channel axis            ┌──────────────── train-time only ───────────────┐
   frame t   ┘  (t = current / annotated)        │ small-object copy-paste augmentation:           │
        │                                         │ downscale instances → alpha-blend back, with    │
        ▼                                         │ optional per-frame motion (moving targets)       │
  (B, 9, H, W)  9-channel input ─────────────────┤  (manufactures the missing small-object regime)  │
        │                                         └──────────────────────────────────────────────────┘
        ▼
 ┌───────────────────────────────────────────────────────────┐
 │  TemporalPreEmbed   (B,9,H,W) → (B,3,H,W)   motion-aware    │
 │                                                             │
 │   preembed : out = cur + R(diffs)            R: zero-init   │
 │   bgsub    : out = cur + α·(cur − mean(past)) + R           │
 │   bgsubcoh : bgsub, but the foreground is coherence-gated   │
 │             (keeps only motion consistent across all        │
 │              transitions; suppresses sensor noise)          │
 └───────────────────────────────────────────────────────────┘
        │  (B, 3, H, W)   ← sharp, in-distribution for pretrained weights
        ▼
  DINOv2 ViT backbone (windowed attention)      ← 3-channel weights load byte-for-byte
        │
        ▼
  Multi-scale projector (ViTDet/LW-DETR FPN)
        │
        ▼
  LW-DETR deformable decoder + detection heads
        │
        ▼
  detections  { pred_boxes, pred_logits }
```

At initialisation `R ≡ 0` and (for `bgsub`/`bgsubcoh`) the add-weight term vanishes on static
input, so the backbone receives exactly the current frame → **identical to the single-frame
baseline**. The temporal contribution is learned from there.

## What's different from upstream

| | Upstream RF-DETR | RF-DETR-Temporal |
|---|---|---|
| Input | 1 frame, `(B,3,H,W)` | 3 stacked frames, `(B,9,H,W)`, last = current |
| Multi-channel | widens the patch-embed conv (sums channels → **averages** frames) | `TemporalPreEmbed` reduces `9→3` before the **unmodified 3ch** patch embed |
| Pretrained patch embed | lossily widened | **loaded unchanged**; new module absorbed by `strict=False` |
| Init behaviour | — | **exact single-frame parity**; temporal learned as a residual |
| New config | — | `in_channels`, `temporal_fusion ∈ {none, preembed, bgsub, bgsubcoh}` |
| Training | PyTorch-Lightning stack | standalone DDP script + manifest loader + temporal-aware augmentation |
| Tooling | — | `diagnostics/` suite (parity, motion-/size-stratified eval, …) |

The detector, decoder, heads, loss, and matcher are **unchanged**; the only model change is the
new module inside the backbone plus two config fields and a one-line gate in the weight loader.

## How it works (short)

- **The pitfall.** Naively widening the patch-embed `Conv2d` to 9 channels makes the embedding
  compute the **temporal average** of the frames — a low-pass / motion-blur op that destroys the
  change signal and feeds the backbone an out-of-distribution blurred image.
- **The fix — `TemporalPreEmbed`.** Reduce `9→3` channels with a small motion-aware module
  *before* the unmodified 3-channel patch embed, so DINOv2/RF-DETR weights load verbatim and the
  model starts at single-frame parity. Three fusion modes (`preembed`, `bgsub`, `bgsubcoh`) trade
  off how the temporal/motion signal is injected.
- **The data lever.** A size census showed the validation set was dominated by large instances
  with the small-object regime nearly empty — so the detection bottleneck was **data, not
  architecture**. A small-object copy-paste augmentation manufactures the missing small (optionally
  moving) targets.

Full details: **[docs/architecture.md](docs/architecture.md)** ·
**[docs/temporal-fusion.md](docs/temporal-fusion.md)** ·
**[docs/training.md](docs/training.md)** ·
**[docs/findings.md](docs/findings.md)**

## Quick start

```bash
pip install uv && uv sync --all-groups        # PyTorch >=2.2,<3; transformers >=5,<6; Python >=3.10

# provide your data as two manifests — data_manifests/{train,valid}.txt — one clip per line:
#   /abs/frame0|/abs/frame1|/abs/frame2|<labels>
# where <labels> are YOLO "cls cx cy w h" boxes for the LAST (current) frame. See docs/training.md.

# train on 4 GPUs (coherence-gated motion add-weight + moving small-object augmentation)
DATA_DIR=data_manifests NUM_GPUS=4 RESOLUTION=952 \
TEMPORAL_FUSION=bgsubcoh AUG_SMALLOBJ_P=0.5 AUG_SMALLOBJ_MOTION=14 \
uv run --no-sync python train_temporal_base_v4.py
```

Env-var reference, data format, ONNX export and inference: **[docs/training.md](docs/training.md)**.

## Results

Accuracy is class-averaged **mAP@0.5**. "Aggregate" is on the real validation set (dominated by
large instances); "small-object" is on a deterministic *synthetic* small-object set, since the
real data has almost no small instances (the data bottleneck — see below).

| Configuration | Aggregate mAP@0.5 | Synthetic small-object mAP@0.5 |
|---|:---:|:---:|
| single-frame baseline | 0.875 | — |
| naive 9-channel (temporal averaging) | ~0.892 | — |
| **`preembed`** (temporal) | **0.907** | ~0.06 |
| `preembed` + small-object augmentation | ~0.88 | **~0.80** |
| `bgsub` (plain motion add-weight) | 0.891 | — |
| `bgsubcoh` + moving-object augmentation | *in progress* | *in progress* |

Reading: the temporal fix recovers and exceeds the single-frame baseline; the plain motion add-weight
**raises motion-region attention ~3×** but did **not** improve aggregate detection (an honest
negative result); and the real lever for the moving/small regime was **data** — synthetic
small-object augmentation lifts small-object mAP from ≈0.06 to ≈0.80. Full record, caveats, and
the motion-stratified breakdown: **[docs/findings.md](docs/findings.md)**.

### Speed

The temporal extension is **near-zero overhead**: the DINOv2 backbone and LW-DETR decoder are
**unchanged**, and the fusion module adds just **438 parameters (0.001% of the model)** — a few
convolutions at input resolution. Per-inference network latency therefore matches upstream
**RF-DETR Base** (real-time class) on the same hardware; the only added runtime cost is decoding
**3 frames instead of 1** per inference (I/O, not compute). Absolute FPS is not separately
benchmarked here.

## Repository layout

```
src/rfdetr/                              # upstream RF-DETR (Apache-2.0), minimally modified
└── models/backbone/temporal_fusion.py  # ★ TemporalPreEmbed (9ch→3ch motion fusion)
train_temporal_base_v4.py               # ★ DDP training entrypoint + dataset + augmentations
diagnostics/                            # ★ probes + stratified evaluators
export_onnx.py                          # ONNX export (9-channel temporal model)
docs/                                   # detailed design / training / findings
```

Generated locally and git-ignored: `runs/`, `onnx_exports/`, `data_manifests/`, and all
`*.pth`/`*.onnx`/`*.mp4` artifacts (see `.gitignore`).

## Attribution & license

Derivative work of **[RF-DETR](https://github.com/roboflow/rf-detr)** by Roboflow, licensed under
the **Apache License 2.0**, and itself released under Apache-2.0. The upstream `LICENSE` is retained
and all upstream source files keep their original license headers. Per Apache §4, this notice
states that **significant changes were made**: a new temporal pre-embedding module and fusion
modes, 9-channel input wiring, a small-object augmentation, and standalone training/diagnostics
tooling. The DINOv2-with-Registers backbone code is itself derived from HuggingFace Transformers
(see that file's header).
