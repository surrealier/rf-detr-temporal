# Architecture

> Part of RF-DETR-Temporal — see the [README](../README.md).

This document describes (a) the **upstream RF-DETR** architecture and repository
layout we build on, and (b) the **deltas introduced by RF-DETR-Temporal** to make
the detector consume short stacks of consecutive CCTV frames instead of a single
image. All file paths are relative to `src/rfdetr/`.

---

## Part A — Upstream RF-DETR

RF-DETR is a DETR-family detector: a windowed DINOv2 ViT backbone, a multi-scale
FPN-style projector, and a deformable-attention LW-DETR decoder with set-based
(Hungarian) matching. The Python surface is layered into a thin user-facing
facade, a lightweight runtime context, and the underlying `nn.Module`.

### A.1 Layering: `RFDETR` facade → `ModelContext` → `LWDETR`

```
RFDETR (detr.py)                 user-facing facade: predict/train/export/deploy
   └─ self.model: ModelContext   (inference.py) runtime wrapper, lazy device placement
        └─ .model: LWDETR        (models/lwdetr.py) the actual nn.Module
```

**`RFDETR`** (`detr.py`) is the public class. Its `__init__` builds a
`ModelConfig` via `get_model_config(**kwargs)`, optionally downloads pretrained
weights, then calls `get_model()` → `_build_model_context()`. It owns the
high-level verbs:

- `predict()` — normalises images (per-3-channel-group `F.normalize` with ImageNet
  mean/std), resizes to a shape divisible by `patch_size * num_windows`, runs the
  module, and converts raw outputs to `supervision.Detections`. Class IDs are
  0-indexed; logit index `num_classes` is the background slot, mapped to
  `__background__`.
- `train()` — delegates to the PyTorch Lightning stack
  (`RFDETRModelModule` / `RFDETRDataModule` / `build_trainer`), syncing the trained
  `nn.Module` back onto `self.model.model` afterward.
- `export()` — deep-copies the module, switches it into export mode, and emits ONNX
  (optionally TFLite). `optimize_for_inference()` traces it with `torch.jit.trace`.
- `from_checkpoint()` — resolves the correct subclass from the checkpoint's
  `model_name` (or legacy `pretrain_weights` filename) and reconstructs it.

**`ModelContext`** (`inference.py`) is a lightweight container holding `model`
(the `LWDETR` `nn.Module`), `postprocess` (a `PostProcess`), `device`,
`resolution`, the derived `args` namespace, and `class_names`. It is built by
`_build_model_context(model_config)`, which:

1. constructs a transitional `SimpleNamespace` via `_namespace_from_configs`,
2. calls `build_model(args)` to assemble the `LWDETR`,
3. loads pretrained weights through `load_pretrain_weights` when configured,
4. optionally applies LoRA.

The module is intentionally **kept on CPU** at construction; `_ensure_model_on_device`
in `detr.py` performs the deferred `.to(device)` on first use. This avoids
initialising CUDA inside `__init__`, which would otherwise break fork/spawn DDP
strategies in notebook environments.

**`LWDETR`** (`models/lwdetr.py`, `nn.Module`) is the detector. `forward` takes a
`NestedTensor`, runs `self.backbone` to obtain multi-scale feature maps + position
embeddings, decomposes them into `srcs`/`masks`, then calls `self.transformer` with
the reference-point embeddings (`refpoint_embed`) and query features (`query_feat`).
Decoder hidden states feed the classification head (`class_embed`, a `nn.Linear`)
and the box head (`bbox_embed`, a 3-layer `MLP`). `forward_export` returns plain
tuples for tracing.

### A.2 Backbone: windowed-attention DINOv2

The backbone is built by `build_backbone` (`models/backbone/__init__.py`) and
wrapped by `Backbone` (`models/backbone/backbone.py`), which holds two parts:

1. **`DinoV2` encoder** (`models/backbone/dinov2.py`). The variant name (e.g.
   `"dinov2_windowed_small"`) is parsed for `registers`/`windowed` flags and a
   size suffix. When windowed attention is requested it instantiates
   `WindowedDinov2WithRegistersBackbone` from a local config
   (`dinov2_configs/dinov2_<size>.json`); otherwise it falls back to a stock
   HuggingFace `AutoBackbone`.

   **Windowed attention** (`models/backbone/dinov2_with_windowed_attn.py`) is a
   local fork of the HF DINOv2-with-Registers implementation (targeting
   `transformers>=5.0`). In `WindowedDinov2WithRegistersEmbeddings.forward`, when
   `num_windows > 1` the patch grid is reshaped into `num_windows²` independent
   windows along the batch dimension so most blocks attend only within a window
   (cheap). Blocks listed in `out_feature_indexes` instead run **full** attention
   (`run_full_attention` in `WindowedDinov2WithRegistersLayer`), temporarily
   un-windowing to mix global context. The backbone returns reshaped 4-D feature
   maps `(B, hidden, H/ps, W/ps)` at the selected stages, with windowing undone.

   > **NOTE — the "Base" variant uses a ViT-Small basis.** `RFDETRBaseConfig` sets
   > `encoder="dinov2_windowed_small"`, i.e. `dinov2_small.json` with
   > `hidden_size=384`, `num_hidden_layers=12`, `num_attention_heads=6`,
   > `patch_size=14`, native `image_size=518` (PE grid 37). The model "size" name
   > (Base/Large/etc.) refers to the *overall RF-DETR configuration*, not the ViT
   > width; both Base and the newer Large configs ride on the ViT-Small encoder.

   Inputs must be divisible by `block_size = patch_size * num_windows`. The encoder
   exposes `_out_feature_channels` (= `size_to_width[size]`, repeated per output
   index) for the projector.

2. **`MultiScaleProjector`** (`models/backbone/projector.py`). This is the
   FPN-style neck. Each requested pyramid level (`projector_scale` ⊂
   {P3, P4, P5, P6}) maps to a resampling factor (P3→2× up, P4→1×, P5→0.5× down,
   P6→0.25× pool). For every level it resamples each input feature map
   (`ConvTranspose2d` to upsample, strided `ConvX` to downsample), concatenates,
   then fuses through a `C2f` (CSP-bottleneck) block + LayerNorm to `out_channels`
   (= `hidden_dim`). The result is the list of per-level feature maps the decoder
   consumes; `len(projector_scale)` sets `num_feature_levels`.

### A.3 Decoder, queries, and heads (LW-DETR)

`build_transformer` (`models/transformer.py`) builds a deformable-attention
decoder (`Transformer` → `TransformerDecoder` of `TransformerDecoderLayer`), using
`MSDeformAttn` (`models/ops/`) for multi-scale deformable cross-attention over the
projector's feature levels. Key LW-DETR mechanics from `LWDETR.__init__`:

- **Queries.** `num_queries` object slots, each with a learnable 4-D reference
  point (`refpoint_embed`, zero-initialised) and a content embedding
  (`query_feat`). With **group DETR** (`group_detr` groups, default 13) the query
  bank is `num_queries * group_detr` during training and sliced to `num_queries`
  at inference for faster convergence.
- **Two-stage** (`two_stage=True`). The encoder proposes box/class priors via
  `enc_out_bbox_embed` / `enc_out_class_embed` before the decoder refines them.
- **Box reparameterisation** (`bbox_reparam=True`). Decoder box deltas are applied
  multiplicatively to reference width/height and additively to center, rather than
  the plain `sigmoid(delta + ref)` scheme.
- **Heads.** `class_embed` (`nn.Linear` → `num_classes + 1` logits, focal-loss
  bias init) and `bbox_embed` (`MLP`, last layer zero-init). `reinitialize_detection_head`
  resizes `class_embed` (and the two-stage enc heads) by tiling/truncating to a new
  class count — used during weight loading.
- **Auxiliary losses** (`aux_loss`) attach a head to every decoder layer.

Set matching (`models/matcher.py`), losses (`models/criterion.py`,
`SetCriterion` with focal/varifocal/IA-BCE options), and `PostProcess`
(`models/postprocess.py`, top-`num_select` selection) complete the pipeline. An
optional `SegmentationHead` (`models/heads/`) adds `pred_masks`.

### A.4 Config variants

Configs are Pydantic models in `config.py`. `BaseConfig` forbids unknown fields
(typo-catching) and validates on assignment. `ModelConfig` is the authoritative
architecture spec (`encoder`, `hidden_dim`, `patch_size`, `num_windows`,
`out_feature_indexes`, `projector_scale`, `resolution`,
`positional_encoding_size`, `num_queries`, `group_detr`, `in_channels`,
`num_classes`, `pretrain_weights`, …). Named variants subclass it and pin the
field values, e.g.:

| Variant config | encoder (ViT basis) | patch_size | num_windows | resolution | PE | projector_scale |
| --- | --- | --- | --- | --- | --- | --- |
| `RFDETRBaseConfig` | small (384) | 14 | 4 | 560 | 37 | [P4] |
| `RFDETRSmallConfig` | small (384) | 16 | 2 | 512 | 32 | [P4] |
| `RFDETRMediumConfig` | small (384) | 16 | 2 | 576 | 36 | [P4] |
| `RFDETRLargeConfig` | small (384) | 16 | 2 | 704 | 44 | [P4] |
| `RFDETRNanoConfig` | small (384) | 16 | 2 | 384 | 24 | [P4] |

`RFDETRBaseConfig` deliberately keeps `positional_encoding_size=37` (DINOv2's
native 518 px grid) while training at 560 px, so PE is *not* recomputed from
resolution — a `model_validator` (`_sync_pe_with_resolution`) only rescales PE when
the class default is formula-derived (`PE == resolution // patch_size`). The
parallel `RFDETR*` *facade* subclasses live in `variants.py`, each pinning
`_model_config_class` to the matching config. `TrainConfig` carries training
hyperparameters; `_namespace_from_configs` (`_namespace.py`) merges
`ModelConfig` + `TrainConfig` + hardcoded `MODEL_DEFAULTS` into the transitional
`SimpleNamespace` the builders consume (ModelConfig wins on overlapping
architecture fields).

### A.5 Pretrained weight loading & patch-embed widening

`load_pretrain_weights` (`models/weights.py`) loads a checkpoint into the freshly
built `LWDETR` (`strict=False`). It:

- normalises PTL `.ckpt` (`state_dict` with `model.` prefix, stripping
  `_orig_mod.` from compiled runs) to the `{"model": {...}}` form;
- extracts `class_names` from the checkpoint `args`;
- aligns the detection head to the checkpoint's class count
  (`reinitialize_detection_head`), auto-adopting the checkpoint count unless the
  user explicitly overrode `num_classes`, then expanding/trimming back as needed;
- trims `refpoint_embed`/`query_feat` rows to `num_queries * group_detr`;
- bicubic-interpolates DINOv2 `position_embeddings` to the configured PE grid
  (`_interpolate_position_embeddings`) so a custom resolution still loads.

**Patch-embed widening (`_expand_patch_embed_channels`).** Upstream, the only way
to feed more than 3 input channels is to widen the patch-embed `Conv2d`. This
helper **tiles** the pretrained `(hidden, 3, ps, ps)` projection weight along the
input-channel axis up to `target_channels` and rescales by `3 / target_channels`
to preserve activation magnitude. Because the conv then *sums* over all input
channels with near-identical per-group weights, the patch embedding effectively
computes a **temporal average** of the stacked frames — the failure mode the
temporal work was designed to fix (see Part B).

---

## Part B — What RF-DETR-Temporal changes

The goal is to detect objects in **CCTV video** by giving the detector temporal
context: a sample is a stack of `num_frames` consecutive frames (default 3),
concatenated channel-wise into a `(3 * num_frames, H, W)` tensor whose **last**
group is the annotated/current frame. Labels apply to the current frame; the
manifest carries a generic per-sample `class id` (no class names are encoded in
the architecture). The architectural surface stays almost identical to upstream —
the changes are concentrated in how multi-frame input is fused *before* the
pretrained backbone.

### B.1 The core problem and the fix

Naively widening the patch embed to `3 * num_frames` channels (Part A.5) makes the
backbone average the frames. Diagnostics confirmed this empirically: per-frame
patch-embed weight groups stayed near-identical (cosine ≈ 0.998) and the output
was essentially invariant to shuffling or zeroing the past frames — i.e. pure
temporal averaging / motion blur that discards the change signal. A 9-channel
averaging run scored mAP@50 ≈ 0.892 vs the single-frame baseline 0.875: better,
but for the wrong reason and not the intended mechanism.

**`TemporalPreEmbed`** (`models/backbone/temporal_fusion.py`) replaces widening.
It is a small learnable module placed *in front of* the unmodified 3-channel patch
embed. It reduces `3 * num_frames` channels back to 3, so the pretrained
DINOv2/RF-DETR weights load **verbatim** (no channel expansion). Three modes,
selected by `ModelConfig.temporal_fusion`:

- **`preembed`** (default for temporal configs). The learnable branch consumes
  temporal *differences* `[f_cur, f_cur − f_0, …]` so it responds to motion. The
  output conv is **zero-initialised** (ControlNet-style "zero conv"), so at init
  the module output equals the current frame exactly — guaranteed parity with the
  single-frame baseline, with the temporal contribution learned as a residual
  growing from zero. This was the best detector: mAP@50 ≈ 0.9074.
- **`bgsub`** (background-subtraction add-weight). Estimates a background as the
  mean of past frames, forms foreground `fg = f_cur − bg`, and add-weights it back
  with a learnable positive per-channel gain `alpha` (a temporal unsharp mask),
  active from step one. The injected enhancement was demonstrably more
  motion-aligned (≈0.505 vs preembed 0.166, ~3×) and concentrated energy on
  ground-truth boxes (1.38× vs preembed's ~random 1.00×) — yet aggregate detection
  did **not** improve (best ≈ 0.8913, below preembed, and declining with training,
  reproducible across two runs). Motion-stratified evaluation explained why: on
  high-motion val it matched preembed, but on low-motion val it regressed by
  amplifying sensor noise on faint inputs.
- **`bgsubcoh`** (coherence-gated `bgsub`). Adds a gate keeping only motion
  *consistent across every frame transition* (per-pixel min of each diff projected
  onto the foreground direction, ReLU-clamped); uncorrelated noise is suppressed to
  zero. Unit-verified: static input → exact parity (0.0), coherent motion →
  amplified, incoherent noise → suppressed. This fixes `bgsub`'s noise
  amplification; a final A/B vs preembed (both with the moving-small augmentation)
  was running at the time of writing — **verdict pending**.

In all modes the residual (and `alpha`'s effect on static regions) is zero at init,
so a temporal model is a strict superset of its single-frame initialisation.

### B.2 The real bottleneck: data, not architecture

A size census of the validation set showed the small-object COCO bin was
essentially **empty** for the primary target category; the data is dominated by
large instances. The detection ceiling is therefore a **data** problem. A
small-object **copy-paste augmentation** (`datasets/synthetic.py`) was added to
manufacture small (and, in a later variant, small *moving*) targets. On a
deterministic synthetic small val (class-averaged AP@0.5), copy-paste lifted small
≈0.06→≈0.80 and medium ≈0.36→≈0.86 while leaving large ≈0.96→≈0.95. Caveat: the
synthetic small distribution is in-distribution to the augmentation, so this proves
the detector *can learn* small objects once they exist in training — it is **not**
proof of generalisation to real distant targets.

### B.3 Files added / modified

| File | Add/Mod | What changed |
| --- | --- | --- |
| `models/backbone/temporal_fusion.py` | **add** | `TemporalPreEmbed`: motion-aware fusion of `3*num_frames`→3 channels (`preembed`/`bgsub`/`bgsubcoh`). |
| `models/backbone/dinov2.py` | mod | Instantiate `TemporalPreEmbed` when `temporal_fusion != "none"`; keep `num_channels=3` so DINOv2 weights load verbatim; apply fusion at the top of `forward`. |
| `models/backbone/backbone.py` | mod | Thread `num_channels` / `temporal_fusion` through to `DinoV2`. |
| `models/backbone/__init__.py` | mod | `build_backbone` accepts and forwards `num_channels` / `temporal_fusion`. |
| `models/lwdetr.py` | mod | `build_model` forwards `in_channels` and `temporal_fusion` (via `getattr`, default safe) to `build_backbone`. |
| `models/weights.py` | mod | Skip `_expand_patch_embed_channels` for preembed-family fusion (patch embed stays 3-ch); new `temporal_pre.*` params absorbed by `strict=False` at identity init. |
| `config.py` | mod | Add `in_channels`, `temporal_fusion` to `ModelConfig`; add `RFDETRTemporalConfig` / `RFDETRTemporalBaseConfig` / `TemporalTrainConfig`; extend `dataset_file` with `"temporal"`. |
| `_namespace.py` | mod | Forward `in_channels` and `temporal_fusion` into the builder namespace. |
| `detr.py` | mod | `predict()` handles `in_channels > 3` (replicate a single frame, per-3-group normalisation); `optimize_for_inference`/`export` use `in_channels` for dummy inputs. |
| `datasets/temporal.py` | **add** | `TemporalCocoDetection` + `build_temporal`: load sequence dirs, stack `num_frames` frames channel-wise, annotate the last frame. |
| `datasets/synthetic.py` | **add** | Small-object (and small-moving) copy-paste augmentation + synthetic COCO generation for size-stratified evaluation. |
| `datasets/__init__.py` | mod | Route `dataset_file == "temporal"` to `build_temporal`. |
| `variants.py` / `__init__.py` / `_namespace`-adjacent | mod | Expose temporal facade variants and wiring. |

Temporal configs (`config.py`): `RFDETRTemporalConfig` rides the Large config
(patch_size 16, `in_channels=9`, `temporal_fusion="preembed"`, `resolution=960`,
PE 60); `RFDETRTemporalBaseConfig` rides the Base config (patch_size **14**,
`in_channels=9`, `resolution=952 = 68*14`, PE 68). `TemporalTrainConfig` defaults
`dataset_file="temporal"` and `num_frames=3`.

### B.4 Conceptual deltas at a glance

- **Input contract:** `(3, H, W)` → `(3*num_frames, H, W)`, last group = current
  frame; `H, W` divisible by `patch_size * num_windows` as before.
- **Fusion point:** a new pre-embed stage compresses frames to 3 channels *before*
  the pretrained patch embed, instead of widening it. Backbone/projector/decoder/
  heads are byte-for-byte upstream.
- **Weight compatibility:** preembed-family fusion preserves the 3-channel patch
  embed, so all pretrained RF-DETR weights load unchanged; only the small
  `temporal_pre.*` module is new (and inert at init).
- **Initialisation guarantee:** at step 0 a temporal model reproduces its
  single-frame baseline exactly (parity), then learns temporal cues as a residual.
- **Evaluation:** add motion-stratified (high/low inter-frame motion) and
  size-stratified (synthetic small/medium/large) splits; report class-averaged mAP
  only.

---

## Appendix — `src/rfdetr/` directory map

```
src/rfdetr/
├── detr.py                 # RFDETR facade (predict/train/export/from_checkpoint)
├── inference.py            # ModelContext + _build_model_context
├── config.py               # Pydantic ModelConfig/TrainConfig + all variants (incl. temporal)
├── variants.py             # RFDETR* facade subclasses
├── _namespace.py           # _namespace_from_configs: configs → builder SimpleNamespace
├── models/
│   ├── lwdetr.py           # LWDETR nn.Module + build_model
│   ├── transformer.py      # deformable decoder (Transformer/Decoder/Layer)
│   ├── matcher.py          # Hungarian matcher
│   ├── criterion.py        # SetCriterion (focal/varifocal/IA-BCE, dice)
│   ├── postprocess.py      # PostProcess (top-num_select)
│   ├── position_encoding.py
│   ├── math.py             # MLP and helpers
│   ├── weights.py          # load_pretrain_weights, apply_lora, patch-embed widening
│   ├── _defaults.py / _types.py / _namespace bridge
│   ├── ops/                # MSDeformAttn CUDA/py ops
│   ├── heads/              # SegmentationHead
│   └── backbone/
│       ├── __init__.py     # build_backbone
│       ├── backbone.py     # Backbone (encoder + projector wrapper)
│       ├── base.py         # BackboneBase
│       ├── dinov2.py       # DinoV2 encoder selector (+ temporal_pre wiring)
│       ├── dinov2_with_windowed_attn.py  # windowed DINOv2-with-registers fork
│       ├── dinov2_configs/ # per-size JSON (dinov2_small.json = ViT-S, hidden 384)
│       ├── projector.py    # MultiScaleProjector (FPN) + C2f/ConvX
│       └── temporal_fusion.py  # [TEMPORAL] TemporalPreEmbed
├── datasets/
│   ├── __init__.py         # build_dataset routing (+ "temporal")
│   ├── temporal.py         # [TEMPORAL] TemporalCocoDetection + build_temporal
│   ├── synthetic.py        # [TEMPORAL] small-object copy-paste + synthetic gen
│   ├── coco.py / yolo.py / o365.py / transforms.py / aug_config.py / ...
├── training/               # PyTorch Lightning module/datamodule/trainer/callbacks
├── export/                 # ONNX (_onnx) + TFLite (_tflite) export
├── evaluation/             # metrics
├── deploy/ · platform/ · cli/ · visualize/ · assets/ · utilities/ · util/
```
