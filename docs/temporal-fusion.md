# Temporal Pre-Embedding Fusion

> Part of RF-DETR-Temporal — see the [README](../README.md).

This document describes the design and the guiding principles of the temporal
pre-embedding fusion used to feed multi-frame (stacked) input into an
otherwise single-frame RF-DETR detector. It covers the failure mode that
motivated the design, the input convention, the three fusion modes with their
math, and how the pretrained patch embedding is preserved through weight
loading.

The implementation lives in
[`src/rfdetr/models/backbone/temporal_fusion.py`](../src/rfdetr/models/backbone/temporal_fusion.py),
is wired in
[`src/rfdetr/models/backbone/dinov2.py`](../src/rfdetr/models/backbone/dinov2.py),
configured in [`src/rfdetr/config.py`](../src/rfdetr/config.py), and supported by
weight-loading logic in
[`src/rfdetr/models/weights.py`](../src/rfdetr/models/weights.py).

## 1. Motivation: the patch-embed averaging pitfall

The obvious way to give a detector temporal context is to stack `num_frames`
RGB frames along the channel dimension into a `3 * num_frames` channel tensor
and widen the patch-embedding `Conv2d` to accept those channels. This is a
trap, and the trap is structural — it does not go away with more training.

DINOv2's patch embedding is a strided `Conv2d` whose kernel is initialised from
a pretrained 3-channel weight `W` of shape `(hidden, 3, ps, ps)`. To accept
`3 * num_frames` channels, that weight is tiled into per-frame groups
`[W, W, ..., W]` (each group a copy of the pretrained 3-channel kernel). For a
single output position, the convolution computes a sum over **all** input
channels:

```
W_wide · [f0, f1, f2] = W·f0 + W·f1 + W·f2
                      = W·(f0 + f1 + f2)
                      = 3 · W·mean(f0, f1, f2)
```

Because each per-frame weight group is an identical copy of `W`, the widened
convolution is algebraically equal (up to the constant front factor that the
loader rescales away; see §5) to the patch embedding applied to the
**temporal mean** of the frames. In other words, the very first operation of
the network low-pass filters the time axis: it produces a motion-blurred image
and hands that out-of-distribution blur to a backbone pretrained on sharp
single frames. The change signal — exactly what a temporal model is supposed to
attend to — is averaged out before any attention layer sees it.

This is not merely theoretical. Empirically, the per-frame patch-embed weight
groups stayed near-identical through training (cosine similarity of the weight
groups ≈ 0.998), and the model output was essentially invariant to shuffling or
zeroing the past frames — confirming the embedding had collapsed to pure
temporal averaging. The naive 9-channel averaging baseline reached only
mAP@50 ≈ 0.892, barely above the single-frame baseline of 0.875, and for the
wrong reason (mild denoising, not motion awareness).

The fix is to **never widen the patch embed**. Instead we keep the pretrained
3-channel patch embedding completely intact and place a small learnable module
in front of it that reduces the `3 * num_frames` channels back down to 3. The
backbone always sees a 3-channel, in-distribution image; all temporal reasoning
happens in the pre-embedding module. This is `TemporalPreEmbed`.

## 2. Input convention

`TemporalPreEmbed.forward` expects the already-normalised channel-concatenation
of `num_frames` frames in temporal order:

```
x : (B, 3 * num_frames, H, W)
```

The channel order is temporal, and **the last frame group is the current /
annotated frame**. Internally the module splits the input with
`x.chunk(num_frames, dim=1)` and takes `current = frames[-1]`
(`temporal_fusion.py:142-143`). All past frames are `frames[:-1]`. Labels and
boxes are always associated with the current (last) frame; the past frames
provide context only.

The default temporal stack is 3 frames (9 channels): `num_frames = num_channels
// 3` is derived in the backbone (`dinov2.py:93-98`).

## 3. The three fusion modes

`temporal_fusion` is a `Literal["none", "preembed", "bgsub", "bgsubcoh"]`
config field (`config.py:99`). For any of the three active modes the backbone
constructs a `TemporalPreEmbed` and then resets its own `num_channels` to 3
(`dinov2.py:87-98`), so the encoder is built as a plain 3-channel DINOv2.

All three modes share two structural invariants:

- **A zero-initialised output convolution** (`conv_out`), ControlNet-style. Its
  weight and bias are zeroed at construction (`temporal_fusion.py:124-125`), so
  the learned residual branch contributes exactly 0 at initialisation while
  gradients still flow into it from step one.
- **The current frame passed through unchanged at init.** Whatever extra
  temporal term a mode adds is constructed to vanish on static input, so the
  module output equals the current frame at initialisation. This gives
  **parity-at-init** with the single-frame baseline: training starts from a
  known-good detector and learns the temporal contribution as a residual that
  grows from zero.

The learnable branch in every mode is the same small bottleneck:
`Conv2d(1×1) → GELU → depthwise Conv2d(3×3) → GELU → Conv2d(1×1, zero-init)`
(`temporal_fusion.py:105-111`), with `mid_channels = 16`.

### 3.1 `preembed` — zero-init motion residual

The learnable branch consumes the current frame concatenated with the
per-transition differences:

```
motion   = [ f_cur, f_cur − f0, f_cur − f1 ]        # 3 * num_frames channels
residual = conv_out(act2(conv_dw(act1(conv_in(motion)))))
out      = f_cur + residual
```

(`temporal_fusion.py:167-170`). Feeding differences `f_cur − f_i` rather than
raw frames makes the branch respond directly to motion. Because `conv_out` is
zero-init, `residual == 0` at init and `out == f_cur` **exactly** — exact
parity-at-init. The temporal contribution is then discovered purely by
gradient descent.

`preembed` is the default and the best-performing mode in practice:
mAP@50 ≈ 0.9074, well above both the single-frame baseline (0.875) and naive
averaging (0.892).

### 3.2 `bgsub` — explicit background-subtraction add-weight

`preembed`'s residual must *discover* motion from scratch, which is a weak
signal for small, distant, semi-transparent moving instances in CCTV. `bgsub`
instead **injects** the change signal explicitly. It estimates a background as
the mean of the per-transition differences (equivalently, the current frame
minus the mean of the past frames), forms a foreground, and add-weights it back
into the current frame with a learnable per-channel gain:

```
diffs       = [ f_cur − f_i  for each past frame i ]
foreground  = mean(diffs)                              # ≈ f_cur − mean(past)
out         = f_cur + alpha · foreground + residual
```

(`temporal_fusion.py:147-149, 162-165`). `alpha` is a per-channel
`nn.Parameter` of shape `(3,)` initialised to `gain_init = 0.5`
(`temporal_fusion.py:117-119`), so the enhancement is active from the very
first step rather than being discovered from zero.

The intuition is a **temporal high-pass / unsharp mask**. Where there is
motion, `foreground` is non-zero and the add-weight boosts local contrast,
raising the patch-embed token magnitude (hence attention) on moving regions.
Where the scene is static, `foreground ≈ 0` and the term vanishes regardless of
`alpha`, so static background stays in-distribution. The learnable branch is
additionally fed the foreground and its per-pixel magnitude
(`magnitude = foreground.abs().sum(dim=1, keepdim=True)`,
`temporal_fusion.py:162-163`) so the residual refinement is motion-aware. Note
parity-at-init is **approximate** here: it holds exactly only when the input is
static (`foreground == 0`).

Empirically the *mechanism* worked but detection did **not** improve. The
injected enhancement was far better aligned with real motion than `preembed`'s
residual (motion-alignment ≈ 0.505 vs 0.166, roughly 3×) and concentrated
energy on ground-truth boxes (≈1.38× vs `preembed`'s ≈1.00×, i.e. random). Yet
aggregate mAP@50 peaked at ≈0.8913, **below** `preembed`'s 0.9074, and it
*declined* with training (≈0.892 → 0.890 → 0.886 across epochs 1/5/10),
reproducibly across two independent runs.

A motion-stratified evaluation (validation split by inter-frame motion)
explained why. On the high-motion subset `bgsub` was roughly on par with
`preembed` (marginally lower). On the low-motion subset it **regressed**: with
little real motion to amplify, the add-weight amplified sensor noise on faint
inputs, degrading those frames. The unconditional add-weight cannot tell
genuine coherent motion from per-pixel noise.

### 3.3 `bgsubcoh` — coherence-gated add-weight

`bgsubcoh` keeps `bgsub`'s explicit injection but adds a **coherence gate** that
removes the noise-amplification failure mode. The principle: genuine motion is
*consistent across every frame transition* — if an object is moving, it shows up
in the same direction in `f_cur − f0` and `f_cur − f1`. Uncorrelated sensor
noise does not: its sign disagrees between transitions. The gate keeps only the
component of the foreground that is present, in the same direction, in **all**
transitions:

```
sign      = sign(foreground)
proj      = [ diff_i · sign  for each transition i ]   # project each diff onto fg direction
coherent  = relu( min_i(proj_i) )                      # per-pixel min across transitions, then relu
foreground = sign · coherent
```

(`temporal_fusion.py:150-161`). Projecting each per-transition diff onto the
foreground sign and taking the per-pixel **minimum** is a conservative
intersection: a pixel survives only if every transition agrees on the motion
(positive projection in all of them). If any transition disagrees, the min goes
negative and `relu` zeroes it. The surviving magnitude is the smallest
consistent motion; the direction is the foreground sign. The gated foreground
then flows through the same add-weight and residual as `bgsub`
(`temporal_fusion.py:162-165`).

Three behaviours are unit-verified:

- **Static input → exact parity** (gated foreground is 0, so `out == f_cur`).
- **Coherent motion → amplified** (survives the min, passes through the
  add-weight).
- **Incoherent input (noise) → suppressed to 0** (killed by the min/relu).

This restores static parity-at-init and prevents the low-motion regression that
sank `bgsub`. A small-object-focused A/B (`bgsubcoh` vs `preembed`, both with a
moving-small-object augmentation) on a synthetic validation set is the subject
of an ongoing sweep; that verdict is pending.

### Mode summary

| Mode | Extra term added to `f_cur` | Learnable params beyond residual | Parity at init | Noise behaviour |
| --- | --- | --- | --- | --- |
| `preembed` | none (residual only) | none | exact | n/a (no explicit injection) |
| `bgsub` | `alpha · mean(diffs)` | per-channel `alpha` (init 0.5) | exact only on static input | amplifies noise on faint input |
| `bgsubcoh` | `alpha · gated(mean(diffs))` | per-channel `alpha` (init 0.5) | exact (gate zeroes static) | suppresses incoherent noise |

## 4. A note on where the bottleneck actually was

The architecture work above is sound and `preembed` is a real improvement, but
a size census of the validation data was decisive: the small-object COCO size
bin was essentially **empty** (≈0 small instances for the primary target
category); the data is dominated by large instances. The detection bottleneck
is therefore **data, not architecture** — no fusion variant can recover targets
that are not represented at the scale of interest.

A small-object copy-paste augmentation, measured on a deterministic *synthetic*
small validation set (class-averaged AP@0.5), lifted small-object AP from ≈0.06
(no augmentation) to ≈0.80, and medium from ≈0.36 to ≈0.86, while leaving large
essentially flat (≈0.96 → 0.95). This proves the detector *learns* small
objects once they exist in training — but because the synthetic small set is
in-distribution to the augmentation, it is **not** proof of generalisation to
genuinely distant real targets. The `bgsubcoh` A/B uses a moving-small
augmentation to manufacture small *moving* targets specifically to probe the
fusion modes under that regime.

## 5. Preserving the pretrained patch embed through weight loading

The entire point of the design is that the backbone stays 3-channel, so the
pretrained 3-channel patch projection loads **verbatim**. The weight loader
enforces this.

`weights.py` contains `_expand_patch_embed_channels`
(`weights.py:108-142`), which tiles the patch-embed conv weight from 3 channels
to `target_channels` and rescales by `3 / target_channels` — this is exactly the
naive-averaging path and is the mechanism that produces the
`3 · W·mean(...)` collapse described in §1 (the rescale removes the constant
front factor, leaving plain averaging). For any preembed-family fusion this path
must **not** run.

The gate is at `weights.py:308-310`:

```python
if getattr(mc, "temporal_fusion", "none") not in ("preembed", "bgsub", "bgsubcoh"):
    _expand_patch_embed_channels(checkpoint["model"], mc.in_channels)
nn_model.load_state_dict(checkpoint["model"], strict=False)
```

When `temporal_fusion` is one of the three active modes, channel expansion is
skipped entirely: the checkpoint's `(hidden, 3, ps, ps)` projection weight
matches the model's (still 3-channel) patch embed and loads unchanged.

The new `temporal_pre.*` parameters (the `conv_in`/`conv_dw`/`conv_out` weights
and, for `bgsub`/`bgsubcoh`, `alpha`) are absent from the pretrained checkpoint.
They are absorbed by `strict=False` on `load_state_dict`
(`weights.py:310`): the loader leaves them at their constructor-set values —
the zero-init residual and the add-weight init — which is precisely the
parity-at-init state the module is designed around.

## 6. Wiring table

| File | Change | Lines |
| --- | --- | --- |
| `src/rfdetr/models/backbone/temporal_fusion.py` | `TemporalPreEmbed` module: input split + current-frame selection; the three forward paths; zero-init `conv_out`; per-channel `alpha`; coherence gate | `142-143` (split/current), `167-170` (`preembed`), `147-165` (`bgsub`/`bgsubcoh`), `124-125` (zero-init), `117-119` (`alpha`), `150-161` (coherence gate) |
| `src/rfdetr/models/backbone/dinov2.py` | Import `TemporalPreEmbed`; `temporal_fusion` arg; construct module for active modes and reset `num_channels` to 3; apply in `forward` before the encoder | `21` (import), `71` (arg), `86-100` (construct + reset), `243-245` (forward) |
| `src/rfdetr/config.py` | `temporal_fusion` literal field on `ModelConfig`; temporal config presets (`in_channels=9`, default `preembed`, resolutions/PE sizes, pretrain weights) | `99` (field), `91` (`in_channels` default), `632-661` (`RFDETRTemporalConfig`, `RFDETRTemporalBaseConfig`), `664-668` (`TemporalTrainConfig`) |
| `src/rfdetr/models/weights.py` | Skip `_expand_patch_embed_channels` for preembed-family fusion so the 3ch projection loads verbatim; new `temporal_pre.*` params absorbed by `strict=False` | `108-142` (expander), `308-310` (gate + `strict=False` load) |
| `src/rfdetr/_namespace.py` | Forward `temporal_fusion` (and `in_channels`) from `ModelConfig` into the legacy builder namespace | `38`, `53` (`_MC_NAMESPACE_FIELDS` entries) |
