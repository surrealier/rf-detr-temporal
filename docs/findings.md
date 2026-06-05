# Empirical findings

> Part of RF-DETR-Temporal — see the [README](../README.md).

This document reports the empirical results of the temporal extension **honestly, negative
results included**. All numbers are aggregate or **class-averaged** (mAP / AP@0.5); no
per-instance-category breakdowns are given. Each result links to the diagnostic script under
[`diagnostics/`](../diagnostics) that produced it, so every claim is reproducible.

The headline arc: the first temporal variant *looked* like it fused frames but was provably
**averaging them** (a bug); fixing the fusion gave the best aggregate score but the learned signal
did not land on objects; a motion-explicit variant raised the right diagnostic ~3x yet **lost mAP**;
and a size census then showed the real ceiling is **data**, not architecture.

---

## 1. The patch-embed averaging bug

The first temporal design widened the DINOv2 patch embedding to a `Conv2d(9, 768, 14, 14)` to ingest
three stacked frames, initialising the three per-frame channel-groups (`ch 0:3`, `3:6`, `6:9`) to
identical scaled copies of the pretrained RGB weight. At init the convolution therefore computes
`W · mean(f0, f1, f2)` — a pure **temporal average / motion blur** — and nothing in training forced
it to diverge.

Two independent diagnostics confirm the model never broke this symmetry:

| Probe | Script | Result | Reading |
|---|---|---|---|
| Per-frame weight-group cosine | [`diag_weight_divergence.py`](../diagnostics/diag_weight_divergence.py) | min pairwise cosine ≈ **0.998** | groups stayed ~identical ⇒ still averaging |
| Output vs. past-frame ablation | [`diag_dup_vs_real.py`](../diagnostics/diag_dup_vs_real.py) | output ~invariant to **zeroing / shuffling** past frames | past frames unused; no temporal reasoning, order-invariant |

A cosine of ≈0.998 against the init value of 1.000 means training moved the frame-groups
essentially not at all. Combined with the ablation — duplicating, zeroing, or permuting the past
frames barely changes detections — the model was behaving as a single-frame detector fed a blurred
average. The change/motion signal was destroyed *before attention ever saw it*.

**Fix (preembed):** keep the pretrained **3-channel** patch embed verbatim and place a small
`TemporalPreEmbed` module *in front of it* that outputs `current + residual`, with the residual
zero-initialised. [`verify_parity.py`](../diagnostics/verify_parity.py) unit-confirms the
guarantees: the patch embed stays `in_channels=3`, the init residual is `<1e-6`, and at init the
output depends on the current frame only (max|Δlogits|, max|Δboxes| `<1e-5` under arbitrary past
frames). Training therefore *starts at exact single-frame parity* and learns temporal cues as a
residual from zero.

---

## 2. preembed — best aggregate, but the residual did not concentrate on objects

The preembed variant is the **best-scoring** model on the aggregate validation set.

| Model | val mAP@50 |
|---|---|
| single-frame baseline | 0.875 |
| naive 9-channel averaging (the buggy design) | 0.892 |
| **preembed (residual)** | **0.9074** |

So the fix recovers and exceeds the single-frame baseline. But "best aggregate" is not the same as
"learned what we wanted". [`residual_activation.py`](../diagnostics/residual_activation.py) probes
*where* the learned temporal residual activates:

| Quantity | preembed | Reference |
|---|---|---|
| residual motion-alignment (cosine vs inter-frame diff) | 0.166 | — |
| GT-box energy concentration (energy-in-box / box-area) | **1.00x** | 1.00x = random |

A concentration of **1.00x is exactly the random baseline**: the residual's energy is spread over
the frame in proportion to area, *not* focused on the objects, and its alignment with actual motion
is weak (0.166). The aggregate gain is real, but it does **not** come from a motion pathway that
attends to the moving targets — it is a generic, mildly helpful residual. This motivated an explicit
motion mechanism (Section 3).

---

## 3. bgsub — a genuine NEGATIVE result

`bgsub` makes the temporal contribution explicit: `enhancement = alpha · (current − mean(past)) +
residual`, an always-on background-subtraction add-weight.
[`bgsub_enhancement.py`](../diagnostics/bgsub_enhancement.py) shows the **mechanism works as
designed** — it injects energy onto moving regions far better than preembed:

| Quantity | preembed | bgsub | Change |
|---|---|---|---|
| motion-alignment of injected enhancement | 0.166 | **0.505** | ~3x |
| GT-box energy concentration | 1.00x (random) | **1.38x** | above random |

And yet detection got **worse**, not better, and degraded *with* training — reproduced across two
independent runs:

| Model | best val mAP@50 | ep1 → ep5 → ep10 |
|---|---|---|
| preembed | **0.9074** | — |
| bgsub | 0.8913 | 0.892 → 0.890 → 0.886 |

The injected signal landed on motion (3x better alignment, above-random box concentration) but the
aggregate mAP fell **below preembed** and **declined monotonically** as training proceeded — a clear
negative result, not noise.

**Why** — motion-stratified evaluation.
[`motion_stratified_eval.py`](../diagnostics/motion_stratified_eval.py) splits the validation set by
inter-frame motion into low/high strata and reports class-averaged mAP@0.5 per stratum:

| Stratum | bgsub vs preembed |
|---|---|
| high-motion subset | ≈ parity (bgsub marginally lower) |
| low-motion subset | **regressed** |

On faint, near-static inputs the `current − mean(past)` term has essentially no foreground to
recover, so the add-weight **amplified sensor noise** instead, and that hurt exactly the inputs where
motion was not the discriminative cue. Because the aggregate set is dominated by low/near-static
content, this noise amplification dragged the headline number below preembed. The mechanism was
right; the always-on, ungated formulation was wrong.

---

## 4. The decisive bottleneck is DATA — the size census

Before iterating further on architecture, we asked whether small/distant targets — the regime the
whole project is aimed at — even exist in the data.
[`size_stratified_eval.py`](../diagnostics/size_stratified_eval.py) bins ground-truth instances by
COCO area (small `<32²`, medium `32²–96²`, large `≥96²`) at the model's input resolution and counts
them.

| COCO size bin | GT instances (validation, primary target category) |
|---|---|
| small (`<32²` px) | **≈ 0 (essentially empty)** |
| medium / large | dominate the distribution |

The small bin is **effectively empty**: there are almost no small instances of the primary target
category to detect, and the data is overwhelmingly large instances. This reframes everything above —
no temporal or attention mechanism can improve a regime that is unobserved at train and eval time.
**The detection bottleneck is the data distribution, not the architecture.** The small-object losses
of bgsub were being measured against a near-absent population.

---

## 5. Small-object augmentation — a POSITIVE result (with a caveat)

Since real small targets are absent, we (a) manufacture small instances at train time via
small-object copy-paste augmentation and (b) measure on a **deterministic synthetic small** val set
(identical pastes per index for every checkpoint).
[`test_smallobj_paste.py`](../diagnostics/test_smallobj_paste.py) visually verifies paste/box
alignment; [`synthetic_small_eval.py`](../diagnostics/synthetic_small_eval.py) scores class-averaged
AP@0.5 per size bin.

| COCO size bin | no-aug baseline | with small-object aug |
|---|---|---|
| small | 0.06 | **0.80** |
| medium | 0.36 | 0.86 |
| large | 0.96 | 0.95 |

Small-object AP@0.5 jumps from **≈0.06 to ≈0.80** (medium ≈0.36→≈0.86) with no meaningful cost to
large objects (≈0.96→≈0.95). This proves the detector **learns small targets once they exist** in
training.

**Caveat (stated plainly):** the synthetic-small val is *in-distribution* to the augmentation that
produced it — same paste generator, same appearance statistics. This result demonstrates the
detector's *capacity* to learn small objects given supervision; it is **not** evidence of
generalisation to *real* distant targets, whose appearance and noise characteristics differ from the
pasted ones. Closing that gap requires real small-target data.

---

## 6. bgsubcoh + moving-small augmentation — running, verdict PENDING

`bgsubcoh` is a **coherence-gated** add-weight designed to fix bgsub's noise amplification (Section
3): the foreground term is gated by directional coherence across the inter-frame transitions, so
incoherent (noise-like) change is suppressed while coherent (genuine-motion) change is amplified.
Unit verification confirms the gate behaves as intended:

| Input | bgsubcoh enhancement |
|---|---|
| static input | exact parity (**0.0**) |
| coherent motion | amplified |
| incoherent / noise | suppressed to **0.0** |

To give it a population to act on, a **moving-small** augmentation manufactures small *moving*
targets (the synthetic-small generator with per-frame displacement). An A/B comparison —
**bgsubcoh vs preembed, both with the moving-small augmentation**, evaluated on a synthetic
moving-small val via [`synthetic_small_eval.py`](../diagnostics/synthetic_small_eval.py) with
`--motion > 0` — is the next experiment to run.

**Verdict: PENDING.** This is the open question — whether a coherence-gated motion mechanism, given
small *moving* targets to fuse, can finally convert the favourable motion-attention diagnostics of
Section 3 into an actual mAP gain over preembed, without the low-motion regression. Results will be
appended here once the sweep completes.

---

## Summary

| Stage | Outcome |
|---|---|
| 9-channel averaging | **Bug** — provably averaged frames (cosine ≈0.998; ablation-invariant) |
| preembed | **Best aggregate** (mAP@50 0.9074) but residual did not concentrate on objects (1.00x = random) |
| bgsub | **Negative** — motion-attention ↑3x yet mAP↓ (0.8913 < 0.9074), regressed on low-motion inputs |
| size census | **Reframe** — small bin ≈ empty; the bottleneck is data, not architecture |
| small-object aug | **Positive** (synthetic small AP 0.06→0.80), but in-distribution caveat |
| bgsubcoh + moving-small | **Pending** — gate unit-verified; A/B sweep in progress |
