# ------------------------------------------------------------------------
# RF-DETR-Temporal — paper figure: 3-way pipeline comparison.
#   (a) RF-DETR baseline (single frame, 3ch)
#   (b) preembed       (3 frames -> learned-residual fusion -> 3ch)
#   (c) bgsub          (3 frames -> background-subtraction add-weight -> 3ch)
# Downstream (patch embed -> DINOv2 -> decoder) is identical in all three.
# ------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})

C_IN, C_SHARED, C_NEW, C_DET = "#D6E4F0", "#ECECEC", "#FFE0AE", "#D9EAD3"
E_IN, E_SHARED, E_NEW, E_DET = "#3D6FA3", "#9AA0A6", "#E07B00", "#6AA84F"

fig, ax = plt.subplots(figsize=(12, 7.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 8.4); ax.axis("off")
H = 0.82


def box(x, y, w, h, text, fc, ec, lw=1.4, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(b1, b2):
    ax.add_patch(FancyArrowPatch(
        (b1[0] + b1[2], b1[1] + b1[3] / 2), (b2[0], b2[1] + b2[3] / 2),
        arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#444444"))


def row(stages, y, x0=0.45, x1=11.7, gap=0.42):
    n = len(stages)
    bw = (x1 - x0 - gap * (n - 1)) / n
    bs = []
    for i, (txt, fc, ec, bold) in enumerate(stages):
        bs.append(box(x0 + i * (bw + gap), y, bw, H, txt, fc, ec,
                      lw=2.0 if bold else 1.4, bold=bold))
    for a, b in zip(bs, bs[1:]):
        arrow(a, b)
    return bs


SHARED = [
    ("Patch Embed\nConv 3→C", C_SHARED, E_SHARED, False),
    ("DINOv2 ViT\nbackbone + FPN", C_SHARED, E_SHARED, False),
    ("LW-DETR\ndecoder", C_SHARED, E_SHARED, False),
    ("Detections", C_DET, E_DET, False),
]
FRAMES = ("3 frames\n(t−2,t−1,t)\n9×H×W", C_IN, E_IN, False)


def label(y, t):
    ax.text(0.45, y, t, fontsize=11, fontweight="bold", ha="left")


def caption(bx, y, formula, note):
    cx = bx[0] + bx[2] / 2
    ax.text(cx, y, formula, ha="center", va="top", fontsize=9.5, color=E_NEW)
    ax.text(cx, y - 0.34, note, ha="center", va="top", fontsize=7.6, color="#7A4A00")


# (a) baseline
ya = 6.95
label(ya + H + 0.12, "(a)  RF-DETR baseline  (single frame)")
row([("Input image\n3×H×W", C_IN, E_IN, False)] + SHARED, ya)

# (b) preembed
yb = 4.15
label(yb + H + 0.12, "(b)  preembed  (learned-residual temporal fusion)")
bb = row([FRAMES, ("Temporal\nPre-Embed\n(residual)", C_NEW, E_NEW, True)] + SHARED, yb)
caption(bb[1], yb - 0.12, r"$\mathbf{x} = f_t + R(\Delta f)$",
        "zero-init learned residual on frame\ndifferences Δf  (identity at init)")

# (c) bgsub
yc = 1.35
label(yc + H + 0.12, "(c)  bgsub  (background-subtraction add-weight)")
bc = row([FRAMES, ("Temporal\nPre-Embed\n(bg-subtract)", C_NEW, E_NEW, True)] + SHARED, yc)
caption(bc[1], yc - 0.12,
        r"$\mathbf{x} = f_t + \alpha\,(f_t - \mathrm{mean}(f_{t-1},f_{t-2})) + R$",
        "explicit background subtraction →\nemphasizes moving regions")

# legend (top band, clear of diagram)
ly = 8.12
for lx, (fc, ec, lab) in [
        (5.7, (C_NEW, E_NEW, "new (this work)")),
        (8.7, (C_SHARED, E_SHARED, "unchanged / pretrained"))]:
    ax.add_patch(FancyBboxPatch((lx, ly), 0.34, 0.22, boxstyle="round,pad=0.01",
                 facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(lx + 0.46, ly + 0.11, lab, fontsize=9, va="center")

plt.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"docs/figures/pipeline_3way.{ext}", dpi=300, bbox_inches="tight")
print("saved docs/figures/pipeline_3way.{pdf,svg,png}")
