# ------------------------------------------------------------------------
# RF-DETR-Temporal — detailed schematic of ONLY the part that differs from
# upstream RF-DETR in the `preembed` configuration: the TemporalPreEmbed module
# (9ch -> 3ch) that sits in front of the unchanged pretrained patch embed.
# ------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})

C_IN, C_MOD, C_CONV, C_OUT, C_SHARED = "#D6E4F0", "#FFF1DC", "#FFE0AE", "#FFCB80", "#ECECEC"
E_IN, E_NEW, E_SHARED = "#3D6FA3", "#E07B00", "#9AA0A6"

fig, ax = plt.subplots(figsize=(13.8, 4.9))
ax.set_xlim(0, 13.8); ax.set_ylim(0, 4.9); ax.axis("off")
YM, H = 1.55, 0.95          # main-row baseline y and box height
yc = YM + H / 2             # vertical centre of the row


def box(x, w, text, fc, ec, lw=1.5, fs=9):
    ax.add_patch(FancyBboxPatch((x, YM), w, H, boxstyle="round,pad=0.015,rounding_size=0.06",
                 linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, yc, text, ha="center", va="center", fontsize=fs)
    return (x, w)


def arr(x1, x2, y=None):
    y = y or yc
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                 mutation_scale=12, lw=1.3, color="#444"))


# --- main left->right flow (the new module) ---
b1 = box(0.25, 1.85, "Stacked frames\n9×H×W\n$[f_{t-2},\\,f_{t-1},\\,f_t]$", C_IN, E_IN)
b2 = box(2.45, 2.55, "Motion features (concat)\n$[\\,f_t,\\ f_t{-}f_{t-1},\\ f_t{-}f_{t-2}\\,]$\n9 ch",
         C_MOD, E_NEW)
b3 = box(5.30, 1.55, "Conv 1×1\n9→16\n+ GELU", C_CONV, E_NEW)
b4 = box(7.05, 1.75, "DWConv 3×3\n16 (depthwise)\n+ GELU", C_CONV, E_NEW)
b5 = box(9.00, 1.55, "Conv 1×1\n16→3\n(zero-init)", C_CONV, E_NEW)
# residual add node
add_cx = 11.05
ax.add_patch(Circle((add_cx, yc), 0.26, facecolor=C_OUT, edgecolor=E_NEW, lw=1.6))
ax.text(add_cx, yc, "+", ha="center", va="center", fontsize=16, fontweight="bold")
b7 = box(11.55, 1.45, "$x = f_t + R$\n3 ch", C_OUT, E_NEW)

for (x, w), nx in [(b1, 2.45), (b2, 5.30), (b3, 7.05), (b4, 9.00)]:
    arr(x + w, nx)
arr(b5[0] + b5[1], add_cx - 0.26)             # conv_out -> (+)
arr(add_cx + 0.26, b7[0])                      # (+) -> x

# --- residual skip: current frame f_t routed over the conv stack into (+) ---
ax.add_patch(FancyArrowPatch((b1[0] + b1[1] / 2, YM + H), (add_cx, yc + 0.26),
             connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
             mutation_scale=12, lw=1.5, color=E_NEW, linestyle="-"))
ax.text(6.0, 3.62, "skip:  current frame  $f_t$  (3 ch)", ha="center",
        fontsize=9, color=E_NEW, style="italic")

# --- boundary: everything to the right is unchanged RF-DETR ---
bx = b7[0] + b7[1] + 0.18
ax.plot([bx, bx], [0.55, 3.15], color=E_SHARED, lw=1.4, linestyle=(0, (5, 4)))
pe = box(bx + 0.2, 1.65, "Patch Embed\nConv 3→C\n(unchanged)", C_SHARED, E_SHARED)
arr(b7[0] + b7[1], bx + 0.2)
ax.text(pe[0] + pe[1] / 2, YM - 0.30, "→ DINOv2 ViT → LW-DETR\n(original RF-DETR, pretrained)",
        ha="center", va="top", fontsize=7.8, color="#5F6368", style="italic")

# --- captions: the zero-init parity property ---
ax.text(b5[0] + b5[1] / 2, YM - 0.22, "$R=0$ at init", ha="center", va="top",
        fontsize=8.5, color=E_NEW)
ax.text(b7[0] + b7[1] / 2, YM - 0.22, "$\\Rightarrow x=f_t$ at init\n(single-frame parity)",
        ha="center", va="top", fontsize=8.0, color="#7A4A00")

# --- top brackets: NEW module vs unchanged ---
def bracket(x0, x1, y, color, txt):
    ax.plot([x0, x0, x1, x1], [y - 0.12, y, y, y - 0.12], color=color, lw=1.4)
    ax.text((x0 + x1) / 2, y + 0.06, txt, ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color=color)

bracket(0.25, b7[0] + b7[1], 4.18, E_NEW, "TemporalPreEmbed  (preembed)  —  NEW")
bracket(bx + 0.2, pe[0] + pe[1], 4.18, E_SHARED, "unchanged RF-DETR")

plt.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"docs/figures/preembed_detail.{ext}", dpi=300, bbox_inches="tight")
print("saved docs/figures/preembed_detail.{pdf,svg,png}")
