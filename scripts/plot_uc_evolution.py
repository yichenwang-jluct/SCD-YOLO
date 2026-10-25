#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 10: evolution of the class-adaptive upper bounds u_c during training.
Reads u_c_evolution.csv, written during training by scripts/train_adaptive.py
(or by scripts/run_experiment.py --loss adaptive).
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ============================================================================
# Data source: CSV with columns epoch, u_c_0 ... u_c_5 (or class names)
# ============================================================================
CSV = os.environ.get("UC_CSV", "runs/scd_yolo_adaptive/u_c_evolution.csv")
WARMUP, U_MIN, U_MAX, U_INIT = 30, 0.70, 0.95, 0.825

CLASSES = ["blight", "curl", "grey_mildew", "healthy", "leaf_spot", "wilt"]
# grey_mildew and healthy carry the argument -> solid, highlighted
STYLE = {
    "grey_mildew": dict(color="#c0392b", lw=2.1, ls="-",  z=6),
    "healthy":     dict(color="#2c5f8a", lw=2.1, ls="-",  z=6),
    "blight":      dict(color="#9a9a9a", lw=1.0, ls="--", z=3),
    "curl":        dict(color="#7a7a7a", lw=1.0, ls="-.", z=3),
    "leaf_spot":   dict(color="#b0b0b0", lw=1.0, ls=":",  z=3),
    "wilt":        dict(color="#8a8a8a", lw=1.0, ls=(0, (5, 1, 1, 1)), z=3),
}

# ---------------------------------------------------------------- load data
if not os.path.isfile(CSV):
    raise FileNotFoundError(
        f"u_c_evolution.csv not found: {CSV}\n"
        "This file is written by attach_adaptive_callback() during training; "
        "point CSV at the run directory produced by scripts/train_adaptive.py."
    )
df = pd.read_csv(CSV)
df.columns = [c.strip() for c in df.columns]
# the callback writes u_c_0 ... u_c_5 in class-index order
if "u_c_0" in df.columns:
    df = df.rename(columns={f"u_c_{i}": c for i, c in enumerate(CLASSES)})

ep = df["epoch"].to_numpy()

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(7.0, 4.0))

ax.axhline(U_MAX, color="#c4c4c4", lw=0.7, ls=(0, (1, 2)), zorder=1)
ax.axhline(U_MIN, color="#c4c4c4", lw=0.7, ls=(0, (1, 2)), zorder=1)
ax.text(37, U_MAX + 0.006, r"$u_{\max}=0.95$", fontsize=7, color="#9a9a9a", va="bottom")
ax.text(37, U_MIN - 0.006, r"$u_{\min}=0.70$", fontsize=7, color="#9a9a9a", va="top")

ax.axvspan(0, WARMUP, color="#f2f2f2", zorder=0, lw=0)
ax.axvline(WARMUP, color="#c4c4c4", lw=0.7, ls=":", zorder=1)
ax.text(WARMUP / 2, (U_MIN + U_MAX) / 2, f"warm-up  ($u_c$ fixed at {U_INIT})",
        fontsize=7.2, color="#a0a0a0", ha="center", va="center",
        rotation=90, style="italic")

for c in CLASSES:
    s = STYLE[c]
    ax.plot(ep, df[c], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=s["z"],
            solid_capstyle="round")
    y = df[c].iloc[-1]
    ax.text(ep[-1] + 4, y, c, fontsize=7.6, va="center", color=s["color"],
            fontweight="bold" if c in ("grey_mildew", "healthy") else "normal")

ax.set_xlabel("Epoch", fontsize=9.5)
ax.set_ylabel(r"Class-adaptive upper bound  $u_c$", fontsize=9.5)
ax.set_xlim(0, 248)
ax.set_ylim(U_MIN - 0.030, U_MAX + 0.040)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
ax.grid(True, axis="y", color="#ececec", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_bounds(U_MIN - 0.03, U_MAX + 0.01)
ax.spines["bottom"].set_bounds(0, 200)
ax.tick_params(length=3, width=0.7)

fig.tight_layout()
fig.savefig("fig_uc_evolution.pdf")
fig.savefig("fig_uc_evolution.png", dpi=600)
print("written:", os.path.abspath("fig_uc_evolution.pdf"))
