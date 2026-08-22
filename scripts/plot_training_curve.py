#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练曲线图：三种 u_c 方案（统一上界 / 手工设定 / 训练自适应）

用途
----
支撑 4.8 节的核心论证：grey_mildew 与 healthy 的自适应终值与手工设定完全相同，
逐类精度却相差 +8.6 / -5.3 个百分点 —— 差异来自演化路径而非终点取值。
表格画不出"路径"，只有训练曲线能。

只画指标不画损失
----------------
三组用的是不同的损失函数（u_c 直接改变 Focaler 聚焦区间），box_loss 数值
量级本就不同，横向比较没有意义。mAP 不受训练目标影响，才是可比的尺子。

用法
----
1. 改下面 RUNS 里的三个路径（Ultralytics 每次训练都会写 results.csv）
2. python plot_training_curve.py
3. 输出 fig_training_curve.pdf / .png（600 dpi）

依赖：matplotlib、numpy、pandas
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ============================================================================
# 配置：把三个 results.csv 的路径填进来（顺序即图例顺序）
# ============================================================================

RUNS = [
    ("Uniform upper bound (u = 0.95)",
     r"runs\scd_yolo_uniform_uc\results.csv"),
    ("Manually set $u_c$",
     r"runs\scd_yolo_manual_uc\results.csv"),
    ("Training-adaptive $u_c$ (Ours)",
     r"runs\scd_yolo_adaptive\results.csv"),
]

COLORS = ["#8a8a8a", "#4a6f8a", "#c0392b"]   # 灰 / 蓝 / 红（本文），黑白打印可区分
STYLES = ["--", "-.", "-"]
WARMUP = 30          # 论文的热身轮数，图上标一条竖线
ZOOM_FROM = 150      # 放大框起始 epoch

# ============================================================================


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]        # Ultralytics 列名带空格
    need = {"epoch": None, "map50": None, "map": None}
    for c in df.columns:
        lc = c.lower()
        if lc == "epoch":
            need["epoch"] = c
        elif "map50-95" in lc:
            need["map"] = c
        elif "map50" in lc and "95" not in lc:
            need["map50"] = c
    miss = [k for k, v in need.items() if v is None]
    if miss:
        raise KeyError(f"{path} 缺少列 {miss}；实际列名：{list(df.columns)}")
    return df[need["epoch"]].to_numpy(), df[need["map50"]].to_numpy(), df[need["map"]].to_numpy()


def panel(ax, idx, ylabel, title, zoom_box):
    ymins, ymaxs = [], []
    for (label, path), c, ls in zip(RUNS, COLORS, STYLES):
        try:
            ep, m50, m95 = load(path)
        except Exception as e:
            print(f"[跳过] {label}: {e}")
            continue
        y = m50 if idx == 0 else m95
        ax.plot(ep, y, color=c, ls=ls, lw=1.5 if "Ours" in label else 1.1,
                label=label, zorder=5 if "Ours" in label else 3)
        tail = y[ep >= ZOOM_FROM]
        if len(tail):
            ymins.append(tail.min()); ymaxs.append(tail.max())

    ax.axvline(WARMUP, color="#b0b0b0", lw=0.7, ls=":", zorder=1)
    ax.annotate(f"warm-up ends ({WARMUP})", xy=(WARMUP, ax.get_ylim()[0]),
                xytext=(WARMUP + 6, ax.get_ylim()[0] + 0.03),
                fontsize=6.8, color="#7a7a7a")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(True, color="#e4e4e4", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 放大框：三条曲线在后期会收敛，不放大看不出最终差距
    if ymins and zoom_box:
        lo, hi = min(ymins), max(ymaxs)
        pad = max((hi - lo) * 0.25, 0.004)
        axi = inset_axes(ax, width="46%", height="38%", loc="lower right",
                         borderpad=1.4)
        for (label, path), c, ls in zip(RUNS, COLORS, STYLES):
            try:
                ep, m50, m95 = load(path)
            except Exception:
                continue
            y = m50 if idx == 0 else m95
            k = ep >= ZOOM_FROM
            axi.plot(ep[k], y[k], color=c, ls=ls,
                     lw=1.4 if "Ours" in label else 1.0)
        axi.set_xlim(ZOOM_FROM, None)
        axi.set_ylim(lo - pad, hi + pad)
        axi.tick_params(labelsize=6, length=2, pad=1.5)
        axi.grid(True, color="#ececec", linewidth=0.4)
        for s in ("top", "right"):
            axi.spines[s].set_visible(False)
        mark_inset(ax, axi, loc1=2, loc2=4, fc="none", ec="#b8b8b8", lw=0.5, ls="--")


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
panel(axes[0], 0, "mAP@0.5", "(a) mAP@0.5", zoom_box=True)
panel(axes[1], 1, "mAP@0.5:0.95", "(b) mAP@0.5:0.95", zoom_box=True)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=3,
           frameon=False, fontsize=8, columnspacing=2.2, handlelength=2.6)
fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.26, wspace=0.24)
fig.savefig("fig_training_curve.pdf")
fig.savefig("fig_training_curve.png", dpi=600)
print("已输出 fig_training_curve.pdf 与 fig_training_curve.png（600 dpi）")
