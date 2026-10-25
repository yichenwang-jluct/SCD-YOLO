#!/usr/bin/env python3
"""
What does the channel gate of Cotton-StarNet learn?  (gate diagnostic of Section 3.3.1,
Supplementary Table S7b and Figure S1.)

For every CottonStarBlock of a trained model:
  (a) Spearman correlation between the mean gate weight g_c of each channel and the mean response
      GAP(x1)_c of the gated branch (a negative correlation means the gate damps high-magnitude
      channels);
  (b) coefficient of variation (CV) of the channel magnitudes before and after gating (a lower CV
      after gating means the magnitude differences between channels are narrowed).
All blocks of all given checkpoints are reported; nothing is selected.  The result can go either way
and is reported as it comes out.

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python plot_gate_bias.py --runs fac_S1D1A1I1 --seeds 0 1 2 --data <cotton_disease.yaml> --split val --out gate
    python plot_gate_bias.py --weights a.pt b.pt --images <folder with images> --out gate

Writes into --out:
  gate_summary.csv / gate_summary.md   one row per checkpoint and block: C, Spearman rho, p, CV before/after
  gate_stats_<k>.csv                   per channel: mean GAP(x1) and mean gate weight (checkpoint k)
  fig_gate_bias.png / .pdf             figure for the first checkpoint
Images are read with np.fromfile + cv2.imdecode, so non-ASCII paths work on Windows.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_images(folder, n, size=640):
    """simple resize to size x size (no letterbox, no augmentation); only used for channel statistics"""
    import cv2
    import torch
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)[:n]
    if not files:
        raise SystemExit(f"no images in {folder}")
    batch = []
    for f in files:
        im = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            print(f"!! cannot read {f}, skipped")
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        batch.append(im.transpose(2, 0, 1))
    x = torch.from_numpy(np.stack(batch)).float() / 255.0
    print(f"{len(batch)} images read from {folder}")
    return x


def collect(model, imgs, batch=8, device="cuda"):
    """hooks on every CottonStarBlock: GAP of the gated branch x1 (output of f1) and the gate weights"""
    import torch
    blocks = [m for m in model.modules() if type(m).__name__ == "CottonStarBlock"]
    if not blocks:
        raise SystemExit("no CottonStarBlock found - is this a Cotton-StarNet checkpoint?")
    print(f"   {len(blocks)} CottonStarBlocks")
    store = {i: {"x1": [], "g": []} for i in range(len(blocks))}
    handles = []
    for i, blk in enumerate(blocks):
        handles.append(blk.f1.register_forward_hook(
            lambda m, inp, out, i=i: store[i]["x1"].append(out.mean(dim=(2, 3)).detach().float().cpu())))
        handles.append(blk.gate.register_forward_hook(
            lambda m, inp, out, i=i: store[i]["g"].append(out.flatten(1).detach().float().cpu())))
    model.eval().float().to(device)
    with torch.no_grad():
        for k in range(0, len(imgs), batch):
            model(imgs[k:k + batch].to(device))
    for h in handles:
        h.remove()
    return {i: (torch.cat(store[i]["x1"]).numpy(), torch.cat(store[i]["g"]).numpy()) for i in store}


def block_stats(x1, g):
    from scipy.stats import spearmanr
    mx1 = x1.mean(axis=0)                     # (C,) mean channel response
    mg = g.mean(axis=0)                       # (C,) mean gate weight
    rho, p = spearmanr(mx1, mg)
    amp_pre = np.maximum(x1, 0).clip(max=6).mean(axis=0)     # ReLU6 magnitude, as multiplied in the block
    amp_post = amp_pre * mg
    cv_pre = amp_pre.std() / (amp_pre.mean() + 1e-12)
    cv_post = amp_post.std() / (amp_post.mean() + 1e-12)
    return mx1, mg, float(rho), float(p), float(cv_pre), float(cv_post)


def figure(panels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300})
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.0), squeeze=False)
    for k, (i, mx1, mg, rho, p, cv_pre, cv_post) in enumerate(panels):
        ax = axes[0][k]
        ax.scatter(mx1, mg, s=7, alpha=0.55, edgecolors="none", color="#2b6cb0")
        ax.set_title(f"Block {i}  (C={len(mx1)})\n" + rf"Spearman $\rho$={rho:+.2f}, p={p:.1e}", fontsize=8)
        ax.set_xlabel(r"channel response  GAP($x_1$)")
        if k == 0:
            ax.set_ylabel(r"gate weight  $g_c$")
        ax.axhline(0.5, ls=":", lw=0.7, color="grey")
        ax.grid(alpha=0.25, lw=0.4)
        ax = axes[1][k]
        ax.bar(["pre-gate", "post-gate"], [cv_pre, cv_post], color=["#a0aec0", "#2b6cb0"], width=0.55)
        ax.set_title(f"CV {cv_pre:.2f} -> {cv_post:.2f}", fontsize=8)
        if k == 0:
            ax.set_ylabel("channel-magnitude CV")
        ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "fig_gate_bias.png", bbox_inches="tight")
    fig.savefig(out / "fig_gate_bias.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[], help="run names in --project, e.g. fac_S1D1A1I1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--images", default=None, help="folder with images (or use --data/--split)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200, help="number of images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gate_bias")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    cks = [Path(w) for w in a.weights]
    for run in a.runs:
        for s in a.seeds:
            c = Path(a.project) / f"{run}_seed{s}" / "weights" / "best.pt"
            if c.exists():
                cks.append(c)
            else:
                print(f"   (not on this computer, skipped: {c})")
    if not cks:
        raise SystemExit("no checkpoints")
    if a.images:
        img_dir = Path(a.images)
    elif a.data:
        import yaml
        d = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
        root = Path(d.get("path") or Path(a.data).parent)
        img_dir = Path(d[a.split]) if Path(d[a.split]).is_absolute() else root / d[a.split]
    else:
        raise SystemExit("give --images or --data")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    imgs = load_images(img_dir, a.n)
    summary = []
    first_panels = None
    for k, ck in enumerate(cks):
        print(f"\n=== {ck}")
        data = collect(YOLO(str(ck)).model, imgs, device=a.device)
        panels, rows = [], []
        for i, (x1, g) in data.items():
            mx1, mg, rho, p, cv_pre, cv_post = block_stats(x1, g)
            panels.append((i, mx1, mg, rho, p, cv_pre, cv_post))
            rows += [dict(block=i, channel=c, gap_x1=float(mx1[c]), gate=float(mg[c])) for c in range(len(mx1))]
            summary.append(dict(checkpoint=str(ck), block=i, channels=len(mx1), spearman_rho=round(rho, 4),
                                p_value=f"{p:.2e}", cv_before=round(cv_pre, 4), cv_after=round(cv_post, 4)))
            print(f"   block {i}: C={len(mx1):4d}  Spearman rho={rho:+.3f}  p={p:.2e}  CV {cv_pre:.3f} -> {cv_post:.3f}")
        with open(out / f"gate_stats_{k}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if first_panels is None:
            first_panels = panels
    with open(out / "gate_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    md = ["| Checkpoint | Block | C | Spearman rho | p | CV before | CV after |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['checkpoint']} | {r['block']} | {r['channels']} | {r['spearman_rho']:+.3f} | {r['p_value']} | "
           f"{r['cv_before']:.3f} | {r['cv_after']:.3f} |" for r in summary]
    (out / "gate_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        figure(first_panels, out)
        print(f"\nfigure: {out / 'fig_gate_bias.png'}")
    except Exception as e:
        print(f"!! figure not written: {type(e).__name__}: {e}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
