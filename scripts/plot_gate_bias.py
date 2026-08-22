"""
测量 Cotton-StarNet 的门控是否在纠正星操作的幅度倾向。

    python plot_gate_bias.py --weights runs/.../best.pt --images datasets/cotton/images/val

产出 fig_gate_bias.png / .pdf 与 gate_stats.csv。

图分两栏：
  (a) 每个通道的门控权重 g_c 对该通道在 x1 上的平均响应 GAP(x1)_c —— 直接画出网络
      学到的门控函数。若为负相关，说明门控确实在压制高幅度通道。
  (b) 门控前后通道幅度离散度（变异系数 CV）的对比 —— 若门控后 CV 下降，
      说明通道间的幅度差距被收窄。

注意：结果可能是任一方向。若 (a) 的 Spearman ρ 不显著为负，说明网络并未学出
纠正行为，3.1 节第二、三段就必须保持当前的弱表述，不能改回强断言。
如实报告，不要挑层、挑图。
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_images(folder, n, size=640):
    """按 letterbox 无关的简单缩放读入，仅用于统计通道响应，不做增强。"""
    import cv2
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)[:n]
    assert files, f'{folder} 下没有图像'
    batch = []
    for f in files:
        im = cv2.imread(str(f))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        batch.append(im.transpose(2, 0, 1))
    x = torch.from_numpy(np.stack(batch)).float() / 255.0
    print(f'读入 {len(files)} 张图像')
    return x


def collect(model, imgs, batch=8, device='cuda'):
    """挂钩子取每个 CottonStarBlock 的 x1 与门控权重。"""
    blocks = [m for m in model.modules() if type(m).__name__ == 'CottonStarBlock']
    assert blocks, '未找到 CottonStarBlock，请确认权重是 Cotton-StarNet 版本'
    print(f'找到 {len(blocks)} 个 CottonStarBlock')

    store = {i: {'x1': [], 'g': []} for i in range(len(blocks))}
    handles = []
    for i, blk in enumerate(blocks):
        handles.append(blk.f1.register_forward_hook(
            lambda m, inp, out, i=i: store[i]['x1'].append(
                out.mean(dim=(2, 3)).detach().float().cpu())))
        handles.append(blk.gate.register_forward_hook(
            lambda m, inp, out, i=i: store[i]['g'].append(
                out.flatten(1).detach().float().cpu())))

    model.eval().to(device)
    with torch.no_grad():
        for k in range(0, len(imgs), batch):
            model(imgs[k:k + batch].to(device))
    for h in handles:
        h.remove()

    out = {}
    for i in store:
        x1 = torch.cat(store[i]['x1']).numpy()      # (N, C) 每图每通道的 GAP(x1)
        g = torch.cat(store[i]['g']).numpy()        # (N, C) 每图每通道的门控权重
        out[i] = (x1, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--n', type=int, default=200, help='采样图像数')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights).model
    imgs = load_images(args.images, args.n)
    data = collect(model, imgs, device=args.device)

    # 逐 block 汇总为逐通道均值
    rows, panels = [], []
    for i, (x1, g) in data.items():
        mx1 = x1.mean(axis=0)                        # (C,) 通道平均响应
        mg = g.mean(axis=0)                          # (C,) 通道平均门控
        rho, p = spearmanr(mx1, mg)
        # 门控前后的通道幅度离散度（用 ReLU6 后的幅度，与实际参与相乘的量一致）
        amp_pre = np.maximum(x1, 0).clip(max=6).mean(axis=0)
        amp_post = amp_pre * mg
        cv_pre = amp_pre.std() / (amp_pre.mean() + 1e-12)
        cv_post = amp_post.std() / (amp_post.mean() + 1e-12)
        panels.append((i, mx1, mg, rho, p, cv_pre, cv_post))
        for c in range(len(mx1)):
            rows.append(dict(block=i, channel=c, gap_x1=float(mx1[c]),
                             gate=float(mg[c])))
        print(f'block {i}: C={len(mx1):4d}  Spearman rho={rho:+.3f}  p={p:.2e}  '
              f'CV {cv_pre:.3f} -> {cv_post:.3f}')

    with open('gate_stats.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── 出图（英文标注，便于直接用于英文投稿）──────────────────────────
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.linewidth': 0.8, 'figure.dpi': 300})
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.0), squeeze=False)

    for k, (i, mx1, mg, rho, p, cv_pre, cv_post) in enumerate(panels):
        ax = axes[0][k]
        ax.scatter(mx1, mg, s=7, alpha=0.55, edgecolors='none', color='#2b6cb0')
        ax.set_title(f'Block {i}  (C={len(mx1)})\n'
                     rf'Spearman $\rho$={rho:+.2f}, p={p:.1e}', fontsize=8)
        ax.set_xlabel(r'channel response  GAP($x_1$)')
        if k == 0:
            ax.set_ylabel(r'gate weight  $g_c$')
        ax.axhline(0.5, ls=':', lw=0.7, color='grey')
        ax.grid(alpha=0.25, lw=0.4)

        ax = axes[1][k]
        ax.bar(['pre-gate', 'post-gate'], [cv_pre, cv_post],
               color=['#a0aec0', '#2b6cb0'], width=0.55)
        ax.set_title(f'CV {cv_pre:.2f} → {cv_post:.2f}', fontsize=8)
        if k == 0:
            ax.set_ylabel('channel-magnitude CV')
        ax.grid(axis='y', alpha=0.25, lw=0.4)

    fig.tight_layout()
    fig.savefig('fig_gate_bias.png', bbox_inches='tight')
    fig.savefig('fig_gate_bias.pdf', bbox_inches='tight')
    print('\n已写出 fig_gate_bias.png / .pdf / gate_stats.csv')


if __name__ == '__main__':
    main()
