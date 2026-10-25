"""
Build the predetermined u_c schedules used as controls for Cotton-IoU (Section 3.7, Table 17)
from the u_c_evolution.csv files written by the three adaptive SCD-YOLO runs.

    python make_uc_schedules.py \
        --csv runs/scd_yolo_adaptive_seed0/u_c_evolution.csv \
              runs/scd_yolo_adaptive_seed1/u_c_evolution.csv \
              runs/scd_yolo_adaptive_seed2/u_c_evolution.csv \
        --out schedules

In the paper only the seed-0 trajectory was available when the schedules were built, so they were made
with a single --csv (the seed-0 u_c_evolution.csv); "mean" then equals that run's values.

Outputs (all in the CSV format read by run_experiment.py --loss schedule):
    replay_mean.csv     C3a  open-loop replay of the MEAN adaptive trajectory (class-wise, time-varying)
    linear_ramp.csv     C3b  0.825 during warm-up, then a linear ramp to the mean final values,
                             reaching them at the "settling epoch" of the adaptive runs, then constant
    uniform_traj.csv    C3c  (optional) class-AGNOSTIC trajectory = per-epoch mean over classes of C3a
    fixed_final.txt     C2   the mean final class-wise bounds, as a --uc string for --loss fixed
    summary.json             per-seed final values, mean/SD, settling epoch

Why the MEAN trajectory and not each seed's own trajectory?  With deterministic=True, replaying
seed s's own trajectory in a new run with the same seed s would reproduce that run bit-for-bit,
which tests nothing.  A single predetermined schedule applied to all three seeds is a genuine
open-loop control: same time course, no feedback from the run being trained.

Row semantics: the row with epoch = e holds the value written into the loss at the END of
epoch e (used during epoch e+1), exactly as in u_c_evolution.csv; the row epoch = -1 holds the
value used during epoch 0.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_uc(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cols = sorted([c for c in rows[0] if c.startswith("u_c_")], key=lambda c: int(c.split("_")[-1]))
    ep = np.array([int(float(r["epoch"])) for r in rows])
    u = np.array([[float(r[c]) for c in cols] for r in rows])
    order = np.argsort(ep)
    return ep[order], u[order]


def write_sched(path, init, ep, u):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + [f"u_c_{i}" for i in range(u.shape[1])])
        w.writerow([-1] + [round(float(x), 5) for x in init])
        for e, row in zip(ep, u):
            w.writerow([int(e)] + [round(float(x), 5) for x in row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="schedules")
    ap.add_argument("--u-init", type=float, default=0.825)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--tol", type=float, default=0.10,
                    help="settling = first epoch after which every class stays within tol*|final-init| of its final value")
    ap.add_argument("--settle-epoch", type=int, default=None, help="override the automatically found settling epoch")
    ap.add_argument("--names", nargs="*", default=["blight", "curl", "grey_mildew", "healthy", "leaf_spot", "wilt"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    eps, us = zip(*[read_uc(p) for p in a.csv])
    n = min(len(e) for e in eps)
    for e in eps:
        assert np.array_equal(e[:n], eps[0][:n]), "epoch columns differ between runs"
    ep = eps[0][:n]
    U = np.stack([u[:n] for u in us])            # (seeds, epochs, classes)
    nc = U.shape[2]
    mean_traj = U.mean(0)
    init = np.full(nc, a.u_init)
    final_each = U[:, -1, :]
    final_mean = final_each.mean(0)

    # settling epoch of the mean trajectory (5-epoch moving average vs. mean of the last 10 epochs,
    # so that epoch-to-epoch jitter does not postpone the settling point)
    k = 5
    pad = np.pad(mean_traj, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([pad[i:i + len(mean_traj)] for i in range(k)]).mean(0)
    final_ref = mean_traj[-10:].mean(0)
    dev = np.abs(smooth - final_ref)                           # (epochs, classes)
    thr = a.tol * np.maximum(np.abs(final_ref - a.u_init), 1e-3)
    ok = (dev <= thr).all(axis=1)
    settle = None
    for i in range(len(ep)):
        if ok[i:].all():
            settle = int(ep[i])
            break
    if a.settle_epoch is not None:
        settle = a.settle_epoch
    if settle is None or settle <= a.warmup:
        settle = int(ep[-1])

    # C3a replay of the mean trajectory
    write_sched(out / "replay_mean.csv", init, ep, mean_traj)

    # C3b linear ramp: logged rows e < warmup stay at init (the adaptive callback first updates at the end of epoch `warmup`)
    lin = np.empty_like(mean_traj)
    for i, e in enumerate(ep):
        if e < a.warmup:
            lin[i] = init
        elif e >= settle:
            lin[i] = final_mean
        else:
            t = (e - a.warmup + 1) / (settle - a.warmup + 1)
            lin[i] = init + t * (final_mean - init)
    write_sched(out / "linear_ramp.csv", init, ep, lin)

    # C3c class-agnostic trajectory
    uni = np.repeat(mean_traj.mean(1, keepdims=True), nc, axis=1)
    write_sched(out / "uniform_traj.csv", init, ep, uni)

    (out / "fixed_final.txt").write_text(",".join(f"{x:.4f}" for x in final_mean) + "\n", encoding="utf-8")
    summary = dict(
        classes=a.names[:nc],
        final_per_seed=final_each.round(4).tolist(),
        final_mean=final_mean.round(4).tolist(),
        final_sd=final_each.std(0, ddof=1).round(4).tolist() if len(us) > 1 else None,
        mean_of_final_bounds=float(final_mean.mean().round(4)),
        settling_epoch=settle, warmup=a.warmup, tol=a.tol, epochs=int(ep[-1]) + 1,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n--uc for the fixed-final control (C2): {(out / 'fixed_final.txt').read_text().strip()}")
    print(f"Mean of the final bounds = {final_mean.mean():.3f}: include a uniform bound near this value in the C1 grid.")


if __name__ == "__main__":
    main()
