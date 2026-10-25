"""
Percentile bootstrap confidence intervals for mAP (overall and per class) and for the paired
difference between two models, resampling INDEPENDENT IMAGES (default) or ACQUISITION GROUPS
(--groups); used for Section 3.6 and Supplementary Tables S9 and S10.

Each replicate draws images (or groups) with replacement, recomputes every metric with exactly
the Ultralytics definitions (apcore.py) for every seed of every model, averages over seeds, and
stores the model values and the difference B - A.  The same resample is used for all models, so
the difference is paired.  Seed-to-seed training variability is reported separately (mean +/- SD
over seeds); the bootstrap quantifies the sampling uncertainty due to the finite evaluation set.

    python bootstrap_ci.py \
        --model "YOLOv13n=stats/base_s0_test.pkl,stats/base_s1_test.pkl,stats/base_s2_test.pkl" \
        --model "SCD-YOLO=stats/scd_s0_test.pkl,stats/scd_s1_test.pkl,stats/scd_s2_test.pkl" \
        --diff YOLOv13n SCD-YOLO --B 2000 --out boot_test

    # cluster bootstrap over groups of near-duplicate field images, groups.csv = "image,group" (file name, id)
    python bootstrap_ci.py ... --groups field_groups.csv --out boot_field

Outputs: <out>_ci.csv (all metrics) and <out>_ci.md (readable table), <out>_replicates.npz.
"""
import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apcore  # noqa: E402


class Packed:
    """All images of one (model, seed) evaluation packed into flat arrays for fast resampling."""

    def __init__(self, records, order):
        by = {Path(r["im_file"]).name: r for r in records}
        recs = [by[k] for k in order]
        self.np_ = np.array([len(r["conf"]) for r in recs])
        self.nt_ = np.array([len(r["target_cls"]) for r in recs])
        self.ps = np.concatenate([[0], np.cumsum(self.np_)[:-1]])
        self.ts = np.concatenate([[0], np.cumsum(self.nt_)[:-1]])
        self.tp, self.conf, self.pc, self.tc = apcore.concat_images(recs)

    @staticmethod
    def _gather(starts, lens, sample):
        L = lens[sample]
        if L.sum() == 0:
            return np.zeros(0, dtype=int)
        S = starts[sample]
        offs = np.repeat(np.cumsum(L) - L, L)
        return np.arange(L.sum()) - offs + np.repeat(S, L)

    def metrics(self, sample=None):
        if sample is None:
            tp, conf, pc, tc = self.tp, self.conf, self.pc, self.tc
        else:
            ip = self._gather(self.ps, self.np_, sample)
            it = self._gather(self.ts, self.nt_, sample)
            tp, conf, pc, tc = self.tp[ip], self.conf[ip], self.pc[ip], self.tc[it]
        res = apcore.ap_per_class(tp, conf, pc, tc)
        out = apcore.summarize(res)
        for k, c in enumerate(res["classes"]):
            out[f"AP50/{c}"] = float(res["ap"][k, 0])
            out[f"AP50_95/{c}"] = float(res["ap"][k].mean())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help='LABEL=a.pkl,b.pkl,c.pkl (one file per seed)')
    ap.add_argument("--diff", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--groups", default=None, help="CSV with columns image,group for a cluster bootstrap")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--out", default="bootstrap")
    a = ap.parse_args()

    models = {}
    names = None
    for spec in a.model:
        lab, files = spec.split("=", 1)
        recs = []
        for f in files.split(","):
            with open(f.strip(), "rb") as fh:
                d = pickle.load(fh)
            recs.append(d["records"])
            names = names or d.get("names")
        models[lab] = recs
    # common image set
    sets = [set(Path(r["im_file"]).name for r in rs) for recs in models.values() for rs in recs]
    common = sorted(set.intersection(*sets))
    if any(len(s) != len(common) for s in sets):
        print(f"warning: evaluations cover different images; using the {len(common)} common images")
    packed = {lab: [Packed(rs, common) for rs in recs] for lab, recs in models.items()}

    if a.groups:
        gmap = {}
        with open(a.groups, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                gmap[Path(r["image"]).name] = r["group"]
        missing = [k for k in common if k not in gmap]
        if missing:
            raise SystemExit(f"{len(missing)} images have no group, e.g. {missing[:3]}")
        gids = sorted(set(gmap[k] for k in common))
        members = [np.array([i for i, k in enumerate(common) if gmap[k] == g]) for g in gids]
        unit = f"{len(gids)} groups"
    else:
        members = [np.array([i]) for i in range(len(common))]
        unit = f"{len(common)} images"

    def evaluate(sample):
        out = {}
        for lab, plist in packed.items():
            vals = [p.metrics(sample) for p in plist]
            keys = set().union(*[v.keys() for v in vals])
            for k in keys:
                out[(lab, k)] = float(np.nanmean([v.get(k, np.nan) for v in vals]))
        if a.diff:
            A, B = a.diff
            for (lab, k) in list(out):
                if lab == B and (A, k) in out:
                    out[("diff", k)] = out[(B, k)] - out[(A, k)]
        return out

    point = evaluate(None)
    rng = np.random.default_rng(a.seed)
    keys = sorted(point)
    reps = np.full((a.B, len(keys)), np.nan)
    t0 = time.time()
    for b in range(a.B):
        g = rng.integers(0, len(members), len(members))
        sample = np.concatenate([members[i] for i in g])
        r = evaluate(sample)
        reps[b] = [r.get(k, np.nan) for k in keys]
        if (b + 1) % max(1, a.B // 10) == 0:
            print(f"  {b + 1}/{a.B} replicates ({time.time() - t0:.0f}s)")

    lo, hi = (1 - a.level) / 2 * 100, (1 + a.level) / 2 * 100
    rows = []
    for j, (lab, k) in enumerate(keys):
        col = reps[:, j]
        cname = k
        if "/" in k:
            met, c = k.split("/")
            cname = f"{met}/{names[int(c)] if names else c}"
        rows.append(dict(model=lab, metric=cname, point=point[(lab, k)],
                         ci_low=np.nanpercentile(col, lo), ci_high=np.nanpercentile(col, hi),
                         boot_sd=np.nanstd(col, ddof=1),
                         p_gt_0=float(np.nanmean(col > 0)) if lab == "diff" else ""))
    with open(f"{a.out}_ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(f"{a.out}_replicates.npz", reps=reps, keys=np.array([f"{l}|{k}" for l, k in keys]))
    lines = [f"Percentile bootstrap, B = {a.B}, resampling unit = {unit}, {int(a.level * 100)}% CI; "
             f"point = mean over seeds on the full set", "",
             "| Model | Metric | Point | CI low | CI high | P(diff>0) |", "|---|---|---|---|---|---|"]
    order = ["P", "R", "mAP50", "mAP50_95"]
    rows.sort(key=lambda r: (r["model"] == "diff", r["model"], r["metric"] not in order,
                             order.index(r["metric"]) if r["metric"] in order else 0, r["metric"]))
    for r in rows:
        lines.append(f"| {r['model']} | {r['metric']} | {r['point']:.3f} | {r['ci_low']:.3f} | {r['ci_high']:.3f} | "
                     f"{r['p_gt_0'] if r['p_gt_0'] == '' else format(r['p_gt_0'], '.3f')} |")
    Path(f"{a.out}_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
