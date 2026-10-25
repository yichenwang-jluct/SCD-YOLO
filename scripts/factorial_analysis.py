"""
Interaction-effect analysis of the full 2^4 factorial ablation (Section 3.2, Supplementary Table S6).

Input: the per-seed CSV(s) from eval_per_seed.py in which the --runs labels encode the
configuration as S?D?A?I?  (S = Cotton-StarNet, D = Star-ADown, A = Cotton-SimAM,
I = Cotton-IoU; 1 = on), e.g. "S1D0A1I0".  All 16 labels x 3 seeds are expected.

    python factorial_analysis.py per_seed_factorial.csv --split val --out fac_val
    python factorial_analysis.py per_seed_factorial.csv --split test --out fac_test

Reported for mAP@0.5 and mAP@0.5:0.95:
  1. all 16 configurations, mean +/- SD over seeds                     (<out>_configs.csv/.md)
  2. factorial effects with +/-1 coding (main effects and all 2-, 3-, 4-way interactions),
     estimated by OLS with seed as a block factor; effect = 2*coefficient, SE, t, p
     (residual df = 48 - 16 - 2 = 30)                                    (<out>_effects.csv/.md)
  3. pairwise interaction contrasts in two backgrounds, per seed then mean +/- SD:
       empty background : I_XY = y(X+Y) - y(X) - y(Y) + y(none)
       full background  : I_XY = y(all) - y(all-X) - y(all-Y) + y(all-X-Y)
     positive = synergy (super-additive), negative = overlap (sub-additive)   (<out>_pairs.csv)
  4. Shapley values: each module's gain averaged over all 4! = 24 insertion orders,
     per seed then mean +/- SD; they sum to the total gain over the baseline (<out>_shapley.csv)
"""
import argparse
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

F = ["S", "D", "A", "I"]
NAMES = {"S": "Cotton-StarNet", "D": "Star-ADown", "A": "Cotton-SimAM", "I": "Cotton-IoU"}
PAT = re.compile(r"^S([01])D([01])A([01])I([01])$")


def load(paths, split, metric):
    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in paths], ignore_index=True)
    df = df[(df["split"] == split) & (df["class"] == "all")]
    df = df[df["label"].str.match(PAT)]
    df = df.drop_duplicates(subset=["label", "seed"], keep="last")
    y = {}
    for _, r in df.iterrows():
        bits = tuple(int(b) for b in PAT.match(r["label"]).groups())
        y[(bits, int(r["seed"]))] = float(r[metric])
    seeds = sorted({s for (_, s) in y})
    cells = list(itertools.product([0, 1], repeat=4))
    missing = [(c, s) for c in cells for s in seeds if (c, s) not in y]
    if missing:
        raise SystemExit(f"missing {len(missing)} config/seed results, e.g. {missing[:4]}")
    return y, seeds, cells


def lab(bits):
    on = [f for f, b in zip(F, bits) if b]
    return "+".join(on) if on else "baseline"


def ols_effects(y, seeds, cells):
    rows, Y = [], []
    terms = [t for k in range(1, 5) for t in itertools.combinations(range(4), k)]
    for c in cells:
        x = np.array([1 if b else -1 for b in c])
        for s in seeds:
            row = [1.0] + [1.0 if s == ss else 0.0 for ss in seeds[1:]] + [float(np.prod(x[list(t)])) for t in terms]
            rows.append(row)
            Y.append(y[(c, s)])
    X, Y = np.array(rows), np.array(Y)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(Y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    from scipy import stats
    out = []
    off = 1 + len(seeds) - 1
    for j, t in enumerate(terms):
        b, se = beta[off + j], math.sqrt(cov[off + j, off + j])
        tval = b / se if se > 0 else np.nan
        out.append(dict(term="x".join(F[i] for i in t), order=len(t), effect=2 * b, se=2 * se, t=tval,
                        p=2 * stats.t.sf(abs(tval), dof) if se > 0 else np.nan, df=dof))
    return pd.DataFrame(out)


def shapley(v):
    """v: dict bits->value. Returns dict factor->Shapley value (gain relative to baseline)."""
    n = 4
    phi = {}
    for i in range(n):
        tot = 0.0
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                b0 = tuple(1 if j in S else 0 for j in range(n))
                b1 = tuple(1 if (j in S or j == i) else 0 for j in range(n))
                tot += w * (v[b1] - v[b0])
        phi[F[i]] = tot
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--split", default="val")
    ap.add_argument("--metrics", nargs="+", default=["AP50", "AP50_95"])
    ap.add_argument("--out", default="factorial")
    a = ap.parse_args()

    cfg_rows, eff_all, pair_rows, shap_rows = [], [], [], []
    for met in a.metrics:
        y, seeds, cells = load(a.csv, a.split, met)
        for c in sorted(cells, key=lambda c: (sum(c), [-b for b in c])):
            v = np.array([y[(c, s)] for s in seeds])
            cfg_rows.append(dict(metric=met, S=c[0], D=c[1], A=c[2], I=c[3], config=lab(c),
                                 mean=v.mean(), sd=v.std(ddof=1)))
        eff = ols_effects(y, seeds, cells)
        eff.insert(0, "metric", met)
        eff_all.append(eff)
        full, base = (1, 1, 1, 1), (0, 0, 0, 0)
        for i, j in itertools.combinations(range(4), 2):
            e_i = tuple(1 if k == i else 0 for k in range(4))
            e_j = tuple(1 if k == j else 0 for k in range(4))
            e_ij = tuple(1 if k in (i, j) else 0 for k in range(4))
            f_i = tuple(0 if k == i else 1 for k in range(4))
            f_j = tuple(0 if k == j else 1 for k in range(4))
            f_ij = tuple(0 if k in (i, j) else 1 for k in range(4))
            emp = np.array([y[(e_ij, s)] - y[(e_i, s)] - y[(e_j, s)] + y[(base, s)] for s in seeds])
            ful = np.array([y[(full, s)] - y[(f_i, s)] - y[(f_j, s)] + y[(f_ij, s)] for s in seeds])
            pair_rows.append(dict(metric=met, pair=f"{F[i]}x{F[j]}", modules=f"{NAMES[F[i]]} x {NAMES[F[j]]}",
                                  empty_bg_mean=emp.mean(), empty_bg_sd=emp.std(ddof=1),
                                  full_bg_mean=ful.mean(), full_bg_sd=ful.std(ddof=1)))
        per_seed = []
        for s in seeds:
            v = {c: y[(c, s)] for c in cells}
            per_seed.append(shapley(v))
        singles = {F[i]: np.array([y[(tuple(1 if k == i else 0 for k in range(4)), s)] - y[(base, s)] for s in seeds])
                   for i in range(4)}
        total = np.array([y[(full, s)] - y[(base, s)] for s in seeds])
        for f in F:
            vals = np.array([p[f] for p in per_seed])
            shap_rows.append(dict(metric=met, module=NAMES[f], shapley_mean=vals.mean(), shapley_sd=vals.std(ddof=1),
                                  alone_gain_mean=singles[f].mean(), alone_gain_sd=singles[f].std(ddof=1)))
        shap_rows.append(dict(metric=met, module="TOTAL (full - baseline)", shapley_mean=total.mean(),
                              shapley_sd=total.std(ddof=1),
                              alone_gain_mean=sum(singles[f].mean() for f in F), alone_gain_sd=np.nan))

    cfg = pd.DataFrame(cfg_rows)
    eff = pd.concat(eff_all, ignore_index=True)
    pairs = pd.DataFrame(pair_rows)
    shap = pd.DataFrame(shap_rows)
    cfg.to_csv(f"{a.out}_configs.csv", index=False)
    eff.to_csv(f"{a.out}_effects.csv", index=False)
    pairs.to_csv(f"{a.out}_pairs.csv", index=False)
    shap.to_csv(f"{a.out}_shapley.csv", index=False)

    md = [f"# Factorial ablation on the {a.split} split (mean ± SD over seeds)", ""]
    for met in a.metrics:
        md += [f"## {met}", "", "| Cotton-StarNet | Star-ADown | Cotton-SimAM | Cotton-IoU | mean ± SD |", "|---|---|---|---|---|"]
        for _, r in cfg[cfg.metric == met].iterrows():
            ck = lambda b: "✓" if b else "✗"
            md.append(f"| {ck(r.S)} | {ck(r.D)} | {ck(r.A)} | {ck(r.I)} | {r['mean']:.3f}±{r['sd']:.3f} |")
        md += ["", "Factorial effects (±1 coding, OLS with seed blocks):", "", "| Term | Effect | SE | t | p |", "|---|---|---|---|---|"]
        for _, r in eff[eff.metric == met].iterrows():
            md.append(f"| {r.term} | {r.effect:+.4f} | {r.se:.4f} | {r.t:+.2f} | {r.p:.3g} |")
        md += ["", "Pairwise interaction contrasts (positive = synergy, negative = overlap):", "",
               "| Pair | Empty background | Full-model background |", "|---|---|---|"]
        for _, r in pairs[pairs.metric == met].iterrows():
            md.append(f"| {r.modules} | {r.empty_bg_mean:+.4f}±{r.empty_bg_sd:.4f} | {r.full_bg_mean:+.4f}±{r.full_bg_sd:.4f} |")
        md += ["", "Shapley decomposition of the total gain (order-independent contribution):", "",
               "| Module | Shapley value | Gain when added alone |", "|---|---|---|"]
        for _, r in shap[shap.metric == met].iterrows():
            alone = "" if np.isnan(r.alone_gain_sd) else f"±{r.alone_gain_sd:.4f}"
            md.append(f"| {r.module} | {r.shapley_mean:+.4f}±{r.shapley_sd:.4f} | {r.alone_gain_mean:+.4f}{alone} |")
        md.append("")
    Path(f"{a.out}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
