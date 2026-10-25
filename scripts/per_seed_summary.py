"""
Aggregate the per-seed CSV(s) written by eval_per_seed.py into mean +/- SD tables and run the
consistency checks of the per-seed values (Supplementary Tables S4 and S5).

    python per_seed_summary.py per_seed_main.csv --metric AP50_95 --out summary_main
    python per_seed_summary.py per_seed_uc.csv --metric AP50 AP50_95 --labels Uniform Manual Adaptive --out summary_t16

Conventions (state them in the table notes):
  * every value is first computed per run (seed); mean and SAMPLE standard deviation (n-1) are
    then taken over the runs;
  * the overall value of a run is the macro-average of its class-wise APs (this is what
    Ultralytics reports), so the SD of the overall value can never exceed the mean of the
    class-wise SDs.  The script verifies both properties and stops if either fails.

Outputs (prefix --out):
  <out>_long.csv        per-seed values (for the Supplementary per-seed tables)
  <out>_summary.csv     mean, SD, n for every label/split/class/metric
  <out>_<split>_<metric>.md   wide Markdown table: rows = classes + Overall, columns = labels
  <out>_diff.csv        (with --diff A B) per-seed paired differences B - A, mean +/- SD
"""
import argparse
import sys

import numpy as np
import pandas as pd


def fmt(m, s, nd=3):
    return f"{m:.{nd}f}±{s:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--metric", nargs="+", default=["AP50_95"], choices=["P", "R", "AP50", "AP50_95"])
    ap.add_argument("--labels", nargs="*", default=None, help="column order (default: order of appearance)")
    ap.add_argument("--diff", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--nd", type=int, default=3)
    ap.add_argument("--out", default="summary")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in a.csv], ignore_index=True)
    df = df.drop_duplicates(subset=["label", "seed", "split", "class"], keep="last")
    labels = a.labels or list(dict.fromkeys(df["label"]))
    df = df[df["label"].isin(labels)]
    df.to_csv(f"{a.out}_long.csv", index=False)

    problems = []
    rows = []
    for (lab, split), g in df.groupby(["label", "split"], sort=False):
        cls = g[g["class"] != "all"]
        allr = g[g["class"] == "all"].set_index("seed")
        seeds = sorted(allr.index)
        for met in ["P", "R", "AP50", "AP50_95"]:
            # (1) overall of each run == macro-average of its classes
            macro = cls.groupby("seed")[met].mean()
            for s in seeds:
                if abs(macro.loc[s] - allr.loc[s, met]) > 5e-4:
                    problems.append(f"{lab}/{split}/seed {s}/{met}: overall {allr.loc[s, met]:.4f} "
                                    f"!= macro-average of classes {macro.loc[s]:.4f}")
            # summaries
            for c, gc in cls.groupby("class", sort=False):
                v = gc.set_index("seed")[met].reindex(seeds)
                rows.append(dict(label=lab, split=split, cls=c, metric=met, mean=v.mean(),
                                 sd=v.std(ddof=1), n=v.notna().sum()))
            v = allr[met].reindex(seeds)
            sd_all = v.std(ddof=1)
            rows.append(dict(label=lab, split=split, cls="Overall", metric=met, mean=v.mean(), sd=sd_all, n=len(v)))
            # (2) SD bound
            mean_cls_sd = cls.groupby("class")[met].std(ddof=1).mean()
            if sd_all > mean_cls_sd + 1e-9:
                problems.append(f"{lab}/{split}/{met}: SD(overall)={sd_all:.4f} exceeds mean class SD={mean_cls_sd:.4f}")

    summ = pd.DataFrame(rows)
    summ.to_csv(f"{a.out}_summary.csv", index=False)

    for split in dict.fromkeys(summ["split"]):
        for met in a.metric:
            s = summ[(summ["split"] == split) & (summ["metric"] == met)]
            classes = [c for c in dict.fromkeys(s["cls"]) if c != "Overall"] + ["Overall"]
            lines = ["| Class | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
            for c in classes:
                cells = []
                for lab in labels:
                    r = s[(s["cls"] == c) & (s["label"] == lab)]
                    cells.append(fmt(r["mean"].iloc[0], r["sd"].iloc[0], a.nd) if len(r) else "-")
                lines.append(f"| {c} | " + " | ".join(cells) + " |")
            path = f"{a.out}_{split}_{met}.md"
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{met} on the {split} split, mean ± SD over seeds (sample SD, n-1)\n\n" + "\n".join(lines) + "\n")
            print(f"\n== {split} / {met} ==\n" + "\n".join(lines))

    if a.diff:
        A, B = a.diff
        drows = []
        for split in dict.fromkeys(df["split"]):
            for c in dict.fromkeys(df["class"]):
                for met in a.metric:
                    va = df[(df.label == A) & (df.split == split) & (df["class"] == c)].set_index("seed")[met]
                    vb = df[(df.label == B) & (df.split == split) & (df["class"] == c)].set_index("seed")[met]
                    common = sorted(set(va.index) & set(vb.index))
                    if len(common) < 2:
                        continue
                    d = (vb.loc[common] - va.loc[common]).astype(float)
                    drows.append(dict(split=split, cls=c, metric=met, diff_mean=d.mean(), diff_sd=d.std(ddof=1),
                                      n=len(d), all_positive=bool((d > 0).all())))
        pd.DataFrame(drows).to_csv(f"{a.out}_diff.csv", index=False)
        print(f"\npaired differences ({B} - {A}) written to {a.out}_diff.csv")

    if problems:
        print("\nCONSISTENCY PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nconsistency checks passed (overall = macro-average of classes; SD bound respected)")


if __name__ == "__main__":
    main()
