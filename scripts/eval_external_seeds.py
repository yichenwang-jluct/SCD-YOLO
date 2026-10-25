#!/usr/bin/env python3
"""
Cross-dataset results over three seeds (Tables 13 and 14).  YOLOv13n and SCD-YOLO on PlantDoc and cotton-cqamc,
seeds 0, 1 and 2 = <project>/ext_<dataset>_<base|scd>_seed<k> (train_external_seeds.py).  With --earlier, the
checkpoints of the original submission in runs/V13train (eval_external.py) are evaluated as well; they are listed
separately and are not part of the means.  Every checkpoint is evaluated with the paper's protocol (rect=False,
conf=0.001, NMS IoU=0.7, imgsz=640) on short-name copies of the official validation and test splits, exactly as in
eval_external.py (Windows paths of 260+ characters cannot be opened).

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python eval_external_seeds.py --plantdoc <folder> --cqamc <folder> --out <dir> [--earlier]

Writes into --out: per_seed_external3.csv (one row per dataset / model / seed / split / class),
external3_summary.txt (everything printed, incl. mean +- sample SD over the seeds), MANIFEST.txt and one zip file
with these three files.  Only reads the run folders and the datasets.
"""
import argparse
import csv
import hashlib
import shutil
import statistics
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def class_names(root, default):
    import yaml
    import eval_external as ee
    dy = root / "data.yaml"
    if dy.exists():
        try:
            with open(ee.lp(dy), encoding="utf-8") as f:
                yn = (yaml.safe_load(f) or {}).get("names")
            if isinstance(yn, dict):
                yn = [yn[k] for k in sorted(yn)]
            if yn:
                return list(yn)
        except Exception:
            pass
    return list(default)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantdoc", required=True)
    ap.add_argument("--cqamc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--earlier", action="store_true", help="also evaluate the original runs in runs/V13train")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    import yaml
    import eval_all
    import eval_external as ee

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_ultralytics_tmp"
    lines, rows = [], []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"eval_external_seeds  {time.strftime('%Y-%m-%d %H:%M:%S')}   seeds {a.seeds}")
    for ds, folder, default in (("plantdoc", a.plantdoc, ee.PLANTDOC_NAMES), ("cqamc", a.cqamc, ee.CQAMC_NAMES)):
        root = Path(folder)
        if not root.is_dir():
            say(f"\n!! {ds}: folder not found: {folder}")
            continue
        names = class_names(root, default)
        say(f"\n=== {ds}: {root}  ({len(names)} classes)")
        base = out / "_ext" / ds
        if base.exists():
            shutil.rmtree(ee.lp(base))
        y = {"path": str(base.resolve()), "nc": len(names), "names": names}
        for key, sname, cands in (("val", "valid", ("valid", "val")), ("test", "test", ("test",))):
            sub = ee.find_split(root, *cands)
            if sub:
                n = len(ee.short_copy(root, sub, base / sname))
                y[key] = f"{sname}/images"
                say(f"  {key:<4} {n} images (short-name copy)")
        if "val" not in y:
            say(f"  !! {ds}: no validation split - nothing evaluated")
            continue
        y["train"] = y["val"]  # required key; not used for evaluation
        ypath = out / f"data_{ds}.yaml"
        ypath.write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
        orig = dict(ee.MODELS[ds])
        jobs = [(label, s, Path(a.project) / f"ext_{ds}_{short}_seed{s}", True)
                for label, short in (("YOLOv13n", "base"), ("SCD-YOLO", "scd")) for s in a.seeds]
        if a.earlier:
            jobs += [(f"{label} (earlier run)", 0, orig[label], False) for label in ("YOLOv13n", "SCD-YOLO")]
        for label, s, run, new in jobs:
            ckpt = run / "weights" / "best.pt"
            if not ckpt.exists():
                say(f"  !! missing {ckpt}")
                continue
            if new:
                fin, note, _ = eval_all.run_finished(run)
                if not fin:
                    say(f"  !! {run} is not finished ({note}) - not evaluated; start this again after the training")
                    continue
            for split in ("val", "test"):
                if split not in y:
                    continue
                try:
                    res = eval_all.evaluate(ckpt, ypath, split, a, tmp)
                except Exception as e:
                    say(f"  !! {label} seed {s} {split}: {type(e).__name__}: {e}")
                    eval_all.free_gpu()
                    continue
                for cls, P, R, A50, A in res:
                    rows.append({"dataset": ds, "label": label, "seed": s, "split": split, "class": cls,
                                 "P": f"{float(P):.6f}", "R": f"{float(R):.6f}", "AP50": f"{float(A50):.6f}",
                                 "AP50_95": f"{float(A):.6f}", "checkpoint": str(ckpt)})
                _, P, R, A50, A = res[0]
                say(f"  {label:<23} seed {s} {split:<5} P {float(P):.3f}  R {float(R):.3f}  "
                    f"mAP50 {float(A50):.3f}  mAP50-95 {float(A):.3f}")

    say("\n=== mean +- sample SD over the seeds (overall values; the earlier runs are listed with n=1, not averaged)")
    for ds in ("plantdoc", "cqamc"):
        for split in ("val", "test"):
            for label in ("YOLOv13n", "SCD-YOLO", "YOLOv13n (earlier run)", "SCD-YOLO (earlier run)"):
                sel = [r for r in rows if r["dataset"] == ds and r["split"] == split and r["label"] == label
                       and r["class"] == "all"]
                if not sel:
                    continue
                txt = []
                for k in ("AP50", "AP50_95"):
                    v = [float(r[k]) for r in sel]
                    sd = f"{statistics.stdev(v):.3f}" if len(v) > 1 else "  -  "
                    txt.append(f"{k} {statistics.mean(v):.3f} +- {sd}")
                say(f"  {ds:<8} {split:<4} {label:<23} n={len(sel)}  " + "   ".join(txt))

    files = []
    if rows:
        p = out / "per_seed_external3.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        files.append(p)
    stamp = time.strftime("%Y%m%d-%H%M")
    zpath = out / f"external3_results_{stamp}.zip"
    say(f"\n[{time.strftime('%m-%d %H:%M')}] ===== finished =====")
    say("   send this ONE file (drag the zip itself into the chat, do not unpack or edit it):")
    say(f"   {zpath.resolve()}")
    s = out / "external3_summary.txt"
    s.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    files.append(s)
    man = out / "MANIFEST.txt"
    man.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.stat().st_size:>10}  {f.name}\n"
                           for f in files), encoding="utf-8")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + [man]:
            z.write(f, f.name)


if __name__ == "__main__":
    main()
