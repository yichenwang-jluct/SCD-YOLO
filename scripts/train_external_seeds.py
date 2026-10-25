#!/usr/bin/env python3
"""
Three-seed cross-dataset runs (Tables 13 and 14): YOLOv13n and SCD-YOLO retrained on PlantDoc and cotton-cqamc with
seeds 0, 1 and 2 and the protocol of Section 2.6 (SCD-YOLO with the adaptive Cotton-IoU loss) - 12 trainings, all
started with the same script, so that the three seeds of each model are trained identically.

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python train_external_seeds.py --plantdoc <PlantDoc.v4i.yolov8 folder> --cqamc <cotton-cqamc folder>

Start the same command in up to THREE windows, one or two minutes apart: each window trains one run at a time and
takes the next run that is neither finished nor being trained by another window (run_experiment.py keeps track with
its .pid files), so three runs are trained at the same time (three trainings need about 14 of 16 GB GPU memory).

The datasets are used in place: the official Roboflow splits, class names from the data.yaml of the release.  A data
YAML with the absolute dataset path is written to ./ext_data/ (the dataset folders are only read; Ultralytics adds its
usual labels.cache files).  Runs are written to
<project>/ext_<dataset>_<scd|base>_seed<k>.  run_experiment.py skips finished seeds and redoes unfinished ones,
so after an interruption simply start the same command again.  Order: cotton-cqamc first (about 1.3-1.5 h per
run alone), then PlantDoc (about twice as long).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(HERE))

from eval_external import CQAMC_NAMES, PLANTDOC_NAMES, find_split, lp  # noqa: E402  (same rules as the evaluation)


RUNS = [  # (dataset, short name, --arch, --loss); same arguments as fac_S1D1A1I1 / fac_S0D0A0I0
    ("cqamc", "scd", "S1_D1_A1", "adaptive"),
    ("cqamc", "base", "S0_D0_A0", "ciou"),
    ("plantdoc", "scd", "S1_D1_A1", "adaptive"),
    ("plantdoc", "base", "S0_D0_A0", "ciou"),
]


def write_yaml(ds, folder, names, outdir):
    import yaml
    root = Path(folder).resolve()
    if not root.is_dir():
        raise SystemExit(f"{ds}: folder not found: {folder}")
    tr, va, te = find_split(root, "train"), find_split(root, "valid", "val"), find_split(root, "test")
    if not tr or not va:
        raise SystemExit(f"{ds}: no train/ or valid/ split with an images/ folder in {root}")
    dy = root / "data.yaml"
    if dy.exists():
        with open(lp(dy), encoding="utf-8") as f:
            yn = (yaml.safe_load(f) or {}).get("names")
        if isinstance(yn, dict):
            yn = [yn[k] for k in sorted(yn)]
        if yn:
            names = list(yn)
    d = {"path": str(root), "train": f"{tr}/images", "val": f"{va}/images"}
    if te:
        d["test"] = f"{te}/images"
    d["nc"] = len(names)
    d["names"] = list(names)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"data_{ds}_train.yaml"
    txt = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
    if not p.exists() or p.read_text(encoding="utf-8") != txt:   # never rewrite it while another window trains
        p.write_text(txt, encoding="utf-8")
    print(f"{ds}: {root}  ({d['nc']} classes) -> {p}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantdoc", required=True, help="PlantDoc.v4i.yolov8 folder (Roboflow YOLOv8 export, v4)")
    ap.add_argument("--cqamc", required=True, help="cotton-cqamc folder (Roboflow YOLOv8 export)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--parallel", type=int, default=1, help="seeds trained at the same time BY THIS WINDOW")
    ap.add_argument("--project", default="runs_revision")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    ys = {"cqamc": write_yaml("cqamc", a.cqamc, CQAMC_NAMES, Path("ext_data")),
          "plantdoc": write_yaml("plantdoc", a.plantdoc, PLANTDOC_NAMES, Path("ext_data"))}
    failed = []
    for ds, short, arch, loss in RUNS:
        name = f"ext_{ds}_{short}"
        cmd = [sys.executable, str(HERE / "run_experiment.py"), "--arch", arch, "--loss", loss, "--name", name,
               "--data", str(ys[ds]), "--seeds", *[str(s) for s in a.seeds], "--parallel", str(a.parallel),
               "--project", a.project]
        print(f"\n[{time.strftime('%m-%d %H:%M')}] ===== {name}, seeds {' '.join(map(str, a.seeds))} =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    import eval_all
    todo = [f"ext_{ds}_{short}_seed{s}" for ds, short, _, _ in RUNS for s in a.seeds]
    done = [r for r in todo if eval_all.run_finished(Path(a.project) / r)[0]]
    print(f"\n[{time.strftime('%m-%d %H:%M')}] ===== this window has nothing left to start =====")
    print(f"   finished: {len(done)} of {len(todo)} runs")
    for r in todo:
        if r not in done:
            print(f"   not finished: {r}  ({eval_all.run_finished(Path(a.project) / r)[1]})")
    if failed:
        print("   errors in: " + ", ".join(failed))
    if len(done) == len(todo):
        print("   all runs are finished - next: eval_external_seeds.py")
    else:
        print("   runs of other windows may still be training; when ALL windows show this message and some runs are")
        print("   not finished, start the same command again (unfinished runs are redone)")


if __name__ == "__main__":
    main()
