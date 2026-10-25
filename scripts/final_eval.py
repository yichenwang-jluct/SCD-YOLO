#!/usr/bin/env python3
"""
Final evaluation pipeline of the paper (run after ALL trainings have finished).

Steps (each step is skipped when its results already exist, so the script can simply be started again,
e.g. after further trainings - then only the new checkpoints are evaluated):

  0  refuses to start while more than --max-running (default 2) trainings are running on this computer
     (GPU memory: three trainings use about 14 of 16 GB); runs that are still training are skipped and
     evaluated the next time this file is started
  1  eval_all.py, seeds 0 1 2, into a NEW folder (default eval_final): every finished run of
     runs_revision and the original runs, val/test (source-disjoint runs: val/test_field/test_public),
     per-seed group files and the per-image statistics for the bootstrap
  2  bootstrap_ci.py: test set, SCD-YOLO vs YOLOv13n (images resampled) and the 427 field images,
     source-disjoint models (groups of near-duplicate images resampled, --groups)
  3  plot_gate_bias.py: what the Cotton-StarNet gate learns, seeds 0 1 2
  4  simam_probe.py --auto: Cotton-SimAM artefact test and attention statistics, seeds 0 1 2
  5  MANIFEST.txt (size and SHA-256 of every result file) and one zip file with all results

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python final_eval.py --data <cotton_disease.yaml> --src-dir <cotton_srcdisjoint> --groups <groups of near-duplicate field images, csv>
                         --out eval_final

Only reads the run folders; writes only into --out.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOG = None


def say(s=""):
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


def run(cmd, title):
    """run a child script, show and log its output; returns the exit code"""
    say(f"\n[{time.strftime('%m-%d %H:%M')}] ===== {title} =====")
    say("  " + " ".join(str(c) for c in cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        if LOG is not None:
            LOG.write(line + "\n")
    p.wait()
    LOG.flush()
    if p.returncode != 0:
        say(f"!! {title}: exit code {p.returncode}")
    return p.returncode


def running_trainings(project):
    """names of runs in <project> that are being trained right now (same test as run_experiment.py)"""
    import run_experiment as rx
    proj = Path(project)
    if not proj.is_dir():
        return []
    cand = set()
    for p in proj.iterdir():
        m = re.match(r"(.+)_seed(\d+)(\.pid|\.log)?$", p.name)
        if m and (p.is_dir() or m.group(3)):
            cand.add((m.group(1), int(m.group(2))))
    busy = []
    for name, s in sorted(cand):
        try:
            st = rx.seed_state(SimpleNamespace(project=str(proj), name=name, epochs=200), s)
        except Exception:
            continue
        if st == "running":
            busy.append(f"{name}_seed{s}")
    return busy


def newer(out_file, inputs):
    """True if out_file exists and is newer than every input file"""
    try:
        t = Path(out_file).stat().st_mtime
        return all(Path(i).stat().st_mtime <= t for i in inputs)
    except OSError:
        return False


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--groups", required=True, help="fold_assignment.csv (columns image, group)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-running", type=int, default=2,
                    help="start only if at most this many trainings are running (GPU memory)")
    ap.add_argument("--force", action="store_true", help="start whatever is running")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG = open(out / "final_log.txt", "a", encoding="utf-8")
    seeds = [str(s) for s in a.seeds]
    say(f"\n########## final evaluation {time.strftime('%Y-%m-%d %H:%M:%S')} ##########")
    say(f"python {sys.executable}")
    try:
        import torch
        import ultralytics
        say(f"torch {torch.__version__} cuda {torch.cuda.is_available()}  ultralytics {ultralytics.__version__}")
    except Exception as e:
        say(f"versions: {type(e).__name__}: {e}")

    # 0 ---------------------------------------------------------------- nothing may be training
    busy = running_trainings(a.project)
    if len(busy) > a.max_running and not a.force:
        say(f"\n!! {len(busy)} trainings are running on this computer: " + ", ".join(busy))
        say(f"   At most {a.max_running} may run during the evaluation (GPU memory). Start this file again when "
            f"one of them has finished.")
        LOG.close()
        sys.exit(2)
    if busy:
        say(f"\nnote: {len(busy)} training(s) running ({', '.join(busy)}): the evaluation shares the GPU and takes "
            f"longer; runs that are still training are evaluated the next time this file is started.")

    py = [sys.executable, "-u"]
    # 1 ---------------------------------------------------------------- evaluation of every finished run
    run(py + [HERE / "eval_all.py", "--seeds", *seeds, "--data", a.data, "--src-dir", a.src_dir, "--out", out],
        "1/4 evaluation of all finished runs (eval_all.py)")

    # 2 ---------------------------------------------------------------- bootstrap confidence intervals
    st = out / "stats"
    boot = out / "boot"
    boot.mkdir(exist_ok=True)
    jobs = [("test set, SCD-YOLO vs YOLOv13n, images resampled", "boot_test",
             {"YOLOv13n": [st / f"fac_S0D0A0I0_s{s}_test.pkl" for s in seeds],
              "SCD-YOLO": [st / f"fac_S1D1A1I1_s{s}_test.pkl" for s in seeds]}, None),
            ("427 field images, source-disjoint models, groups of near-duplicates resampled", "boot_field",
             {"YOLOv13n": [st / f"src_base_s{s}_test_field.pkl" for s in seeds],
              "SCD-YOLO": [st / f"src_scd_s{s}_test_field.pkl" for s in seeds]}, a.groups)]
    for title, name, models, groups in jobs:
        files = [f for fs in models.values() for f in fs]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            say(f"\n!! 2/4 bootstrap {name}: missing {', '.join(missing)} - skipped")
            continue
        if groups and not Path(groups).exists():
            say(f"\n!! 2/4 bootstrap {name}: {groups} not found - skipped")
            continue
        if newer(boot / f"{name}_ci.csv", files + ([groups] if groups else [])):
            say(f"\n[skip] 2/4 bootstrap {name}: results already there")
            continue
        cmd = py + [HERE / "bootstrap_ci.py"]
        for lab, fs in models.items():
            cmd += ["--model", f"{lab}=" + ",".join(str(f) for f in fs)]
        cmd += ["--diff", "YOLOv13n", "SCD-YOLO", "--B", "2000", "--out", boot / name]
        if groups:
            cmd += ["--groups", groups]
        run(cmd, f"2/4 bootstrap: {title}")

    # 3 ---------------------------------------------------------------- gate statistics
    gate = out / "gate"
    if (gate / "gate_summary.md").exists():
        say("\n[skip] 3/4 gate statistics: results already there")
    else:
        run(py + [HERE / "plot_gate_bias.py", "--runs", "fac_S1D1A1I1", "--seeds", *seeds, "--data", a.data,
                  "--split", "val", "--out", gate], "3/4 gate statistics (plot_gate_bias.py)")

    # 4 ---------------------------------------------------------------- Cotton-SimAM artefact test
    probe = out / "simam_probe"
    if (probe / "summary.md").exists():
        say("\n[skip] 4/4 Cotton-SimAM artefact test: results already there")
    else:
        run(py + [HERE / "simam_probe.py", "--auto", "--seeds", *seeds, "--data", a.data, "--split", "test",
                  "--out", probe], "4/4 Cotton-SimAM artefact test (simam_probe.py)")

    # 5 ---------------------------------------------------------------- manifest and zip
    keep = []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out)
        parts = rel.parts
        if parts[0] in ("stats", "_ultralytics_tmp") \
                or any(x.startswith(("_ultralytics", "perturbed_")) for x in parts):
            continue
        if p.suffix.lower() in (".pkl", ".npz", ".pt", ".zip", ".sig") or rel.name in ("MANIFEST.txt", "final_log.txt"):
            continue
        if p.stat().st_size > 20 * 1024 * 1024:
            continue
        keep.append(p)
    stamp = time.strftime("%Y%m%d-%H%M")
    lines = [f"MANIFEST {time.strftime('%Y-%m-%d %H:%M:%S')}  ({len(keep)} files; written by final_eval.py)"]
    for p in keep:
        lines.append(f"{sha256(p)}  {p.stat().st_size:>10}  {p.relative_to(out).as_posix()}")
    (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.flush()
    zpath = out / f"final_results_{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in keep + [out / "MANIFEST.txt", out / "final_log.txt"]:
            z.write(p, p.relative_to(out).as_posix())
    say(f"\n[{time.strftime('%m-%d %H:%M')}] ===== finished =====")
    say(f"   all results are collected in this file:")
    say(f"   {zpath.resolve()}")
    LOG.close()


if __name__ == "__main__":
    main()
