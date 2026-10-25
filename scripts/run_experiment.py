"""
Unified launcher for the three-seed training runs of the SCD-YOLO paper
(full-factorial ablation, controls and upper-bound settings; the 30 configurations are listed in
docs/REPRODUCE.md).

It reproduces the original protocol exactly (Section 2.6 of the paper):
    epochs=200, batch=8, imgsz=640, workers=0, amp=False, pretrained=False,
    deterministic=True, patience=100, optimizer='auto', seeds 0/1/2.
Only the architecture YAML (--arch) and the box-regression loss (--loss) change.

Run it from the root of your YOLOv13 source tree with that root on PYTHONPATH
(exactly like the scripts in the SCD-YOLO repository):

    set PYTHONPATH=%CD%                         (Windows)   |  export PYTHONPATH=$PWD
    python <repo>/scripts/run_experiment.py --arch S1_D0_A1 --loss ciou --name fac_S1D0A1_ciou

Architectures (--arch)                      file in cfg/ablation/
    S{0|1}_D{0|1}_A{0|1}                    abl_S?_D?_A?.yaml   (2^3 architectural factorial)
        S = Cotton-StarNet, D = Star-ADown, A = Cotton-SimAM
    ctrl_S1_ADown_A0                        Table 7 ADown control
    ctrl_ungated_D1_A1 / ctrl_ungated_D0_A0 ungated StarNet controls (Table 6, Suppl. Table S7)
    or any path to a model YAML

Losses (--loss)
    ciou       stock Ultralytics loss (CIoU), no patch
    adaptive   Cotton-IoU with online class-adaptive upper bounds (the proposed method;
               identical settings to scripts/train_adaptive.py: r=1.2, [0.70,0.95], beta=0.9,
               30 warm-up epochs, min_samples=10)
    uniform    Cotton-IoU components with ONE fixed upper bound for all classes   (--u 0.85)
    fixed      Cotton-IoU components with fixed class-wise upper bounds          (--uc 0.796,0.845,...)
    schedule   Cotton-IoU components with a PREDETERMINED (open-loop) schedule of class-wise
               upper bounds read from a CSV produced by make_uc_schedules.py   (--schedule file.csv)
All non-CIoU variants use the same loss class (CottonDetectionLossAdaptive); for uniform /
fixed / schedule the online update is disabled and u_c is written by a callback, so the
ONLY difference from the adaptive run is where the u_c values come from.

Several seeds are run as separate child processes (one process per seed, so the
monkey-patched loss can never leak between runs). --parallel N runs N seeds at once on
the same GPU (YOLOv13n at batch 8 needs < 4 GB, so 2-3 concurrent runs fit on 16 GB).

IMPORTANT: do not use Ultralytics `resume=True` for adaptive/schedule runs - the u_c state
lives in the loss object, which is not stored in last.pt. Re-run the seed from scratch.

Restarting is safe: a seed whose folder <project>/<name>_seed<s> already holds a finished run
(weights/best.pt plus results.png, or all epochs in results.csv) is skipped; an unfinished folder
is renamed to <name>_seed<s>_incomplete_<time> and that seed is trained again from scratch.
So after a crash or a Windows restart, simply start the same command again.

Several queues on one computer: while a seed is training, <project>/<name>_seed<s>.pid holds the
process id; another window that reaches this seed sees the live process and skips it (it is never
renamed). Runs started by an older version of this script have no .pid file and are recognised
by a write to their log or results.csv during the last 20 minutes. So the same command can be
started in a second window to train two settings at the same time. The state of every seed is
checked again right before it is launched, so a command with several seeds never starts a seed
that another window has started in the meantime.

Moving work to another computer: a line "<name>_seed<s>" (one seed) or "<name>" (all seeds) in
<project>/SKIP.txt makes this computer skip that run ([moved]); lines starting with # are ignored.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]
ABL_DIR = KIT_ROOT / "cfg" / "ablation"

ADAPTIVE_CFG = dict(inner_ratio=1.2, u_min=0.70, u_max=0.95, warmup_epochs=30,
                    beta=0.9, min_samples=10, init_u=None)


# --------------------------------------------------------------------------- helpers
def resolve_arch(arch: str) -> Path:
    p = Path(arch)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    if arch.startswith("S") and len(arch) == 8:          # e.g. S1_D0_A1
        cand = ABL_DIR / f"abl_{arch}.yaml"
    else:
        cand = ABL_DIR / f"{arch}.yaml"
    if not cand.exists():
        raise FileNotFoundError(f"unknown --arch '{arch}' (looked for {cand})")
    return cand


def read_data_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return int(d.get("nc", len(names))), list(names)


class Schedule:
    """u_c values per epoch.  `init` is used during epoch 0; after_epoch[e] is written into the
    loss at the end of epoch e (i.e. it is used during epoch e+1) - the same timing as the
    adaptive callback, whose u_c_evolution.csv row `e` is the value written at the end of epoch e."""

    def __init__(self, init, after_epoch):
        self.init = [float(x) for x in init]
        self.after_epoch = {int(k): [float(x) for x in v] for k, v in after_epoch.items()}

    def value_after_epoch(self, e):
        if e in self.after_epoch:
            return self.after_epoch[e]
        prev = [k for k in self.after_epoch if k <= e]
        return self.after_epoch[max(prev)] if prev else self.init

    @classmethod
    def constant(cls, u):
        return cls(u, {0: u})

    @classmethod
    def from_csv(cls, path, nc):
        rows = {}
        init = None
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                e = int(float(r["epoch"]))
                u = [float(r[f"u_c_{i}"]) for i in range(nc)]
                if e < 0:
                    init = u
                else:
                    rows[e] = u
        if init is None:
            init = [(ADAPTIVE_CFG["u_min"] + ADAPTIVE_CFG["u_max"]) / 2.0] * nc
        return cls(init, rows)


def build_schedule(a, nc):
    if a.loss == "uniform":
        if a.u is None:
            raise SystemExit("--loss uniform needs --u")
        return Schedule.constant([a.u] * nc)
    if a.loss == "fixed":
        if not a.uc:
            raise SystemExit("--loss fixed needs --uc v0,v1,...")
        u = [float(x) for x in a.uc.split(",")]
        if len(u) != nc:
            raise SystemExit(f"--uc has {len(u)} values but the dataset has nc={nc}")
        return Schedule.constant(u)
    if a.loss == "schedule":
        if not a.schedule:
            raise SystemExit("--loss schedule needs --schedule file.csv")
        return Schedule.from_csv(a.schedule, nc)
    return None


def attach_schedule_callback(model, schedule, logfile="u_c_evolution.csv"):
    """Writes the predetermined u_c into the loss at the end of every epoch and logs it."""
    import torch
    state = {"header": False}

    def _on_epoch_end(trainer):
        crit = getattr(trainer.model, "criterion", None)
        bl = getattr(crit, "bbox_loss", None) if crit is not None else None
        if bl is None or not hasattr(bl, "class_u"):
            return
        u = schedule.value_after_epoch(int(trainer.epoch))
        with torch.no_grad():
            bl.class_u.copy_(torch.tensor(u, dtype=bl.class_u.dtype, device=bl.class_u.device))
            if hasattr(bl, "iou_sum"):
                bl.iou_sum.zero_()
                bl.iou_cnt.zero_()
        mode = "a" if state["header"] else "w"
        with open(os.path.join(trainer.save_dir, logfile), mode, newline="") as f:
            w = csv.writer(f)
            if not state["header"]:
                w.writerow(["epoch"] + [f"u_c_{i}" for i in range(len(u))])
                state["header"] = True
            w.writerow([trainer.epoch] + [round(x, 4) for x in u])

    model.add_callback("on_train_epoch_end", _on_epoch_end)


# --------------------------------------------------------------------------- child
def train_one(a, seed):
    import warnings
    warnings.filterwarnings("ignore")
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    yaml_path = resolve_arch(a.arch)
    nc, names = read_data_yaml(a.data)
    sched = build_schedule(a, nc)

    if a.loss != "ciou":
        from ultralytics.nn.newsAddmodules.cotton_iou_loss_adaptive import (
            CottonDetectionLossAdaptive, attach_adaptive_callback)
        cfg = dict(ADAPTIVE_CFG)
        if a.loss != "adaptive":
            cfg["warmup_epochs"] = 10 ** 9          # online update never switched on
            cfg["init_u"] = sched.init

        def _patched_init_criterion(self):
            return CottonDetectionLossAdaptive(self, **cfg)

        DetectionModel.init_criterion = _patched_init_criterion   # must precede YOLO()

    model = YOLO(str(yaml_path))
    if a.loss == "adaptive":
        attach_adaptive_callback(model, logfile="u_c_evolution.csv")
    elif a.loss != "ciou":
        attach_schedule_callback(model, sched, logfile="u_c_evolution.csv")

    cache = False if a.cache == "False" else a.cache
    model.train(
        data=a.data, cache=cache, imgsz=640, epochs=a.epochs, workers=a.workers,
        batch=a.batch, amp=False, pretrained=False, seed=seed, deterministic=True,
        patience=100, device=a.device, project=a.project, name=f"{a.name}_seed{seed}",
        exist_ok=False, plots=True,
    )


def dry_run(a):
    """Build the model, print parameter counts (unfused / fused) and GFLOPs, then exit."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    nc, _ = read_data_yaml(a.data)
    m = DetectionModel(str(resolve_arch(a.arch)), nc=nc, verbose=False)
    p_unf = get_num_params(m)
    m.eval().fuse(verbose=False)
    print(f"{a.arch}: params unfused={p_unf:,}  fused={get_num_params(m):,}  "
          f"GFLOPs@640={get_flops(m, 640):.2f}")


# --------------------------------------------------------------------------- parent
BUSY_WINDOW = 20 * 60          # seconds without any write before an unfinished run counts as dead


def _recent(path, now):
    try:
        return now - path.stat().st_mtime < BUSY_WINDOW
    except OSError:
        return False


def _pid_alive(pidfile):
    """True / False if the training process recorded in pidfile is / is not running, None if unknown."""
    if not pidfile.exists():
        return None
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        import psutil
    except Exception:
        return None
    try:
        if not psutil.pid_exists(pid):
            return False
        return "python" in psutil.Process(pid).name().lower()
    except Exception:
        return False


def seed_state(a, seed):
    """'done' | 'running' | 'incomplete' | 'new' for <project>/<name>_seed<seed>."""
    d = Path(a.project) / f"{a.name}_seed{seed}"
    log = Path(a.project) / f"{a.name}_seed{seed}.log"
    res = d / "results.csv"
    now = time.time()
    alive = _pid_alive(Path(a.project) / f"{a.name}_seed{seed}.pid")
    if alive is None and (Path(a.project) / f"{a.name}_seed{seed}.exit").exists():
        alive = False      # this script saw the training process end (finished or failed)
    if alive is None:      # started by an older version of this script: judge by recent writes
        alive = _recent(log, now) or _recent(res, now) or _recent(d / "weights" / "last.pt", now)
    if not d.exists():
        return "running" if alive else "new"
    rows = 0
    if res.exists():
        try:
            with open(res, encoding="utf-8", errors="replace") as f:
                rows = max(0, sum(1 for line in f if line.strip()) - 1)
        except Exception:
            rows = 0
    if (d / "weights" / "best.pt").exists() and ((d / "results.png").exists() or rows >= a.epochs):
        return "done"
    return "running" if alive else "incomplete"


def skipped(a, seed):
    """True if <project>/SKIP.txt lists <name>_seed<seed> or <name> (run moved to another computer)."""
    f = Path(a.project) / "SKIP.txt"
    if not f.exists():
        return False
    try:
        items = {l.strip() for l in f.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                 if l.strip() and not l.strip().startswith("#")}
    except OSError:
        return False
    return a.name in items or f"{a.name}_seed{seed}" in items


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--loss", required=True, choices=["ciou", "adaptive", "uniform", "fixed", "schedule"])
    ap.add_argument("--u", type=float, default=None)
    ap.add_argument("--uc", type=str, default=None)
    ap.add_argument("--schedule", type=str, default=None)
    ap.add_argument("--name", required=False, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--data", default=str(Path("cfg") / "cotton_disease.yaml"))
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cache", default="False", choices=["False", "disk", "ram"],
                    help="'disk' stores decoded originals as .npy (pixel-identical, faster); default False = original protocol")
    ap.add_argument("--device", default="0")
    ap.add_argument("--dry-run", action="store_true", help="only build the model and print params/GFLOPs")
    ap.add_argument("--child-seed", type=int, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    a = parse()
    if a.dry_run:
        dry_run(a)
        return
    if a.name is None:
        raise SystemExit("--name is required for training runs")
    if a.child_seed is not None:
        train_one(a, a.child_seed)
        return

    Path(a.project).mkdir(parents=True, exist_ok=True)
    manifest = dict(vars(a), arch_yaml=str(resolve_arch(a.arch)), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(Path(a.project) / f"{a.name}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    base = [sys.executable, str(Path(__file__).resolve())]
    passthru = ["--arch", a.arch, "--loss", a.loss, "--name", a.name, "--data", a.data,
                "--project", a.project, "--epochs", str(a.epochs), "--batch", str(a.batch),
                "--workers", str(a.workers), "--cache", a.cache, "--device", a.device]
    for k in ("u", "uc", "schedule"):
        v = getattr(a, k)
        if v is not None:
            passthru += [f"--{k}", str(v)]

    queue = []
    for s in a.seeds:
        if skipped(a, s):
            print(f"[moved]  {a.name} seed {s}: listed in {Path(a.project) / 'SKIP.txt'} (trained on another computer), skipped")
            continue
        state = seed_state(a, s)
        if state == "done":
            print(f"[skip]   {a.name} seed {s}: already finished ({Path(a.project) / f'{a.name}_seed{s}'})")
            continue
        if state == "running":
            print(f"[busy]   {a.name} seed {s}: being trained in another window (written in the last 20 min), skipped")
            continue
        if state == "incomplete":
            d = Path(a.project) / f"{a.name}_seed{s}"
            moved = d.with_name(d.name + "_incomplete_" + time.strftime("%Y%m%d-%H%M%S"))
            d.rename(moved)
            print(f"[redo]   {a.name} seed {s}: unfinished run moved to {moved.name}, training again")
        queue.append(s)
    running = []
    failed = []
    while queue or running:
        while queue and len(running) < max(1, a.parallel):
            s = queue.pop(0)
            state = seed_state(a, s)        # check again: another window may have started it meanwhile
            if skipped(a, s) or state in ("done", "running"):
                print(f"[skip]   {a.name} seed {s}: {'moved' if skipped(a, s) else state} (checked again before launch)")
                continue
            if state == "incomplete":
                d = Path(a.project) / f"{a.name}_seed{s}"
                moved = d.with_name(d.name + "_incomplete_" + time.strftime("%Y%m%d-%H%M%S"))
                d.rename(moved)
                print(f"[redo]   {a.name} seed {s}: unfinished run moved to {moved.name}, training again")
            log = open(Path(a.project) / f"{a.name}_seed{s}.log", "w", encoding="utf-8")
            try:
                (Path(a.project) / f"{a.name}_seed{s}.exit").unlink()
            except OSError:
                pass
            p = subprocess.Popen(base + passthru + ["--child-seed", str(s)], stdout=log, stderr=subprocess.STDOUT)
            (Path(a.project) / f"{a.name}_seed{s}.pid").write_text(str(p.pid), encoding="utf-8")
            running.append((s, p, log))
            print(f"[launch] {a.name} seed {s} (pid {p.pid})")
        time.sleep(10)
        for item in list(running):
            s, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                try:
                    (Path(a.project) / f"{a.name}_seed{s}.pid").unlink()
                except OSError:
                    pass
                (Path(a.project) / f"{a.name}_seed{s}.exit").write_text(str(p.returncode), encoding="utf-8")
                status = "OK" if p.returncode == 0 else f"FAILED (code {p.returncode})"
                if p.returncode != 0:
                    failed.append(s)
                print(f"[done]   {a.name} seed {s}: {status}")
    if failed:
        raise SystemExit(f"seeds failed: {failed} - see the .log files in {a.project}")


if __name__ == "__main__":
    main()
