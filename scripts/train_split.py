#!/usr/bin/env python3
"""
Train one share of the 27 settings (A/B/C groups + the two source-disjoint runs) for the given seeds, so that
several computers together train every setting exactly once:

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python train_split.py PART OF SEED [SEED ...] --data <cotton_disease.yaml> --src <data_srcdisjoint.yaml>

    one computer, seeds 1 and 2:      1 1 1 2
    three computers, seeds 1 and 2:   computer 1: 1 3 1 2     computer 2: 2 3 1 2     computer 3: 3 3 1 2
    two computers, seed 1 only:       computer 1: 1 2 1       computer 2: 2 2 1

Start the same command in TWO windows on each computer: the second window skips the seed the first one is
training and takes the next one (run_experiment.py keeps track of this).  Finished seeds are skipped and
unfinished ones are redone, so after a crash or a restart just start it again.  Every setting is trained
with the arguments listed in docs/REPRODUCE.md; only the split over computers is new.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (name, --arch, --loss, extra arguments, uses the source-disjoint data); the ADown control of Table 7
# (ctrl_adown: --arch ctrl_S1_ADown_A0 --loss ciou) was trained separately with run_experiment.py
SETTINGS = [
    # most important first, so that the main comparisons are available early
    ("fac_S1D1A1I1", "S1_D1_A1", "adaptive", [], False),        # SCD-YOLO
    ("fac_S0D0A0I0", "S0_D0_A0", "ciou", [], False),            # YOLOv13n
    ("src_scd", "S1_D1_A1", "adaptive", [], True),              # source-disjoint (Section 3.6)
    ("src_base", "S0_D0_A0", "ciou", [], True),
    ("fac_S1D0A0I0", "S1_D0_A0", "ciou", [], False),            # factorial (Tables 4, 5)
    ("fac_S0D1A0I0", "S0_D1_A0", "ciou", [], False),
    ("fac_S0D0A1I0", "S0_D0_A1", "ciou", [], False),
    ("fac_S0D0A0I1", "S0_D0_A0", "adaptive", [], False),
    ("fac_S1D1A0I0", "S1_D1_A0", "ciou", [], False),
    ("fac_S1D1A1I0", "S1_D1_A1", "ciou", [], False),
    ("fac_S1D1A0I1", "S1_D1_A0", "adaptive", [], False),
    ("fac_S1D0A1I0", "S1_D0_A1", "ciou", [], False),
    ("fac_S1D0A0I1", "S1_D0_A0", "adaptive", [], False),
    ("fac_S0D1A1I0", "S0_D1_A1", "ciou", [], False),
    ("fac_S0D1A0I1", "S0_D1_A0", "adaptive", [], False),
    ("fac_S0D0A1I1", "S0_D0_A1", "adaptive", [], False),
    ("fac_S1D0A1I1", "S1_D0_A1", "adaptive", [], False),
    ("fac_S0D1A1I1", "S0_D1_A1", "adaptive", [], False),
    ("uc_uniform0.75", "S1_D1_A1", "uniform", ["--u", "0.75"], False),   # upper-bound settings (Section 3.7)
    ("uc_uniform0.80", "S1_D1_A1", "uniform", ["--u", "0.80"], False),
    ("uc_uniform0.85", "S1_D1_A1", "uniform", ["--u", "0.85"], False),
    ("uc_uniform0.90", "S1_D1_A1", "uniform", ["--u", "0.90"], False),
    ("uc_fixedfinal", "S1_D1_A1", "fixed", ["--uc", "@fixed_final"], False),
    ("uc_replay", "S1_D1_A1", "schedule", ["--schedule", r"schedules\replay_mean.csv"], False),
    ("uc_linear", "S1_D1_A1", "schedule", ["--schedule", r"schedules\linear_ramp.csv"], False),
    ("gate_ungated_full", "ctrl_ungated_D1_A1", "adaptive", [], False),  # gate controls (Section 3.3.1)
    ("gate_ungated_only", "ctrl_ungated_D0_A0", "ciou", [], False),
]


def share(part, of):
    """round robin over SETTINGS: setting i goes to part (i mod of) + 1"""
    return [s for i, s in enumerate(SETTINGS) if i % of == part - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", type=int)
    ap.add_argument("of", type=int)
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True, help="data_srcdisjoint.yaml")
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--list", action="store_true", help="only print the share of every computer")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if not (1 <= a.part <= a.of):
        raise SystemExit(f"PART must be between 1 and OF (got {a.part} of {a.of})")
    if a.list:
        for k in range(1, a.of + 1):
            print(f"computer {k}: " + ", ".join(s[0] for s in share(k, a.of)))
        return

    mine = share(a.part, a.of)
    seeds = [str(s) for s in a.seeds]
    print(f"this computer: part {a.part} of {a.of}, seeds {' '.join(seeds)}, {len(mine)} settings "
          f"= {len(mine) * len(seeds)} trainings:")
    print("   " + ", ".join(s[0] for s in mine))
    skip_file = Path(a.project) / "SKIP.txt"
    if skip_file.exists():
        print(f"   note: {skip_file} exists - settings listed there are skipped on this computer")
    failed = []
    for name, arch, loss, extra, src in mine:
        data = a.src if src else a.data
        if src and not Path(data).exists():
            print(f"!! {name}: {data} not found - copy the folder cotton_srcdisjoint to this computer; skipped")
            failed.append(name)
            continue
        args = list(extra)
        if "@fixed_final" in args:
            f = Path("schedules") / "fixed_final.txt"
            if not f.exists():
                print(f"!! {name}: {f} not found - copy the schedules folder into the source tree; skipped")
                failed.append(name)
                continue
            args[args.index("@fixed_final")] = f.read_text(encoding="utf-8").strip()
        if "--schedule" in args and not Path(args[args.index("--schedule") + 1]).exists():
            print(f"!! {name}: {args[args.index('--schedule') + 1]} not found - copy the schedules folder; skipped")
            failed.append(name)
            continue
        cmd = [sys.executable, str(HERE / "run_experiment.py"), "--arch", arch, "--loss", loss, "--name", name,
               "--data", str(data), "--seeds", *seeds, "--parallel", "1", "--project", a.project] + args
        print(f"\n[{time.strftime('%m-%d %H:%M')}] ===== {name} seeds {' '.join(seeds)} =====", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failed.append(name)
    print(f"\n[{time.strftime('%m-%d %H:%M')}] ===== share {a.part} of {a.of} finished =====")
    if failed:
        print("   with problems: " + ", ".join(failed) + "  (see runs_revision\\<name>_seed<s>.log)")


if __name__ == "__main__":
    main()
