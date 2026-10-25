#!/usr/bin/env python3
"""
Evaluate every FINISHED run in runs_revision/ with the protocol stated in the paper
(rect=False, conf=0.001, NMS IoU=0.7, imgsz=640, batch 8) and write everything the tables need.

    (from the YOLOv13 source-tree root, with PYTHONPATH = that root)
    python eval_all.py --seeds 0 --data <cotton_disease.yaml> --src-dir <cotton_srcdisjoint> --out eval
    python eval_all.py --seeds 0 1 2 ...     (runs trained on other computers copied into runs_revision/)

Files written into --out:
  per_seed_all.csv        master file, one row per run / seed / split / class (kept between calls)
  per_seed_factorial.csv  labels S?D?A?I? (16 cells)           -> factorial_analysis.py, Tables 4/5
  per_seed_uc.csv         Cotton-IoU upper-bound controls      -> Tables 16 and 17
  per_seed_gate.csv       gated vs ungated StarNet             -> Table 6, Suppl. Table S7
  per_seed_down.csv       Conv / ADown / Star-ADown downsamplers in the Cotton-StarNet model -> Table 7
  per_seed_src.csv        source-disjoint runs: val, test_field, test_public -> Section 3.6
  per_seed_mix.csv        field images in training, 2 folds (cotton_fieldmix): per fold val / test_field
                          (the half the fold never saw) / test_public, and <model>_pooled on test_field_427
                          = both folds' held-out halves together, i.e. all 427 field images (not used in the paper)
  per_seed_orig.csv       runs of the original experiments kept in runs/V13train (consistency check), same protocol
  stats/<run>_s<seed>_<split>.pkl   per-image statistics for bootstrap_ci.py
  first_look.txt          mAP50 / mAP50-95 of every run and seed, training hours

The field-mix folder is taken from --mix-dir, or else <folder of --src-dir>/cotton_fieldmix when it exists.

A run counts as finished when weights/best.pt exists and results.png exists or results.csv holds
all epochs; anything else (still training, missing) is listed and skipped.  An evaluation already
in per_seed_all.csv for the same checkpoint file (same size and time stamp) is not repeated, so the
command can be started again at any time.  It only reads the run folders.
"""
import argparse
import csv
import gc
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FACTORIAL = [f"S{s}D{d}A{a}I{i}" for s in (0, 1) for d in (0, 1) for a in (0, 1) for i in (0, 1)]
UC = [("CIoU", "fac_S1D1A1I0"), ("Uniform0.75", "uc_uniform0.75"), ("Uniform0.80", "uc_uniform0.80"),
      ("Uniform0.85", "uc_uniform0.85"), ("Uniform0.90", "uc_uniform0.90"), ("LinearRamp", "uc_linear"),
      ("FixedFinal", "uc_fixedfinal"), ("Replay", "uc_replay"), ("Adaptive", "fac_S1D1A1I1"),
      ("Uniform0.95", "uc_uniform0.95"), ("Manual", "uc_manual")]   # extra runs (train_extra.py), Tables 16/17
GATE = [("Gated_full", "fac_S1D1A1I1"), ("Ungated_full", "gate_ungated_full"),
        ("Gated_StarNet_only", "fac_S1D0A0I0"), ("Ungated_StarNet_only", "gate_ungated_only")]
SRC = [("YOLOv13n", "src_base"), ("SCD-YOLO", "src_scd")]
# field images in training: two-fold cross-validation over the 427 field images (make_field_mix_split.py)
MIX = [("YOLOv13n", "mix_base"), ("SCD-YOLO", "mix_scd")]
MIX_RUNS = [f"{base}_f{k}" for _, base in MIX for k in (1, 2)]
POOLED = [f"{base}_pooled" for _, base in MIX]
# Table 7: the two neck downsamplers in the Cotton-StarNet model, CIoU, no SimAM
DOWN = [("Conv", "fac_S1D0A0I0"), ("ADown", "ctrl_adown"), ("Star-ADown", "fac_S1D1A0I0")]
# original seed-0 runs of the manuscript (folder names in runs/V13train); evaluated once, seed 0
ORIG = [("orig_S0D0A0I0", "exp v13n"), ("orig_S1D0A0I0", "（消融）CottonStarNet"),
        ("orig_S0D1A0I0", "（消融）star-adown"), ("orig_S0D0A1I0", "（消融）COTTON-SIMAM"),
        ("orig_S0D0A0I1", "新损失（单独损失函数）"), ("orig_S1D1A0I0", "（消融）COTTON-STARTNET+STAR-ADOWN"),
        ("orig_S1D1A1I0", "（消融）3in1"), ("orig_S1D1A1I1", "新损失4in1")]
MAIN_RUNS = ([f"fac_{c}" for c in FACTORIAL] + [r for _, r in UC if not r.startswith("fac_")]
             + [r for _, r in GATE if not r.startswith("fac_")] + [r for _, r in DOWN if not r.startswith("fac_")])
SRC_RUNS = [r for _, r in SRC]
# per-image statistics for the bootstrap: run -> splits
STATS = {"fac_S0D0A0I0": ("val", "test"), "fac_S1D1A1I1": ("val", "test"),
         "src_base": ("test_field",), "src_scd": ("test_field",),
         **{run: ("test_field",) for run in MIX_RUNS}}
FIELDS = ["label", "seed", "split", "class", "P", "R", "AP50", "AP50_95", "checkpoint", "ckpt_sig", "evaluated"]
OUT_FIELDS = FIELDS[:9]


def run_finished(d):
    """(finished?, note, training hours or None)"""
    best = d / "weights" / "best.pt"
    hours, rows, epochs = None, 0, 200
    res = d / "results.csv"
    if res.exists():
        try:
            with open(res, newline="", encoding="utf-8", errors="replace") as f:
                data = [r for r in csv.reader(f) if r]
            rows = max(0, len(data) - 1)
            head = [h.strip().lower() for h in data[0]]
            if "time" in head and rows:
                hours = float(data[-1][head.index("time")]) / 3600
        except Exception:
            pass
    try:
        import yaml
        epochs = int(yaml.safe_load((d / "args.yaml").read_text(encoding="utf-8", errors="replace"))["epochs"])
    except Exception:
        pass
    if not d.exists():
        return False, "not started / not copied here", hours
    if not best.exists():
        return False, f"no weights/best.pt ({rows}/{epochs} epochs)", hours
    if (d / "results.png").exists() or rows >= epochs:
        return True, "", hours
    return False, f"still training or stopped ({rows}/{epochs} epochs)", hours


def sig(p):
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate(ckpt, data, split, a, tmp):
    from ultralytics import YOLO
    model = YOLO(str(ckpt))
    m = model.val(data=str(data), split=split, imgsz=640, batch=a.batch, workers=0, rect=False, conf=0.001,
                  iou=0.7, device=a.device, plots=False, verbose=False, project=str(tmp), name="val", exist_ok=True)
    b = m.box
    names = getattr(m, "names", None) or model.names
    rows = [("all", b.mp, b.mr, b.map50, b.map)]
    for j, c in enumerate(b.ap_class_index):
        rows.append((names[int(c)], b.p[j], b.r[j], b.ap50[j], b.ap[j]))
    del model, m
    free_gpu()
    return rows


def collect_stats(ckpt, data, split, out, a, tmp):
    import apcore
    import collect_val_stats as cvs
    from ultralytics import YOLO
    model = YOLO(str(ckpt))
    V = cvs.make_validator_class()
    args = {**model.overrides, "data": str(data), "split": split, "imgsz": 640, "batch": a.batch, "workers": 0,
            "conf": 0.001, "iou": 0.7, "device": a.device, "plots": False, "verbose": False, "mode": "val",
            "rect": False, "project": str(tmp), "name": "stats", "exist_ok": True}
    v = V(args=args, _callbacks=model.callbacks)
    v(model=model.model)
    recs = v.captured or []
    names = v.names if hasattr(v, "names") else model.names
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    res = apcore.summarize(apcore.ap_per_class(*apcore.concat_images(recs)))
    box = v.metrics.box
    ok = abs(res["mAP50"] - box.map50) < 1e-4 and abs(res["mAP50_95"] - box.map) < 1e-4
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(dict(records=recs, names=names, weights=str(ckpt), split=split, data=str(data),
                             validator=dict(map50=float(box.map50), map=float(box.map),
                                            mp=float(box.mp), mr=float(box.mr))), f)
        out.with_suffix(".sig").write_text(sig(ckpt), encoding="utf-8")
    del model, v
    free_gpu()
    return ok, len(recs), float(box.map50), float(res["mAP50"])


def read_master(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def latest(rows):
    """keep the last row of every (label, seed, split, class)"""
    d = {}
    for r in rows:
        d[(r["label"], str(r["seed"]), r["split"], r["class"])] = r
    return list(d.values())


def write_group(path, rows, mapping, fixed_seed=None):
    out = []
    for r in rows:
        for lab in mapping.get(r["label"], []):
            if fixed_seed is not None and str(r["seed"]) != str(fixed_seed):
                continue
            out.append({**{k: r[k] for k in OUT_FIELDS}, "label": lab})
    if not out:
        return 0
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(out)
    return len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="cotton_disease.yaml (train/val/test of the paper)")
    ap.add_argument("--src-dir", default=None, help="folder made by make_source_disjoint_split.py")
    ap.add_argument("--mix-dir", default=None, help="folder made by make_field_mix_split.py "
                                                    "(default: cotton_fieldmix next to --src-dir, if it exists)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--orig-dir", default=str(Path("runs") / "V13train"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-stats", action="store_true", help="skip the per-image statistics for the bootstrap")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_ultralytics_tmp"
    master = out / "per_seed_all.csv"
    proj = Path(a.project)
    data = Path(a.data)
    src_yaml = src_pub = None
    if a.src_dir:
        src_yaml = Path(a.src_dir) / "data_srcdisjoint.yaml"
        src_pub = Path(a.src_dir) / "data_srcdisjoint_public.yaml"
        if not (src_yaml.exists() and src_pub.exists()):
            print(f"!! {a.src_dir} has no data_srcdisjoint*.yaml - the src runs are skipped")
            src_yaml = src_pub = None
    mix_dir = Path(a.mix_dir) if a.mix_dir else (Path(a.src_dir).parent / "cotton_fieldmix" if a.src_dir else None)
    mix_yaml = {}
    if mix_dir is not None:
        cand = {k: (mix_dir / f"fold{k}" / "data_fieldmix.yaml", mix_dir / f"fold{k}" / "data_fieldmix_public.yaml")
                for k in (1, 2)}
        if all(p.exists() for pair in cand.values() for p in pair):
            mix_yaml = cand
        elif a.mix_dir or mix_dir.exists():
            print(f"!! {mix_dir} has no fold1/fold2 data_fieldmix*.yaml - the mix runs are skipped")

    # ---------------------------------------------------------------- what to evaluate
    run_splits = {}  # run -> [(split_name, data_yaml, ultralytics_split)]
    for run in MAIN_RUNS:
        run_splits[run] = [("val", data, "val"), ("test", data, "test")]
    if src_yaml:
        for run in SRC_RUNS:
            run_splits[run] = [("val", src_yaml, "val"), ("test_field", src_yaml, "test"), ("test_public", src_pub, "test")]
    if mix_yaml:
        for run in MIX_RUNS:
            y, yp = mix_yaml[int(run[-1])]
            run_splits[run] = [("val", y, "val"), ("test_field", y, "test"), ("test_public", yp, "test")]
    jobs = []        # (label, seed, run_dir, [(split_name, data_yaml, ultralytics_split)])
    for group in (MAIN_RUNS, SRC_RUNS, MIX_RUNS):
        for s in a.seeds:
            for run in group:
                if run in run_splits:
                    jobs.append((run, s, proj / f"{run}_seed{s}", run_splits[run]))
    for lab, folder in ORIG:
        d = Path(a.orig_dir) / folder
        if d.exists():
            jobs.append((lab, 0, d, [("val", data, "val"), ("test", data, "test")]))

    have = {}
    for r in read_master(master):
        if r["class"] == "all":
            have[(r["label"], str(r["seed"]), r["split"])] = r.get("ckpt_sig", "")

    new_file = not master.exists()
    f = open(master, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
        f.flush()

    not_ready, hours = [], {}
    n_eval = n_skip = 0
    t0 = time.time()
    for label, seed, d, splits in jobs:
        ok, note, h = run_finished(d)
        hours[(label, seed)] = h
        if not ok:
            if not label.startswith("orig_"):
                not_ready.append(f"{d.name}: {note}")
            continue
        ckpt = d / "weights" / "best.pt"
        cs = sig(ckpt)
        for split_name, yml, usplit in splits:
            if have.get((label, str(seed), split_name)) == cs:
                n_skip += 1
                continue
            try:
                rows = evaluate(ckpt, yml, usplit, a, tmp)
            except Exception as e:  # keep going with the other runs
                print(f"!! {d.name} {split_name}: evaluation failed: {type(e).__name__}: {e}")
                free_gpu()
                continue
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for cls, P, R, A50, A in rows:
                w.writerow(dict(label=label, seed=seed, split=split_name, **{"class": cls}, P=f"{float(P):.6f}",
                                R=f"{float(R):.6f}", AP50=f"{float(A50):.6f}", AP50_95=f"{float(A):.6f}",
                                checkpoint=str(ckpt), ckpt_sig=cs, evaluated=stamp))
            f.flush()
            have[(label, str(seed), split_name)] = cs
            n_eval += 1
            print(f"[eval] {label:<22} seed {seed} {split_name:<11} mAP50 {float(rows[0][3]):.3f}  "
                  f"mAP50-95 {float(rows[0][4]):.3f}   ({(time.time() - t0) / 60:.0f} min)", flush=True)
    f.close()

    # ---------------------------------------------------------------- per-image statistics
    stats_msgs = []
    if not a.no_stats:
        for s in a.seeds:
            for run, splits in STATS.items():
                if run not in run_splits:          # src / mix data not on this computer
                    continue
                d = proj / f"{run}_seed{s}"
                ok, _, _ = run_finished(d)
                if not ok:
                    continue
                srcmap = {sn: (y, us) for sn, y, us in run_splits[run]}
                ckpt = d / "weights" / "best.pt"
                for split_name in splits:
                    pkl = out / "stats" / f"{run}_s{s}_{split_name}.pkl"
                    sg = pkl.with_suffix(".sig")
                    if pkl.exists() and sg.exists() and sg.read_text(encoding="utf-8").strip() == sig(ckpt):
                        continue
                    yml, usplit = srcmap[split_name]
                    try:
                        good, n, m50, r50 = collect_stats(ckpt, yml, usplit, pkl, a, tmp)
                    except Exception as e:
                        stats_msgs.append(f"!! stats {pkl.name}: {type(e).__name__}: {e}")
                        free_gpu()
                        continue
                    msg = (f"[stats] {pkl.name}: {n} images, mAP50 validator {m50:.5f} / recomputed {r50:.5f} -> "
                           f"{'saved' if good else 'MISMATCH, not saved'}")
                    print(msg, flush=True)
                    stats_msgs.append(msg)

    # ---------------------------------------------------------------- field mix: both held-out halves pooled
    # every field image was tested by the fold model that did NOT train on it; pooling the per-image
    # statistics of fold 1 (half 1) and fold 2 (half 2) gives one AP over all 427 field images
    if mix_yaml and not a.no_stats:
        import apcore
        fpool = open(master, "a", newline="", encoding="utf-8-sig")
        wpool = csv.DictWriter(fpool, fieldnames=FIELDS)
        for s in a.seeds:
            for _, base in MIX:
                parts = []
                for k in (1, 2):
                    d = proj / f"{base}_f{k}_seed{s}"
                    ck = d / "weights" / "best.pt"
                    pkl = out / "stats" / f"{base}_f{k}_s{s}_test_field.pkl"
                    sg = pkl.with_suffix(".sig")
                    if not (run_finished(d)[0] and pkl.exists() and sg.exists()
                            and sg.read_text(encoding="utf-8").strip() == sig(ck)):
                        parts = None
                        break
                    parts.append((pkl, ck))
                if not parts:
                    continue
                label = f"{base}_pooled"
                combined = "|".join(sig(ck) for _, ck in parts)
                if have.get((label, str(s), "test_field_427")) == combined:
                    continue
                recs, names, n_each = [], None, []
                for pkl, _ in parts:
                    with open(pkl, "rb") as fh:
                        dd = pickle.load(fh)
                    recs += dd["records"]
                    names = dd["names"]
                    n_each.append(len(dd["records"]))
                files = [Path(str(r.get("im_file", ""))).name for r in recs]
                if len(set(files)) != len(files):
                    msg = f"!! pooled {label} seed {s}: an image appears in both folds - not written"
                    print(msg)
                    stats_msgs.append(msg)
                    continue
                res = apcore.ap_per_class(*apcore.concat_images(recs))
                summ = apcore.summarize(res)
                prow = [("all", summ["P"], summ["R"], summ["mAP50"], summ["mAP50_95"])]
                for j, c in enumerate(res["classes"]):
                    prow.append((names[int(c)], res["p"][j], res["r"][j], res["ap"][j, 0], res["ap"][j].mean()))
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for cls, P, R, A50, A in prow:
                    wpool.writerow(dict(label=label, seed=s, split="test_field_427", **{"class": cls},
                                        P=f"{float(P):.6f}", R=f"{float(R):.6f}", AP50=f"{float(A50):.6f}",
                                        AP50_95=f"{float(A):.6f}", checkpoint=" + ".join(str(ck) for _, ck in parts),
                                        ckpt_sig=combined, evaluated=stamp))
                fpool.flush()
                have[(label, str(s), "test_field_427")] = combined
                msg = (f"[pool]  {label} seed {s}: {n_each[0]} + {n_each[1]} = {len(recs)} field images -> "
                       f"mAP50 {summ['mAP50']:.3f}  mAP50-95 {summ['mAP50_95']:.3f}")
                print(msg, flush=True)
                stats_msgs.append(msg)
        fpool.close()

    # ---------------------------------------------------------------- group files + first look
    rows = latest(read_master(master))
    fac = {f"fac_{c}": [c] for c in FACTORIAL}
    uc, gate, down = {}, {}, {}
    for lab, run in DOWN:
        down.setdefault(run, []).append(lab)
    for lab, run in UC:
        uc.setdefault(run, []).append(lab)
    for lab, run in GATE:
        gate.setdefault(run, []).append(lab)
    src = {run: [lab] for lab, run in SRC}
    mixg = {}
    for lab, base in MIX:
        for k in (1, 2):
            mixg[f"{base}_f{k}"] = [f"{lab}_fold{k}"]
        mixg[f"{base}_pooled"] = [f"{lab}_pooled"]
    orig = {lab: [lab[5:]] for lab, _ in ORIG}
    counts = {name: write_group(out / f"per_seed_{name}.csv", rows, m)
              for name, m in (("factorial", fac), ("uc", uc), ("gate", gate), ("down", down), ("src", src),
                              ("mix", mixg), ("orig", orig))}

    allrows = {(r["label"], str(r["seed"]), r["split"]): r for r in rows if r["class"] == "all"}
    seeds_present = sorted({int(r["seed"]) for r in rows})
    lines = [f"first look  {time.strftime('%Y-%m-%d %H:%M')}   values: mAP50 / mAP50-95 (rect=False, conf 0.001, IoU 0.7)",
             f"{'run':<24}{'split':<16}" + "".join(f"{'seed ' + str(s):<16}" for s in seeds_present) + "mean"]
    order = MAIN_RUNS + SRC_RUNS + MIX_RUNS + POOLED + [lab for lab, _ in ORIG]
    for run in order:
        splits = (("val", "test_field", "test_public") if run in SRC_RUNS or run in MIX_RUNS
                  else ("test_field_427",) if run in POOLED else ("val", "test"))
        for sp in splits:
            vals = []
            cells = ""
            for s in seeds_present:
                r = allrows.get((run, str(s), sp))
                if r:
                    v = (float(r["AP50"]), float(r["AP50_95"]))
                    vals.append(v)
                    cells += f"{v[0]:.3f}/{v[1]:.3f}     "
                else:
                    cells += f"{'-':<16}"
            if not vals:
                continue
            mean = f"{sum(v[0] for v in vals) / len(vals):.3f}/{sum(v[1] for v in vals) / len(vals):.3f}" if len(vals) > 1 else ""
            lines.append(f"{run:<24}{sp:<16}{cells}{mean}")
    lines.append("\ntraining time per run (h, from results.csv):")
    th = [f"{lab}_seed{s} {h:.1f}" for (lab, s), h in hours.items() if h is not None and not lab.startswith("orig_")]
    for i in range(0, len(th), 4):
        lines.append("  " + "   ".join(th[i:i + 4]))
    if not_ready:
        lines.append("\nnot evaluated (not finished on this computer):")
        lines += ["  " + x for x in not_ready]
    if stats_msgs:
        lines.append("\nper-image statistics:")
        lines += ["  " + x for x in stats_msgs]
    lines.append(f"\nevaluated now {n_eval}, already done {n_skip}; rows in group files: {counts}")
    text = "\n".join(lines) + "\n"
    (out / "first_look.txt").write_text(text, encoding="utf-8-sig")
    print("\n" + text)
    print(f"all files are in {out.resolve()}")


if __name__ == "__main__":
    main()
