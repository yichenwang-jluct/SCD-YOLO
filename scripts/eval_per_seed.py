"""
Evaluate one checkpoint per seed on one or more splits and write a tidy per-seed, per-class
CSV (overall P, R, mAP@0.5, mAP@0.5:0.95 and class-wise P, R, AP@0.5, AP@0.5:0.95).

Used for the per-seed values behind Tables 4, 5 and 15-17 (Supplementary Tables S4 and S5), the
test splits of Tables 13 and 14 and the source-disjoint field test (Section 3.6).

Same evaluation protocol as the paper: rect=False, conf=0.001, NMS IoU=0.7, imgsz=640.

    python eval_per_seed.py --data cfg/cotton_disease.yaml --splits val test --seeds 0 1 2 \
        --runs "YOLOv13n=runs/yolov13n_baseline_seed{seed}/weights/best.pt" \
               "SCD-YOLO=runs/scd_yolo_adaptive_seed{seed}/weights/best.pt" \
        --out per_seed_main.csv

A --runs entry is LABEL=PATTERN.  PATTERN either contains {seed} or is a comma-separated list
of checkpoints given in the same order as --seeds (handy for the original run folders, whose
names do not contain the seed).
"""
import argparse
import csv
from pathlib import Path


def expand(entry, seeds):
    label, pat = entry.split("=", 1)
    if "{seed}" in pat:
        return label, [(s, pat.format(seed=s)) for s in seeds]
    paths = [p.strip() for p in pat.split(",") if p.strip()]
    if len(paths) != len(seeds):
        raise SystemExit(f"{label}: {len(paths)} checkpoints given for {len(seeds)} seeds")
    return label, list(zip(seeds, paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="per_seed_results.csv")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    fields = ["label", "seed", "split", "class", "P", "R", "AP50", "AP50_95", "checkpoint"]
    new_file = not Path(a.out).exists()
    with open(a.out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for entry in a.runs:
            label, items = expand(entry, a.seeds)
            for seed, ckpt in items:
                for split in a.splits:
                    if not Path(ckpt).exists():
                        print(f"!! missing {ckpt}")
                        continue
                    model = YOLO(ckpt)
                    m = model.val(data=a.data, split=split, imgsz=a.imgsz, batch=a.batch, workers=0,
                                  rect=False, conf=0.001, iou=0.7, device=a.device, plots=False, verbose=False)
                    b = m.box
                    names = m.names if hasattr(m, "names") else model.names
                    w.writerow(dict(label=label, seed=seed, split=split, **{"class": "all"},
                                    P=f"{b.mp:.6f}", R=f"{b.mr:.6f}", AP50=f"{b.map50:.6f}",
                                    AP50_95=f"{b.map:.6f}", checkpoint=ckpt))
                    for j, c in enumerate(b.ap_class_index):
                        w.writerow(dict(label=label, seed=seed, split=split, **{"class": names[int(c)]},
                                        P=f"{b.p[j]:.6f}", R=f"{b.r[j]:.6f}", AP50=f"{b.ap50[j]:.6f}",
                                        AP50_95=f"{b.ap[j]:.6f}", checkpoint=ckpt))
                    f.flush()
                    print(f"{label:<22} seed {seed} {split:<5}  P {b.mp:.3f}  R {b.mr:.3f}  "
                          f"mAP50 {b.map50:.3f}  mAP50-95 {b.map:.3f}")
                    del model
                    torch.cuda.empty_cache()
    print(f"\nappended to {a.out}")


if __name__ == "__main__":
    main()
