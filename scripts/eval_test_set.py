"""
Evaluate one or more checkpoints on a given split, using the exact protocol
reported in the paper.

    python scripts/eval_test_set.py \
        --data cfg/cotton_disease.yaml \
        --split test \
        --weights weights/yolov13n_cotton.pt weights/scd_yolo_cotton.pt

The cross-dataset experiments (Tables 13 and 14; --split val and --split test)
use the same command with a different --data / --weights pair.

IMPORTANT: rect=False is set explicitly. Ultralytics defaults to rectangular
inference for validation, which shifts mAP by a few tenths of a point; every
number in the paper was measured with it disabled.
"""

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset YAML")
    ap.add_argument("--weights", required=True, nargs="+", help="one or more .pt files")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None, help="output CSV (default: results_<split>.csv)")
    return ap.parse_args()


def main():
    args = parse_args()
    val_kw = dict(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        workers=0, rect=False, conf=0.001, iou=0.7, device=args.device,
        plots=False, verbose=False,
    )

    rows = []
    print(f"{'checkpoint':<28}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for w in args.weights:
        model = None
        try:
            if not Path(w).exists():
                raise FileNotFoundError("checkpoint not found")
            model = YOLO(w).val(**val_kw)
            b = model.box
            per_class = {
                model.names[int(c)]: round(float(b.maps[int(c)]), 3)
                for c in b.ap_class_index
            }
            rows.append([
                Path(w).stem,
                round(float(b.mp), 4), round(float(b.mr), 4),
                round(float(b.map50), 4), round(float(b.map), 4),
                per_class,
            ])
            print(f"{Path(w).stem:<28}{b.mp:>8.3f}{b.mr:>8.3f}"
                  f"{b.map50:>9.3f}{b.map:>10.3f}")
        except Exception as e:
            msg = str(e).splitlines()[0][:70]
            rows.append([Path(w).stem, "ERROR", "", "", "", msg])
            print(f"{Path(w).stem:<28}  skipped: {msg}")
        finally:
            del model
            torch.cuda.empty_cache()

    out = args.out or f"results_{args.split}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "P", "R", "mAP50", "mAP50-95", "per_class"])
        writer.writerows(rows)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
