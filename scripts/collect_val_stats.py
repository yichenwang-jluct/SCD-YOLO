"""
Run the standard Ultralytics validation (rect=False, conf=0.001, NMS IoU=0.7, imgsz=640) and
save the PER-IMAGE matching statistics (tp, conf, pred_cls, target_cls, image file) that the
validator computes internally.  These files feed bootstrap_ci.py (Section 3.6).

    python collect_val_stats.py --data cfg/cotton_disease.yaml --split test \
        --weights runs/scd_yolo_adaptive_seed0/weights/best.pt --out stats/scd_seed0_test.pkl

The script re-computes mAP from the saved per-image statistics with apcore.py and checks that
it reproduces the validator's own numbers (printed side by side) - if they differ by more than
1e-4 something is wrong and the bootstrap must not be used.

Works with Ultralytics 8.3.x before and after the 2025 validator refactor (stats kept either in
validator.stats or in validator.metrics.stats).
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apcore  # noqa: E402


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def make_validator_class():
    from ultralytics.models.yolo.detect import DetectionValidator

    class CaptureValidator(DetectionValidator):
        captured = None

        def _stat_lists(self):
            st = getattr(getattr(self, "metrics", None), "stats", None)
            if isinstance(st, dict) and "tp" in st:
                return st
            st = getattr(self, "stats", None)
            if isinstance(st, dict) and "tp" in st:
                return st
            raise RuntimeError("per-image stats not found in this Ultralytics version")

        def update_metrics(self, preds, batch):
            if self.captured is None:
                self.captured = []
            n0 = len(self._stat_lists()["tp"])
            super().update_metrics(preds, batch)
            st = self._stat_lists()
            n1 = len(st["tp"])
            files = list(batch["im_file"])
            nb = len(files)
            if n1 - n0 == nb:
                idx = list(range(nb))
            else:  # older versions skip images that have neither predictions nor labels
                idx = []
                for si in range(nb):
                    p = preds[si]
                    npr = len(p["cls"]) if isinstance(p, dict) else len(p)
                    nl = int((batch["batch_idx"] == si).sum())
                    if npr > 0 or nl > 0:
                        idx.append(si)
                if len(idx) != n1 - n0:
                    raise RuntimeError("could not align per-image statistics with image files")
            for j, si in enumerate(idx):
                k = n0 + j
                self.captured.append(dict(
                    im_file=files[si],
                    tp=_np(st["tp"][k]).astype(bool).reshape(-1, 10),
                    conf=_np(st["conf"][k]).astype(np.float64).reshape(-1),
                    pred_cls=_np(st["pred_cls"][k]).astype(np.int64).reshape(-1),
                    target_cls=_np(st["target_cls"][k]).astype(np.int64).reshape(-1)))
            for si in sorted(set(range(nb)) - set(idx)):
                self.captured.append(dict(im_file=files[si], tp=np.zeros((0, 10), bool), conf=np.zeros(0),
                                          pred_cls=np.zeros(0, np.int64), target_cls=np.zeros(0, np.int64)))

    return CaptureValidator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    V = make_validator_class()
    args = {**model.overrides, "rect": True, "data": a.data, "split": a.split, "imgsz": a.imgsz,
            "batch": a.batch, "workers": 0, "conf": 0.001, "iou": 0.7, "device": a.device,
            "plots": False, "verbose": False, "mode": "val"}
    args["rect"] = False
    v = V(args=args, _callbacks=model.callbacks)
    v(model=model.model)
    recs = v.captured or []
    names = v.names if hasattr(v, "names") else model.names
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    res = apcore.summarize(apcore.ap_per_class(*apcore.concat_images(recs)))
    box = v.metrics.box
    print(f"validator : mAP50 {box.map50:.5f}  mAP50-95 {box.map:.5f}  P {box.mp:.5f}  R {box.mr:.5f}")
    print(f"recomputed: mAP50 {res['mAP50']:.5f}  mAP50-95 {res['mAP50_95']:.5f}  P {res['P']:.5f}  R {res['R']:.5f}")
    ok = abs(res["mAP50"] - box.map50) < 1e-4 and abs(res["mAP50_95"] - box.map) < 1e-4
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(records=recs, names=names, weights=a.weights, split=a.split, data=a.data,
                         validator=dict(map50=float(box.map50), map=float(box.map),
                                        mp=float(box.mp), mr=float(box.mr))), f)
    print(f"{len(recs)} images saved to {a.out}  ->  {'CONSISTENT' if ok else 'MISMATCH - do not use'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
