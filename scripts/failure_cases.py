"""
Automatic, fully specified failure-case classification for Section 3.8 / Table 18
(Supplementary Note S2).  Every rule below is applied by code; no manual relabelling is involved.

    python failure_cases.py --data cfg/cotton_disease.yaml --split test \
        --weights runs/scd_yolo_adaptive_seed0/weights/best.pt --out failures_scd_seed0

Procedure
---------
1. Detections: imgsz 640, NMS IoU 0.7, confidence >= 0.25 (operating point of Table 18).
2. Per-image mAP@0.5 (classes present in that image's ground truth, Ultralytics AP definition).
   An image is a FAILURE if its per-image mAP@0.5 < 0.7.  Images without ground truth are
   reported separately (they cannot have an mAP).
3. Instance-level errors (thresholds t_f = 0.5, t_b = 0.1, after the TIDE taxonomy of Bolya et
   al., 2020).  Detections are matched greedily in descending confidence to the unmatched
   same-class ground truth of highest IoU >= t_f (true positives).  Every unmatched detection is
   then given exactly ONE type, checked in this order:
     BKG  background false positive : IoU < t_b with every ground-truth box (any class)
     LOC  localization error        : a same-class ground truth with IoU >= t_b exists
                                      (t_b <= IoU < t_f, or IoU >= t_f with an already matched
                                      ground truth, i.e. a duplicate box)
     CLS  misclassification         : otherwise (overlaps a ground truth of another class with
                                      IoU >= t_b, but no same-class one)
   Every unmatched ground truth that is not overlapped (IoU >= t_b) by any LOC or CLS detection
   is a MISS (missed detection); it is additionally flagged SMALL if its area in the 640x640
   network input is < 32^2 px.
4. Dominant type of a failure image (one image -> exactly one category, so categories are
   mutually exclusive at image level although an image may contain several error types):
   the type whose ORACLE correction gives the largest increase of that image's mAP@0.5
     LOC fixed: the detection's box and class are set to its ground truth (duplicates removed)
     CLS fixed: the detection's class (and box) are set to the overlapped ground truth
     BKG fixed: background detections removed
     MISS fixed: a perfect detection (confidence 1) added for each missed ground truth
   Ties in the gain are broken by the number of errors of each type, then by the fixed order
   MISS > BKG > LOC > CLS (change with --priority).  The number of tie-broken images is reported.
5. Table 18 columns: images per type, share of failures, share of the test set, mAP@0.5 of the
   subset of images assigned to the type (same detections), and the indicative loss
   contribution -(1 - subset mAP@0.5) x (subset size / test-set size).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apcore  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TYPES = ["LOC", "BKG", "MISS", "CLS"]
LONG = {"LOC": "Blurred lesion boundaries (localization offset)", "BKG": "False positives on background",
        "MISS": "Missed detections of small or occluded targets", "CLS": "Misclassification of similar diseases"}


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    for k in range(len(parts) - 1, -1, -1):
        if parts[k] == "images":
            parts[k] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_gt(img, w, h):
    lp = label_path(img)
    cls, box = [], []
    if lp.exists():
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) >= 5:
                c, x, y, bw, bh = int(float(v[0])), *map(float, v[1:5])
                cls.append(c)
                box.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(cls, int), np.array(box, float).reshape(-1, 4)


def image_map50(pb, pc, pf, gb, gc):
    if len(gc) == 0:
        return np.nan
    iou = apcore.box_iou(gb, pb)
    tp = apcore.match_predictions(pc, gc, iou)
    res = apcore.ap_per_class(tp.astype(float), pf, pc, gc)
    return float(res["ap"][:, 0].mean())


def classify(pb, pc, pf, gb, gc, tf=0.5, tb=0.1):
    """Returns per-detection type list (None for TP) and per-GT status."""
    n, m = len(pc), len(gc)
    iou = apcore.box_iou(pb, gb) if n and m else np.zeros((n, m))
    order = np.argsort(-pf, kind="stable")
    gt_used = np.zeros(m, bool)
    dtype = [None] * n
    match = [-1] * n
    for i in order:
        best, bj = tf, -1
        for j in range(m):
            if not gt_used[j] and gc[j] == pc[i] and iou[i, j] >= best:
                best, bj = iou[i, j], j
        if bj >= 0:
            gt_used[bj] = True
            match[i] = bj
    for i in range(n):
        if match[i] >= 0:
            continue
        if m == 0 or iou[i].max() < tb:
            dtype[i] = "BKG"
            continue
        same = [j for j in range(m) if gc[j] == pc[i]]
        if same and iou[i, same].max() >= tb:
            dtype[i] = "LOC"
            match[i] = same[int(np.argmax(iou[i, same]))]
        else:
            other = [j for j in range(m) if gc[j] != pc[i]]
            dtype[i] = "CLS"
            match[i] = other[int(np.argmax(iou[i, other]))]
    covered = np.zeros(m, bool)
    for i in range(n):
        if dtype[i] in ("LOC", "CLS") and match[i] >= 0:
            covered[match[i]] = True
    missed = [j for j in range(m) if not gt_used[j] and not covered[j]]
    return dtype, match, missed


def oracle_gain(pb, pc, pf, gb, gc, dtype, match, missed, base):
    gains = {}
    pb, pc, pf = pb.copy(), pc.copy(), pf.copy()
    for t in ("LOC", "CLS", "BKG"):
        keep = np.ones(len(pc), bool)
        b2, c2 = pb.copy(), pc.copy()
        seen = set()
        for i, ty in enumerate(dtype):
            if ty != t:
                continue
            if t == "BKG":
                keep[i] = False
            else:
                j = match[i]
                if j in seen:          # duplicate fix: drop the extra box
                    keep[i] = False
                b2[i], c2[i] = gb[j], gc[j]
                seen.add(j)
        gains[t] = image_map50(b2[keep], c2[keep], pf[keep], gb, gc) - base
    if missed:
        b3 = np.concatenate([pb, gb[missed]])
        c3 = np.concatenate([pc, gc[missed]])
        f3 = np.concatenate([pf, np.ones(len(missed))])
        gains["MISS"] = image_map50(b3, c3, f3, gb, gc) - base
    else:
        gains["MISS"] = 0.0
    return gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--fail-thr", type=float, default=0.7)
    ap.add_argument("--priority", default="MISS,BKG,LOC,CLS")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="failures")
    a = ap.parse_args()
    prio = a.priority.split(",")

    d = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    root = Path(d.get("path", Path(a.data).parent))
    imgs = sorted(p for p in (root / d[a.split]).rglob("*") if p.suffix.lower() in IMG)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows, all_det = [], []
    for k in range(0, len(imgs), 16):
        batch = imgs[k:k + 16]
        res = model.predict([str(p) for p in batch], imgsz=640, conf=a.conf, iou=a.iou, max_det=300,
                            device=a.device, verbose=False, rect=False)
        for p, r in zip(batch, res):
            h, w = r.orig_shape
            gc, gb = read_gt(p, w, h)
            bx = r.boxes
            pb = bx.xyxy.cpu().numpy().astype(float).reshape(-1, 4)
            pc = bx.cls.cpu().numpy().astype(int)
            pf = bx.conf.cpu().numpy().astype(float)
            all_det.append((p.name, pb, pc, pf, gb, gc))
            m50 = image_map50(pb, pc, pf, gb, gc)
            dtype, match, missed = classify(pb, pc, pf, gb, gc)
            cnt = {t: sum(1 for x in dtype if x == t) for t in ("LOC", "BKG", "CLS")}
            cnt["MISS"] = len(missed)
            scale = 640.0 / max(w, h)
            small = sum(1 for j in missed if (gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1]) * scale ** 2 < 32 ** 2)
            row = dict(image=p.name, n_gt=len(gc), n_det=len(pc), map50=m50,
                       **{f"n_{t}": cnt[t] for t in TYPES}, missed_small=small,
                       failure=bool(len(gc) and m50 < a.fail_thr))
            if row["failure"]:
                g = oracle_gain(pb, pc, pf, gb, gc, dtype, match, missed, m50)
                cand = [t for t in TYPES if cnt[t] > 0] or TYPES
                key = lambda t: (round(g[t], 9), cnt[t], -prio.index(t))
                best = max(cand, key=key)
                tie = sum(1 for t in cand if round(g[t], 9) == round(g[best], 9)) > 1
                row.update({f"gain_{t}": g[t] for t in TYPES}, dominant=best, tie_broken=tie)
            rows.append(row)

    fields = ["image", "n_gt", "n_det", "map50", "failure", "dominant", "tie_broken"] + \
             [f"n_{t}" for t in TYPES] + ["missed_small"] + [f"gain_{t}" for t in TYPES]
    with open(f"{a.out}_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_test = len(rows)
    fails = [r for r in rows if r["failure"]]
    det_by = {x[0]: x for x in all_det}

    def subset_map(names):
        tp, cf, pcl, tcl = [], [], [], []
        for nm in names:
            _, pb, pc, pf, gb, gc = det_by[nm]
            iou = apcore.box_iou(gb, pb)
            tp.append(apcore.match_predictions(pc, gc, iou).astype(float))
            cf.append(pf); pcl.append(pc); tcl.append(gc)
        res = apcore.ap_per_class(np.concatenate(tp) if tp else np.zeros((0, 10)), np.concatenate(cf),
                                  np.concatenate(pcl), np.concatenate(tcl))
        return float(res["ap"][:, 0].mean()) if res["ap"].size else float("nan")

    lines = [f"Failure images (per-image mAP@0.5 < {a.fail_thr}, conf >= {a.conf}): {len(fails)} of {n_test} "
             f"({100 * len(fails) / n_test:.1f}%); tie-broken assignments: {sum(r.get('tie_broken', False) for r in fails)}",
             "", "| Failure type | Images | % of failed | % of test set | Subset mAP@0.5 | mAP@0.5 loss contribution | Error instances (all images) |",
             "|---|---|---|---|---|---|---|"]
    tot_loss = 0.0
    for t in ["LOC", "BKG", "MISS", "CLS"]:
        sub = [r["image"] for r in fails if r.get("dominant") == t]
        sm = subset_map(sub) if sub else float("nan")
        loss = -(1 - sm) * len(sub) / n_test if sub else 0.0
        tot_loss += loss
        inst = sum(r[f"n_{t}"] for r in rows)
        lines.append(f"| {LONG[t]} | {len(sub)} | {100 * len(sub) / max(1, len(fails)):.1f}% | {100 * len(sub) / n_test:.1f}% | "
                     f"{sm:.3f} | {loss:.3f} | {inst} |")
    lines.append(f"| Total | {len(fails)} | 100% | {100 * len(fails) / n_test:.1f}% | - | {tot_loss:.3f} | - |")
    ms = sum(r["missed_small"] for r in rows)
    mi = sum(r["n_MISS"] for r in rows)
    lines += ["", f"Missed ground truths: {mi}, of which small (< 32^2 px at 640 input): {ms} ({100 * ms / max(1, mi):.1f}%)"]
    Path(f"{a.out}_table17.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
