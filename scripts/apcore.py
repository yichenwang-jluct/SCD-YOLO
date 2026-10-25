"""
NumPy re-implementation of Ultralytics' detection metrics (ultralytics/utils/metrics.py:
ap_per_class / compute_ap / smooth) so that resampled metrics are computed exactly the way the
paper's numbers were computed.  No torch / ultralytics import needed.

Inputs follow Ultralytics' validator statistics:
    tp         (N_pred, 10) bool   correct at IoU 0.50:0.05:0.95
    conf       (N_pred,)    float
    pred_cls   (N_pred,)    int
    target_cls (N_gt,)      int
"""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)                      # 101-point interpolation (COCO), as Ultralytics
    return _trapz(np.interp(x, mrec, mpre), x)


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Returns dict(classes, ap (nc_present, 10), p, r) exactly like Ultralytics (P/R at max mean-F1)."""
    i = np.argsort(-conf, kind="stable")
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve, r_curve = np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        m = pred_cls == c
        n_l, n_p = nt[ci], m.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[m]).cumsum(0)
        tpc = tp[m].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[m], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[m], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    k = smooth(f1.mean(0), 0.1).argmax() if nc else 0
    return dict(classes=unique_classes.astype(int), ap=ap, p=p_curve[:, k], r=r_curve[:, k], nt=nt)


def summarize(res):
    """Overall metrics as reported by Ultralytics (macro-average over classes present in the targets)."""
    ap = res["ap"]
    if ap.size == 0:
        return dict(P=np.nan, R=np.nan, mAP50=np.nan, mAP50_95=np.nan)
    return dict(P=float(res["p"].mean()), R=float(res["r"].mean()),
                mAP50=float(ap[:, 0].mean()), mAP50_95=float(ap.mean()))


def concat_images(records, idx=None):
    """Concatenate per-image records (dicts with tp, conf, pred_cls, target_cls) for the given indices."""
    recs = records if idx is None else [records[k] for k in idx]
    tp = np.concatenate([r["tp"] for r in recs], 0) if recs else np.zeros((0, 10), bool)
    conf = np.concatenate([r["conf"] for r in recs], 0) if recs else np.zeros(0)
    pc = np.concatenate([r["pred_cls"] for r in recs], 0) if recs else np.zeros(0)
    tc = np.concatenate([r["target_cls"] for r in recs], 0) if recs else np.zeros(0)
    return tp.astype(float), conf, pc, tc


def box_iou(a, b):
    """IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(pred_cls, true_cls, iou, iouv=np.linspace(0.5, 0.95, 10)):
    """Ultralytics' greedy matching (DetectionValidator.match_predictions, use_scipy=False).
    iou: (N_gt, N_pred).  Returns (N_pred, len(iouv)) bool."""
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    if len(pred_cls) == 0 or len(true_cls) == 0:
        return correct
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(iouv):
        matches = np.array(np.nonzero(iou >= t)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct
