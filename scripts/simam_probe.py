"""
Quantitative probe of what Cotton-SimAM does to isolated artefacts (Section 3.3.3, Supplementary Table S8 and Note S3).
Inference only - uses trained checkpoints
(SimAM vs Cotton-SimAM at the same construction stage; optionally the no-attention control).

Two analyses
------------
(A) Controlled artefact injection (detection level).  Synthetic specular highlights, water-droplet
    spots and impulse-noise bursts are pasted into the TEST images at random positions that do not
    overlap any annotated lesion (fixed RNG seed, identical for all models).  For every model we
    report, per artefact type,
      * artefact-induced false positives: detections (conf >= 0.25) that overlap an injected
        artefact (IoU >= 0.1 with its box, or containing its centre) and match no ground truth
        (IoU < 0.5 with every same-class box), per 100 artefacts, minus the same count at the
        same positions in the clean image;
      * mAP@0.5 and mAP@0.5:0.95 on the perturbed test set vs the clean test set.
(B) Attention statistics (feature level).  Forward hooks on every SimAM / Cotton-SimAM module
    record the attention weight a = sigmoid(e) actually applied, and the mean weight inside
    (i) injected artefacts, (ii) annotated lesion boxes and (iii) the remaining background, at
    each detection scale.  Because Cotton-SimAM multiplies the energy by (1 + gamma*rho) >= 1 it can
    never lower a weight below its SimAM value for the SAME features; (B) therefore shows whether
    the trained network ends up giving artefacts a lower ABSOLUTE weight (suppression) or only a
    lower weight RELATIVE to lesions (relative enhancement).  The script also reports, on the same
    feature maps, the weights that gamma = 0 would have produced, to separate the two effects.

    python simam_probe.py --data cfg/cotton_disease.yaml --split test --out probe \
        --model "SimAM=ckS0.pt,ckS1.pt,ckS2.pt" --model "Cotton-SimAM=ckC0.pt,ckC1.pt,ckC2.pt" \
        [--model "No attention=ckN0.pt,ckN1.pt,ckN2.pt"] --types specular droplet impulse --per-image 5

    # revision runs: matched factorial pairs with and without Cotton-SimAM, every seed found
    python simam_probe.py --auto --seeds 0 1 2 --data <cotton_disease.yaml> --out probe
      (full model S1D1A1I1 vs S1D1A0I1 and baseline S0D0A1I0 vs S0D0A0I0, from runs_revision/<run>_seed<s>)
No plain-SimAM checkpoints exist for the revision, so the SimAM reference is the gamma = 0 counterfactual
computed on the same feature maps (section B), and the detection-level control is the matched model
without attention.  Image files are read and written with np.fromfile/imdecode (non-ASCII paths).
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apcore  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# matched pairs of the 2^4 factorial (same architecture and loss except Cotton-SimAM)
# "@g0": the same trained Cotton-SimAM model with gamma set to 0 at inference (= SimAM computation)
AUTO_PAIRS = [("CottonSimAM_full", "fac_S1D1A1I1"), ("CottonSimAM_full@g0", "fac_S1D1A1I1"),
              ("NoAttention_full", "fac_S1D1A0I1"),
              ("CottonSimAM_base", "fac_S0D0A1I0"), ("CottonSimAM_base@g0", "fac_S0D0A1I0"),
              ("NoAttention_base", "fac_S0D0A0I0")]
VAL_DIR = None      # set in main(): where Ultralytics writes its validation folders


def set_gamma(model, value):
    """set gamma of every (Cotton-)SimAM module of a loaded YOLO model; returns how many were changed"""
    n = 0
    for m in model.model.modules():
        if "simam" in type(m).__name__.lower() and hasattr(m, "gamma"):
            m.gamma = value
            n += 1
    return n


def imread_u(path):
    """cv2.imread that also works for non-ASCII paths on Windows"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(path, im, params=None):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", im, params or [])
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


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


# ----------------------------------------------------------------------------- artefacts
def paint(im, kind, cx, cy, r, rng):
    """Draw one artefact of radius r (pixels of the ORIGINAL image) centred at (cx, cy)."""
    h, w = im.shape[:2]
    R = int(math.ceil(3 * r))
    x0, x1, y0, y1 = max(0, cx - R), min(w, cx + R + 1), max(0, cy - R), min(h, cy + R + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    patch = im[y0:y1, x0:x1].astype(np.float32)
    if kind == "specular":
        amp = rng.uniform(150, 255)
        patch += amp * np.exp(-d2 / (2 * r * r))[..., None]
    elif kind == "droplet":
        inside = d2 <= r * r
        ring = np.exp(-((np.sqrt(d2) - r) ** 2) / (2 * (0.25 * r) ** 2))
        patch[inside] *= 0.85
        patch += 70 * ring[..., None]
        hx, hy = cx - 0.35 * r, cy - 0.35 * r
        patch += 220 * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * (0.2 * r) ** 2))[..., None]
    elif kind == "impulse":
        sq = (np.abs(xx - cx) <= 1.5 * r) & (np.abs(yy - cy) <= 1.5 * r)
        u = rng.random(sq.shape)
        patch[sq & (u < 0.15)] = 255.0
        patch[sq & (u > 0.85)] = 0.0
    im[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    rr = 1.5 * r if kind == "impulse" else (1.6 * r if kind == "specular" else 1.1 * r)
    return [cx - rr, cy - rr, cx + rr, cy + rr]


def make_perturbed(imgs, out_dir, kind, per_image, seed):
    rng = np.random.default_rng(seed)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    meta = {}
    ranges = {"specular": (2, 6), "droplet": (4, 10), "impulse": (2, 5)}   # radius in 640-input pixels
    for p in imgs:
        im = imread_u(p)
        if im is None:
            print(f"!! cannot read {p}, skipped")
            continue
        h, w = im.shape[:2]
        s = max(h, w) / 640.0
        gc, gb = read_gt(p, w, h)
        margin = 0.02 * max(h, w)
        boxes = []
        for _ in range(per_image):
            for _try in range(200):
                r = rng.uniform(*ranges[kind]) * s
                cx, cy = int(rng.uniform(3 * r, w - 3 * r)), int(rng.uniform(3 * r, h - 3 * r))
                cand = np.array([[cx - 3 * r - margin, cy - 3 * r - margin, cx + 3 * r + margin, cy + 3 * r + margin]])
                if len(gb) and apcore.box_iou(cand, gb).max() > 0:
                    continue
                if boxes and apcore.box_iou(cand, np.array(boxes)).max() > 0:
                    continue
                boxes.append(paint(im, kind, cx, cy, r, rng))
                break
        imwrite_u(out_dir / "images" / p.name, im,
                  [cv2.IMWRITE_JPEG_QUALITY, 98] if p.suffix.lower() in (".jpg", ".jpeg") else [])
        lp = label_path(p)
        if lp.exists():
            shutil.copy2(lp, out_dir / "labels" / lp.name)
        meta[p.name] = boxes
    (out_dir / "artefacts.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- (A) detection level
def artefact_fps(model, img_dir, meta, gt_dir_imgs, conf, device):
    names = sorted(meta)
    n_art, n_fp = 0, 0
    for k in range(0, len(names), 16):
        batch = names[k:k + 16]
        res = model.predict([str(img_dir / n) for n in batch], imgsz=640, conf=conf, iou=0.7, device=device,
                            verbose=False, rect=False)
        for n, r in zip(batch, res):
            arts = np.array(meta[n], float).reshape(-1, 4)
            n_art += len(arts)
            if not len(arts) or not len(r.boxes):
                continue
            h, w = r.orig_shape
            gc, gb = read_gt(gt_dir_imgs[n], w, h)
            pb = r.boxes.xyxy.cpu().numpy()
            pc = r.boxes.cls.cpu().numpy().astype(int)
            for b, c in zip(pb, pc):
                same = gb[gc == c] if len(gb) else np.zeros((0, 4))
                if len(same) and apcore.box_iou(b[None], same).max() >= 0.5:
                    continue                                   # a correct detection
                ctr = ((arts[:, 0] + arts[:, 2]) / 2, (arts[:, 1] + arts[:, 3]) / 2)
                inside = (ctr[0] >= b[0]) & (ctr[0] <= b[2]) & (ctr[1] >= b[1]) & (ctr[1] <= b[3])
                if apcore.box_iou(b[None], arts).max() >= 0.1 or inside.any():
                    n_fp += 1
    return n_fp, n_art


def val_map(model, data_yaml, split, device):
    kw = dict(project=str(VAL_DIR), name="val", exist_ok=True) if VAL_DIR else {}
    m = model.val(data=str(data_yaml), split=split, imgsz=640, batch=8, workers=0, rect=False, conf=0.001, iou=0.7,
                  device=device, plots=False, verbose=False, **kw)
    return float(m.box.map50), float(m.box.map)


# ----------------------------------------------------------------------------- (B) feature level
def letterbox(im, size=640):
    h, w = im.shape[:2]
    r = size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    im2 = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    im2 = cv2.copyMakeBorder(im2, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im2, r, left, top


def attention_stats(model, img_dir, meta, gt_imgs, device, max_images=200):
    import torch
    net = model.model.to(device).float().eval()
    mods = [m for m in net.modules() if "simam" in type(m).__name__.lower()]
    if not mods:
        return None
    m0 = mods[0]
    found = {k: getattr(m0, k) for k in ("gamma", "kernel_size", "lambda_param", "e_lambda") if hasattr(m0, k)}
    print(f"   attention modules: {len(mods)} x {type(m0).__name__}; attributes {found}"
          + ("" if "gamma" in found else "  (no 'gamma' attribute: the 'actual' rows equal gamma = 0; use the g05 rows)"))
    store = []

    def hook(mod, inp, out):
        x = inp[0].detach().float()
        lam = float(getattr(mod, "lambda_param", getattr(mod, "e_lambda", 1e-4)))
        k = int(getattr(mod, "kernel_size", 3))
        g = float(getattr(mod, "gamma", 0.0))
        _, _, hh, ww = x.shape
        n = max(hh * ww - 1, 1)
        mu = x.mean(dim=[2, 3], keepdim=True)
        d = (x - mu).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e = d / (4 * (v + lam)) + 0.5
        rho = (torch.nn.functional.avg_pool2d(d, k, 1, k // 2) / (d + lam)).clamp(0, 1)
        a_actual = torch.sigmoid(e * (1 + g * rho))
        a_g0 = torch.sigmoid(e)
        a_g05 = torch.sigmoid(e * (1 + 0.5 * rho))
        store.append(dict(stride=640 // hh, actual=a_actual.mean(1)[0].cpu().numpy(),
                          g0=a_g0.mean(1)[0].cpu().numpy(), g05=a_g05.mean(1)[0].cpu().numpy()))

    hs = [m.register_forward_hook(hook) for m in mods]
    acc = {}
    for n in sorted(meta)[:max_images]:
        im = imread_u(img_dir / n)
        if im is None:
            continue
        h, w = im.shape[:2]
        lb, r, left, top = letterbox(im)
        t = torch.from_numpy(lb[..., ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(device)
        store.clear()
        with torch.no_grad():
            net(t)
        gc, gb = read_gt(gt_imgs[n], w, h)
        arts = np.array(meta[n], float).reshape(-1, 4)
        for rec in store:
            S = rec["stride"]
            H = rec["actual"].shape[0]
            m_art = np.zeros((H, H), bool)
            m_les = np.zeros((H, H), bool)
            for boxes, mask in ((arts, m_art), (gb, m_les)):
                for b in boxes:
                    x0 = int(max(0, (b[0] * r + left) // S)); x1 = int(min(H - 1, (b[2] * r + left) // S))
                    y0 = int(max(0, (b[1] * r + top) // S)); y1 = int(min(H - 1, (b[3] * r + top) // S))
                    mask[y0:y1 + 1, x0:x1 + 1] = True
            m_bg = ~(m_art | m_les)
            for key in ("actual", "g0", "g05"):
                for reg, mask in (("artefact", m_art & ~m_les), ("lesion", m_les), ("background", m_bg)):
                    if mask.any():
                        acc.setdefault((S, key, reg), []).append(float(rec[key][mask].mean()))
    for hh in hs:
        hh.remove()
    return {f"{S}|{key}|{reg}": float(np.mean(v)) for (S, key, reg), v in acc.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", default=[], help="LABEL=ck0.pt,ck1.pt,ck2.pt")
    ap.add_argument("--auto", action="store_true", help="matched factorial pairs from --project (see AUTO_PAIRS)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--project", default="runs_revision")
    ap.add_argument("--types", nargs="+", default=["specular", "droplet", "impulse"])
    ap.add_argument("--per-image", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="simam_probe")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if a.auto:
        for label, run in AUTO_PAIRS:
            cks = [Path(a.project) / f"{run}_seed{s}" / "weights" / "best.pt" for s in a.seeds]
            have = [str(c) for c in cks if c.exists()]
            for c in cks:
                if not c.exists():
                    print(f"   (not on this computer, skipped: {c})")
            if have:
                a.model.append(f"{label}={','.join(have)}")
    if not a.model:
        raise SystemExit("no checkpoints: give --model LABEL=ck.pt,... or --auto with runs in --project")
    global VAL_DIR
    VAL_DIR = Path(a.out) / "_ultralytics_val"

    from ultralytics import YOLO
    d = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    root = Path(d.get("path", Path(a.data).parent))
    names = d["names"]
    imgs = sorted(p for p in (root / d[a.split]).rglob("*") if p.suffix.lower() in IMG)
    gt_imgs = {p.name: p for p in imgs}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sets = {}
    for t in a.types:
        pdir = out / f"perturbed_{t}"
        meta = make_perturbed(imgs, pdir, t, a.per_image, a.seed + a.types.index(t))
        dy = out / f"data_{t}.yaml"
        dy.write_text(yaml.safe_dump({"path": str(pdir.resolve()), "train": "images", "val": "images",
                                      "test": "images", "names": names}, sort_keys=False), encoding="utf-8")
        sets[t] = (pdir, meta, dy)
        print(f"{t}: {sum(len(v) for v in meta.values())} artefacts in {len(meta)} images -> {pdir}")

    rows, attn = [], {}
    for spec in a.model:
        label, files = spec.split("=", 1)
        g0 = label.endswith("@g0")
        for seed_i, ck in enumerate(f.strip() for f in files.split(",")):
            model = YOLO(ck)
            if g0:
                n_set = set_gamma(model, 0.0)
                print(f"{label} run {seed_i}: gamma set to 0 in {n_set} attention modules")
                if not n_set:
                    print(f"!! {label}: no module with a 'gamma' attribute - skipped")
                    continue
            c50, c5095 = val_map(model, a.data, a.split, a.device)
            for t, (pdir, meta, dy) in sets.items():
                fp_p, n_art = artefact_fps(model, pdir / "images", meta, gt_imgs, a.conf, a.device)
                clean_dir = (root / d[a.split])
                fp_c, _ = artefact_fps(model, clean_dir, meta, gt_imgs, a.conf, a.device)
                p50, p5095 = val_map(model, dy, "test", a.device)
                rows.append(dict(model=label, run=seed_i, artefact=t, artefacts=n_art,
                                 fp_per100=100 * (fp_p - fp_c) / max(1, n_art), fp_raw_perturbed=fp_p, fp_raw_clean=fp_c,
                                 map50_clean=c50, map50_pert=p50, map5095_clean=c5095, map5095_pert=p5095))
                print(f"{label} run {seed_i} {t}: artefact FPs/100 = {rows[-1]['fp_per100']:.2f}; "
                      f"mAP50 {c50:.3f} -> {p50:.3f}; mAP50-95 {c5095:.3f} -> {p5095:.3f}")
            if g0:
                continue                      # attention statistics: the g0 rows of the gamma = 0.5 model cover this
            pdir, meta, _ = sets[a.types[0]]
            st = attention_stats(model, pdir / "images", meta, gt_imgs, "cuda:" + a.device if a.device.isdigit() else a.device)
            if st:
                attn[f"{label}|{seed_i}"] = st

    with open(out / "artefact_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "attention_stats.json").write_text(json.dumps(attn, indent=1), encoding="utf-8")

    import pandas as pd
    df = pd.DataFrame(rows)
    df["map50_drop"] = df["map50_clean"] - df["map50_pert"]
    df["map5095_drop"] = df["map5095_clean"] - df["map5095_pert"]
    g = df.groupby(["artefact", "model"], sort=False)
    lines = ["| Artefact | Model | Artefact-induced FPs per 100 artefacts | mAP@0.5 drop | mAP@0.5:0.95 drop |",
             "|---|---|---|---|---|"]
    def ms(v, nd):
        return f"{v.mean():.{nd}f}" + (f"±{v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")

    for (t, m), x in g:
        lines.append(f"| {t} | {m} (n={len(x)}) | {ms(x.fp_per100, 2)} | {ms(x.map50_drop, 3)} | {ms(x.map5095_drop, 3)} |")
    if attn:
        lines += ["", f"Mean attention weight a = sigmoid(e) (artefact type '{a.types[0]}'), averaged over runs:", "",
                  "| Model | Stride | Weight source | Artefact | Lesion | Background | Lesion/Artefact |", "|---|---|---|---|---|---|---|"]
        labels = list(dict.fromkeys(k.split("|")[0] for k in attn))
        for lab in labels:
            runs = [v for k, v in attn.items() if k.startswith(lab + "|")]
            for S in (8, 16, 32):
                for key in ("actual", "g0", "g05"):
                    vals = {reg: np.mean([r.get(f"{S}|{key}|{reg}", np.nan) for r in runs]) for reg in ("artefact", "lesion", "background")}
                    if np.isnan(vals["artefact"]):
                        continue
                    lines.append(f"| {lab} | {S} | {key} | {vals['artefact']:.4f} | {vals['lesion']:.4f} | {vals['background']:.4f} | "
                                 f"{vals['lesion'] / vals['artefact']:.4f} |")
        lines += ["", "actual = weights applied by the trained module; g0 / g05 = weights that gamma = 0 / 0.5 "
                  "would give on the same features (isolates the formula from the learned features)."]
    lines += ["", "@g0 = the same trained Cotton-SimAM model with gamma set to 0 at inference (SimAM computation, "
              "no retraining); NoAttention = the matched factorial model trained without the attention module."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
