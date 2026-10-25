#!/usr/bin/env python3
"""
External datasets (Section 3.5, Tables 13 and 14, Supplementary Table S3): split sizes and per-class instance counts of PlantDoc and
cotton-cqamc, and the original PlantDoc / cotton-cqamc models (runs/V13train) re-evaluated with the
paper's protocol on the validation AND the official test split.

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python eval_external.py --plantdoc <PlantDoc.v4i.yolov8 folder> --cqamc <cotton-cqamc folder> --out eval

Each dataset folder is a Roboflow YOLOv8 export: train/, valid/ (or val/) and test/, each with
images/ and labels/.

Long file names: many PlantDoc file names are longer than 200 characters, so their full paths exceed the
260-character limit of Windows programs without long-path support; Ultralytics cannot open such files and
silently skips them as "corrupt".  This script therefore
  1. reads the labels.cache files that Ultralytics left in the dataset (read only): they record where the
     dataset was when it was last used and how many images were skipped as corrupt;
  2. counts every split with long-path-safe file access and reports how many files exceed the limit at the
     current location and at the original location (from the cache, else <Desktop>\\<folder name>);
  3. evaluates on short-name copies (<out>/_ext/<dataset>/): "val" and "test" = every image of the official
     splits; "val_orig" = only the validation images that could be opened at the original location, i.e. what
     a run there without long-path support saw.
Nothing in the dataset folders is changed.

Writes into --out:
  data_plantdoc.yaml, data_cqamc.yaml   dataset files of the short-name copies
  external_splits.csv                   images and instances per split and class (+ files over the limit)
  per_seed_external.csv                 P, R, mAP50, mAP50-95 overall and per class
  external_summary.txt                  everything printed
"""
import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIMIT = 260  # Windows MAX_PATH including the terminating null: paths of 260+ characters cannot be opened
PLANTDOC_NAMES = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf spot', 'Bell_pepper leaf',
                  'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf',
                  'Peach leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Potato leaf',
                  'Raspberry leaf', 'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf',
                  'Strawberry leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot',
                  'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus',
                  'Tomato leaf yellow virus', 'Tomato leaf', 'Tomato mold leaf',
                  'Tomato two spotted spider mites leaf', 'grape leaf black rot', 'grape leaf']
CQAMC_NAMES = ['curl_stage1', 'curl_stage2', 'healthy', 'leaf_enation', 'sooty']
RUNS = Path("runs") / "V13train"
MODELS = {
    "plantdoc": [("YOLOv13n", RUNS / "30类数据集v13n"), ("SCD-YOLO", RUNS / "30类数据集改进后")],
    "cqamc": [("YOLOv13n", RUNS / "5类数据集v13n"), ("SCD-YOLO", RUNS / "5类数据集改进后")],
}


def lp(p):
    """path string that open()/copy accept even beyond 260 characters on Windows (\\\\?\\ prefix)"""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s
    return s


def long_paths_enabled():
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def find_split(root, *names):
    for n in names:
        if (root / n / "images").is_dir():
            return n
    return None


def list_images(d):
    with os.scandir(lp(d)) as it:
        return sorted(e.name for e in it if e.is_file() and Path(e.name).suffix.lower() in IMG)


def read_label(path):
    try:
        with open(lp(path), encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return None


def read_cache(path):
    """summary of an Ultralytics labels.cache (read only), or None"""
    if not os.path.exists(lp(path)):
        return None
    try:
        import numpy as np
        c = np.load(lp(path), allow_pickle=True).item()
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")
    labels = c.get("labels") or []
    first = str(labels[0].get("im_file", "")) if labels else ""
    msgs = [str(m) for m in (c.get("msgs") or [])]
    return dict(version=c.get("version"), results=c.get("results"), n_labels=len(labels), first=first,
                msgs=msgs, mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(lp(path)))))


def describe(root, orig):
    """per split: images, instances per class id, files whose path is >= LIMIT at the current / original location,
    images without label file, names readable at the original location"""
    splits = {"train": find_split(root, "train"), "val": find_split(root, "valid", "val"),
              "test": find_split(root, "test")}
    stats = {}
    for key, sub in splits.items():
        if not sub:
            continue
        idir, ldir = root / sub / "images", root / sub / "labels"
        imgs = list_images(idir)
        inst, n_long, n_long_orig, n_nolabel, n_bad, longest, readable_orig = Counter(), 0, 0, 0, 0, 0, []
        for name in imgs:
            stem = Path(name).stem
            length = max(len(str(idir / name)), len(str(ldir / (stem + ".txt"))))
            length_orig = max(len(str(orig / sub / "images" / name)), len(str(orig / sub / "labels" / (stem + ".txt"))))
            longest = max(longest, length)
            n_long += length >= LIMIT
            n_long_orig += length_orig >= LIMIT
            if length_orig < LIMIT:
                readable_orig.append(name)
            lines = read_label(ldir / (stem + ".txt"))
            if lines is None:
                n_nolabel += 1
                continue
            for line in lines:
                if line.strip():
                    try:
                        inst[int(float(line.split()[0]))] += 1
                    except ValueError:
                        n_bad += 1
        stats[key] = dict(sub=sub, images=len(imgs), inst=inst, long=n_long, long_orig=n_long_orig,
                          nolabel=n_nolabel, bad=n_bad, longest=longest, readable_orig=readable_orig)
    return splits, stats


def short_copy(root, sub, dst, only=None):
    """copy one split (or only the given image names) under short names 00000.jpg / 00000.txt; returns the name map"""
    idir, ldir = root / sub / "images", root / sub / "labels"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    keep = None if only is None else set(only)
    rows = []
    for i, name in enumerate(list_images(idir)):
        if keep is not None and name not in keep:
            continue
        new = f"{i:05d}"
        shutil.copyfile(lp(idir / name), lp(dst / "images" / (new + Path(name).suffix)))
        lab = ldir / (Path(name).stem + ".txt")
        has = os.path.exists(lp(lab))
        if has:
            shutil.copyfile(lp(lab), lp(dst / "labels" / (new + ".txt")))
        rows.append(dict(split=dst.name, original=name, short=new + Path(name).suffix, label=int(has)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantdoc", required=True)
    ap.add_argument("--cqamc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    import yaml
    import eval_all

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_ultralytics_tmp"
    lines, split_rows, eval_rows = [], [], []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    def run_eval(ds, label, ckpt, ypath, split, tag):
        try:
            rows = eval_all.evaluate(ckpt, ypath, split, a, tmp)
        except Exception as e:
            say(f"  !! {label} {tag}: {type(e).__name__}: {e}")
            eval_all.free_gpu()
            return
        for cls, P, R, A50, A in rows:
            eval_rows.append(dict(label=f"{ds}/{label}", seed=0, split=tag, **{"class": cls}, P=f"{float(P):.6f}",
                                  R=f"{float(R):.6f}", AP50=f"{float(A50):.6f}", AP50_95=f"{float(A):.6f}",
                                  checkpoint=str(ckpt)))
        _, P, R, A50, A = rows[0]
        say(f"  {label:<9} {tag:<9} P {float(P):.3f}  R {float(R):.3f}  mAP50 {float(A50):.3f}  "
            f"mAP50-95 {float(A):.3f}")

    def process(ds, folder, names):
        root = Path(folder)
        if not root.is_dir():
            say(f"\n!! {ds}: folder not found: {folder}")
            return
        say(f"\n=== {ds}: {folder}")
        dy = root / "data.yaml"
        if dy.exists():
            try:
                with open(lp(dy), encoding="utf-8") as f:
                    yn = (yaml.safe_load(f) or {}).get("names")
                if isinstance(yn, dict):
                    yn = [yn[k] for k in sorted(yn)]
                if yn and list(yn) != list(names):
                    say(f"  note: data.yaml lists other class names ({len(yn)}); they are used: {yn}")
                    names = list(yn)
            except Exception as e:
                say(f"  note: data.yaml not readable ({type(e).__name__}); expected class names used")

        # 1. caches left by earlier Ultralytics runs (where was the dataset, how many images were skipped)
        orig = None
        for sub in ("train", "valid", "val", "test"):
            c = read_cache(root / sub / "labels.cache")
            if c is None:
                continue
            if "error" in c:
                say(f"  cache {sub}/labels.cache: not readable ({c['error']})")
                continue
            res = c["results"]
            txt = (f"found {res[0]}, missing {res[1]}, empty {res[2]}, corrupt {res[3]}, total {res[4]}"
                   if isinstance(res, (tuple, list)) and len(res) >= 5 else str(res))
            say(f"  cache {sub}/labels.cache (written {c['mtime']}, Ultralytics {c['version']}): {txt}; "
                f"{len(c['msgs'])} warnings")
            if c["first"]:
                say(f"      dataset location recorded in it: {Path(c['first']).parents[2]}")
                if orig is None and sub in ("train", "valid", "val"):
                    orig = Path(c["first"]).parents[2]
            for m in c["msgs"][:3]:
                say(f"      e.g. {m[:220]}")
        src = "recorded in labels.cache"
        if orig is None:
            orig, src = Path.home() / "Desktop" / root.name, "assumed (dataset directly on the Desktop)"
        say(f"  original location ({src}): {orig}")

        # 2. counts
        splits, stats = describe(root, orig)
        for key in ("train", "val", "test"):
            if key not in stats:
                say(f"  {key:<5}  -- not found")
                continue
            s = stats[key]
            say(f"  {key:<5} {s['images']:>5} images, {sum(s['inst'].values()):>6} instances, "
                f"{len(s['inst'])} classes present, {s['nolabel']} without label file; paths of {LIMIT}+ characters: "
                f"{s['long']} here (longest {s['longest']}), {s['long_orig']} at the original location"
                + (f"; {s['bad']} unreadable label lines" if s['bad'] else ""))
            for k in range(len(names)):
                split_rows.append(dict(dataset=ds, split=key, cls=names[k], images=s["images"],
                                       instances=s["inst"].get(k, 0), over_limit_here=s["long"],
                                       over_limit_original_location=s["long_orig"]))

        # 3. short-name copies and evaluation
        base = out / "_ext" / ds
        if base.exists():
            shutil.rmtree(lp(base))
        ymap = {"path": str(base.resolve()), "nc": len(names), "names": names}
        name_map = []
        for key, sname in (("val", "valid"), ("test", "test")):
            if key in stats:
                name_map += short_copy(root, splits[key], base / sname)
                ymap[key] = f"{sname}/images"
        if "val" not in ymap:
            say(f"  !! {ds}: no validation split - nothing evaluated")
            return
        ymap["train"] = ymap["val"]  # required key; not used for evaluation
        ypath = out / f"data_{ds}.yaml"
        ypath.write_text(yaml.safe_dump(ymap, sort_keys=False, allow_unicode=True), encoding="utf-8")

        yorig = None
        if stats["val"]["long_orig"]:
            name_map += short_copy(root, splits["val"], base / "valid_orig", only=stats["val"]["readable_orig"])
            yo = dict(ymap, val="valid_orig/images", train="valid_orig/images")
            yo.pop("test", None)
            yorig = out / f"data_{ds}_orig.yaml"
            yorig.write_text(yaml.safe_dump(yo, sort_keys=False, allow_unicode=True), encoding="utf-8")
            say(f"  val_orig = the {len(stats['val']['readable_orig'])} of {stats['val']['images']} validation "
                f"images that can be opened at the original location without long-path support")
        with open(base / "short_names.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["split", "original", "short", "label"])
            w.writeheader()
            w.writerows(name_map)
        say(f"  short-name copies: {base}")

        for label, run in MODELS[ds]:
            ckpt = run / "weights" / "best.pt"
            if not ckpt.exists():
                say(f"  !! missing {ckpt}")
                continue
            run_eval(ds, label, ckpt, ypath, "val", "val")
            if "test" in ymap:
                run_eval(ds, label, ckpt, ypath, "test", "test")
            if yorig is not None:
                run_eval(ds, label, ckpt, yorig, "val", "val_orig")

    say(f"eval_external  {time.strftime('%Y-%m-%d %H:%M:%S')}   Windows long-path support: {long_paths_enabled()}")
    for ds, folder, names in (("plantdoc", a.plantdoc, PLANTDOC_NAMES), ("cqamc", a.cqamc, CQAMC_NAMES)):
        try:
            process(ds, folder, names)
        except Exception as e:
            say(f"  !! {ds} stopped: {type(e).__name__}: {e}")

    if split_rows:
        with open(out / "external_splits.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
            w.writeheader()
            w.writerows(split_rows)
    if eval_rows:
        with open(out / "per_seed_external.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0]))
            w.writeheader()
            w.writerows(eval_rows)
    (out / "external_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"\nfiles written to {out.resolve()}")


if __name__ == "__main__":
    main()
