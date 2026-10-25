"""
Source-disjoint evaluation split (Section 3.6): train and select models on the PUBLIC
Roboflow-derived images only and test on the authors' FIELD images, which come from a different
source (site, device, acquisition protocol) and never enter training or model selection.

    python make_source_disjoint_split.py --data cfg/cotton_disease.yaml --out D:/datasets/cotton_srcdisjoint
    # field images are identified by NOT matching --public-regex (Roboflow exports contain ".rf.")
    # or listed explicitly:  --field-list field_images.txt   (one file name per line)

Creates
    <out>/images|labels/train        public images of the original training split
    <out>/images|labels/val          public images of the original validation split
    <out>/images|labels/test_field   ALL field images (from the original train/val/test splits)
    <out>/images|labels/test_public  public images of the original test split (in-source reference)
    <out>/data_srcdisjoint.yaml          (test = test_field)
    <out>/data_srcdisjoint_public.yaml   (test = test_public)
    <out>/field_groups.csv           capture-day groups from EXIF (with --exif), for a cluster bootstrap
Files are hard-linked when possible (same drive) and copied otherwise; originals are untouched.

Expected counts for the paper's dataset: 3,746 public + 427 field images.
"""
import argparse
import csv
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    for k in range(len(parts) - 1, -1, -1):
        if parts[k] == "images":
            parts[k] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def exif_day(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            v = ex.get(36867) or ex.get(306)          # DateTimeOriginal / DateTime
            if not v and hasattr(ex, "get_ifd"):
                v = ex.get_ifd(0x8769).get(36867)
        return str(v)[:10].replace(":", "-") if v else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public-regex", default=r"\.rf\.")
    ap.add_argument("--field-list", default=None)
    ap.add_argument("--exif", action="store_true", help="write field_groups.csv from EXIF capture dates")
    a = ap.parse_args()

    d = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    root = Path(d.get("path", Path(a.data).parent))
    names = d["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    field_set = None
    if a.field_list:
        field_set = {Path(l.strip()).name for l in Path(a.field_list).read_text(encoding="utf-8").splitlines() if l.strip()}
    pub_re = re.compile(a.public_regex)

    out = Path(a.out)
    counts = Counter()
    inst = {k: Counter() for k in ("train", "val", "test_field", "test_public")}
    groups = []
    for split in ("train", "val", "test"):
        sdir = root / d[split]
        for im in sorted(p for p in sdir.rglob("*") if p.suffix.lower() in IMG):
            is_field = (im.name in field_set) if field_set is not None else (pub_re.search(im.name) is None)
            if is_field:
                tgt = "test_field"
            else:
                tgt = {"train": "train", "val": "val", "test": "test_public"}[split]
            place(im, out / "images" / tgt / im.name)
            lp = label_path(im)
            if lp.exists():
                place(lp, out / "labels" / tgt / lp.name)
                for line in lp.read_text().splitlines():
                    if line.strip():
                        inst[tgt][int(float(line.split()[0]))] += 1
            counts[(tgt, split)] += 1
            if is_field and a.exif:
                groups.append(dict(image=im.name, group=exif_day(im) or "unknown", original_split=split))

    base = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "nc": len(names), "names": names}
    (out / "data_srcdisjoint.yaml").write_text(yaml.safe_dump({**base, "test": "images/test_field"}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "data_srcdisjoint_public.yaml").write_text(yaml.safe_dump({**base, "test": "images/test_public"}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if groups:
        with open(out / "field_groups.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "group", "original_split"])
            w.writeheader()
            w.writerows(groups)
        g = Counter(r["group"] for r in groups)
        print(f"field capture-day groups: {len(g)} ({dict(sorted(g.items()))})")

    print("\nimages per new split (by original split):")
    for tgt in ("train", "val", "test_field", "test_public"):
        tot = sum(v for (t, _), v in counts.items() if t == tgt)
        by = {s: counts[(tgt, s)] for s in ("train", "val", "test") if counts[(tgt, s)]}
        print(f"  {tgt:<12} {tot:>5}  {by}")
        print("     instances: " + ", ".join(f"{names[k]} {inst[tgt][k]}" for k in range(len(names))))
    n_field = sum(v for (t, _), v in counts.items() if t == "test_field")
    if n_field != 427:
        print(f"\n!! {n_field} field images found (paper: 427). Check --public-regex or pass --field-list.")


if __name__ == "__main__":
    main()
