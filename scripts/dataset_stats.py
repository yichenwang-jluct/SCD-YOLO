"""
Split and class statistics of a YOLO-format dataset (Supplementary Table S3): number of images and
labelled instances per split, per-class instance counts and the number of images containing
each class.  Pure Python (PyYAML only), no GPU needed.

    python dataset_stats.py --data cfg/plantdoc.yaml cfg/cotton_cqamc.yaml cfg/cotton_disease.yaml --out dataset_stats

Writes <out>.csv (long format) and <out>.md (one table per dataset) - these become
Supplementary Table S3 and the split description in Section 2.5.
"""
import argparse
import csv
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


def list_images(root: Path, spec):
    specs = spec if isinstance(spec, list) else [spec]
    out = []
    for s in specs:
        p = Path(s)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*")) if q.suffix.lower() in IMG]
        elif p.suffix == ".txt" and p.exists():
            out += [Path(l.strip()) if Path(l.strip()).is_absolute() else root / l.strip()
                    for l in p.read_text().splitlines() if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="dataset_stats")
    a = ap.parse_args()
    rows, md = [], []
    for dy in a.data:
        d = yaml.safe_load(Path(dy).read_text(encoding="utf-8"))
        root = Path(d.get("path", Path(dy).parent))
        names = d["names"]
        names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        md += [f"## {Path(dy).stem}  ({root})", ""]
        header = ["Class"]
        table = {n: [] for n in names}
        totals = {"Images": [], "Instances": [], "Images without labels": []}
        for split in ("train", "val", "test"):
            if split not in d or not d[split]:
                continue
            imgs = list_images(root, d[split])
            inst, imgs_with, empty = Counter(), Counter(), 0
            for im in imgs:
                lp = label_path(im)
                cls = []
                if lp.exists():
                    for line in lp.read_text().splitlines():
                        if line.strip():
                            cls.append(int(float(line.split()[0])))
                if not cls:
                    empty += 1
                inst.update(cls)
                imgs_with.update(set(cls))
            header.append(f"{split} (inst / img)")
            for k, n in enumerate(names):
                table[n].append(f"{inst[k]} / {imgs_with[k]}")
                rows.append(dict(dataset=Path(dy).stem, split=split, cls=n, instances=inst[k], images_with_class=imgs_with[k]))
            totals["Images"].append(str(len(imgs)))
            totals["Instances"].append(str(sum(inst.values())))
            totals["Images without labels"].append(str(empty))
            rows.append(dict(dataset=Path(dy).stem, split=split, cls="ALL", instances=sum(inst.values()), images_with_class=len(imgs)))
        md += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        md += [f"| {n} | " + " | ".join(v) + " |" for n, v in table.items()]
        md += [f"| **{k}** | " + " | ".join(v) + " |" for k, v in totals.items()] + [""]
    with open(f"{a.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "split", "cls", "instances", "images_with_class"])
        w.writeheader()
        w.writerows(rows)
    Path(f"{a.out}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
