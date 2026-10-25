#!/usr/bin/env python3
"""
Find the authors' 427 field images inside the merged cotton dataset (read-only on the dataset).

Every file of the merged dataset went through a Roboflow export ("<stem>_jpg.rf.<hash>.jpg"), so
the ".rf." rule of make_source_disjoint_split.py cannot tell the two sources apart.  This script
collects other evidence for every image:

  * pixel size            field photos: 48-MP phone saved at 3000 x 3000 (unless the export resized)
  * EXIF camera and date  only if the export kept EXIF
  * original name stem    the part before "_jpg.rf.<hash>", and its pattern with digits -> #
  * date in the name      e.g. IMG_20230815_101530 -> 2023-08-15

and lists every simple rule that selects exactly --expect (427) images, with how those images are
spread over train/val/test (a random 8:1:1 re-split puts about 342/43/43 of them in each).

    python find_field_images.py --data <cotton_disease.yaml> --report field_report.txt
    python find_field_images.py --data ... --auto --write field_list.txt
    python find_field_images.py --data ... --rule "size=3000x3000" --write field_list.txt
    python find_field_images.py --data ... --originals <folder with the 427 images> --auto --write ...

--originals: a folder holding the field images (copies taken out of the dataset, or the original
photos).  They are matched to the dataset by identical file name, then by the original file name
(the part before "_jpg.rf.<hash>"), then by image content (difference hash, robust to resizing and
re-compression).  With --auto, the matched images are used when they are exactly --expect.
Without --originals, --auto writes the list only if the strong rules (one pixel size, one EXIF
camera) and single name patterns that select exactly --expect images all agree; otherwise it
writes nothing and exits with code 2.
--write saves the selected file names (one per line) and, next to it, <name>_groups.csv
(image, group = capture day from EXIF or from the file name, original_split) for the cluster
bootstrap.
Rules (several --rule are OR-ed): size=WxH | camera=<text in EXIF make/model> |
    pattern=<pattern as printed> | regex=<python regex on the file name> | namedate | exifdate
"""
import argparse
import csv
import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RF = re.compile(r"^(?P<stem>.+?)(?:_(?:jpe?g|png|bmp|webp|tiff?|heic))?\.rf\.[0-9a-zA-Z]{6,}$", re.I)
DATE = re.compile(r"(?<!\d)(20[12]\d)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?!\d)")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
HEX = re.compile(r"(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{16,}")
SPLITS = ("train", "val", "test")


def original_stem(name):
    base = name.rsplit(".", 1)[0]
    m = RF.match(base)
    return m.group("stem") if m else base


def pattern(stem):
    s = UUID.sub("<uuid>", stem)
    s = HEX.sub("<hex>", s)
    return re.sub(r"\d+", "#", s)


def label_path(img):
    parts = list(img.parts)
    for k in range(len(parts) - 1, -1, -1):
        if parts[k] == "images":
            parts[k] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def probe(path):
    """(width, height, camera, exif_day) - camera/exif_day are '' when the file has no EXIF."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            cam, day = "", ""
            try:
                ex = im.getexif()
                cam = " ".join(str(ex.get(t, "")).strip("\x00 ") for t in (271, 272)).strip()
                v = ex.get(36867) or ex.get(306)
                if not v and hasattr(ex, "get_ifd"):
                    v = ex.get_ifd(0x8769).get(36867)
                if v:
                    day = str(v)[:10].replace(":", "-")
            except Exception:
                pass
            return w, h, cam, day
    except Exception:
        return 0, 0, "", ""


def collect(data_yaml):
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(d.get("path") or Path(data_yaml).parent)
    names = d["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    recs = []
    for split in SPLITS:
        if not d.get(split):
            continue
        sdir = root / d[split]
        for im in sorted(p for p in sdir.rglob("*") if p.suffix.lower() in IMG):
            w, h, cam, eday = probe(im)
            stem = original_stem(im.name)
            m = DATE.search(stem)
            nday = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
            cls = Counter()
            lp = label_path(im)
            if lp.exists():
                for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip():
                        try:
                            cls[int(float(line.split()[0]))] += 1
                        except ValueError:
                            pass
            recs.append(dict(name=im.name, split=split, stem=stem, pattern=pattern(stem), size=f"{w}x{h}",
                             mp=w * h / 1e6, camera=cam, exif_day=eday, name_day=nday, cls=cls, path=im))
    return recs, names


def dhash(path):
    """64-bit difference hash of the (EXIF-oriented) image; robust to resizing and JPEG re-compression."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        try:
            im.draft("L", (256, 256))
        except Exception:
            pass
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        px = list(im.convert("L").resize((9, 8)).tobytes())
    v = 0
    for r in range(8):
        for c in range(8):
            v = (v << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
    return v


def match_folder(recs, folder, max_dist=10):
    """Dataset records matching the images in `folder`: same name, then same original name, then content."""
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG)
    notes = [f"\n--originals: {len(files)} image files in {folder}"]
    by_name = {r["name"].lower(): r for r in recs}
    by_stem = defaultdict(list)
    for r in recs:
        by_stem[r["stem"].lower()].append(r)
    hit = {}                      # folder file -> record
    for p in files:
        r = by_name.get(p.name.lower())
        if r is not None:
            hit[p] = r
    n1 = len(hit)
    for p in files:
        if p in hit:
            continue
        cand = by_stem.get(original_stem(p.name).lower(), [])
        if len(cand) == 1:
            hit[p] = cand[0]
    n2 = len(hit) - n1
    n3, worst = 0, 0
    rest = [p for p in files if p not in hit]
    if rest:
        notes.append(f"  {len(rest)} files not matched by name -> comparing image content (about 1 minute)")
        taken = {id(r) for r in hit.values()}
        pool = [(dhash(r["path"]), r) for r in recs if id(r) not in taken]
        for p in rest:
            try:
                h = dhash(p)
            except Exception:
                continue
            ds = sorted((bin(h ^ hh).count("1"), k) for k, (hh, _) in enumerate(pool))
            if ds and ds[0][0] <= max_dist and (len(ds) == 1 or ds[1][0] - ds[0][0] >= 3):
                hit[p] = pool[ds[0][1]][1]
                n3 += 1
                worst = max(worst, ds[0][0])
    sel, seen, dup = [], set(), 0
    for p, r in hit.items():
        if id(r) in seen:
            dup += 1
            continue
        seen.add(id(r))
        _, _, _, day = probe(p)          # capture date of the folder copy (original photos keep EXIF)
        if day and not r["exif_day"]:
            r["exif_day"] = day
        sel.append(r)
    notes.append(f"  matched: {n1} by identical name, {n2} by original name, {n3} by image content"
                 + (f" (largest hash distance {worst})" if n3 else "")
                 + f"; not matched: {len(files) - len(hit)}" + (f"; {dup} matched an image twice" if dup else ""))
    return sel, notes


def spread(sel):
    c = Counter(r["split"] for r in sel)
    return "/".join(str(c[s]) for s in SPLITS)


def instances(sel, names):
    c = Counter()
    for r in sel:
        c.update(r["cls"])
    return ", ".join(f"{names[k]} {c[k]}" for k in range(len(names)))


def plausible(sel, expect):
    """A random 8:1:1 re-split: each of val/test should hold roughly 1/10 of the field images."""
    c = Counter(r["split"] for r in sel)
    lo, hi = 0.04 * expect, 0.18 * expect
    return all(c[s] > 0 for s in SPLITS) and lo <= c["val"] <= hi and lo <= c["test"] <= hi


def select(recs, rules, extra=None):
    out = []
    for r in recs:
        hit = False
        for rule in rules:
            k, _, v = rule.partition("=")
            k = k.strip().lower()
            if k == "size":
                hit = r["size"] == v.strip().lower()
            elif k == "camera":
                hit = v.strip().lower() in r["camera"].lower() and r["camera"] != ""
            elif k == "pattern":
                hit = r["pattern"] == v
            elif k == "regex":
                hit = re.search(v, r["name"]) is not None
            elif k == "namedate":
                hit = r["name_day"] != ""
            elif k == "exifdate":
                hit = r["exif_day"] != ""
            else:
                raise SystemExit(f"unknown rule '{rule}'")
            if hit:
                break
        if not hit and extra is not None:
            hit = any(r is x for x in extra)
        if hit:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--expect", type=int, default=427)
    ap.add_argument("--rule", action="append", default=[])
    ap.add_argument("--originals", default=None, help="folder with the original field photos (matched by file name)")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--write", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    recs, names = collect(a.data)
    E = a.expect
    say(f"images: {len(recs)}  (train/val/test {spread(recs)})   looking for {E} field images "
        f"(random 8:1:1 split -> about {round(E * .8)}/{round(E * .1)}/{round(E * .1)})")

    # ---- pixel sizes
    by_size = defaultdict(list)
    for r in recs:
        by_size[r["size"]].append(r)
    say("\npixel sizes (count, train/val/test):")
    for s, sel in sorted(by_size.items(), key=lambda kv: -len(kv[1]))[:10]:
        say(f"  {s:<12} {len(sel):>5}  {spread(sel)}")
    if len(by_size) > 10:
        say(f"  ... {len(by_size) - 10} more sizes")
    big = [r for r in recs if r["mp"] >= 4.0]
    say(f"  >= 4 megapixels: {len(big)}  {spread(big)}")

    # ---- EXIF
    by_cam = defaultdict(list)
    for r in recs:
        if r["camera"]:
            by_cam[r["camera"]].append(r)
    n_exday = sum(1 for r in recs if r["exif_day"])
    say(f"\nEXIF: camera in {sum(len(v) for v in by_cam.values())} images, capture date in {n_exday} images")
    for c, sel in sorted(by_cam.items(), key=lambda kv: -len(kv[1]))[:6]:
        say(f"  {c[:40]:<40} {len(sel):>5}  {spread(sel)}")

    # ---- dates in names
    nd = [r for r in recs if r["name_day"]]
    days = Counter(r["name_day"] for r in nd)
    say(f"\ndate in the file name: {len(nd)} images on {len(days)} days  {spread(nd) if nd else ''}")
    if days:
        months = Counter(r["name_day"][:7] for r in nd)
        say("  by month: " + ", ".join(f"{m} {n}" for m, n in sorted(months.items())[:36])
            + (" ..." if len(months) > 36 else ""))

    # ---- name patterns
    by_pat = defaultdict(list)
    for r in recs:
        by_pat[r["pattern"]].append(r)
    say(f"\nfile-name patterns (digits -> #): {len(by_pat)} patterns; the 25 largest:")
    say(f"  {'n':>5}  {'tr/va/te':<12} {'main size':<11} pattern   (example)")
    for p, sel in sorted(by_pat.items(), key=lambda kv: -len(kv[1]))[:25]:
        ms = Counter(r["size"] for r in sel).most_common(1)[0][0]
        say(f"  {len(sel):>5}  {spread(sel):<12} {ms:<11} {p[:45]}   ({sel[0]['name'][:60]})")

    # ---- rules selecting exactly E images
    strong, weak = [], []
    for s, sel in by_size.items():
        if len(sel) == E:
            strong.append((f"size={s}", sel))
    for c, sel in by_cam.items():
        if len(sel) == E:
            strong.append((f"camera={c}", sel))
    folder_sel = None
    if a.originals:
        folder_sel, notes = match_folder(recs, a.originals)
        for n in notes:
            say(n)
        say(f"  matched dataset images: {len(folder_sel)}  (train/val/test {spread(folder_sel)})")
        say(f"           instances: {instances(folder_sel, names)}")
        if len(folder_sel) == E:
            strong.append(("originals", folder_sel))
    if len(nd) == E:
        weak.append(("namedate", nd))
    if n_exday == E:
        weak.append(("exifdate", [r for r in recs if r["exif_day"]]))
    if len(big) == E and not any(k.startswith("size=") for k, _ in strong):
        weak.append((">= 4 megapixels (use --rule regex/size)", big))
    small = sorted(((p, sel) for p, sel in by_pat.items() if len(sel) <= E), key=lambda kv: -len(kv[1]))[:60]
    for k in (1, 2, 3):
        for combo in itertools.combinations(small, k):
            if sum(len(sel) for _, sel in combo) == E:
                weak.append((" ".join(f'"pattern={p}"' for p, _ in combo), [r for _, sel in combo for r in sel]))
        if len(weak) > 12:
            break
    say(f"\nrules that select exactly {E} images (train/val/test, plausible 8:1:1?):")
    if not strong and not weak:
        say("  none - send this report")
    for tag, (rule, sel) in [("STRONG", x) for x in strong] + [("weak", x) for x in weak[:12]]:
        say(f"  [{tag}] {rule:<45} {spread(sel):<12} {'yes' if plausible(sel, E) else 'NO'}")
        say(f"           instances: {instances(sel, names)}")

    chosen = None
    if a.rule or (a.originals and not a.auto):
        chosen = select(recs, a.rule, folder_sel)
        say(f"\n--rule {a.rule}{' + originals' if folder_sel is not None else ''}: {len(chosen)} images  {spread(chosen)}")
        say(f"           instances: {instances(chosen, names)}")
    elif a.auto and a.originals:
        if len(folder_sel) == E:
            chosen = folder_sel
            same = {x["name"] for x in chosen}
            agree = [r for r, s_ in strong + weak if r != "originals" and {x["name"] for x in s_} == same]
            say(f"\n--auto: using the {E} images matched in {a.originals}"
                + (f" (the same set as {', '.join(agree)})" if agree else ""))
            if not plausible(chosen, E):
                say("  note: their spread over train/val/test is not the usual 8:1:1 - send this report")
        else:
            say(f"\n--auto: {len(folder_sel)} images matched in {a.originals}, not {E} - nothing written, send this report")
    elif a.auto:
        # candidates: strong rules and single name patterns with a plausible 8:1:1 spread; accepted only
        # if they all select the SAME set of images and no other rule selects a different set
        cands = [(r, s) for r, s in strong if plausible(s, E)]
        cands += [(r, s) for r, s in weak if r.startswith('"pattern=') and r.count('"pattern=') == 1 and plausible(s, E)]
        sets = {frozenset(x["name"] for x in s) for _, s in cands}
        others = {frozenset(x["name"] for x in s) for _, s in strong + weak} - sets
        if len(sets) == 1 and not others:
            chosen = cands[0][1]
            say(f"\n--auto: using {' = '.join(r for r, _ in cands)} ({len(chosen)} images, {spread(chosen)})")
        else:
            say(f"\n--auto: {len(sets)} candidate sets ({len(others)} other) - nothing written, send this report")

    if a.report:
        Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        print(f"\nreport saved to {a.report}")
    if a.write:
        if not chosen:
            print("nothing written")
            sys.exit(2)
        if len(chosen) != E:
            print(f"!! {len(chosen)} images selected, not {E}")
        out = Path(a.write)
        out.write_text("\n".join(r["name"] for r in chosen) + "\n", encoding="utf-8")
        g = out.with_name(out.stem + "_groups.csv")
        with open(g, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "group", "original_split"])
            for r in chosen:
                w.writerow([r["name"], r["exif_day"] or r["name_day"] or "unknown", r["split"]])
        gd = Counter(r["exif_day"] or r["name_day"] or "unknown" for r in chosen)
        print(f"{len(chosen)} names written to {out}\ncapture-day groups ({len(gd)}) written to {g}")
        if len(chosen) != E:
            sys.exit(3)
    elif a.auto and not chosen:
        sys.exit(2)


if __name__ == "__main__":
    main()
