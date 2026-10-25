#!/usr/bin/env python3
"""
Deployment accuracy and timing on a desktop PC (Table 20, Supplementary Table S11), used because the Jetson / Coral
boards are no longer available.  Run from the YOLOv13 source-tree root (PYTHONPATH = that root):

  --stage fp    in the training environment (GPU):
                FP32 and FP16 accuracy of YOLOv13n and SCD-YOLO on the test split, and GPU latency
                with the repeated-measurement protocol of benchmark_edge.py
  --stage int8  in a SEPARATE environment (it installs OpenVINO / NNCF):
                OpenVINO export in FP32 and in INT8 (post-training quantization with NNCF,
                calibrated on the validation images - never the test images), accuracy and CPU latency

Accuracy: batch 1, imgsz 640, rect=False, conf=0.001, NMS IoU=0.7 (the paper's protocol).
Timing: one in-memory test image per call, pre-processing + inference + NMS, synchronous;
camera acquisition, image decoding, data transfer and display are excluded; W warm-up frames,
then N timed frames, repeated R times (mean +/- SD over repeats).
Timing is skipped while trainings of this computer are running (they would distort it); run the
stage again later with --timing-only.

Outputs in --out: deployed_accuracy.csv, bench_<model>_<tag>_summary.json/_repeats.csv,
versions_<stage>.json, export_<model>_<tag>.json, edge_summary_<stage>.txt
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = [("YOLOv13n", Path("runs_revision") / "fac_S0D0A0I0_seed0" / "weights" / "best.pt"),
          ("SCD-YOLO", Path("runs_revision") / "fac_S1D1A1I1_seed0" / "weights" / "best.pt")]


def versions(stage):
    v = {"stage": stage, "python": sys.version.split()[0], "platform": platform.platform(),
         "cpu": platform.processor()}
    for mod in ("ultralytics", "torch", "torchvision", "openvino", "nncf", "onnx", "numpy", "cv2"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
            v["cuda"] = torch.version.cuda
            v["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return v


def trainings_running():
    """names of runs whose training process (pid file written by run_experiment.py) is alive"""
    alive = []
    try:
        import psutil
    except Exception:
        return alive
    for pf in Path("runs_revision").glob("*.pid"):
        try:
            pid = int(pf.read_text(encoding="utf-8").strip())
            if psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
                alive.append(pf.stem)
        except Exception:
            pass
    return alive


def test_dir(data):
    import yaml
    d = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    root = Path(d.get("path") or Path(data).parent)
    t = Path(d["test"])
    return t if t.is_absolute() else root / t


def accuracy(model, label, tag, runtime, data, device, half, out_rows, tmp):
    from ultralytics import YOLO
    m = YOLO(str(model), task="detect").val(data=str(data), split="test", imgsz=640, batch=1, workers=0,
                                            rect=False, conf=0.001, iou=0.7, device=device, half=half,
                                            plots=False, verbose=False, project=str(tmp), name="val", exist_ok=True)
    b = m.box
    names = getattr(m, "names", None) or {}
    row = dict(model=label, precision=tag, runtime=runtime, device=str(device), P=f"{b.mp:.4f}", R=f"{b.mr:.4f}",
               mAP50=f"{b.map50:.4f}", mAP50_95=f"{b.map:.4f}", file=str(model))
    for j, c in enumerate(b.ap_class_index):
        row[f"AP50_95_{names.get(int(c), int(c))}"] = f"{b.ap[j]:.4f}"
    out_rows.append(row)
    print(f"[acc] {label:<9} {tag:<14} P {b.mp:.4f}  R {b.mr:.4f}  mAP50 {b.map50:.4f}  mAP50-95 {b.map:.4f}", flush=True)
    return row


def timing(model, label, tag, device, half, images, out, warmup, frames, repeats, cooldown):
    cmd = [sys.executable, str(HERE / "benchmark_edge.py"), "--model", str(model), "--images", str(images),
           "--device", str(device), "--warmup", str(warmup), "--frames", str(frames), "--repeats", str(repeats),
           "--cooldown", str(cooldown), "--out", str(out / f"bench_{label}_{tag}")]
    if half:
        cmd.append("--half")
    print(f"[time] {label} {tag}: {repeats} x {frames} frames after {warmup} warm-up frames ...", flush=True)
    r = subprocess.run(cmd)
    js = out / f"bench_{label}_{tag}_summary.json"
    if r.returncode != 0 or not js.exists():
        print(f"!! timing of {label} {tag} failed")
        return None
    s = json.loads(js.read_text(encoding="utf-8"))
    return s


def write_rows(path, rows):
    if not rows:
        return
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            old = list(csv.DictReader(f))
    keyset = {(r["model"], r["precision"]) for r in rows}
    old = [r for r in old if (r.get("model"), r.get("precision")) not in keyset]
    allrows = old + rows
    fields = []
    for r in allrows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(allrows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["fp", "int8"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--warmup", type=int, default=None, help="default 100 (fp) / 30 (int8)")
    ap.add_argument("--frames", type=int, default=None, help="default 1000 (fp) / 300 (int8)")
    ap.add_argument("--repeats", type=int, default=None, help="default 5 (fp) / 3 (int8)")
    ap.add_argument("--cooldown", type=float, default=None, help="seconds, default 20 (fp) / 10 (int8)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_ultralytics_tmp"
    vers = versions(a.stage)
    (out / f"versions_{a.stage}.json").write_text(json.dumps(vers, indent=2), encoding="utf-8")
    images = test_dir(a.data)
    busy = trainings_running()
    do_timing = not a.no_timing
    if busy and do_timing:
        print(f"!! trainings are running on this computer ({', '.join(busy)}): timing is skipped. "
              f"Run this stage again with --timing-only when they have finished.")
        do_timing = False
    rows, lines = [], []
    for label, pt in MODELS:
        if not pt.exists():
            print(f"!! missing {pt}")
            continue
        if a.stage == "fp":
            variants = [("FP32", pt, "PyTorch", 0, False), ("FP16", pt, "PyTorch", 0, True)]
        else:
            from ultralytics import YOLO
            variants = []
            for tag, int8 in (("OpenVINO-FP32", False), ("OpenVINO-INT8", True)):
                kw = dict(format="openvino", imgsz=640, batch=1, int8=int8, half=False, dynamic=False)
                if int8:
                    kw["data"] = str(a.data)          # calibration images: the validation split
                man = dict(model=label, tag=tag, weights=str(pt), export_args=kw)
                try:
                    t0 = time.time()
                    path = YOLO(str(pt)).export(**kw)
                    man.update(output=str(path), seconds=round(time.time() - t0, 1))
                    if int8:
                        try:
                            from ultralytics.data.utils import check_det_dataset
                            vdir = Path(check_det_dataset(str(a.data))["val"])
                            man["calibration_split"] = "val"
                            man["calibration_images_available"] = sum(
                                1 for p in vdir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
                            man["quantizer"] = "NNCF post-training quantization (Ultralytics export defaults)"
                        except Exception as e:
                            man["calibration_images_available"] = f"unknown ({e})"
                    variants.append((tag, Path(path), "OpenVINO", "cpu", False))
                except Exception as e:
                    man["error"] = f"{type(e).__name__}: {e}"
                    print(f"!! export {label} {tag} failed: {man['error']}")
                (out / f"export_{label}_{tag}.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
        for tag, model, runtime, device, half in variants:
            if not a.timing_only:
                try:
                    accuracy(model, label, tag, runtime, a.data, device, half, rows, tmp)
                except Exception as e:
                    print(f"!! accuracy {label} {tag} failed: {type(e).__name__}: {e}")
            if do_timing:
                dflt = (100, 1000, 5, 20) if a.stage == "fp" else (30, 300, 3, 10)
                prm = [v if v is not None else d for v, d in zip((a.warmup, a.frames, a.repeats, a.cooldown), dflt)]
                s = timing(model, label, tag, device, half, images, out, *prm)
                if s:
                    lines.append(f"{label:<9} {tag:<14} {runtime:<8} {str(device):<4} "
                                 f"FPS {s['fps'][0]:.1f} ± {s['fps'][1]:.1f}   latency {s['latency_mean_ms'][0]:.2f} ± "
                                 f"{s['latency_mean_ms'][1]:.2f} ms (p95 {s['latency_p95_ms_pooled']:.2f})")
    write_rows(out / "deployed_accuracy.csv", rows)
    text = [f"stage {a.stage}  {time.strftime('%Y-%m-%d %H:%M')}", json.dumps(vers)]
    if rows:
        text.append("\naccuracy on the test split (batch 1, rect=False, conf 0.001, IoU 0.7):")
        text += [f"  {r['model']:<9} {r['precision']:<14} P {r['P']}  R {r['R']}  mAP50 {r['mAP50']}  mAP50-95 {r['mAP50_95']}"
                 for r in rows]
    if lines:
        text.append("\nlatency (mean ± SD over repeats):")
        text += ["  " + x for x in lines]
    elif not do_timing:
        text.append("\ntiming skipped" + (f" (trainings running: {', '.join(busy)})" if busy else ""))
    (out / f"edge_summary_{a.stage}.txt").write_text("\n".join(text) + "\n", encoding="utf-8-sig")
    print("\n" + "\n".join(text))


if __name__ == "__main__":
    main()
