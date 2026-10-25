"""
Edge timing / power protocol with repeated measurements (Section 3.9, Supplementary Table S11).

  # Jetson (power from tegrastats, sampled every 100 ms, rail VDD_IN = whole module input)
  sudo nvpmodel -m <mode>; sudo jetson_clocks            # fix the power mode and clocks first
  python benchmark_edge.py --model scd_yolo_cotton.engine --images <dataset>/images/test \
         --tegrastats --out orin_scd

  # Raspberry Pi 4 + Coral USB (power from an external USB meter that logs timestamp,watts)
  python benchmark_edge.py --model scd_yolo_cotton_full_integer_quant_edgetpu.tflite \
         --images <dataset>/images/test --power-csv meter_log.csv --out coral_scd

Protocol (Section 3.9, Supplementary Table S11):
  * all test images are decoded and held in RAM before timing; frames are fed one at a time
    (batch 1), cycling through the test set;
  * W warm-up frames are run and discarded (default 100), then N timed frames (default 1,000);
  * the whole run is repeated R times (default 5) with a cool-down between repeats (default 60 s);
  * per frame, the wall-clock time of one model.predict() call on an in-memory image is recorded
    = letterbox pre-processing + inference + post-processing (NMS), synchronous, no overlap
    between frames;  FPS = N / total wall time of the timed window;
  * EXCLUDED: camera acquisition, image decoding/file I/O, data transfer to other devices,
    rendering/display and any downstream control actions;
  * power = mean of all power samples inside the timed window; energy per frame = power / FPS;
  * reported values = mean +/- SD over the R repeats (latency percentiles pooled over repeats).
"""
import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
RAIL = re.compile(r"(VDD_IN|POM_5V_IN)\s+(\d+)(?:mW)?/(\d+)")


def read_power_csv(path, t0, t1):
    vals = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ts = r.get("timestamp") or r.get("time")
            try:
                t = float(ts)
            except ValueError:
                t = datetime.fromisoformat(ts).timestamp()
            if t0 <= t <= t1:
                vals.append(float(r.get("watts") or r.get("power_w") or r.get("power")))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--half", action="store_true", help="only for .pt models; engines carry their own precision")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=60)
    ap.add_argument("--tegrastats", action="store_true")
    ap.add_argument("--interval", type=int, default=100, help="tegrastats sampling interval (ms)")
    ap.add_argument("--power-csv", default=None, help="external meter log with columns timestamp,watts")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="bench")
    a = ap.parse_args()

    from ultralytics import YOLO
    files = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG)
    # np.fromfile + imdecode also works for non-ASCII paths on Windows (cv2.imread does not)
    frames = [cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR) for p in files]
    frames = [f for f in frames if f is not None]
    print(f"{len(frames)} images held in memory")
    model = YOLO(a.model, task="detect")
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, verbose=False)
    if a.device is not None:
        kw["device"] = a.device
    if a.half:
        kw["half"] = True

    for i in range(a.warmup):
        model.predict(frames[i % len(frames)], **kw)

    reps = []
    all_lat = []
    for r in range(a.repeats):
        if r:
            time.sleep(a.cooldown)
        tlog = Path(f"{a.out}_tegrastats_rep{r}.txt")
        proc = None
        if a.tegrastats:
            proc = subprocess.Popen(["tegrastats", "--interval", str(a.interval)], stdout=open(tlog, "w"),
                                    stderr=subprocess.STDOUT)
            time.sleep(1.0)
        lat, pre, inf, post = [], [], [], []
        t_start_wall = time.time()
        t0 = time.perf_counter()
        for i in range(a.frames):
            f = frames[i % len(frames)]
            s = time.perf_counter()
            res = model.predict(f, **kw)
            lat.append((time.perf_counter() - s) * 1000)
            sp = res[0].speed
            pre.append(sp.get("preprocess", np.nan)); inf.append(sp.get("inference", np.nan)); post.append(sp.get("postprocess", np.nan))
        wall = time.perf_counter() - t0
        t_end_wall = time.time()
        power, n_samples = float("nan"), 0
        if proc is not None:
            proc.terminate()
            proc.wait()
            mw = [int(m.group(2)) for line in tlog.read_text(errors="ignore").splitlines() for m in [RAIL.search(line)] if m]
            mw = mw[1:] if len(mw) > 2 else mw
            power, n_samples = (float(np.mean(mw)) / 1000.0, len(mw)) if mw else (float("nan"), 0)
        elif a.power_csv:
            power, n_samples = read_power_csv(a.power_csv, t_start_wall, t_end_wall)
        fps = a.frames / wall
        rep = dict(repeat=r, window_start=datetime.fromtimestamp(t_start_wall).isoformat(timespec="seconds"),
                   window_end=datetime.fromtimestamp(t_end_wall).isoformat(timespec="seconds"),
                   fps=fps, latency_mean_ms=float(np.mean(lat)), latency_p50_ms=float(np.percentile(lat, 50)),
                   latency_p95_ms=float(np.percentile(lat, 95)), pre_ms=float(np.nanmean(pre)),
                   inf_ms=float(np.nanmean(inf)), post_ms=float(np.nanmean(post)), power_w=power,
                   power_samples=n_samples, energy_j_per_frame=power / fps if power == power else float("nan"))
        reps.append(rep)
        all_lat += lat
        print(json.dumps(rep))

    with open(f"{a.out}_repeats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reps[0]))
        w.writeheader()
        w.writerows(reps)

    def ms(key):
        v = [r[key] for r in reps if r[key] == r[key]]
        return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), float("nan"))

    summ = {k: ms(k) for k in ("fps", "latency_mean_ms", "pre_ms", "inf_ms", "post_ms", "power_w", "energy_j_per_frame")}
    summ["latency_p50_ms_pooled"] = float(np.percentile(all_lat, 50))
    summ["latency_p95_ms_pooled"] = float(np.percentile(all_lat, 95))
    summ["protocol"] = dict(warmup=a.warmup, frames=a.frames, repeats=a.repeats, cooldown_s=a.cooldown,
                            imgsz=a.imgsz, batch=1, conf=a.conf, iou=a.iou, images=len(frames),
                            power_source="tegrastats VDD_IN" if a.tegrastats else ("external meter" if a.power_csv else "none"))
    Path(f"{a.out}_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\nmean ± SD over repeats:")
    for k in ("fps", "latency_mean_ms", "power_w", "energy_j_per_frame"):
        print(f"  {k:<20} {summ[k][0]:.3f} ± {summ[k][1]:.3f}")


if __name__ == "__main__":
    main()
