# SCD-YOLO

Code accompanying the paper
**"SCD-YOLO: A Lightweight Model for Cotton Leaf Disease Detection and an
Adaptive Localization Loss for Training."**

SCD-YOLO builds on YOLOv13n and contributes four components, of which the
class-adaptive Cotton-IoU loss is the core one:

| # | Component | What it does | File |
|---|---|---|---|
| 1 | **Cotton-StarNet** | adds channel-adaptive gating to the star operation, so channels are recalibrated by content before multiplicative fusion | `modules/cotton_starnet.py` |
| 2 | **Star-ADown** | replaces ADown's channel concatenation with multiplicative fusion, preserving lesion boundaries under downsampling | `modules/star_adown.py` |
| 3 | **Cotton-SimAM** | adds a spatial-coherence constraint to parameter-free attention, separating clustered lesions from isolated specular noise | `modules/cotton_simam.py` |
| 4 | **Cotton-IoU** | lets each class's localisation target evolve during training from its real-time localisation quality, removing per-class manual tuning | `modules/cotton_iou_loss_adaptive.py` |

On the cotton disease dataset (6 classes, 4,173 images), SCD-YOLO raises
mAP@0.5 from 0.890 to 0.945 and mAP@0.5:0.95 from 0.624 to 0.705 while cutting
parameters by about 22% (2.47 M → 1.92 M) and computation by about 8%
(6.2 → 5.7 GFLOPs).

---

## Repository layout

```
modules/    the four contributions, plus the fixed-bound Cotton-IoU used for ablation
cfg/        model YAML and dataset YAML
scripts/    training, evaluation and plotting
docs/       INSTALL.md (how to register the modules) and REPRODUCE.md
```

## Quick start

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

SCD-YOLO runs against a **modified Ultralytics YOLOv13 source tree**, not the
PyPI package: four small edits to `ultralytics/nn/tasks.py` are needed before
the YAML can be parsed. Follow **[docs/INSTALL.md](docs/INSTALL.md)** — it is
the step most reproductions get stuck on.

Then point `path:` in `cfg/cotton_disease.yaml` at the dataset and run:

```bash
python scripts/train_baseline.py     # baseline YOLOv13n -> runs/yolov13n_baseline/
python scripts/train_adaptive.py     # SCD-YOLO          -> runs/scd_yolo_adaptive/

python scripts/eval_test_set.py --data cfg/cotton_disease.yaml --split test \
    --weights weights/yolov13n_cotton.pt weights/scd_yolo_cotton.pt
```

Run the scripts from the root of the YOLOv13 tree with that root on
`PYTHONPATH` (`set PYTHONPATH=%CD%` on Windows, `export PYTHONPATH=$PWD`
elsewhere) — Ultralytics is a source tree here, not an installed package.

Sanity check: built with `nc=6` and fused, the model should report
**1,915,507 parameters** and **5.7 GFLOPs**. Anything else means a
registration edit did not take. See docs/INSTALL.md step 7.

For the ablations, the loss variants and the table-by-table mapping, see
**[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

---

## Two things worth knowing before you run it

**Always evaluate with `rect=False`.** Ultralytics uses rectangular inference
for validation by default; every number in the paper was measured with it
disabled. This is worth a few tenths of a mAP point and is the most likely
cause of small mismatches.

**Single-GPU only.** The adaptive loss accumulates per-class localisation
quality in process-local buffers with no all-reduce, so under DDP each rank
would only see its own shard. All reported results are single-GPU.

---

## Dataset

The dataset (4,173 images, 6,012 instances, 8:1:1 split) combines a curated
subset of a public Roboflow Universe dataset (CC BY 4.0) with 427 images
collected and annotated in the field by the authors. See the Data Availability
statement of the paper for the download link and the licence terms.

The class order in `cfg/cotton_disease.yaml` is load-bearing — it indexes
`class_u` in the Cotton-IoU loss:

```
0 blight   1 curl   2 grey_mildew   3 healthy   4 leaf_spot   5 wilt
```

---

## Licence

Released under **AGPL-3.0**, matching the licence of the upstream Ultralytics
project this work builds on. See `LICENSE`.

## Citation

```bibtex
@article{scdyolo,
  title  = {SCD-YOLO: A Lightweight Model for Cotton Leaf Disease Detection
            and an Adaptive Localization Loss for Training},
  author = {Wang, Yichen and Liang, Jiaming and Li, Shuangyuan},
  year   = {2026}
}
```

## Acknowledgements

Built on [Ultralytics](https://github.com/ultralytics/ultralytics) and YOLOv13.
Cotton-StarNet extends StarNet (Ma et al., CVPR 2024); Star-ADown extends ADown
(YOLOv9); Cotton-SimAM extends SimAM (Yang et al., ICML 2021); Cotton-IoU builds
on Inner-IoU and Focaler-IoU.
