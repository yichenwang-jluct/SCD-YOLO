# SCD-YOLO

Code accompanying the paper
**"SCD-YOLO: A Lightweight Cotton Leaf Disease Detector with a Class-Adaptive
Localization Loss"** (Wang, Li, Ge and He; Frontiers in Plant Science, under review).

SCD-YOLO builds on YOLOv13n and has four components:

| # | Component | What it does | File |
|---|---|---|---|
| 1 | **Cotton-StarNet** | star-operation backbone (StarNet-s050) with a channel gate on one branch of every star block | `modules/cotton_starnet.py` |
| 2 | **Star-ADown** | ADown-style downsampling whose two branches are fused by element-wise multiplication instead of concatenation; replaces the two stride-2 3×3 convolutions of the YOLOv13n neck | `modules/star_adown.py` |
| 3 | **Cotton-SimAM** | parameter-free SimAM attention with a spatial-coherence term that raises spatially coherent responses relative to isolated ones | `modules/cotton_simam.py` |
| 4 | **Cotton-IoU** | Inner-IoU + Focaler-IoU regression loss whose class-wise upper bound evolves during training from each class's localization quality, without per-class manual tuning | `modules/cotton_iou_loss_adaptive.py` |

On the cotton disease dataset (6 classes, 4,173 images), SCD-YOLO raises
mAP@0.5 from 0.897 to 0.942 and mAP@0.5:0.95 from 0.624 to 0.693 on the
validation set (means of three seeds; mAP@0.5:0.95 0.605 → 0.687 on the
independent test set) while cutting parameters by about 22% (2.47 M → 1.92 M)
and computation by about 8% (6.2 → 5.7 GFLOPs).

The full-factorial ablation in the paper attributes the gain mainly to the
star-operation backbone and Cotton-IoU, with a smaller contribution from
Star-ADown; the channel gate and Cotton-SimAM add no measurable accuracy, and
the adaptive upper bounds of Cotton-IoU perform on par with fixed bounds
(uniform, hand-set and validation-selected) and, on the test set, with plain
CIoU. Trained without the 427 field images and tested on them, SCD-YOLO is no
better than the baseline overall and is lower on four of the six classes, so its
advantage is established within the training source.

---

## Repository layout

```
modules/        the four components, the fixed-bound Cotton-IoU (loss ablation)
                and the ungated StarNet used as gate control
cfg/            model YAML (SCD-YOLO.yaml) and dataset YAMLs
cfg/ablation/   YAMLs of every ablation and control architecture (Tables 4-7)
scripts/        training, evaluation, analysis and plotting
scripts/edge/   accuracy and timing of the exported models (Table 20)
docs/           INSTALL.md (module registration) and REPRODUCE.md (run by run)
```

## Quick start

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

SCD-YOLO runs against a **modified Ultralytics YOLOv13 source tree**, not the
PyPI package: a few small edits to `ultralytics/nn/tasks.py` are needed before
the YAML can be parsed. Follow **[docs/INSTALL.md](docs/INSTALL.md)** — it is
the step most reproductions get stuck on.

Then point `path:` in `cfg/cotton_disease.yaml` at the dataset and run, from
the root of the YOLOv13 tree with that root on `PYTHONPATH`:

```bash
# SCD-YOLO and the YOLOv13n baseline, three seeds each (as in the paper)
python <repo>/scripts/run_experiment.py --arch S1_D1_A1 --loss adaptive --name fac_S1D1A1I1 --seeds 0 1 2 --data <repo>/cfg/cotton_disease.yaml
python <repo>/scripts/run_experiment.py --arch S0_D0_A0 --loss ciou     --name fac_S0D0A0I0 --seeds 0 1 2 --data <repo>/cfg/cotton_disease.yaml

# evaluation with the protocol of the paper (rect=False, conf=0.001, NMS IoU=0.7)
python <repo>/scripts/eval_test_set.py --data <repo>/cfg/cotton_disease.yaml --split test \
    --weights runs_revision/fac_S0D0A0I0_seed0/weights/best.pt runs_revision/fac_S1D1A1I1_seed0/weights/best.pt
```

Set `PYTHONPATH` with `set PYTHONPATH=%CD%` on Windows or
`export PYTHONPATH=$PWD` elsewhere — Ultralytics is a source tree here, not an
installed package. The single-run scripts `scripts/train_adaptive.py` and
`scripts/train_baseline.py` train seed 0 with the same settings.

Sanity check: built with `nc=6` and fused, the model should report
**1,915,507 parameters** and **5.7 GFLOPs**. Anything else means a
registration edit did not take. See docs/INSTALL.md step 7.

Every configuration of the paper, with its exact command, and the evaluation
and analysis scripts are listed in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

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

## Checkpoints

Trained checkpoints are attached to the GitHub releases (not tracked in git):

* **v1.1.0** — `SCD-YOLO_checkpoints_v1.1.0.zip`: the 102 models evaluated with
  three seeds in the paper, named `<run>_seed<k>.pt`: 30 configurations on the
  cotton dataset, and SCD-YOLO and YOLOv13n on PlantDoc and cotton-cqamc
  (Tables 13 and 14), each with seeds 0, 1, 2; the run names are explained in
  docs/REPRODUCE.md.
* **v1.0.0** — the checkpoints released with the original submission (PlantDoc,
  cotton-cqamc and the cotton dataset); the revised paper uses those of v1.1.0.

---

## Dataset

The dataset (4,173 images, 6,012 instances) combines a curated subset of a
public Roboflow Universe dataset (CC BY 4.0) with 427 images collected and
annotated in the field by the authors. All 427 field images are in the
training split; the public images were split randomly into the remaining
training images and the validation and test splits (3,338 / 417 / 418 images
in total). See the Data Availability Statement of the paper for the download
link and the licence terms.

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
@article{wang2026scdyolo,
  title   = {SCD-YOLO: A Lightweight Cotton Leaf Disease Detector with a
             Class-Adaptive Localization Loss},
  author  = {Wang, Yichen and Li, Shuangyuan and Ge, Ji and He, Yuli},
  journal = {Frontiers in Plant Science},
  note    = {under review},
  year    = {2026}
}
```

## Acknowledgements

Built on [Ultralytics](https://github.com/ultralytics/ultralytics) and YOLOv13.
Cotton-StarNet extends StarNet (Ma et al., CVPR 2024); Star-ADown extends ADown
(YOLOv9); Cotton-SimAM extends SimAM (Yang et al., ICML 2021); Cotton-IoU builds
on Inner-IoU and Focaler-IoU.
