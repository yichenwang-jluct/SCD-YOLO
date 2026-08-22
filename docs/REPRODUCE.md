# Reproducing the results in the paper

All runs use identical hyper-parameters; only the model YAML and the loss patch
differ between configurations.

```
epochs=200  batch=8  imgsz=640  workers=0  amp=False
pretrained=False  seed=0  patience=100  device=0
```

Optimiser, learning-rate schedule, warm-up and loss weights are the Ultralytics
defaults for `optimizer='auto'`. With 3,338 training images and a nominal batch
size of 64, `ceil(3338/64) x 200 = 10,600 > 10,000` iterations, so `auto`
selects SGD with an initial learning rate of 0.01.

> **Evaluation caveat.** Every number in the paper was measured with
> `rect=False`. Ultralytics defaults to rectangular inference for validation,
> which shifts mAP by a few tenths of a point. Always pass `rect=False`
> explicitly, as `scripts/eval_test_set.py` does.

---

## Main result — SCD-YOLO (Tables 3, 4, 5, 9, 15, 16)

```bash
python scripts/train_adaptive.py          # -> runs/scd_yolo_adaptive/
python scripts/eval_test_set.py           # test-set metrics
```

Produces `u_c_evolution.csv` in the run directory, which is the input to
`scripts/plot_uc_evolution.py` (Figure 10).

Expected: 1.92 M parameters, 5.7 GFLOPs, mAP@0.5 = 0.945, mAP@0.5:0.95 = 0.705
on the validation set (mean of three seeds).

---

## Run directories

The names below are the ones the released scripts produce. The original run
directories used for the paper had legacy names, listed in the last column for
traceability.

| Paper | Script | Released run name | Original directory |
|---|---|---|---|
| Baseline YOLOv13n (Tables 3, 5, 13-16) | `train_baseline.py` | `runs/yolov13n_baseline/` | `exp v13n` |
| SCD-YOLO / Table 16 "Training-adaptive" | `train_adaptive.py` | `runs/scd_yolo_adaptive/` | `新损失4in1` |
| Table 16 "Manually set u_c" | `train_manual_uc.py` | `runs/scd_yolo_manual_uc/` | `exp_(4in1gongkai)_CottonIoU_ClassAware` |
| Table 16 "Uniform (u = 0.95)" | `train_loss_ablation.py`, `CONFIG_VARIANT = 4` | `runs/scd_yolo_uniform_uc/` | `exp_3mod_CSN_SAD_CottonIoU_ClassAware` |

All four use the same architecture (all four contributions) except the baseline,
which is stock YOLOv13n. The `3mod` in the last directory name is legacy: that
run also uses the full four-module architecture.

---

## Ablation of the four modules (Tables 4 and 5)

The eight rows are combinations of the four contributions. Three of them live
in the model YAML, the fourth is the loss patch:

| # | Cotton-StarNet | Star-ADown | Cotton-SimAM | Cotton-IoU | How to build it |
|---|:---:|:---:|:---:|:---:|---|
| 1 | ✗ | ✗ | ✗ | ✗ | `scripts/train_baseline.py` |
| 2 | ✓ | ✗ | ✗ | ✗ | stock YAML, backbone line -> `cotton_starnet_s050` |
| 3 | ✗ | ✓ | ✗ | ✗ | stock YAML, lines 23 / 27 -> `StarADown` |
| 4 | ✗ | ✗ | ✓ | ✗ | stock YAML, lines 31-33 -> `CottonSimAM, [0.0001, 3, 0.5]` |
| 5 | ✗ | ✗ | ✗ | ✓ | stock YAML + `train_adaptive.py`'s loss patch |
| 6 | ✓ | ✓ | ✗ | ✗ | rows 2 and 3 combined |
| 7 | ✓ | ✓ | ✓ | ✗ | rows 2, 3 and 4 combined |
| 8 | ✓ | ✓ | ✓ | ✓ | `cfg/SCD-YOLO.yaml` + `train_adaptive.py` |

Row 8 is the released configuration. The line numbers refer to the comments in
`cfg/SCD-YOLO.yaml`, which mark every position that differs from stock YOLOv13n.

---

## Ablation of the loss (Table 9)

`scripts/train_loss_ablation.py` carries all five variants; set
`CONFIG_VARIANT` at the top of the file.

| Variant | Name | `inner_ratio` | `use_focaler` | `class_u` |
|---|---|---|---|---|
| 1 | CIoU (baseline) | — | False | — |
| 2 | Inner-CIoU | 1.2 | False | — |
| 3 | Focaler-CIoU | 1.0 | True | `[0.95] * 6` |
| 4 | Inner + Focaler (uniform) | 1.2 | True | `[0.95] * 6` |
| 5 | Cotton-IoU (class-aware, fixed) | 1.2 | True | `[0.9, 0.9, 0.7, 0.95, 0.95, 0.8]` |

The `Cotton-IoU` row of Table 9 is the **adaptive** variant, i.e.
`train_adaptive.py`, not variant 5.

---

## Upper-bound strategies (Table 16, Figure 11)

Three runs, same architecture (all four modules), differing only in how the
Focaler upper bound `u_c` is set:

| Table 16 column | Script | Setting | Run directory |
|---|---|---|---|
| Uniform (u = 0.95) | `train_loss_ablation.py` | `CONFIG_VARIANT = 4` | `runs/scd_yolo_uniform_uc/` |
| Manually set `u_c` | `train_manual_uc.py` | `class_u = [0.9, 0.9, 0.7, 0.95, 0.95, 0.8]` | `runs/scd_yolo_manual_uc/` |
| Training-adaptive `u_c` | `train_adaptive.py` | `u_min=0.70, u_max=0.95, beta=0.9, warmup=30` | `runs/scd_yolo_adaptive/` |

`scripts/plot_training_curve.py` reads the three `results.csv` files and
produces Figure 11.

> The curves in Figure 11 are **per-epoch validation** results logged during
> training. Table 16 reports the metrics of `best.pt`, i.e. the epoch that
> maximises the Ultralytics fitness `0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95`, which
> is generally not the last epoch. The two therefore do not have to agree at
> epoch 200.

---

## Comparisons against other backbones / downsamplers / attention (Tables 6, 7, 8)

These rows swap a single component of `cfg/SCD-YOLO.yaml` for a third-party
module (MobileNetV3, ShuffleNetV2, ADown, SPD-Conv, AKConv, SE, CBAM, ECA,
CoordAtt, SimAM). Their configuration files are **not redistributed here**;
please obtain the modules from the original authors and cite the corresponding
references given in the paper. Everything else — data split, hyper-parameters,
evaluation protocol — is unchanged.

---

## Figures

| Figure | Script | Input |
|---|---|---|
| 9 — per-class gain vs. baseline difficulty | — | values from Table 15 |
| 10 — evolution of `u_c` | `scripts/plot_uc_evolution.py` | `u_c_evolution.csv` |
| 11 — training curves of the three `u_c` strategies | `scripts/plot_training_curve.py` | three `results.csv` |

`scripts/plot_gate_bias.py` is an additional diagnostic, not used for any figure
in the paper. It measures whether the Cotton-StarNet gate actually counteracts
the amplitude bias of the star operation, by correlating the learned gate
weights with the mean channel response.

---

## Generalisation experiments (Tables 13 and 14)

Same protocol as the main experiments; only the dataset YAML changes. Neither
dataset is redistributed here — download them from the sources cited in the
paper and set `path:` in the corresponding YAML.

```bash
# PlantDoc — cross-crop (Table 13)
python scripts/eval_test_set.py --data cfg/plantdoc.yaml --split test \
    --weights weights/yolov13n_plantdoc.pt weights/scd_yolo_plantdoc.pt

# cotton-cqamc — in-domain, different source (Table 14)
python scripts/eval_test_set.py --data cfg/cotton_cqamc.yaml --split val \
    --weights weights/yolov13n_cqamc.pt weights/scd_yolo_cqamc.pt
```

To retrain rather than evaluate, point `data_yaml_path` in
`scripts/train_adaptive.py` at the corresponding YAML; nothing else changes.

---

## Released checkpoints

Attached to the GitHub Release, not tracked in git.

| File | Architecture | Dataset | Paper |
|---|---|---|---|
| `scd_yolo_cotton.pt` | SCD-YOLO (1.92 M) | cotton disease (6 classes) | Tables 3, 4, 5, 9, 15, 16 |
| `yolov13n_cotton.pt` | YOLOv13n (2.47 M) | cotton disease (6 classes) | baseline row of the same tables |
| `scd_yolo_plantdoc.pt` | SCD-YOLO | PlantDoc (30 classes) | Table 13 |
| `yolov13n_plantdoc.pt` | YOLOv13n | PlantDoc (30 classes) | Table 13 baseline |
| `scd_yolo_cqamc.pt` | SCD-YOLO | cotton-cqamc (5 classes) | Table 14 |
| `yolov13n_cqamc.pt` | YOLOv13n | cotton-cqamc (5 classes) | Table 14 baseline |
