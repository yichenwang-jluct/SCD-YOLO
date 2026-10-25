# Reproducing the results in the paper

All commands are run from the root of the YOLOv13 source tree with that root on
`PYTHONPATH` (see INSTALL.md); `<repo>` is this repository.

## Protocol

Every run uses the same settings; only the architecture YAML and the
box-regression loss differ between configurations.

```
epochs=200  batch=8  imgsz=640  workers=0  amp=False  pretrained=False
deterministic=True  patience=100  optimizer=auto  seeds 0, 1, 2
```

`optimizer='auto'` selects SGD with an initial learning rate of 0.01, because
Ultralytics computes the iterations with its nominal batch size of 64:
`ceil(3338/64) x 200 = 10,600 > 10,000`.

Evaluation (all tables): `rect=False, conf=0.001, NMS IoU=0.7, imgsz=640` on
`weights/best.pt`, i.e. the epoch with the highest validation fitness
`0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95`. The overall value of a run is the
macro-average of its class-wise values; tables report the mean and the sample
standard deviation (n - 1) over the three seeds.

> **Evaluation caveat.** Ultralytics defaults to rectangular inference for
> validation, which shifts mAP by a few tenths of a point. Always pass
> `rect=False` explicitly, as all evaluation scripts here do.

---

## The 90 runs (30 configurations x 3 seeds)

Every run is started with `scripts/run_experiment.py`, which writes
`runs_revision/<run>_seed<k>/weights/best.pt`; the released checkpoint
`<run>_seed<k>.pt` is that file.

```bash
python <repo>/scripts/run_experiment.py --arch <arch> --loss <loss> [options] \
    --name <run> --seeds 0 1 2 --data <repo>/cfg/cotton_disease.yaml
```

`--arch` names a YAML in `cfg/ablation/` (`S1_D0_A1` -> `abl_S1_D0_A1.yaml`).
S = Cotton-StarNet, D = Star-ADown, A = Cotton-SimAM, I = Cotton-IoU (1 = on).

| Run | `--arch` | `--loss` and options | Paper |
|---|---|---|---|
| `fac_S?D?A?I0` (8 runs) | `S?_D?_A?` | `ciou` | Tables 4, 5 |
| `fac_S?D?A?I1` (8 runs) | `S?_D?_A?` | `adaptive` | Tables 4, 5 |
| `ctrl_adown` | `ctrl_S1_ADown_A0` | `ciou` | Table 7 (ADown) |
| `gate_ungated_only` | `ctrl_ungated_D0_A0` | `ciou` | Table 6, Suppl. Table S7 |
| `gate_ungated_full` | `ctrl_ungated_D1_A1` | `adaptive` | Suppl. Table S7 |
| `uc_uniform0.75` ... `uc_uniform0.95` (5 runs) | `S1_D1_A1` | `uniform --u 0.75` ... `--u 0.95` | Tables 16, 17 (C1), Suppl. Table S5 |
| `uc_manual` | `S1_D1_A1` | `fixed --uc 0.90,0.90,0.70,0.95,0.95,0.80` | Tables 16, 17 |
| `uc_fixedfinal` (C2) | `S1_D1_A1` | `fixed --uc 0.7956,0.8450,0.7000,0.9500,0.7620,0.8586` | Table 17 |
| `uc_replay` (C3a) | `S1_D1_A1` | `schedule --schedule schedules/replay_mean.csv` | Table 17 |
| `uc_linear` (C3b) | `S1_D1_A1` | `schedule --schedule schedules/linear_ramp.csv` | Table 17 |
| `src_scd` | `S1_D1_A1` | `adaptive`, `--data <split>/data_srcdisjoint.yaml` | Section 3.6 |
| `src_base` | `S0_D0_A0` | `ciou`, `--data <split>/data_srcdisjoint.yaml` | Section 3.6 |

`fac_S1D1A1I1` is SCD-YOLO and `fac_S0D0A0I0` is YOLOv13n (Tables 3, 15 and 16).
The rows of our modules in Tables 6-12 are the matching factorial runs:
Cotton-StarNet `fac_S1D0A0I0`, Star-ADown `fac_S1D1A0I0` (Table 7, together with
`ctrl_adown`; the stride-2 convolutions are `fac_S1D0A0I0`), no attention
`fac_S1D1A0I0` and Cotton-SimAM `fac_S1D1A1I0` (Table 8), CIoU `fac_S1D1A1I0`
and Cotton-IoU `fac_S1D1A1I1` (Table 9), and the default settings of Tables
10-12. `scripts/train_split.py` and `scripts/train_extra.py` hold the same list
for distributing the runs over several computers.

`--loss` values: `ciou` = stock Ultralytics loss; `adaptive` = Cotton-IoU with
online class-adaptive upper bounds (r = 1.2, range [0.70, 0.95], beta = 0.9,
30 warm-up epochs at 0.825, min_samples = 10); `uniform`, `fixed` and
`schedule` = the same loss with the upper bounds set from outside (one bound,
class-wise fixed bounds, or a predetermined per-epoch schedule).

### Schedules of C2, C3a and C3b

The C2 bounds are the final bounds of the seed-0 SCD-YOLO run (last column of
Table 16); the schedules are generated from the `u_c_evolution.csv` of that run:

```bash
python <repo>/scripts/make_uc_schedules.py --csv runs_revision/fac_S1D1A1I1_seed0/u_c_evolution.csv --out schedules
```

`replay_mean.csv` replays the seed-0 trajectory (C3a); `linear_ramp.csv` holds
0.825 during the warm-up and then rises linearly to the final bounds, which are
reached at the end of training (C3b); `fixed_final.txt` holds the C2 bounds.

### Source-disjoint split (Section 3.6)

Train/validation = the public images of the training and validation splits,
test = all 427 field images (plus the public test images as in-source reference).
`find_field_images.py` identifies the field images inside the merged dataset (we
matched them by file name to the field-collected images with `--originals`):

```bash
python <repo>/scripts/find_field_images.py --data <repo>/cfg/cotton_disease.yaml --originals <folder with the 427 field images> --auto --write field_list.txt
python <repo>/scripts/make_source_disjoint_split.py --data <repo>/cfg/cotton_disease.yaml --out <split> --field-list field_list.txt
```

---

## Evaluation and analysis

| Script | Output | Paper |
|---|---|---|
| `final_eval.py` | runs the four steps below in order and collects the results | — |
| `eval_all.py` | per-seed, per-class P, R, mAP of every finished run (val and test; the source-disjoint runs also on the field images) and the per-image statistics for the bootstrap | Tables 3-9, 15-17, Suppl. Tables S4, S5, S7 |
| `bootstrap_ci.py` (with `apcore.py`) | percentile bootstrap intervals, images or near-duplicate groups resampled | Suppl. Tables S9, S10 |
| `plot_gate_bias.py` | what the Cotton-StarNet gate learns | Suppl. Table S7, Figure S1 |
| `simam_probe.py` | artefact-injection test and attention statistics of Cotton-SimAM | Suppl. Table S8, Note S3 |
| `per_seed_summary.py` | mean +/- SD tables and consistency checks | Suppl. Tables S4, S5 |
| `factorial_analysis.py` | Shapley values and factorial effects of the 2^4 ablation | Suppl. Table S6 |
| `failure_cases.py` | automatic failure-case classification | Table 18, Suppl. Note S2 |
| `profile_models.py` | parameters and GFLOPs (fused), layer-by-layer table | Tables 4, 7, Suppl. Table S1 |
| `eval_external_seeds.py` | PlantDoc and cotton-cqamc: per-seed results on the validation and test split | Tables 13, 14 |
| `eval_external.py`, `dataset_stats.py` | split statistics of the external datasets | Suppl. Table S3 |
| `edge/edge_pc.py` (with `edge/benchmark_edge.py`) | FP32 / FP16 / INT8 accuracy and repeated timing | Table 20, Suppl. Table S11 |
| `eval_per_seed.py`, `collect_val_stats.py` | stand-alone versions of the per-seed evaluation and of the per-image statistics | — |

```bash
python <repo>/scripts/final_eval.py --data <repo>/cfg/cotton_disease.yaml --src-dir <split> \
    --groups <near-duplicate groups of the field images, csv: image,group> --out eval_final
python <repo>/scripts/factorial_analysis.py eval_final/per_seed_factorial.csv --split val --out factorial_val
python <repo>/scripts/failure_cases.py --data <repo>/cfg/cotton_disease.yaml --split test \
    --weights runs_revision/fac_S1D1A1I1_seed0/weights/best.pt --out failures_scd_seed0
python <repo>/scripts/edge/edge_pc.py --stage fp --data <repo>/cfg/cotton_disease.yaml --out edge
python <repo>/scripts/edge/edge_pc.py --stage int8 --data <repo>/cfg/cotton_disease.yaml --out edge
```

The INT8 stage needs OpenVINO, NNCF and ONNX in a separate environment
(versions in Supplementary Table S11); it is calibrated on the validation
images only.

---

## Single-run scripts of the original experiments

| Script | Setting | Paper |
|---|---|---|
| `train_baseline.py` | YOLOv13n, seed 0 | baseline |
| `train_adaptive.py` | SCD-YOLO, seed 0; writes `u_c_evolution.csv` | Figure 10 |
| `train_manual_uc.py` | SCD-YOLO with fixed manual bounds, seed 0 | Figure 11 |
| `train_loss_ablation.py` | loss variants 1-5 (`CONFIG_VARIANT`); variants 2 and 3 are the Inner-CIoU and Focaler-CIoU rows, variant 4 the uniform bound of Figure 11 | Table 9, Figure 11 |
| `plot_uc_evolution.py` | evolution of u_c (seed 0) | Figure 10 |
| `plot_training_curve.py` | validation curves of the three upper-bound settings (seed 0) | Figure 11 |

They use the same settings as `run_experiment.py` for seed 0. The Cotton-IoU
row of Table 9 is the adaptive loss (`train_adaptive.py`), not variant 5.

---

## Rows of other methods and settings

The rows of the other detectors (Table 3), of the third-party modules in
Tables 6-8 (MobileNetV3, ShuffleNetV2, GhostNet, AKConv, SPD-Conv, SE, CBAM,
ECA, CoordAtt, SimAM), of the other losses in Table 9 and of the non-default
hyperparameter settings in Tables 10-12 are means +/- SD over three runs each,
evaluated with the same settings as our configurations (rect=False,
conf=0.001, NMS IoU=0.7). Their checkpoints are not part of the release.
The third-party modules are not redistributed here; please obtain them from
their authors. The non-default settings are obtained by changing the
`CottonSimAM` arguments `[lambda, k, gamma]` in the YAML (Table 10) or the
`ADAPTIVE_CFG` values `inner_ratio`, `beta`, `u_min`, `u_max` (Tables 11, 12).

## Cross-dataset experiments (Tables 13 and 14)

Both models are retrained with seeds 0, 1 and 2 on each dataset under the same
protocol (12 runs, `runs_revision/ext_<plantdoc|cqamc>_<scd|base>_seed<k>`;
SCD-YOLO with the adaptive Cotton-IoU loss) and evaluated on the official
validation and test splits of each release:

```bash
python <repo>/scripts/train_external_seeds.py --plantdoc <PlantDoc.v4i.yolov8 folder> --cqamc <cotton-cqamc folder>
python <repo>/scripts/eval_external_seeds.py --plantdoc <PlantDoc.v4i.yolov8 folder> --cqamc <cotton-cqamc folder> --out eval_ext3
```

`train_external_seeds.py` may be started in several windows at once: each
window trains the next run that is neither finished nor being trained by
another window. `eval_external_seeds.py` writes the per-seed, per-class values
and the mean ± SD of Tables 13 and 14; it evaluates on short-name copies of the
splits because some PlantDoc file names exceed the Windows path limit.
`eval_external.py` gives the split statistics of Supplementary Table S3.

Neither dataset is redistributed here; download the Roboflow YOLOv8 exports
from the sources cited in the paper.

---

## Released checkpoints

Attached to the GitHub releases, not tracked in git.

| Release | File | Content |
|---|---|---|
| v1.1.0 | `SCD-YOLO_checkpoints_v1.1.0.zip` | `<run>_seed<k>.pt` for the 30 runs above and the 4 cross-dataset runs `ext_<plantdoc\|cqamc>_<scd\|base>`, seeds 0, 1, 2 (102 files) |
| v1.0.0 | `scd_yolo_plantdoc.pt`, `yolov13n_plantdoc.pt` | PlantDoc, released with the original submission |
| v1.0.0 | `scd_yolo_cqamc.pt`, `yolov13n_cqamc.pt` | cotton-cqamc, released with the original submission |
| v1.0.0 | `scd_yolo_cotton.pt`, `yolov13n_cotton.pt` | cotton dataset, released with the original submission |
