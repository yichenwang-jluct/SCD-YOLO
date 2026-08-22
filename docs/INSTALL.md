# Installation

SCD-YOLO is implemented as a set of drop-in modules for a **local Ultralytics
YOLOv13 source tree**. It is not a standalone package: the modules have to be
registered with `parse_model` before the YAML in `cfg/` can be parsed.

Do **not** `pip install ultralytics` and expect this to work — the PyPI package
does not contain the registration hooks described below.

---

## 1. Environment

```bash
conda create -n scd-yolo python=3.10
conda activate scd-yolo

# PyTorch first, from the official index
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Reference configuration: Python 3.10.9, Windows 11, NVIDIA RTX 5070 Ti (16 GB),
CUDA 12.8, PyTorch 2.7.0, Ultralytics 8.3.174, timm 1.0.14.

---

## 2. Get the YOLOv13 source tree

Clone the YOLOv13 repository so that you have a writable `ultralytics/`
directory. All paths below are relative to that tree.

---

## 3. Copy the modules

```
modules/cotton_starnet.py           ->  ultralytics/nn/newsAddmodules/cotton_starnet.py
modules/star_adown.py               ->  ultralytics/nn/newsAddmodules/star_adown.py
modules/cotton_simam.py             ->  ultralytics/nn/newsAddmodules/cotton_simam.py
modules/cotton_iou_loss.py          ->  ultralytics/nn/newsAddmodules/cotton_iou_loss.py
modules/cotton_iou_loss_adaptive.py ->  ultralytics/nn/newsAddmodules/cotton_iou_loss_adaptive.py
```

An `__init__.py` is included in `modules/` — copy it across as well if the
target directory does not already have one.

---

## 4. Register the modules in `ultralytics/nn/tasks.py`

Four edits. Line numbers refer to a stock YOLOv13 `tasks.py`; search for the
anchor text instead of trusting the numbers.

### 4.1 Imports (top of the file)

```python
from ultralytics.nn.newsAddmodules.cotton_starnet import (
    cotton_starnet_s050, cotton_starnet_s100, cotton_starnet_s150,
)
from ultralytics.nn.newsAddmodules.star_adown import StarADown
from ultralytics.nn.newsAddmodules.cotton_simam import CottonSimAM
```

### 4.2 Register `StarADown` as a channel-taking module

Find the large `elif m in {...}` set inside `parse_model` that handles modules
whose first argument is the output channel count (it contains `Conv`, `C3k2`,
`ADown`, ...) and add `StarADown` to it:

```python
elif m in {Conv, ..., ADown, StarADown, ...}:
```

### 4.3 Add a branch for `CottonSimAM`

`CottonSimAM` is shape-preserving and takes its hyper-parameters straight from
the YAML, so its `args` must **not** be rewritten:

```python
elif m is CottonSimAM:
    c2 = ch[f]      # shape-preserving, zero learnable parameters
    # keep the YAML args = [lambda_param, kernel_size, gamma] untouched,
    # so that gamma can be swept (gamma = 0 recovers the original SimAM)
```

Place it next to the existing `elif m is SimAM:` branch.

### 4.4 Register `Cotton-StarNet` as a backbone

Find the `elif m in {...}` set whose body is

```python
m = m(*args)
c2 = m.channel
backbone = True
```

and add the three factory functions:

```python
elif m in {..., cotton_starnet_s050, cotton_starnet_s100, cotton_starnet_s150}:
```

`CottonStarNet` exposes `self.channel = [32, 16, 32, 64, 128]`, which is what
`c2 = m.channel` reads.

---

## 5. The loss is injected at run time, not registered

Cotton-IoU is not a YAML module. The training scripts monkey-patch
`DetectionModel.init_criterion` **before** `YOLO(...)` is constructed:

```python
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.newsAddmodules.cotton_iou_loss_adaptive import (
    CottonDetectionLossAdaptive, attach_adaptive_callback,
)

def _patched_init_criterion(self):
    return CottonDetectionLossAdaptive(self, **ADAPTIVE_CFG)

DetectionModel.init_criterion = _patched_init_criterion   # must precede YOLO()
model = YOLO("cfg/SCD-YOLO.yaml")
attach_adaptive_callback(model)                            # writes u_c_evolution.csv
```

The order matters: patching after `YOLO(...)` has no effect, because the
criterion is built lazily on the first `loss()` call of an already-constructed
model.

---

## 6. Running the scripts

Because Ultralytics is used as a **source tree** rather than an installed
package, Python must be able to find it. Run everything from the root of the
YOLOv13 tree and put that root on `PYTHONPATH`:

```cmd
cd <path-to>\YOLOv13_20260401
set PYTHONPATH=%CD%
python <path-to>\SCD-YOLO\scripts\eval_test_set.py --data ... --weights ...
```

```bash
cd <path-to>/YOLOv13_20260401
export PYTHONPATH=$PWD
python <path-to>/SCD-YOLO/scripts/eval_test_set.py --data ... --weights ...
```

Without this you will get `ModuleNotFoundError: No module named 'ultralytics'`,
because the scripts live outside the source tree.

## 7. Check the installation

```python
from ultralytics.nn.tasks import DetectionModel

m = DetectionModel("cfg/SCD-YOLO.yaml", nc=6, verbose=False)
print("training mode :", sum(p.numel() for p in m.parameters()))   # 1,920,723
m.eval().fuse()
print("fused (eval)  :", sum(p.numel() for p in m.parameters()))   # 1,915,507
```

The paper reports the **fused** count, 1,915,507 parameters and 5.7 GFLOPs at
640x640 — this is also what Ultralytics prints during `val()`. The unfused
training-mode graph has 1,920,723 parameters; the 5,216 difference is the
BatchNorm weights and biases that fusing folds into the preceding convolutions.

Note that `nc` must be passed explicitly: the YAML carries the placeholder
`nc: 80`, which Ultralytics normally overrides from the dataset YAML.

A different number means one of the four registration edits did not take.

---

## Known limitation: single-GPU only

`CottonBboxLossAdaptive` accumulates the per-class localisation quality `q_c` in
process-local buffers (`iou_sum`, `iou_cnt`) with no all-reduce. Under DDP each
rank would only see its own shard, so the adaptive upper bounds would be driven
by partial statistics. All results in the paper were produced on a single GPU.
