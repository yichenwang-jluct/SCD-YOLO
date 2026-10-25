"""
Train SCD-YOLO with fixed, manually set class-wise upper bounds (Table 16, "Manually set u_c").

Model: Cotton-StarNet + Star-ADown + Cotton-SimAM (cfg/SCD-YOLO.yaml). Loss: Cotton-IoU components
(Inner-IoU with r = 1.2 and Focaler-IoU) with u_c = [0.90, 0.90, 0.70, 0.95, 0.95, 0.80] for
[blight, curl, grey_mildew, healthy, leaf_spot, wilt], fixed during training.

Training protocol of Section 2.6, seed 0. The three-seed runs reported in the paper were started with
    python scripts/run_experiment.py --arch S1_D1_A1 --loss fixed --uc 0.90,0.90,0.70,0.95,0.95,0.80 \
        --name uc_manual --seeds 0 1 2 --data ...
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.newsAddmodules.cotton_iou_loss import CottonDetectionLoss


# ============================================================================
# CONFIG: loss variant (5 = fixed, manually set class-wise bounds)
# ============================================================================
CONFIG_VARIANT = 5

ABLATION_CONFIGS = {
    1: {"name": "CIoU_baseline", "use_cotton": False, "kwargs": {}},
    2: {"name": "Inner_CIoU", "use_cotton": True,
        "kwargs": dict(inner_ratio=1.2, use_focaler=False, class_u=None)},
    3: {"name": "Focaler_CIoU_uniform", "use_cotton": True,
        "kwargs": dict(inner_ratio=1.0, use_focaler=True, class_u=[0.95] * 6)},
    4: {"name": "Inner_Focaler_uniform", "use_cotton": True,
        "kwargs": dict(inner_ratio=1.2, use_focaler=True, class_u=[0.95] * 6)},
    5: {"name": "CottonIoU_ClassAware",   # ★ proposed method
        "use_cotton": True,
        "kwargs": dict(
            inner_ratio=1.2,
            use_focaler=True,
            # class order: [blight, curl, grey_mildew, healthy, leaf_spot, wilt]
            class_u=[0.90, 0.90, 0.70, 0.95, 0.95, 0.80],
        )},
}


# model YAML (all four modules) and dataset YAML
# (adjust the paths if your layout differs)
model_yaml_path = "cfg/SCD-YOLO.yaml"
data_yaml_path  = "cfg/cotton_disease.yaml"


if __name__ == '__main__':
    cfg = ABLATION_CONFIGS[CONFIG_VARIANT]
    print(f"\n{'=' * 70}")
    print(f"  SCD-YOLO training with fixed, manually set u_c")
    print(f"    Backbone   : Cotton-StarNet")
    print(f"    Downsample : Star-ADown")
    print(f"    Attention  : Cotton-SimAM (in the YAML)")
    print(f"    Loss       : {cfg['name']} (fixed class-wise bounds)")
    print(f"    Loss kwargs: {cfg['kwargs']}")
    print(f"{'=' * 70}\n")

    # ---- inject the Cotton-IoU loss (must precede YOLO()) ----
    if cfg["use_cotton"]:
        def _patched_init_criterion(self):
            return CottonDetectionLoss(self, **cfg["kwargs"])
        DetectionModel.init_criterion = _patched_init_criterion
        print("[Cotton-IoU] DetectionModel.init_criterion monkey-patched ✓\n")
    else:
        print("[Cotton-IoU] Using vanilla v8DetectionLoss (CIoU)\n")

    # ---- training (protocol of Section 2.6) ----
    model = YOLO(model_yaml_path)

    results = model.train(
        data=data_yaml_path,
        cache=False,
        imgsz=640,
        epochs=200,
        workers=0,
        batch=8,
        amp=False,                          # FP32 training
        pretrained=False,                   # trained from scratch
        seed=0,
        patience=100,
        device='0',
        project="runs",
        name='scd_yolo_manual_uc',
    )
