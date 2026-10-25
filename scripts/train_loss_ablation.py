"""
Training script for the loss ablation (Table 9); choose the variant with CONFIG_VARIANT below.

All variants use the SCD-YOLO architecture (cfg/SCD-YOLO.yaml) and the training protocol of
Section 2.6 (epochs=200, batch=8, imgsz=640, workers=0, amp=False, pretrained=False).
The Cotton-IoU row of Table 9 is the adaptive loss of scripts/train_adaptive.py, not variant 5
(variant 5 uses the fixed, manually set class-wise bounds).
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.newsAddmodules.cotton_iou_loss import CottonDetectionLoss


# ============================================================================
# CONFIG: choose the ablation variant here
# ============================================================================
CONFIG_VARIANT = 5    # variant to train (1-5)


ABLATION_CONFIGS = {
    1: {
        "name": "CIoU_baseline",
        "use_cotton": False,
        "kwargs": {},
    },
    2: {
        "name": "Inner_CIoU",
        "use_cotton": True,
        "kwargs": dict(inner_ratio=1.2, use_focaler=False, class_u=None),
    },
    3: {
        "name": "Focaler_CIoU_uniform",
        "use_cotton": True,
        "kwargs": dict(inner_ratio=1.0, use_focaler=True, class_u=[0.95] * 6),
    },
    4: {
        "name": "Inner_Focaler_uniform",
        "use_cotton": True,
        "kwargs": dict(inner_ratio=1.2, use_focaler=True, class_u=[0.95] * 6),
    },
    5: {
        "name": "CottonIoU_ClassAware",   # ★ proposed method
        "use_cotton": True,
        "kwargs": dict(
            inner_ratio=1.2,
            use_focaler=True,
            # class order: [blight, curl, grey_mildew, healthy, leaf_spot, wilt]
            class_u=[0.9, 0.9, 0.7, 0.95, 0.95, 0.8],
        ),
    },
}


# model and dataset YAML
model_yaml_path = "cfg/SCD-YOLO.yaml"
data_yaml_path = "cfg/cotton_disease.yaml"


if __name__ == '__main__':
    # ====================================================================
    # apply the Cotton-IoU monkey-patch (must precede YOLO())
    # ====================================================================
    cfg = ABLATION_CONFIGS[CONFIG_VARIANT]
    print(f"\n{'='*70}")
    print(f"  Running ablation variant {CONFIG_VARIANT}: {cfg['name']}")
    print(f"  Config: {cfg['kwargs']}")
    print(f"{'='*70}\n")

    if cfg["use_cotton"]:
        def _patched_init_criterion(self):
            return CottonDetectionLoss(self, **cfg["kwargs"])
        DetectionModel.init_criterion = _patched_init_criterion
        print(f"[Cotton-IoU] DetectionModel.init_criterion monkey-patched ✓\n")
    else:
        print(f"[Cotton-IoU] Using vanilla v8DetectionLoss (CIoU)\n")

    # ====================================================================
    # training (protocol of Section 2.6)
    # ====================================================================
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
        project="runs",
        name=f'loss_ablation_{cfg["name"]}',
    )
