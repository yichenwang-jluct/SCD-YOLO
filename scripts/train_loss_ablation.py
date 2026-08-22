"""
Training script for Cotton-IoU ablation experiments.

★ 严格对照用户之前的 train.py 设置:
   - workers=0 (与原脚本一致)
   - amp=False (与原脚本一致, 防 NaN)
   - pretrained=False (原脚本未使用预训练)
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.newsAddmodules.cotton_iou_loss import CottonDetectionLoss


# ============================================================================
# CONFIG: 改这里来切换不同的 ablation variant
# ============================================================================
CONFIG_VARIANT = 5    # ← 跑哪个 variant 改这一个数字 (1-5)


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
            # 类别顺序: [blight, curl, grey_mildew, healthy, leaf_spot, wilt]
            class_u=[0.9, 0.9, 0.7, 0.95, 0.95, 0.8],
        ),
    },
}


# 模型 yaml 和数据集 yaml (★ 用与 baseline 0.946 完全一致的 yaml)
model_yaml_path = "cfg/SCD-YOLO.yaml"
data_yaml_path = "cfg/cotton_disease.yaml"


if __name__ == '__main__':
    # ====================================================================
    # 应用 Cotton-IoU monkey-patch (在 YOLO() 之前必须执行)
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
    # 训练 (★ 所有超参严格匹配 baseline train.py)
    # ====================================================================
    model = YOLO(model_yaml_path)

    results = model.train(
        data=data_yaml_path,
        cache=False,
        imgsz=640,
        epochs=200,
        workers=0,                          # ← 与原脚本一致
        batch=8,
        amp=False,                          # ← 与原脚本一致 (防 NaN)
        pretrained=False,                   # ★ 新增: 显式禁用预训练 (避免下载 yolo13n.pt)
        project="runs",
        name=f'loss_ablation_{cfg["name"]}',
    )
