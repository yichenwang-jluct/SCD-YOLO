"""
四创新点全集成训练脚本
============================================================================
配置: Cotton-StarNet + Star-ADown + SimAM + Cotton-IoU (四个全上)

★ 超参严格对齐你的最优实验 (StarNet+Snu77ADown+SimAM+Cotton-IoU = 0.9519 mAP):
   - epochs=200, batch=8, imgsz=640
   - workers=0, amp=False (防 NaN), pretrained=False
   - seed=0, patience=100   ← 与最优实验的 args.yaml 完全一致, 保证可对比

唯一变量: 模型 yaml 从 [StarNet+Snu77ADown+SimAM] 换成
          [（消融）CottonStarNet+StarADown+SimAM]
   → 这样跑出来的 mAP 才能直接和 0.9519 对比,
     干净地证明"两个二次创新模块是否真的有效"
============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.newsAddmodules.cotton_iou_loss import CottonDetectionLoss


# ============================================================================
# CONFIG: 用哪个损失变体 (5 = 你的 proposed Cotton-IoU, 与最优实验一致)
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
            # 类别顺序: [blight, curl, grey_mildew, healthy, leaf_spot, wilt]
            class_u=[0.90, 0.90, 0.70, 0.95, 0.95, 0.80],
        )},
}


# ★★★ 核心改动: 模型 yaml 指向四创新点全集成版 ★★★
# (如果你的 yaml 放到了别的文件夹, 改这一行的路径即可)
model_yaml_path = "cfg/SCD-YOLO.yaml"
data_yaml_path  = "cfg/cotton_disease.yaml"


if __name__ == '__main__':
    cfg = ABLATION_CONFIGS[CONFIG_VARIANT]
    print(f"\n{'=' * 70}")
    print(f"  四创新点全集成训练")
    print(f"    Backbone   : Cotton-StarNet   (二次创新 #1)")
    print(f"    Downsample : Star-ADown       (二次创新 #2)")
    print(f"    Attention  : SimAM            (二次创新 #3, 在 yaml 里)")
    print(f"    Loss       : {cfg['name']}    (二次创新 #4)")
    print(f"    Loss kwargs: {cfg['kwargs']}")
    print(f"{'=' * 70}\n")

    # ---- 注入 Cotton-IoU 损失 (必须在 YOLO() 之前) ----
    if cfg["use_cotton"]:
        def _patched_init_criterion(self):
            return CottonDetectionLoss(self, **cfg["kwargs"])
        DetectionModel.init_criterion = _patched_init_criterion
        print("[Cotton-IoU] DetectionModel.init_criterion monkey-patched ✓\n")
    else:
        print("[Cotton-IoU] Using vanilla v8DetectionLoss (CIoU)\n")

    # ---- 训练 (所有超参严格匹配最优实验) ----
    model = YOLO(model_yaml_path)

    results = model.train(
        data=data_yaml_path,
        cache=False,
        imgsz=640,
        epochs=200,
        workers=0,                          # ← 与最优实验一致
        batch=8,
        amp=False,                          # ← 与最优实验一致 (防 NaN)
        pretrained=False,                   # ← 与最优实验一致
        seed=0,                             # ★ 与最优实验一致 (可复现/可对比的灵魂)
        patience=100,                       # ★ 与最优实验一致
        device='0',
        project="runs",
        name='scd_yolo_manual_uc',
    )
