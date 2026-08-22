"""
============================================================================
自适应 Cotton-IoU 训练脚本  (Adaptive Cotton-IoU)
============================================================================
★ 与手调版的区别（一眼区分）：
   - 手调版 (train_4in1_cotton.py): u_c 固定 = [0.9,0.9,0.7,0.95,0.95,0.8]
   - 本脚本 (train_adaptive_cotton.py): u_c 训练中【自动演化】，无需手工设定

配置: Cotton-StarNet + Star-ADown + SimAM + Adaptive-Cotton-IoU (四个全上)

★ 超参严格对齐你的最优实验，保证可对比：
   - epochs=200, batch=8, imgsz=640
   - workers=0, amp=False (防 NaN), pretrained=False
   - seed=0, patience=100
   唯一变量: 损失从【手调 u_c】换成【自适应 u_c】
            → 跑出来直接和手调版 (0.922/0.709) 对比，
              证明"免人工调参也能达到甚至超过手调"

★ 跑完产物:
   - 常规训练结果 (results.csv, best.pt 等)
   - u_c_evolution.csv  ← 各类 u_c 随 epoch 的演化曲线数据（画图用！）
============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.newsAddmodules.cotton_iou_loss_adaptive import (
    CottonDetectionLossAdaptive,
    attach_adaptive_callback,
)


# ============================================================================
# CONFIG: 自适应机制的超参（这些是"元超参"，控制 u_c 怎么演化，不是 u_c 本身）
# ============================================================================
ADAPTIVE_CFG = dict(
    inner_ratio=1.2,      # r：Inner-IoU 尺度因子，沿用你验证过的最优值
    u_min=0.70,           # u_c 动态下界（放宽上限，= 你手调里最低的 grey_mildew）
    u_max=0.95,           # u_c 动态上界（严格上限，= 你手调里最高的 healthy）
    warmup_epochs=30,     # 前 30 轮固定 init_u 热身，之后启动自适应
    beta=0.9,             # EMA 平滑系数（越大越平滑；不稳可调 0.95）
    min_samples=10,       # 某类本 epoch 前景样本 < 此值则跳过更新
                          #   （保护 grey_mildew：验证仅 13 框，训练也少）
    init_u=None,          # 初始 u_c：None = 从中点 0.825 起步（中性起点）
                          #   若想从手调值起步做对比，可传：
                          #   init_u=[0.9, 0.9, 0.7, 0.95, 0.95, 0.8]
)


# ============================================================================
# 路径（如与你的目录不同，改这两行即可）
# ============================================================================
model_yaml_path = "cfg/SCD-YOLO.yaml"
data_yaml_path  = "cfg/cotton_disease.yaml"

project_dir = "runs"
run_name    = 'scd_yolo_adaptive'


if __name__ == '__main__':
    print(f"\n{'=' * 74}")
    print(f"  自适应 Cotton-IoU 训练  (Adaptive Cotton-IoU)")
    print(f"    Backbone   : Cotton-StarNet   (二次创新 #1)")
    print(f"    Downsample : Star-ADown       (二次创新 #2)")
    print(f"    Attention  : SimAM            (二次创新 #3, 在 yaml 里)")
    print(f"    Loss       : Adaptive-Cotton-IoU  (★ 升级: u_c 训练中自动演化)")
    print(f"    自适应元超参 : {ADAPTIVE_CFG}")
    print(f"{'=' * 74}\n")

    # ---- 注入自适应 Cotton-IoU 损失 (必须在 YOLO() 之前) ----
    def _patched_init_criterion(self):
        return CottonDetectionLossAdaptive(self, **ADAPTIVE_CFG)
    DetectionModel.init_criterion = _patched_init_criterion
    print("[Adaptive Cotton-IoU] DetectionModel.init_criterion monkey-patched ✓")

    # ---- 建模型 ----
    model = YOLO(model_yaml_path)

    # ---- ★ 注册回调: 每 epoch 更新 u_c + 把演化写进 u_c_evolution.csv ----
    attach_adaptive_callback(model, logfile="u_c_evolution.csv")
    print("[Adaptive Cotton-IoU] epoch-end callback attached ✓")
    print("[Adaptive Cotton-IoU] u_c 演化将记录到 run 目录下的 u_c_evolution.csv\n")

    # ---- 训练 (所有超参严格匹配最优实验，保证可对比) ----
    results = model.train(
        data=data_yaml_path,
        cache=False,
        imgsz=640,
        epochs=200,
        workers=0,                          # ← 与最优实验一致
        batch=8,
        amp=False,                          # ← 与最优实验一致 (防 NaN)
        pretrained=False,                   # ← 与最优实验一致
        seed=0,                             # ★ 与最优实验一致
        patience=100,                       # ★ 与最优实验一致
        device='0',
        project=project_dir,
        name=run_name,
    )

    print(f"\n{'=' * 74}")
    print(f"  训练完成。请检查：")
    print(f"    1) 参数量是否为 1.915M（自适应不加参数，变了说明配错）")
    print(f"    2) {project_dir}/{run_name}/ 下是否生成了 u_c_evolution.csv")
    print(f"    3) 最终 mAP 与手调版 (0.922/0.709) 对比")
    print(f"{'=' * 74}\n")
