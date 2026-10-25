"""
Train SCD-YOLO with the adaptive Cotton-IoU loss (the proposed configuration).

Model: Cotton-StarNet + Star-ADown + Cotton-SimAM (cfg/SCD-YOLO.yaml). Loss: Cotton-IoU with
class-adaptive upper bounds u_c that are updated during training from the localization quality of
each class, without per-class manual setting (scripts/train_manual_uc.py instead fixes
u_c = [0.90, 0.90, 0.70, 0.95, 0.95, 0.80]).

Training protocol of Section 2.6: epochs=200, batch=8, imgsz=640, workers=0, amp=False,
pretrained=False, patience=100, seed 0. The three-seed runs reported in the paper were started with
    python scripts/run_experiment.py --arch S1_D1_A1 --loss adaptive --name fac_S1D1A1I1 --seeds 0 1 2 --data ...
which uses the same settings.

Outputs (runs/scd_yolo_adaptive/): the usual Ultralytics results (results.csv, weights/best.pt) and
u_c_evolution.csv, the u_c of every class after each epoch (input of scripts/plot_uc_evolution.py).
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
# CONFIG: settings of the adaptive mechanism (they control how u_c evolves; they are not u_c itself)
# ============================================================================
ADAPTIVE_CFG = dict(
    inner_ratio=1.2,      # r: Inner-IoU scale factor
    u_min=0.70,           # lower end of the u_c range (most lenient bound)
    u_max=0.95,           # upper end of the u_c range (strictest bound)
    warmup_epochs=30,     # u_c is held at init_u for the first 30 epochs, then adapted
    beta=0.9,             # EMA momentum of the update (larger = smoother)
    min_samples=10,       # a class with fewer foreground samples in an epoch is not updated
                          #   (rare-class protection)
    init_u=None,          # initial u_c: None = midpoint of the range, 0.825
                          #   to start from the manual values instead, pass
                          #   init_u=[0.9, 0.9, 0.7, 0.95, 0.95, 0.8]
)


# ============================================================================
# Paths (relative to the repository root; adjust them if your layout differs)
# ============================================================================
model_yaml_path = "cfg/SCD-YOLO.yaml"
data_yaml_path  = "cfg/cotton_disease.yaml"

project_dir = "runs"
run_name    = 'scd_yolo_adaptive'


if __name__ == '__main__':
    print(f"\n{'=' * 74}")
    print(f"  SCD-YOLO training with the adaptive Cotton-IoU loss")
    print(f"    Backbone   : Cotton-StarNet")
    print(f"    Downsample : Star-ADown")
    print(f"    Attention  : Cotton-SimAM (in the YAML)")
    print(f"    Loss       : Cotton-IoU with class-adaptive u_c updated during training")
    print(f"    Settings   : {ADAPTIVE_CFG}")
    print(f"{'=' * 74}\n")

    # ---- inject the adaptive Cotton-IoU loss (must precede YOLO()) ----
    def _patched_init_criterion(self):
        return CottonDetectionLossAdaptive(self, **ADAPTIVE_CFG)
    DetectionModel.init_criterion = _patched_init_criterion
    print("[Adaptive Cotton-IoU] DetectionModel.init_criterion monkey-patched ✓")

    # ---- build the model ----
    model = YOLO(model_yaml_path)

    # ---- epoch-end callback: updates u_c and logs it to u_c_evolution.csv ----
    attach_adaptive_callback(model, logfile="u_c_evolution.csv")
    print("[Adaptive Cotton-IoU] epoch-end callback attached ✓")
    print("[Adaptive Cotton-IoU] u_c is logged to u_c_evolution.csv in the run directory\n")

    # ---- training (protocol of Section 2.6) ----
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
        project=project_dir,
        name=run_name,
    )

    print(f"\n{'=' * 74}")
    print(f"  Training finished. Check that")
    print(f"    1) the fused model has 1,915,507 parameters (the adaptive loss adds none)")
    print(f"    2) {project_dir}/{run_name}/u_c_evolution.csv was written")
    print(f"    3) weights/best.pt is evaluated with rect=False (scripts/eval_test_set.py)")
    print(f"{'=' * 74}\n")
