"""
Baseline YOLOv13n — the reference row of Tables 3, 5 and 13-16.

No Cotton module and no loss patch: stock YOLOv13n with the vanilla
v8DetectionLoss (CIoU). Every hyper-parameter matches the SCD-YOLO runs, so the
two are directly comparable.

    python scripts/train_baseline.py

Expected: 2,467,497 parameters, 6.2 GFLOPs, and on the cotton validation set
mAP@0.5 = 0.890, mAP@0.5:0.95 = 0.624 (mean of three seeds).
"""

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

# Stock YOLOv13n from the upstream repository — no modules from this project.
model_yaml_path = "ultralytics/cfg/models/addv13/yolov13n.yaml"
data_yaml_path = "cfg/cotton_disease.yaml"

project_dir = "runs"
run_name = "yolov13n_baseline"


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Baseline YOLOv13n (no Cotton modules, vanilla CIoU loss)")
    print("=" * 70 + "\n")

    model = YOLO(model_yaml_path)

    results = model.train(
        data=data_yaml_path,
        cache=False,
        imgsz=640,
        epochs=200,
        workers=0,
        batch=8,
        amp=False,
        pretrained=False,
        seed=0,
        patience=100,
        device='0',
        project=project_dir,
        name=run_name,
    )
