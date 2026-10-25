#!/usr/bin/env python3
"""
Two further settings for Tables 16 and 17, trained after the runs of train_split.py with exactly the
same protocol and the full SCD-YOLO architecture:

    uc_uniform0.95   Cotton-IoU with one uniform upper bound u = 0.95 for all classes (Table 16 "Uniform",
                     and the fifth candidate of the validation-selected uniform bound C1)
    uc_manual        Cotton-IoU with the manually set class-wise bounds of the paper
                     (blight, curl, grey_mildew, healthy, leaf_spot, wilt = 0.90, 0.90, 0.70, 0.95, 0.95, 0.80)

seeds 0, 1 and 2 = 6 trainings.  The command can be started in several windows, one or two minutes apart:
each window takes the next training that is neither finished nor running (run_experiment.py keeps track),
so after a crash or a restart just start it again.

    (from the YOLOv13 source-tree root, PYTHONPATH = that root)
    python train_extra.py 1 1 0 1 2 --data <cotton_disease.yaml> --src <data_srcdisjoint.yaml>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_split  # noqa: E402  (same loop, same run_experiment.py call; only the list of settings differs)

train_split.SETTINGS = [
    ("uc_uniform0.95", "S1_D1_A1", "uniform", ["--u", "0.95"], False),
    ("uc_manual", "S1_D1_A1", "fixed", ["--uc", "0.90,0.90,0.70,0.95,0.95,0.80"], False),
]

if __name__ == "__main__":
    train_split.main()
