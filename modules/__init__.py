"""SCD-YOLO modules.

Copy the files in this directory into ``ultralytics/nn/newsAddmodules/`` of a
YOLOv13 source tree, then register them in ``ultralytics/nn/tasks.py`` as
described in ``docs/INSTALL.md``.
"""

from .cotton_starnet import (
    CottonStarNet,
    CottonStarBlock,
    cotton_starnet_s050,
    cotton_starnet_s100,
    cotton_starnet_s150,
)
from .star_adown import StarADown
from .cotton_simam import CottonSimAM
from .cotton_iou_loss import CottonDetectionLoss, CottonBboxLoss
from .cotton_iou_loss_adaptive import (
    CottonDetectionLossAdaptive,
    CottonBboxLossAdaptive,
    attach_adaptive_callback,
)

__all__ = [
    "CottonStarNet", "CottonStarBlock",
    "cotton_starnet_s050", "cotton_starnet_s100", "cotton_starnet_s150",
    "StarADown", "CottonSimAM",
    "CottonDetectionLoss", "CottonBboxLoss",
    "CottonDetectionLossAdaptive", "CottonBboxLossAdaptive",
    "attach_adaptive_callback",
]
