# ============================================================================
# Class-Aware Cotton-IoU Loss
# ============================================================================
# File:  ultralytics/nn/newsAddmodules/cotton_iou_loss.py
#
# Innovation summary:
#   We propose Cotton-IoU, a class-aware IoU loss for cotton disease detection.
#   It builds on three components:
#     [Component 1] Inner-IoU (Zhang et al., 2023):
#                   Compute IoU on auxiliary bboxes scaled by ratio>1, which
#                   relaxes the metric for slightly misaligned boxes — important
#                   for cotton diseases with fuzzy boundaries.
#     [Component 2] Focaler-IoU (Zhang et al., 2024):
#                   Linearly remap IoU into [0, 1] via (IoU - d)/(u - d), letting
#                   us control which IoU range receives most gradient pressure.
#     [Component 3] ★ Class-Aware adaptation (our contribution):
#                   Make Focaler's upper threshold u class-specific, based on:
#                     (a) per-class sample frequency (rare classes get lower u);
#                     (b) per-class localization difficulty measured by baseline
#                         mAP@0.5:.95 (poorly-localized classes get lower u).
#
# Why this is appropriate for cotton disease detection (paper-ready):
#   Cotton disease imagery has two structural challenges that standard CIoU
#   does not address:
#     (i) Boundary fuzziness — disease lesions have gradient boundaries; the
#         "ground truth" bbox has an inherent annotation tolerance of several
#         pixels. → Inner-IoU with ratio=1.2 mitigates this by enlarging the
#         auxiliary bboxes so small misalignments do not dominate the loss.
#    (ii) Severe class imbalance and uneven localization difficulty — see
#         per-class mAP@0.5:0.95 of the baseline (validation set, Table 15 of
#         the paper): grey_mildew=0.433 (13 instances) and wilt=0.573 (117)
#         lag far behind healthy=0.695 (42). → Class-aware Focaler u lets rare
#         and hard classes use a relaxed IoU target during training.
#
# Per-class Focaler u (calibrated on cotton dataset):
#   class index → name        → u value → rationale
#     0 → blight              → 0.90    → adequate samples, moderate difficulty
#     1 → curl                → 0.90    → moderate samples, moderate difficulty
#     2 → grey_mildew         → 0.70 ★  → rare + hardest, lenient
#     3 → healthy             → 0.95    → easy class, strict
#     4 → leaf_spot           → 0.95    → best-localized class, strict
#     5 → wilt                → 0.80    → moderately difficult, semi-lenient
#
# Ablation experiments supported by this implementation (toggle args in init):
#   ① CIoU baseline             : inner_ratio=1.0, use_focaler=False
#   ② Inner-CIoU                : inner_ratio=1.2, use_focaler=False
#   ③ Focaler-CIoU (uniform)    : inner_ratio=1.0, use_focaler=True, u=[0.95]*6
#   ④ Inner+Focaler (uniform)   : inner_ratio=1.2, use_focaler=True, u=[0.95]*6
#   ⑤ ★ Cotton-IoU (Class-Aware): inner_ratio=1.2, use_focaler=True,
#                                  u=[0.9, 0.9, 0.7, 0.95, 0.95, 0.8]
# ============================================================================

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ultralytics.utils.loss import DFLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist


# ===========================================================================
# Core IoU function
# ===========================================================================

def class_aware_cotton_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    class_u: Optional[torch.Tensor] = None,
    inner_ratio: float = 1.0,
    focaler_d: float = 0.0,
    use_focaler: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Class-Aware Cotton-IoU between two sets of boxes.

    Args:
        box1: (N, 4) predicted boxes in xyxy format.
        box2: (N, 4) target boxes in xyxy format.
        target_labels: (N,) long tensor of target class indices.
                       Required iff use_focaler=True and class_u is non-uniform.
        class_u: (num_classes,) tensor of per-class Focaler upper thresholds.
                 Required iff use_focaler=True.
        inner_ratio: Inner-IoU scaling ratio applied around box centers.
                     1.0 = identity (regular CIoU). >1 = expand boxes.
        focaler_d: Focaler lower threshold (default 0).
        use_focaler: If True, apply class-aware Focaler re-mapping to CIoU.
        eps: Numerical stability constant.

    Returns:
        iou: (N, 1) tensor in [0, 1] (if use_focaler) or unconstrained CIoU.
             The loss is computed as (1 - iou) outside this function.
    """
    # ---- Parse xyxy to center-width-height -------------------------------
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b1_w = (b1_x2 - b1_x1).clamp(min=eps)
    b1_h = (b1_y2 - b1_y1).clamp(min=eps)

    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    b2_w = (b2_x2 - b2_x1).clamp(min=eps)
    b2_h = (b2_y2 - b2_y1).clamp(min=eps)

    # ---- Inner-IoU: scale boxes around centers ---------------------------
    r_half = inner_ratio / 2.0
    b1_x1_s, b1_x2_s = b1_cx - b1_w * r_half, b1_cx + b1_w * r_half
    b1_y1_s, b1_y2_s = b1_cy - b1_h * r_half, b1_cy + b1_h * r_half
    b2_x1_s, b2_x2_s = b2_cx - b2_w * r_half, b2_cx + b2_w * r_half
    b2_y1_s, b2_y2_s = b2_cy - b2_h * r_half, b2_cy + b2_h * r_half

    # ---- Intersection / Union on scaled boxes ----------------------------
    inter_w = (torch.minimum(b1_x2_s, b2_x2_s) - torch.maximum(b1_x1_s, b2_x1_s)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2_s, b2_y2_s) - torch.maximum(b1_y1_s, b2_y1_s)).clamp(min=0)
    inter = inter_w * inter_h

    b1_area_s = (b1_x2_s - b1_x1_s) * (b1_y2_s - b1_y1_s)
    b2_area_s = (b2_x2_s - b2_x1_s) * (b2_y2_s - b2_y1_s)
    union = b1_area_s + b2_area_s - inter + eps
    iou_base = inter / union

    # ---- CIoU penalties: enclosing box, center distance, aspect ratio ----
    enc_x1 = torch.minimum(b1_x1_s, b2_x1_s)
    enc_y1 = torch.minimum(b1_y1_s, b2_y1_s)
    enc_x2 = torch.maximum(b1_x2_s, b2_x2_s)
    enc_y2 = torch.maximum(b1_y2_s, b2_y2_s)
    c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps  # diagonal²

    # Centre distance: scaling is centre-preserving, so the original centres are used.
    rho_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect-ratio penalty: uniform scaling preserves the aspect ratio, so the
    # original w/h are used.
    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h),
        2,
    )
    with torch.no_grad():
        alpha = v / (1 - iou_base + v + eps)

    ciou = iou_base - rho_sq / c_sq - alpha * v   # shape (N, 1)

    # ---- Class-Aware Focaler re-mapping ----------------------------------
    if use_focaler:
        if class_u is None:
            raise ValueError("class_u must be provided when use_focaler=True")
        if target_labels is None:
            raise ValueError("target_labels must be provided when use_focaler=True")

        # u_per_sample shape: (N, 1)
        u_per_sample = class_u[target_labels].unsqueeze(-1)
        ciou = ((ciou - focaler_d) / (u_per_sample - focaler_d + eps)).clamp(min=0, max=1)

    return ciou


# ===========================================================================
# CottonBboxLoss (drop-in replacement of BboxLoss)
# ===========================================================================

class CottonBboxLoss(nn.Module):
    """BboxLoss with Class-Aware Cotton-IoU support and ablation toggles."""

    def __init__(
        self,
        reg_max: int = 16,
        inner_ratio: float = 1.0,
        class_u: Optional[List[float]] = None,
        use_focaler: bool = False,
        focaler_d: float = 0.0,
    ):
        """
        Args:
            reg_max: DFL bin count, same as ultralytics default.
            inner_ratio: 1.0 (off) or >1 (enabled, e.g. 1.2).
            class_u: list of per-class Focaler u thresholds. Length must equal nc.
            use_focaler: enable Focaler re-mapping.
            focaler_d: lower threshold (default 0).
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.inner_ratio = inner_ratio
        self.use_focaler = use_focaler
        self.focaler_d = focaler_d

        if class_u is not None:
            # register_buffer keeps the tensor on the same device as the module
            self.register_buffer(
                "class_u",
                torch.tensor(class_u, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.class_u = None  # type: ignore

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        target_labels: torch.Tensor,  # ★ NEW: needed for class-aware Focaler
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # ---- IoU loss (with Cotton-IoU) ----------------------------------
        iou = class_aware_cotton_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            target_labels=target_labels[fg_mask] if self.use_focaler else None,
            class_u=self.class_u if self.use_focaler else None,
            inner_ratio=self.inner_ratio,
            focaler_d=self.focaler_d,
            use_focaler=self.use_focaler,
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # ---- DFL loss (unchanged from ultralytics) ----------------------
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


# ===========================================================================
# CottonDetectionLoss (subclass of v8DetectionLoss)
# ===========================================================================

class CottonDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with Class-Aware Cotton-IoU.

    Inherits everything; overrides __init__ to swap bbox_loss and __call__ to
    capture target_labels from the assigner and forward them to bbox_loss.
    """

    def __init__(
        self,
        model,
        inner_ratio: float = 1.2,
        class_u: Optional[List[float]] = None,
        use_focaler: bool = True,
    ):
        super().__init__(model)

        # Guard against a silent class-index mismatch
        if use_focaler:
            if class_u is None:
                class_u = [0.95] * self.nc  # falls back to a uniform Focaler bound
            if len(class_u) != self.nc:
                raise ValueError(
                    f"class_u length ({len(class_u)}) must equal num_classes ({self.nc}). "
                    f"Check that your data.yaml class order matches class_u indexing."
                )

        self.bbox_loss = CottonBboxLoss(
            reg_max=self.reg_max,
            inner_ratio=inner_ratio,
            class_u=class_u,
            use_focaler=use_focaler,
        ).to(self.device)

    def __call__(self, preds, batch):
        """Compute losses. Mirrors v8DetectionLoss.__call__ with 2 changes:
            (1) capture target_labels from assigner output;
            (2) pass target_labels to self.bbox_loss.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # ---- Targets -----------------------------------------------------
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # ---- Pred boxes --------------------------------------------------
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # KEY CHANGE 1: capture `target_labels` (was `_` in v8DetectionLoss)
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ---- Cls loss (unchanged) ----------------------------------------
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # ---- Bbox loss with Cotton-IoU ----------------------------------
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # KEY CHANGE 2: forward target_labels to bbox_loss
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                target_labels,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()
