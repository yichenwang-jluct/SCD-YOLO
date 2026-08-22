# ============================================================================
# Adaptive Class-Aware Cotton-IoU Loss  (Training-driven dynamic u_c)
# ============================================================================
# File:  ultralytics/nn/newsAddmodules/cotton_iou_loss_adaptive.py
#
# Innovation (upgraded):
#   Cotton-IoU-Adaptive: the per-class Focaler upper bound u_c is NO LONGER
#   hand-set. Instead it EVOLVES DURING TRAINING, driven by each class's
#   real-time localization quality (mean IoU of its foreground samples),
#   smoothed by EMA. Poorly-localized classes automatically receive a lower
#   (more lenient) u_c; well-localized classes automatically tighten toward
#   u_max. No per-class hyper-parameter tuning is required.
#
#   Mapping rule (per epoch), Eq. (18)-(20) of the paper:
#       q_c        = mean Inner-IoU of class c foreground samples this epoch
#       q~_c       = (q_c - min_k q_k) / (max_k q_k - min_k q_k)   (cross-class
#                    min-max normalisation; training IoUs are uniformly high, so a
#                    direct map would collapse every u_c into a narrow band)
#       u_c^new    = u_min + (u_max - u_min) * q~_c
#       u_c        = beta * u_c^old + (1 - beta) * u_c^new         (EMA)
#
#   Robustness handling:
#       - warmup_epochs: keep u_c at init for the first N epochs (early IoU is
#         uninformative; without warmup all classes collapse to u_min).
#       - min_samples : if a class has < min_samples fg boxes this epoch, SKIP
#         its update (critical for grey_mildew: only 13 val / few train boxes).
#       - EMA(beta=0.9): damps epoch-to-epoch jitter.
#
# Baseline per-class mAP@0.5:0.95 on the cotton validation set (Table 15 of the paper):
#   blight=0.682(282) curl=0.687(80) grey_mildew=0.433(13)
#   healthy=0.695(42) leaf_spot=0.675(57) wilt=0.573(117)
#
# Usage (train script): patch DetectionModel.init_criterion to return
#   CottonDetectionLossAdaptive(model, inner_ratio=1.2, u_min=0.70, u_max=0.95,
#                               warmup_epochs=30, beta=0.9, min_samples=10)
# and register the epoch-end callback (see attach_adaptive_callback below).
# ============================================================================

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ultralytics.utils.loss import DFLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist


# ===========================================================================
# Core IoU function  (returns raw IoU too, for online quality statistics)
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Class-Aware Cotton-IoU.

    Returns:
        iou_out : (N,1) value used for the loss (Focaler-remapped if enabled).
        iou_raw : (N,1) plain Inner-IoU overlap in [0,1], used ONLY as the
                  localization-quality signal q_c (detached by caller).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    b1_cx = (b1_x1 + b1_x2) / 2; b1_cy = (b1_y1 + b1_y2) / 2
    b1_w = (b1_x2 - b1_x1).clamp(min=eps); b1_h = (b1_y2 - b1_y1).clamp(min=eps)
    b2_cx = (b2_x1 + b2_x2) / 2; b2_cy = (b2_y1 + b2_y2) / 2
    b2_w = (b2_x2 - b2_x1).clamp(min=eps); b2_h = (b2_y2 - b2_y1).clamp(min=eps)

    # ---- Inner-IoU: scale boxes around centers ---------------------------
    r_half = inner_ratio / 2.0
    b1_x1_s, b1_x2_s = b1_cx - b1_w * r_half, b1_cx + b1_w * r_half
    b1_y1_s, b1_y2_s = b1_cy - b1_h * r_half, b1_cy + b1_h * r_half
    b2_x1_s, b2_x2_s = b2_cx - b2_w * r_half, b2_cx + b2_w * r_half
    b2_y1_s, b2_y2_s = b2_cy - b2_h * r_half, b2_cy + b2_h * r_half

    inter_w = (torch.minimum(b1_x2_s, b2_x2_s) - torch.maximum(b1_x1_s, b2_x1_s)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2_s, b2_y2_s) - torch.maximum(b1_y1_s, b2_y1_s)).clamp(min=0)
    inter = inter_w * inter_h
    b1_area_s = (b1_x2_s - b1_x1_s) * (b1_y2_s - b1_y1_s)
    b2_area_s = (b2_x2_s - b2_x1_s) * (b2_y2_s - b2_y1_s)
    union = b1_area_s + b2_area_s - inter + eps
    iou_base = inter / union                         # plain Inner-IoU overlap

    enc_x1 = torch.minimum(b1_x1_s, b2_x1_s); enc_y1 = torch.minimum(b1_y1_s, b2_y1_s)
    enc_x2 = torch.maximum(b1_x2_s, b2_x2_s); enc_y2 = torch.maximum(b1_y2_s, b2_y2_s)
    c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps
    rho_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h), 2)
    with torch.no_grad():
        alpha = v / (1 - iou_base + v + eps)
    ciou = iou_base - rho_sq / c_sq - alpha * v      # (N,1)

    iou_out = ciou
    if use_focaler:
        if class_u is None or target_labels is None:
            raise ValueError("class_u and target_labels required when use_focaler=True")
        u_per_sample = class_u[target_labels].unsqueeze(-1)
        iou_out = ((ciou - focaler_d) / (u_per_sample - focaler_d + eps)).clamp(min=0, max=1)

    # iou_base in [0,1] is the clean localization-quality signal
    return iou_out, iou_base.detach()


# ===========================================================================
# CottonBboxLossAdaptive
# ===========================================================================
class CottonBboxLossAdaptive(nn.Module):
    """BboxLoss with adaptive class-aware Cotton-IoU + online q_c accumulation."""

    def __init__(
        self,
        reg_max: int = 16,
        num_classes: int = 6,
        inner_ratio: float = 1.2,
        u_min: float = 0.70,
        u_max: float = 0.95,
        beta: float = 0.9,
        min_samples: int = 10,
        focaler_d: float = 0.0,
        init_u: Optional[List[float]] = None,
        use_focaler: bool = True,
    ):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.nc = num_classes
        self.inner_ratio = inner_ratio
        self.u_min = u_min
        self.u_max = u_max
        self.beta = beta
        self.min_samples = min_samples
        self.focaler_d = focaler_d
        self.use_focaler = use_focaler
        self.adaptive_enabled = False   # toggled on by callback after warmup

        # initial u_c: midpoint by default (neutral start), or user-provided
        if init_u is None:
            init_u = [(u_min + u_max) / 2.0] * num_classes
        self.register_buffer("class_u", torch.tensor(init_u, dtype=torch.float32), persistent=True)

        # online accumulators for q_c (reset each epoch)
        self.register_buffer("iou_sum", torch.zeros(num_classes), persistent=False)
        self.register_buffer("iou_cnt", torch.zeros(num_classes), persistent=False)

    @torch.no_grad()
    def _accumulate(self, iou_raw: torch.Tensor, labels: torch.Tensor):
        """Accumulate per-class IoU sum & count for this epoch."""
        iou_raw = iou_raw.squeeze(-1)                       # (N,)
        self.iou_sum.index_add_(0, labels, iou_raw.to(self.iou_sum.dtype))
        self.iou_cnt.index_add_(0, labels, torch.ones_like(labels, dtype=self.iou_cnt.dtype))

    @torch.no_grad()
    def update_class_u(self) -> torch.Tensor:
        """Epoch end: rank-normalize q_c across classes -> [u_min,u_max], EMA.

        Training foreground IoUs are uniformly high and the absolute gaps between
        classes are compressed, so a direct q_c -> u_c map would collapse every
        u_c into a narrow high band. We therefore min-max normalise q_c ACROSS
        classes, stretching the correctly-ordered but compressed difficulty signal
        over the full [u_min, u_max]: hardest class -> u_min, easiest -> u_max.
        Note: the min/max are taken over the classes that pass `min_samples`.
        """
        if self.adaptive_enabled:
            enough = self.iou_cnt >= self.min_samples
            q = torch.where(self.iou_cnt > 0, self.iou_sum / self.iou_cnt.clamp(min=1),
                            torch.zeros_like(self.iou_sum))
            # --- cross-class min-max normalisation, Eq. (19) ---------------
            if enough.sum() >= 2:
                q_valid = q[enough]
                q_lo, q_hi = q_valid.min(), q_valid.max()
                q_norm = ((q - q_lo) / (q_hi - q_lo + 1e-6)).clamp(0, 1)
            else:
                q_norm = q.clamp(0, 1)
            u_new = (self.u_min + (self.u_max - self.u_min) * q_norm).clamp(self.u_min, self.u_max)
            ema = self.beta * self.class_u + (1 - self.beta) * u_new
            self.class_u = torch.where(enough, ema, self.class_u)
        self.iou_sum.zero_(); self.iou_cnt.zero_()
        return self.class_u.clone()

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, target_labels):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        labels_fg = target_labels[fg_mask]

        iou_out, iou_raw = class_aware_cotton_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask],
            target_labels=labels_fg if self.use_focaler else None,
            class_u=self.class_u if self.use_focaler else None,
            inner_ratio=self.inner_ratio, focaler_d=self.focaler_d,
            use_focaler=self.use_focaler,
        )
        # accumulate localization-quality signal (detached, no grad)
        self._accumulate(iou_raw, labels_fg)

        loss_iou = ((1.0 - iou_out) * weight).sum() / target_scores_sum

        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


# ===========================================================================
# CottonDetectionLossAdaptive
# ===========================================================================
class CottonDetectionLossAdaptive(v8DetectionLoss):
    def __init__(self, model, inner_ratio: float = 1.2, u_min: float = 0.70,
                 u_max: float = 0.95, beta: float = 0.9, warmup_epochs: int = 30,
                 min_samples: int = 10, init_u: Optional[List[float]] = None):
        super().__init__(model)
        self.warmup_epochs = warmup_epochs
        self.bbox_loss = CottonBboxLossAdaptive(
            reg_max=self.reg_max, num_classes=self.nc, inner_ratio=inner_ratio,
            u_min=u_min, u_max=u_max, beta=beta, min_samples=min_samples,
            init_u=init_u, use_focaler=True,
        ).to(self.device)

    # ---- identical to v8DetectionLoss.__call__ except target_labels capture ----
    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
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

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                             batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, target_labels,
            )
        loss[0] *= self.hyp.box; loss[1] *= self.hyp.cls; loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


# ===========================================================================
# Epoch-end callback: enable adaptation after warmup + update u_c + log
# ===========================================================================
def attach_adaptive_callback(model, logfile: str = "u_c_evolution.csv"):
    """Register on a YOLO() instance BEFORE calling .train().

    Writes each epoch's u_c to CSV so you can plot the evolution curve.
    """
    header_written = {"done": False}

    def _on_epoch_end(trainer):
        crit = getattr(trainer.model, "criterion", None)
        bl = getattr(crit, "bbox_loss", None) if crit is not None else None
        if bl is None or not hasattr(bl, "update_class_u"):
            return
        # enable adaptation once past warmup
        if trainer.epoch >= crit.warmup_epochs:
            bl.adaptive_enabled = True
        u = bl.update_class_u().detach().cpu().tolist()
        # log
        import csv, os
        mode = "a" if header_written["done"] else "w"
        with open(os.path.join(trainer.save_dir, logfile), mode, newline="") as f:
            w = csv.writer(f)
            if not header_written["done"]:
                w.writerow(["epoch"] + [f"u_c_{i}" for i in range(len(u))])
                header_written["done"] = True
            w.writerow([trainer.epoch] + [round(x, 4) for x in u])

    model.add_callback("on_train_epoch_end", _on_epoch_end)
    return model
