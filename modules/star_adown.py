# ============================================================================
# Star-ADown : Twice innovation of ADown for cotton pest/disease detection
# ============================================================================
# File:  ultralytics/nn/modules/star_adown.py
# Ref:   ADown (Wang et al., YOLOv9) + Star Operation (Ma et al., CVPR 2024)
#
# Innovation: We replace the conventional concatenation-based fusion in ADown
#             with the star operation (element-wise multiplication), extending
#             the unified "star design philosophy" from feature extraction to
#             feature downsampling.
#
# Mathematical formulation:
#   Standard ADown:   out = Concat[ Conv_s2(x1), Conv_1x1(MaxPool(x2)) ]
#   Star-ADown:       y1  = Conv_s2(x1)                          [spatial]
#                     y2  = Conv_1x1(MaxPool(x2))                 [texture]
#                     out = Proj1x1( ReLU6(y1) ⊙ y2 )            [★ star fusion]
#   where x1, x2 = split(AvgPool(x)) along channel axis.
#
# Motivation (paper-ready):
#   Element-wise multiplication induces implicit high-dimensional feature
#   interaction (Ma et al., 2024). We extend this design from feature
#   extraction (Cotton-StarNet) to feature downsampling, forming a unified
#   design philosophy throughout the entire backbone-neck pipeline. Compared
#   with the simple concatenation of complementary branches, multiplicative
#   fusion enforces stronger cross-branch dependency, which is particularly
#   beneficial for preserving fine-grained disease boundaries during spatial
#   compression.
#
# Parameter overhead: ~20% over standard ADown per call (still lightweight).
#                     The added 1x1 projection is necessary because the
#                     multiplicative result has half the channel count.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard YOLO Conv (Conv2d + BN + SiLU)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True
                    else (act if isinstance(act, nn.Module) else nn.Identity()))

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StarADown(nn.Module):
    """
    Star-ADown: Downsampling via Star Operation (★ Twice Innovation #2).

    Args:
        c1 (int): Input channels (auto-passed by YOLO framework).
        c2 (int): Output channels.

    Note on parameters: Each pre-multiplication branch produces c2//2 channels.
    Their elementwise product also has c2//2 channels, which is then projected
    to c2 via a 1x1 conv. This keeps parameter count close to the original ADown.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)   # stride-2 spatial branch
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)   # 1x1 texture branch (after max-pool)
        self.proj = Conv(self.c, c2, 1, 1, 0)       # restore channels to c2
        self.act = nn.ReLU6()

    def forward(self, x):
        # Pre-downsampling smoothing (same as ADown)
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)

        # Spatial branch: stride-2 3x3 conv
        y1 = self.cv1(x1)

        # Texture branch: max-pool then 1x1 conv
        y2 = F.max_pool2d(x2, 3, 2, 1)
        y2 = self.cv2(y2)

        # ★ Star operation replaces concatenation
        fused = self.act(y1) * y2                   # [B, c2//2, H', W']
        out = self.proj(fused)                      # [B, c2, H', W']
        return out


__all__ = ['StarADown']