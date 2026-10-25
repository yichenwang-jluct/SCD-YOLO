# ============================================================================
# StarNet-s050 WITHOUT the channel-adaptive gate  (gate control: Table 6, Supplementary Table S7)
# ============================================================================
# Copy to:  ultralytics/nn/newsAddmodules/starnet_ungated.py
#
# This backbone is a line-by-line copy of CottonStarNet / CottonStarBlock in
# cotton_starnet.py with exactly one change: the gate MLP is removed and the
# star operation is computed as ReLU6(x1) * x2 (i.e. g == 1).  Everything else
# - stem, stage layout (1, 1, 3, 1), widths (16, 32, 64, 128), expansion 3,
# 7x7 depth-wise convs, conv bias settings, BatchNorm placement, weight
# initialisation and the `channel` list read by parse_model - is identical, so
# that the difference to Cotton-StarNet isolates the gate.
#
# Expected parameter counts (nc = 6, after fuse):
#   ctrl_ungated_D0_A0.yaml : 1,969,177  (= 2,037,811 - 68,634 gate parameters)
#   ctrl_ungated_D1_A1.yaml : 1,846,873  (= 1,915,507 - 68,634)
#
# Registration in ultralytics/nn/tasks.py (two edits, same places as INSTALL.md):
#   1) imports:   from ultralytics.nn.newsAddmodules.starnet_ungated import starnet_s050_ungated
#   2) add `starnet_s050_ungated` to the same `elif m in {..., cotton_starnet_s050, ...}`
#      set whose body is `m = m(*args); c2 = m.channel; backbone = True`.
# ============================================================================

import torch
import torch.nn as nn
from timm.models.layers import DropPath

__all__ = ["StarNetUngated", "UngatedStarBlock", "starnet_s050_ungated"]


class ConvBN(nn.Sequential):
    """Conv + BatchNorm (identical to cotton_starnet.ConvBN)."""

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                          padding, dilation, groups, bias=False))
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class UngatedStarBlock(nn.Module):
    """CottonStarBlock with the channel gate removed:  out = ReLU6(W1 x') * (W2 x')."""

    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2          # plain star operation (no gate)
        x = self.dwconv2(self.g(x))
        return identity + self.drop_path(x)


class StarNetUngated(nn.Module):
    """StarNet backbone with the same layout and API as CottonStarNet, minus the gate."""

    def __init__(self, base_dim=16, depths=(1, 1, 3, 1), mlp_ratio=3, drop_path_rate=0.0, **kwargs):
        super().__init__()
        self.in_channel = 32
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        self.channel = [self.in_channel]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [UngatedStarBlock(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            self.channel.append(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = [self.stem(x)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        return features


def starnet_s050_ungated(**kwargs):
    """Ungated StarNet-s050. channel = [32, 16, 32, 64, 128] (same as cotton_starnet_s050)."""
    return StarNetUngated(base_dim=16, depths=(1, 1, 3, 1), mlp_ratio=3, **kwargs)
