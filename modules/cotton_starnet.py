# ============================================================================
# Cotton-StarNet : Twice innovation of StarNet for cotton pest/disease detection
# ============================================================================
# File:  ultralytics/nn/newsAddmodules/cotton_starnet.py
# Ref:   Ma et al., "Rewrite the Stars", CVPR 2024 (original StarNet)
#
# CHANGELOG:
#   v1.1 (FIX): Added self.channel list to expose stage output channels —
#               required by ultralytics parse_model (line `c2 = m.channel`).
#   v1.0     : Initial release with channel-wise adaptive gate.
# ============================================================================

import torch
import torch.nn as nn
from timm.models.layers import DropPath


# ---- Explicit exports — prevents 'Conv'/'ConvBN' namespace pollution ----
__all__ = [
    'cotton_starnet_s050',
    'cotton_starnet_s100',
    'cotton_starnet_s150',
    'CottonStarNet',
    'CottonStarBlock',
]


class ConvBN(nn.Sequential):
    """Conv + BatchNorm convenience module (matches original StarNet style)."""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                          padding, dilation, groups, bias=False))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class CottonStarBlock(nn.Module):
    """
    Cotton-aware Star Block (★ Twice Innovation #1).

    Adds a channel-wise adaptive gate to one branch of the star operation,
    enabling input-dependent emphasis of disease-relevant feature channels.

    Mathematical formulation:
        Standard:  out = ReLU6(W1·x') ⊙ (W2·x')
        Ours:      g   = σ(MLP(GAP(W1·x')))
                   out = (ReLU6(W1·x') ⊙ g) ⊙ (W2·x')
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., gate_reduction=8):
        super().__init__()
        # Standard StarBlock components
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ★ Cotton-aware adaptive gate
        hidden = max(mlp_ratio * dim // gate_reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mlp_ratio * dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, mlp_ratio * dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x1 = self.f1(x)   # branch 1: activated (ReLU6) then channel-gated
        x2 = self.f2(x)   # branch 2: content branch (kept linear)

        # ★ Gated star operation
        gate = self.gate(x1)            # channel-wise weights ∈ (0,1)^(3C)
        gated_x1 = self.act(x1) * gate  # disease-relevant channel emphasis
        x = gated_x1 * x2               # star op with gated branch

        x = self.dwconv2(self.g(x))
        x = identity + self.drop_path(x)
        return x


class CottonStarNet(nn.Module):
    """
    Cotton-StarNet backbone (★ Twice Innovation #1).

    CRITICAL: Exposes `self.channel` — a list of stage output channels —
    required by ultralytics parse_model (line: `c2 = m.channel`).
    """
    def __init__(self, base_dim=16, depths=(1, 1, 3, 1), mlp_ratio=3,
                 drop_path_rate=0.0, **kwargs):
        super().__init__()
        self.in_channel = 32

        # Stem (stride 2)
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6()
        )

        # ★★★ FIX: Track output channels for parse_model
        self.channel = [self.in_channel]   # index 0: stem output (32)

        # Build 4 stages, each with a downsampler + CottonStarBlocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [
                CottonStarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

            # ★★★ FIX: Append this stage's output channel count
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
        """Returns features at [stem, stage1, stage2, stage3, stage4].
        Index 2/3/4 correspond to P3/P4/P5 (matches original StarNet API)."""
        features = [self.stem(x)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        return features


# ============================================================================
# Factory functions — mirror original starnet_s050/s100/s150 naming
# ============================================================================

def cotton_starnet_s050(**kwargs):
    """Cotton-StarNet-s050. channel = [32, 16, 32, 64, 128]."""
    return CottonStarNet(base_dim=16, depths=(1, 1, 3, 1), mlp_ratio=3, **kwargs)

def cotton_starnet_s100(**kwargs):
    """Cotton-StarNet-s100. channel = [32, 20, 40, 80, 160]."""
    return CottonStarNet(base_dim=20, depths=(1, 2, 4, 1), mlp_ratio=4, **kwargs)

def cotton_starnet_s150(**kwargs):
    """Cotton-StarNet-s150. channel = [32, 24, 48, 96, 192]."""
    return CottonStarNet(base_dim=24, depths=(1, 2, 4, 2), mlp_ratio=3, **kwargs)
