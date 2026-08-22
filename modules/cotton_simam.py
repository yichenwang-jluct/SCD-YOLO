# ============================================================================
# Cotton-SimAM : Spatially-Coherent Parameter-Free Attention
#                for Cotton Disease Detection
# ============================================================================
# File:  ultralytics/nn/newsAddmodules/cotton_simam.py
#
# Contribution #3: a spatially-coherent replacement for SimAM.
#
# NOTE: gamma modulates the SimAM energy multiplicatively as (1 + gamma*rho),
#       so the factor is always >= 1 and coherent lesion regions are amplified
#       while isolated spikes stay at their SimAM baseline. gamma=0 recovers
#       the original SimAM exactly.
#
# Base:  SimAM (Yang et al., ICML 2021) — parameter-free 3-D attention.
#
# ────────────────────────────────────────────────────────────────────────
#  Motivation (paper-ready)
# ────────────────────────────────────────────────────────────────────────
#  Original SimAM measures the importance of each neuron *independently* by
#  its squared deviation from the channel mean:
#
#         e_inv(t) = (t - μ_c)² / (4(σ²_c + λ)) + 0.5
#
#  This per-pixel formulation has a blind spot for cotton-disease imagery:
#  an ISOLATED outlier — a specular highlight, a water droplet, a sensor
#  noise spike on the leaf surface — also deviates strongly from the channel
#  mean and is therefore *wrongly* assigned high attention. Real disease
#  lesions, however, are SPATIALLY CONTIGUOUS regions, not isolated points.
#
#  Cotton-SimAM adds a SPATIAL-COHERENCE constraint: a neuron is *amplified*
#  when its local neighbourhood also deviates from the mean (i.e. the high
#  response is part of a contiguous region), while isolated spikes are left at
#  their baseline SimAM response. This raises the contrast between real lesions
#  and isolated noise by lifting the lesions — without ever attenuating the
#  discriminative lesion boundaries. The neighbourhood statistic is computed
#  with a FIXED average pooling operator, so the module remains STRICTLY
#  PARAMETER-FREE — keeping the lightweight advantage that motivated SimAM.
#
# ────────────────────────────────────────────────────────────────────────
#  Mathematical formulation
# ────────────────────────────────────────────────────────────────────────
#    d(t)        = (t - μ_c)²                              # deviation map
#    σ²_c        = (1/(N-1)) Σ_t d(t)                      # channel variance
#    e_inv(t)    = d(t) / (4(σ²_c + λ)) + 0.5              # SimAM energy
#
#    d_local(t)  = AvgPool_{k×k}( d )(t)                   # neighbourhood dev.
#    ρ(t)        = clamp( d_local(t) / (d(t) + λ), 0, 1 )  # coherence ∈ [0,1]
#                  # isolated spike: d large, d_local small  → ρ → 0 (no boost)
#                  # contiguous lesion: d ≈ d_local          → ρ → 1 (boosted)
#
#    ẽ_inv(t)    = e_inv(t) · ( 1 + γ·ρ(t) )               # coherence-BOOST
#                  # factor ∈ [1, 1+γ] ≥ 1 : amplifies coherent lesion regions
#                  # and never attenuates lesion peaks or boundaries.
#    out         = x · sigmoid( ẽ_inv )
#
#  γ ∈ [0,1] controls the coherence strength (γ=0 recovers vanilla SimAM).
#
# ────────────────────────────────────────────────────────────────────────
#  Properties
# ────────────────────────────────────────────────────────────────────────
#    ✅ ZERO learnable parameters (avg-pool is a fixed operator)
#    ✅ Negligible FLOPs (one extra k×k average pooling)
#    ✅ Shape-preserving: (B,C,H,W) → (B,C,H,W)
#    ✅ γ=0 exactly recovers the original SimAM (clean ablation baseline)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CottonSimAM']


class CottonSimAM(nn.Module):
    """Spatially-coherent parameter-free attention (★ Twice Innovation #3).

    Drop-in replacement for SimAM. Same I/O shape, still zero learnable
    parameters. Set gamma=0.0 to recover the original SimAM exactly (useful
    as an ablation baseline).

    Args:
        lambda_param (float): numerical-stability term. SimAM default 1e-4.
        kernel_size (int): neighbourhood size for the coherence term (odd).
                           3 or 5 recommended; 3 is the default.
        gamma (float): coherence strength in [0,1]. 0 = vanilla SimAM,
                       1 = full coherence modulation. Default 0.5.
    """

    def __init__(self, lambda_param: float = 1e-4,
                 kernel_size: int = 3, gamma: float = 0.5):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert 0.0 <= gamma <= 1.0, "gamma must be in [0, 1]"
        self.lambda_param = lambda_param
        self.kernel_size = kernel_size
        self.gamma = gamma
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        _, _, h, w = x.size()
        n = max(h * w - 1, 1)

        # ---- SimAM energy ------------------------------------------------
        mu = x.mean(dim=[2, 3], keepdim=True)
        d = (x - mu).pow(2)                       # (B,C,H,W) deviation
        v = d.sum(dim=[2, 3], keepdim=True) / n   # (B,C,1,1) channel variance
        e_inv = d / (4.0 * (v + self.lambda_param)) + 0.5

        # ---- ★ spatial-coherence modulation (parameter-free) -------------
        if self.gamma > 0.0:
            k = self.kernel_size
            # neighbourhood deviation via FIXED average pooling (zero params)
            d_local = F.avg_pool2d(d, kernel_size=k, stride=1, padding=k // 2)
            # coherence: isolated spike -> small; contiguous lesion -> ~1
            rho = (d_local / (d + self.lambda_param)).clamp(0.0, 1.0)
            # factor = 1 + γ·ρ ∈ [1, 1+γ] ≥ 1: coherent lesion regions (ρ→1) are
            # amplified; isolated spikes (ρ→0) stay at their SimAM baseline.
            # γ=0 recovers the original SimAM exactly.
            e_inv = e_inv * (1.0 + self.gamma * rho)

        return x * self.activation(e_inv)
