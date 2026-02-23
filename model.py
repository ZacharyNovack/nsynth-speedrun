"""
model.py — TinyFlowNet, UNet2DFlowNet, DiTFlowNet (all pitch-conditioned, CFG-ready)

All three models share the same forward interface:
    forward(x, t, pitch) → same shape as x

where:
    x     : (B, 2, FREQ_BINS, TIME_FRAMES)  noisy spectrogram
    t     : (B,)  time in [0, 1]
    pitch : (B,)  MIDI pitch 0–127, or NULL_PITCH (128) for unconditional

Classifier-Free Guidance (CFG)
-------------------------------
Pitch index 128 is reserved as the null / unconditional token.
During training (see train.py --p_uncond), some pitch labels are randomly
replaced with 128. At inference (see infer.py --guidance_scale), the model
is run twice and the outputs are combined:
    v = v_uncond + guidance_scale * (v_cond - v_uncond)

Model overview
--------------
  TinyFlowNet   (~88k params)  : flat stack of ResBlocks, no downsampling
  UNet2DFlowNet (~213k params) : 2-level encoder-decoder with skip connections
  DiTFlowNet    (~221k params) : patch-based Diffusion Transformer (adaLN-Zero)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NULL_PITCH = 128   # reserved index for unconditional (CFG null token)


# ── Shared building blocks ─────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Fixed sinusoidal embedding of a scalar time value t ∈ [0, 1].
    No learnable parameters — the MLP after it does the heavy lifting.
    """

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half).float() / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B,)
        emb = t[:, None] * self.freqs[None]          # (B, half)
        return torch.cat([emb.sin(), emb.cos()], -1)  # (B, dim)


def _make_t_emb(t_dim: int) -> nn.Sequential:
    """Shared time-embedding MLP: sinusoidal → 2-layer MLP → (B, t_dim)."""
    return nn.Sequential(
        SinusoidalEmbedding(t_dim),
        nn.Linear(t_dim, t_dim * 2),
        nn.SiLU(),
        nn.Linear(t_dim * 2, t_dim),
    )


class ResBlock(nn.Module):
    """
    Pre-norm residual conv block with combined time+pitch conditioning.

      x  ──►  GroupNorm ──► Conv ──► SiLU  ──► + cond_shift ──► GroupNorm ──► Conv ──► SiLU  ──► + x
    """

    def __init__(self, channels: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.norm1  = nn.GroupNorm(groups, channels)
        self.conv1  = nn.Conv2d(channels, channels, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, channels)     # conditioning → additive shift
        self.norm2  = nn.GroupNorm(groups, channels)
        self.conv2  = nn.Conv2d(channels, channels, 3, padding=1)
        self.act    = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, t_dim) — combined time + pitch embedding
        h  = self.act(self.conv1(self.norm1(x)))
        h  = h + self.t_proj(self.act(cond))[:, :, None, None]  # broadcast over (F,T)
        h  = self.act(self.conv2(self.norm2(h)))
        return x + h


# ── TinyFlowNet ────────────────────────────────────────────────────────────────

class TinyFlowNet(nn.Module):
    """
    Predicts the vector field v_θ(x_t, t, pitch) for flow matching.

    A flat stack of ResBlocks — no spatial downsampling.
    Simple and fast; good baseline.

    Default config (hidden=32, n_blocks=4, t_dim=32) ≈ 88k parameters.
    """

    def __init__(self, hidden: int = 32, n_blocks: int = 4, t_dim: int = 32):
        super().__init__()
        groups = min(8, hidden)

        self.t_emb     = _make_t_emb(t_dim)
        self.pitch_emb = nn.Embedding(NULL_PITCH + 1, t_dim)

        self.input_proj = nn.Conv2d(2, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden, t_dim, groups=groups) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        cond = self.t_emb(t) + self.pitch_emb(pitch)  # (B, t_dim)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)


# ── UNet2DFlowNet ──────────────────────────────────────────────────────────────

class UNet2DFlowNet(nn.Module):
    """
    2D UNet vector-field network for flow matching on spectrograms.

    Two spatial downsampling levels with skip connections:

      Encoder:  [2→C] → ResBlock(C) ─── ↓ → ResBlock(C) ─── ↓ → ResBlock(2C)
                                skip1 ↗              skip2 ↗
      Decoder:  ↑+skip2 → merge(3C→C) → ResBlock(C) → ↑+skip1 → merge(2C→C) → ResBlock(C) → [C→2]

    Bilinear upsampling (size read from skip tensor) handles odd input dimensions
    (129 × 63) without size-mismatch issues.

    Default config (hidden=32, t_dim=32): ~213k parameters.
    """

    def __init__(self, hidden: int = 32, t_dim: int = 32):
        super().__init__()
        C = hidden
        g = min(8, C)

        self.t_emb     = _make_t_emb(t_dim)
        self.pitch_emb = nn.Embedding(NULL_PITCH + 1, t_dim)

        # Encoder
        self.input_proj = nn.Conv2d(2, C, 3, padding=1)
        self.enc1       = ResBlock(C,     t_dim, groups=g)
        self.down1      = nn.AvgPool2d(2)
        self.chan_up1   = nn.Conv2d(C, C, 1)           # identity channel change (C→C)
        self.enc2       = ResBlock(C,     t_dim, groups=g)
        self.down2      = nn.AvgPool2d(2)
        self.chan_up2   = nn.Conv2d(C, C * 2, 1)       # C → 2C at bottleneck
        self.bottleneck = ResBlock(C * 2, t_dim, groups=min(8, C * 2))

        # Decoder — merge convs reduce concatenated channels before ResBlock
        self.merge1 = nn.Conv2d(C * 2 + C, C, 3, padding=1)   # cat(2C, C) → C
        self.dec1   = ResBlock(C, t_dim, groups=g)
        self.merge2 = nn.Conv2d(C + C, C, 3, padding=1)        # cat(C, C) → C
        self.dec2   = ResBlock(C, t_dim, groups=g)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(g, C),
            nn.SiLU(),
            nn.Conv2d(C, 2, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        cond = self.t_emb(t) + self.pitch_emb(pitch)  # (B, t_dim)

        # Encoder
        h  = self.input_proj(x)                         # (B, C, F, T)
        s1 = self.enc1(h, cond)                         # (B, C, F, T)   — skip1
        h  = self.chan_up1(self.down1(s1))               # (B, C, F//2, T//2)
        s2 = self.enc2(h, cond)                         # (B, C, F//2, T//2) — skip2
        h  = self.chan_up2(self.down2(s2))               # (B, 2C, F//4, T//4)
        h  = self.bottleneck(h, cond)                    # (B, 2C, F//4, T//4)

        # Decoder — upsample to match skip spatial size, cat, reduce, ResBlock
        h = F.interpolate(h, size=s2.shape[2:], mode='bilinear', align_corners=False)
        h = self.merge1(torch.cat([h, s2], dim=1))       # (B, C, F//2, T//2)
        h = self.dec1(h, cond)

        h = F.interpolate(h, size=s1.shape[2:], mode='bilinear', align_corners=False)
        h = self.merge2(torch.cat([h, s1], dim=1))       # (B, C, F, T)
        h = self.dec2(h, cond)

        return self.output_proj(h)                        # (B, 2, F, T)


# ── DiTFlowNet ─────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaLN-Zero conditioning.

    Given a conditioning vector cond ∈ R^{t_dim}, a learned MLP produces six
    per-sample parameters (scale1, shift1, gate1, scale2, shift2, gate2) that
    modulate the attention and FFN sublayers independently.

    The final linear in the adaLN MLP is zero-initialized so each block starts
    as a near-identity residual connection (gate=0, scale≈1, shift≈0).
    """

    def __init__(self, d_model: int, n_heads: int, t_dim: int, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        # adaLN-Zero: (B, t_dim) → 6 × (B, d_model) for scale/shift/gate × 2 sublayers
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_mlp[-1].weight)
        nn.init.zeros_(self.adaLN_mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, t_dim);  x: (B, n_tokens, d_model)
        g1, b1, a1, g2, b2, a2 = self.adaLN_mlp(cond).chunk(6, dim=-1)

        # Attention sub-block
        h = (1 + g1[:, None]) * self.norm1(x) + b1[:, None]
        h, _ = self.attn(h, h, h)
        x = x + a1[:, None] * h

        # FFN sub-block
        h = (1 + g2[:, None]) * self.norm2(x) + b2[:, None]
        h = self.ffn(h)
        x = x + a2[:, None] * h

        return x


class DiTFlowNet(nn.Module):
    """
    Diffusion Transformer vector-field network for flow matching on spectrograms.

    Patchifies the (2, freq_bins, time_frames) input into tokens, applies N
    transformer blocks with adaLN-Zero conditioning, then unpatches back.

    Input padding: the spectrogram is zero-padded to the nearest multiple of
    patch_size in each spatial dimension before patchification and cropped back
    to the original size at output.

    Default config (d_model=64, n_layers=3, patch_size=8): ~221k parameters.
    For (2, 129, 63): pads to (2, 136, 64) → 17×8 = 136 tokens, patch_dim=128.

    Parameters
    ----------
    freq_bins   : input frequency dimension (e.g. 129)
    time_frames : input time dimension (e.g. 63)
    d_model     : transformer hidden dimension
    n_layers    : number of DiT blocks
    n_heads     : attention heads (must divide d_model)
    t_dim       : conditioning embedding dimension
    patch_size  : spatial patch size applied to both freq and time axes
    """

    def __init__(
        self,
        freq_bins: int  = 129,
        time_frames: int = 63,
        d_model: int    = 64,
        n_layers: int   = 3,
        n_heads: int    = 4,
        t_dim: int      = 32,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        p = patch_size
        patch_dim = 2 * p * p   # 2 channels × p × p pixels per patch

        # Number of tokens for the fixed-size spectrograms
        nf = math.ceil(freq_bins  / p)
        nt = math.ceil(time_frames / p)
        n_tokens = nf * nt

        self.t_emb     = _make_t_emb(t_dim)
        self.pitch_emb = nn.Embedding(NULL_PITCH + 1, t_dim)

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, t_dim) for _ in range(n_layers)
        ])
        self.norm         = nn.LayerNorm(d_model)
        self.unpatch_proj = nn.Linear(d_model, patch_dim, bias=False)

    def _patchify(self, x: torch.Tensor) -> tuple:
        """(B, 2, freq, time) → (B, nf*nt, 2*p*p)"""
        B, C, freq, time = x.shape
        p = self.patch_size
        pad_f = (-freq) % p
        pad_t = (-time) % p
        if pad_f or pad_t:
            x = F.pad(x, (0, pad_t, 0, pad_f))
        _, _, Fp, Tp = x.shape
        nf, nt = Fp // p, Tp // p
        # (B, C, nf, p, nt, p) → (B, nf, nt, C, p, p) → (B, nf*nt, C*p*p)
        x = x.reshape(B, C, nf, p, nt, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, nf * nt, C * p * p)
        return x, (freq, time, nf, nt)

    def _unpatchify(self, x: torch.Tensor, freq_orig: int, time_orig: int,
                    nf: int, nt: int) -> torch.Tensor:
        """(B, nf*nt, 2*p*p) → (B, 2, freq_orig, time_orig)"""
        B = x.shape[0]
        p = self.patch_size
        x = x.reshape(B, nf, nt, 2, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, 2, nf * p, nt * p)
        return x[:, :, :freq_orig, :time_orig]

    def forward(self, x: torch.Tensor, t: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        cond = self.t_emb(t) + self.pitch_emb(pitch)          # (B, t_dim)
        tokens, (freq_orig, time_orig, nf, nt) = self._patchify(x)
        tokens = self.patch_embed(tokens) + self.pos_embed     # (B, n_tokens, d_model)
        for block in self.blocks:
            tokens = block(tokens, cond)
        tokens = self.unpatch_proj(self.norm(tokens))          # (B, n_tokens, patch_dim)
        return self._unpatchify(tokens, freq_orig, time_orig, nf, nt)


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model_from_config(cfg: dict) -> nn.Module:
    """Reconstruct the correct model class from a saved checkpoint config dict."""
    model_type = cfg.get("model_type", "tiny")
    if model_type == "tiny":
        return TinyFlowNet(
            hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], t_dim=cfg["t_dim"]
        )
    elif model_type == "unet":
        return UNet2DFlowNet(hidden=cfg["hidden"], t_dim=cfg["t_dim"])
    elif model_type == "dit":
        return DiTFlowNet(
            freq_bins=cfg["freq_bins"], time_frames=cfg["time_frames"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], t_dim=cfg["t_dim"],
            patch_size=cfg["patch_size"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
