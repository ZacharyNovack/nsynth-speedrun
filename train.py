"""
train.py — Pitch-conditioned flow matching training on NSynth spectrograms

Usage
-----
    python train.py                              # defaults below
    python train.py --epochs 100 --max_files 20000 --batch_size 256
    python train.py --p_uncond 0.1               # enable CFG training (recommended)

Algorithm: OT-CFM (Optimal-Transport Conditional Flow Matching)
  Given a data sample x1, noise x0 ~ N(0,I), and pitch label p:
    x_t    = (1 - t) * x0  +  t * x1
    target = x1 - x0
    loss   = MSE( model(x_t, t, p), target )

Classifier-Free Guidance (CFG) training
  When --p_uncond > 0, each pitch label is independently replaced with the
  null token (index 128) with probability p_uncond. This teaches the model
  to also generate unconditionally, enabling CFG at inference time.

Note on loss curves
  Flow matching loss can appear noisy / flat even while sample quality improves,
  because the loss is an expectation over all noise levels. Don't read too much
  into small fluctuations — use eval.py --fd_every to track actual quality.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


# ── Muon optimizer (Newton-Schulz orthogonalized momentum) ────────────────────
# Based on modded-nanogpt by Keller Jordan (https://github.com/KellerJordan/modded-nanogpt)
# For ≥2-D weights: Nesterov momentum + NS5 orthogonalization → unit-scale updates.
# For 1-D params (biases, norms): plain AdamW at lr/20.
# Typical LR for Muon: 0.02  (much higher than AdamW due to norm-bounded updates).

def _zeropower_via_ns5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """G → orthogonal matrix closest to G, via 5 Newton-Schulz quintic iterations."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon — MomentUm Orthogonalized by Newton-schulz."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr  = group['lr']
            mom = group['momentum']
            ns  = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g     = p.grad
                state = self.state[p]
                if p.ndim >= 2:
                    # Reshape to 2D, apply Nesterov + NS orthogonalization
                    g2 = g.reshape(p.shape[0], -1)
                    if 'buf' not in state:
                        state['buf'] = torch.zeros_like(g2)
                    buf = state['buf'].mul_(mom).add_(g2)
                    u   = g2.add(buf, alpha=mom) if group['nesterov'] else buf
                    u   = _zeropower_via_ns5(u, steps=ns)
                    u  *= max(1, g2.shape[0] / g2.shape[1]) ** 0.5
                    p.data.add_(u.reshape(p.shape), alpha=-lr)
                else:
                    # AdamW fallback for biases / LayerNorm scales
                    alr = lr / 20
                    if 'step' not in state:
                        state.update(step=0,
                                     m1=torch.zeros_like(p),
                                     m2=torch.zeros_like(p))
                    state['step'] += 1
                    t  = state['step']
                    m1 = state['m1'].mul_(0.9).add_(g, alpha=0.1)
                    m2 = state['m2'].mul_(0.999).addcmul_(g, g, value=0.001)
                    p.data.mul_(1 - alr * 1e-4)
                    p.data.addcdiv_(m1 / (1 - 0.9**t),
                                    (m2 / (1 - 0.999**t)).sqrt_().add_(1e-8),
                                    value=-alr)
# ──────────────────────────────────────────────────────────────────────────────

from dataset import (
    FREQ_BINS, TIME_FRAMES,
    NSynthSpecDataset,
)
from model import (
    TinyFlowNet, UNet2DFlowNet, DiTFlowNet,
    count_params, NULL_PITCH,
)

# ── Defaults ───────────────────────────────────────────────────────────────────
TRAIN_DIR    = "/graft3/datasets/znovack/nsynth/nsynth-train/audio/"
SAVE_PATH    = "flow_model.pt"

MAX_FILES    = 5_000    # subset size — 5k ≈ fast, 50k ≈ 10 min/epoch on good GPU
# TinyFlowNet defaults
HIDDEN   = 32
N_BLOCKS = 4
T_DIM    = 32

# UNet defaults
UNET_HIDDEN = 32          # base channel width; bottleneck = 2×

# DiT defaults
DIT_D_MODEL    = 64
DIT_N_LAYERS   = 3
DIT_N_HEADS    = 4
DIT_PATCH_SIZE = 8

BATCH_SIZE = 64
LR         = 3e-4
N_EPOCHS   = 30
P_UNCOND   = 0.1      # CFG dropout probability; 0 = no CFG support at inference
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────


def cfm_loss(
    model,
    x1: torch.Tensor,
    pitch: torch.Tensor,
    p_uncond: float = 0.0,
    t_sample: str = "uniform",
) -> torch.Tensor:
    """
    One mini-batch of OT-CFM training (pitch-conditioned, optional CFG dropout).

    Parameters
    ----------
    model    : flow model
    x1       : (B, 2, F, T)  normalized spectrograms
    pitch    : (B,)           MIDI pitch labels, dtype=long
    p_uncond : probability of replacing pitch with NULL_PITCH (for CFG)
    t_sample : 'uniform' samples t ~ U[0,1]; 'logit_normal' concentrates
               mass near t=0.5, skipping trivially-easy endpoints
    """
    B  = x1.shape[0]
    if t_sample == "logit_normal":
        # t ~ sigmoid(N(0,1)) — concentrates in (0.2, 0.8), avoids trivial extremes
        t = torch.sigmoid(torch.randn(B, device=x1.device))
    else:
        t = torch.rand(B, device=x1.device)
    x0 = torch.randn_like(x1)

    xt     = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    target = x1 - x0

    # CFG dropout: mask some pitch labels with the null token
    if p_uncond > 0:
        pitch = pitch.clone()
        mask  = torch.rand(B, device=x1.device) < p_uncond
        pitch[mask] = NULL_PITCH

    pred = model(xt, t, pitch)
    return F.mse_loss(pred, target)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.p_uncond > 0:
        print(f"CFG training enabled: p_uncond={args.p_uncond} "
              f"(null token = pitch index {NULL_PITCH})")
    else:
        print("CFG training disabled (--p_uncond 0). "
              "Guidance scale > 1 at inference won't work.")

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = NSynthSpecDataset(
        args.train_dir,
        max_files=args.max_files,
        instrument_filter=args.instrument_filter,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    mt = args.model_type
    if mt == "tiny":
        model = TinyFlowNet(
            hidden=args.hidden, n_blocks=args.n_blocks, t_dim=args.t_dim
        ).to(device)
        model_cfg = {
            "model_type": "tiny",
            "hidden": args.hidden, "n_blocks": args.n_blocks, "t_dim": args.t_dim,
            "freq_bins": FREQ_BINS, "time_frames": TIME_FRAMES,
            "p_uncond": args.p_uncond,
        }
    elif mt == "unet":
        model = UNet2DFlowNet(hidden=args.unet_hidden, t_dim=args.t_dim).to(device)
        model_cfg = {
            "model_type": "unet",
            "hidden": args.unet_hidden, "t_dim": args.t_dim,
            "freq_bins": FREQ_BINS, "time_frames": TIME_FRAMES,
            "p_uncond": args.p_uncond,
        }
    elif mt == "dit":
        model = DiTFlowNet(
            freq_bins=FREQ_BINS, time_frames=TIME_FRAMES,
            d_model=args.dit_d_model, n_layers=args.dit_n_layers,
            n_heads=args.dit_n_heads, t_dim=args.t_dim,
            patch_size=args.dit_patch_size,
        ).to(device)
        model_cfg = {
            "model_type": "dit",
            "d_model": args.dit_d_model, "n_layers": args.dit_n_layers,
            "n_heads": args.dit_n_heads, "patch_size": args.dit_patch_size,
            "t_dim": args.t_dim,
            "freq_bins": FREQ_BINS, "time_frames": TIME_FRAMES,
            "p_uncond": args.p_uncond,
        }
    else:
        raise ValueError(f"Unknown --model_type: {mt!r}")

    n_params = count_params(model)
    print(f"Model ({mt}): {n_params:,} parameters  ({n_params/1e3:.1f}k)")
    print(f"Spec shape: (2, {FREQ_BINS}, {TIME_FRAMES})")
    print(f"Steps/epoch: {len(loader)}  |  Total steps: {len(loader) * args.epochs}")
    print(f"t_sample: {args.t_sample}  |  optimizer: {args.optimizer}  |  lr: {args.lr}")
    if args.bf16:
        if not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 not natively supported on this GPU — falling back to fp32")
            args.bf16 = False
        else:
            print("bf16 autocast enabled")

    t0 = time.time()

    # ── Optimizer + schedule ───────────────────────────────────────────────────
    if args.optimizer == "muon":
        optimizer = Muon(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    train_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for x1, pitch in tqdm(loader):
            x1    = x1.to(device, non_blocking=True)
            pitch = pitch.to(device, non_blocking=True)

            with torch.autocast(device, dtype=torch.bfloat16, enabled=args.bf16):
                loss = cfm_loss(model, x1, pitch, p_uncond=args.p_uncond,
                                t_sample=args.t_sample)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(batch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Save ───────────────────────────────────────────────────────────────────
    torch.save(
        {
            "model_state":  model.state_dict(),
            "config":       model_cfg,
            "train_losses": train_losses,
            "n_params":     n_params,
        },
        args.save_path,
    )
    elapsed = time.time() - t0
    print(f"\nCheckpoint saved → {args.save_path}")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train pitch-conditioned flow model on NSynth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train_dir",    default=TRAIN_DIR)
    p.add_argument("--save_path",    default=SAVE_PATH)
    p.add_argument("--max_files",    type=int,   default=MAX_FILES,
                   help="Training files to use. 5k≈fast, 50k≈thorough.")
    # ── Model selection ────────────────────────────────────────────────────────
    p.add_argument("--model_type",  choices=["tiny", "unet", "dit"], default="tiny",
                   help="tiny=flat ResBlocks (~88k), unet=2-level UNet (~213k), "
                        "dit=Diffusion Transformer (~221k)")
    # TinyFlowNet args
    p.add_argument("--hidden",      type=int,   default=HIDDEN)
    p.add_argument("--n_blocks",    type=int,   default=N_BLOCKS)
    p.add_argument("--t_dim",       type=int,   default=T_DIM)
    # UNet args
    p.add_argument("--unet_hidden", type=int,   default=UNET_HIDDEN,
                   help="UNet base channel width (bottleneck = 2×)")
    # DiT args
    p.add_argument("--dit_d_model",    type=int, default=DIT_D_MODEL)
    p.add_argument("--dit_n_layers",   type=int, default=DIT_N_LAYERS)
    p.add_argument("--dit_n_heads",    type=int, default=DIT_N_HEADS)
    p.add_argument("--dit_patch_size", type=int, default=DIT_PATCH_SIZE)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE,
                   help="Bigger batch = fewer steps/epoch but more stable gradients.")
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--epochs",       type=int,   default=N_EPOCHS)
    p.add_argument("--p_uncond",  type=float, default=P_UNCOND,
                   help="CFG dropout probability. 0=disabled, 0.1=recommended.")
    p.add_argument("--t_sample", choices=["uniform", "logit_normal"], default="uniform",
                   help="Timestep sampling: uniform=U[0,1], logit_normal concentrates "
                        "near t=0.5 for faster convergence (SD3/Flux trick)")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="adamw: standard AdamW (default). muon: orthogonalized momentum "
                        "(use --lr 0.02, gives ~2× faster convergence on transformers/CNNs).")
    p.add_argument("--instrument_filter", type=str,   default=None,
                   help="Only train on files whose name starts with this prefix. "
                        "E.g. 'keyboard_synthetic' or 'keyboard_synthetic_099'. "
                        "Narrower = easier task, better FD.")
    p.add_argument("--seed",              type=int,   default=SEED)
    p.add_argument("--bf16", action="store_true",
                   help="Enable bfloat16 autocast (free ~1.5× speedup on A100/L40S/H100; "
                        "no gradient scaler needed; falls back to fp32 if unsupported).")
    args = p.parse_args()

    train(args)
