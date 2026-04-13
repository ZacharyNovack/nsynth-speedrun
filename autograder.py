"""
autograder.py — Autograder for Flow Matching Assignment

Imports the student's assignment.py and tests each part.

Usage
-----
    python autograder.py                          # grade all parts
    python autograder.py --submission_dir ./sub    # custom submission directory

Expected files in submission directory:
    assignment.py                   — student's converted notebook
    submission_q4/submission.npz    — Q4 generated samples + noises + metadata
    model_ft_q4.pt                  — Q4 fine-tuned checkpoint
    submission_q5/submission.npz    — (bonus) Q5 samples
    model_q5.pt                     — (bonus) Q5 checkpoint
"""

import argparse
import copy
import importlib.util
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(os.path.dirname(__file__), "pretrained_keyboard.pt")

POINTS = {
    "Q1: euler_sample":        20,
    "Q2a: naive_scale_sample": 10,
    "Q2b: cfg_sample":         20,
    "Q3a: heun_sample":        20,
    "Q3b: rk4_sample":         10,
    "Q4a: sample_timesteps":    5,
    "Q4b: flow_loss":           5,
    "Q4: reproducibility":     10,
    "Q5: beat_baseline":       10,
}

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"


# ── Dummy models for analytical testing (Q1–Q3) ───────────────────────────────
# These have no learnable parameters. They behave predictably so we can verify
# sampler implementations with exact, analytically-derived expected outputs.

class _ConstantVelModel:
    """Always returns `c * ones_like(x)`, regardless of x, t, or pitch.

    With this model, any correct sampler integrating over [1→0] should give:
        x_final = x0 - c   (since the total integral of a constant over Δt=1 is c)
    """
    def __init__(self, c=0.1):
        self.c = c

    def __call__(self, x, t, pitch):
        return torch.full_like(x, self.c)


class _CFGTestModel:
    """Returns c_cond for real pitches, c_null for NULL_PITCH.

    At guidance_scale=gs, the CFG formula gives:
        v = c_null + gs * (c_cond - c_null)
    so x_final = x0 - v.
    """
    def __init__(self, c_cond=0.2, c_null=0.1):
        self.c_cond = c_cond
        self.c_null = c_null

    def __call__(self, x, t, pitch):
        from model import NULL_PITCH
        null_mask = (pitch == NULL_PITCH).float()[:, None, None, None]
        return torch.ones_like(x) * (
            null_mask * self.c_null + (1 - null_mask) * self.c_cond
        )


class _LinearTimeModel:
    """Returns v = t * ones_like(x).

    Heun's method is *exact* for this model (trapezoidal rule is exact for
    degree-1 polynomials), so:
        x_final = x0 - ∫₁⁰ t dt = x0 - 0.5   (exactly, for any n_steps ≥ 1)

    Euler is NOT exact: it gives x0 - 0.5 - 1/(2*n_steps).
    """
    def __call__(self, x, t, pitch):
        return torch.ones_like(x) * t[:, None, None, None]


class _QuadTimeModel:
    """Returns v = t² * ones_like(x).

    RK4 (Simpson's rule) is *exact* for degree-≤3 polynomials, so:
        x_final = x0 - ∫₁⁰ t² dt = x0 - 1/3   (exactly, for any n_steps ≥ 1)

    Heun (trapezoidal) is NOT exact for t²: it gives ≈ x0 - 0.34375 (N=4),
    which differs from x0 - 1/3 ≈ x0 - 0.33333.
    """
    def __call__(self, x, t, pitch):
        return torch.ones_like(x) * (t[:, None, None, None] ** 2)


class _RecordingModel(nn.Module):
    """Records the x_t and pitch seen on each forward call; predicts zero.

    Has a dummy parameter so gradients can flow from a loss computed on its
    output (needed to verify flow_loss returns a differentiable tensor).
    """
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.last_xt = None
        self.last_pitch = None

    def forward(self, x, t, pitch):
        self.last_xt = x.detach().clone()
        self.last_pitch = pitch.detach().clone()
        # self.w * zeros keeps a gradient path without affecting the value
        return self.w * torch.zeros_like(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_submission(submission_dir: str):
    """Import assignment.py from the submission directory as a module."""
    assignment_path = os.path.join(submission_dir, "assignment.py")
    if not os.path.exists(assignment_path):
        print(f"ERROR: {assignment_path} not found")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("assignment", assignment_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pretrained_model(device):
    """Load the pretrained keyboard model (wrapped for diffusion convention)."""
    from model import load_flow_model
    return load_flow_model(CKPT_PATH, device=device)


def run_test(name: str, fn, results: dict):
    """Run a test function and record PASS/FAIL."""
    try:
        fn()
        results[name] = "PASS"
        print(f"  {PASS}  {name}  [{POINTS[name]} pts]")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"  {FAIL}  {name}  [0/{POINTS[name]} pts]")
        print(f"         {e}")


# ── Test functions ────────────────────────────────────────────────────────────

def test_q1(sub, model, device):
    """Q1: euler_sample — basic sanity + analytical check with dummy model."""
    from dataset import FREQ_BINS, TIME_FRAMES

    # --- Sanity check with pretrained model ---
    torch.manual_seed(0)
    x0 = torch.randn(4, 2, FREQ_BINS, TIME_FRAMES, device=device)
    p  = torch.tensor([60, 62, 64, 67], dtype=torch.long, device=device)

    out = sub.euler_sample(model, x0.clone(), p, n_steps=20)

    assert out.shape == (4, 2, FREQ_BINS, TIME_FRAMES), \
        f"Wrong output shape: {out.shape}"
    assert not torch.allclose(out, x0, atol=1e-3), \
        "Output equals input — integration loop not implemented"
    assert out.isfinite().all(), "Output contains NaN or Inf"
    assert out.std() > 0.05, f"Output std={out.std():.4f} is suspiciously low"

    # --- Analytical check with constant-velocity dummy model ---
    # v(x,t,p) = 0.1  always  →  x_final = x0 - 0.1  (regardless of n_steps)
    x0s = torch.randn(2, 2, 8, 8)
    ps  = torch.zeros(2, dtype=torch.long)
    dummy = _ConstantVelModel(c=0.1)

    out_dummy = sub.euler_sample(dummy, x0s.clone(), ps, n_steps=10)
    assert torch.allclose(out_dummy, x0s - 0.1, atol=1e-5), \
        "Euler with constant v=0.1 should give x_final = x0 - 0.1 exactly. " \
        "Check that t decrements from 1→0 and you subtract (not add) v*dt."


def test_q2a(sub, model, device):
    """Q2a: naive_scale_sample — sanity + analytical scale verification."""
    from dataset import FREQ_BINS, TIME_FRAMES

    # --- Sanity: scale=1 matches euler, scale=2 differs ---
    torch.manual_seed(0)
    x0 = torch.randn(4, 2, FREQ_BINS, TIME_FRAMES, device=device)
    p  = torch.tensor([60, 62, 64, 67], dtype=torch.long, device=device)

    out_s1 = sub.naive_scale_sample(model, x0.clone(), p, n_steps=20, scale=1.0)
    out_e  = sub.euler_sample(model, x0.clone(), p, n_steps=20)
    out_s2 = sub.naive_scale_sample(model, x0.clone(), p, n_steps=20, scale=2.0)

    assert torch.allclose(out_s1, out_e, atol=1e-5), \
        "naive_scale_sample(scale=1.0) must match euler_sample exactly"
    assert not torch.allclose(out_s2, out_e, atol=1e-3), \
        "naive_scale_sample(scale=2.0) should differ from scale=1.0"

    # --- Analytical: constant model, verify exact output at multiple scales ---
    x0s = torch.randn(2, 2, 8, 8)
    ps  = torch.zeros(2, dtype=torch.long)
    dummy = _ConstantVelModel(c=0.1)

    # scale=3 → v_eff = 0.3 → x_final = x0 - 0.3
    out_s3 = sub.naive_scale_sample(dummy, x0s.clone(), ps, n_steps=10, scale=3.0)
    assert torch.allclose(out_s3, x0s - 0.3, atol=1e-5), \
        "With constant v=0.1 and scale=3, expected x_final = x0 - 0.3"

    # scale=0 → no movement → x_final = x0
    out_s0 = sub.naive_scale_sample(dummy, x0s.clone(), ps, n_steps=10, scale=0.0)
    assert torch.allclose(out_s0, x0s, atol=1e-5), \
        "With scale=0, x_final should equal x0 (no movement)"


def test_q2b(sub, model, device):
    """Q2b: cfg_sample — sanity + analytical CFG formula check."""
    from dataset import FREQ_BINS, TIME_FRAMES

    # --- Sanity with pretrained model ---
    torch.manual_seed(0)
    x0 = torch.randn(4, 2, FREQ_BINS, TIME_FRAMES, device=device)
    p  = torch.tensor([60, 62, 64, 67], dtype=torch.long, device=device)

    out_cfg1   = sub.cfg_sample(model, x0.clone(), p, n_steps=20, guidance_scale=1.0)
    out_euler  = sub.euler_sample(model, x0.clone(), p, n_steps=20)
    out_cfg6   = sub.cfg_sample(model, x0.clone(), p, n_steps=20, guidance_scale=6.0)
    out_naive2 = sub.naive_scale_sample(model, x0.clone(), p, n_steps=20, scale=2.0)

    assert torch.allclose(out_cfg1, out_euler, atol=1e-5), \
        "cfg_sample(guidance_scale=1.0) must equal euler_sample"
    assert not torch.allclose(out_cfg6, out_euler, atol=1e-3), \
        "cfg_sample(guidance_scale=6.0) should differ from scale=1.0"
    assert not torch.allclose(out_cfg6, out_naive2, atol=1e-3), \
        "CFG (scale=6) must differ from naive scaling (scale=2)"

    # --- Analytical: split dummy model, verify exact CFG formula ---
    # c_cond=0.2, c_null=0.1
    # v_cfg = c_null + gs*(c_cond - c_null) = 0.1 + gs*0.1
    # x_final = x0 - v_cfg
    x0s = torch.randn(2, 2, 8, 8)
    ps  = torch.zeros(2, dtype=torch.long)  # non-null pitches
    cfg_dummy = _CFGTestModel(c_cond=0.2, c_null=0.1)

    # gs=1 uses only cond velocity (no uncond call) → v=0.2 → x0 - 0.2
    out_d1 = sub.cfg_sample(cfg_dummy, x0s.clone(), ps, n_steps=8, guidance_scale=1.0)
    assert torch.allclose(out_d1, x0s - 0.2, atol=1e-5), \
        "cfg(gs=1) with c_cond=0.2: expected x_final = x0 - 0.2"

    # gs=2 → v = 0.1 + 2*(0.2-0.1) = 0.3 → x0 - 0.3
    out_d2 = sub.cfg_sample(cfg_dummy, x0s.clone(), ps, n_steps=8, guidance_scale=2.0)
    assert torch.allclose(out_d2, x0s - 0.3, atol=1e-5), \
        "cfg(gs=2) with c_cond=0.2, c_null=0.1: expected x_final = x0 - 0.3 " \
        "(formula: v = c_null + gs*(c_cond - c_null) = 0.3, not 0.4)"


def test_q3a(sub, model, device):
    """Q3a: heun_sample — sanity + Heun exactness check with linear velocity model."""
    from dataset import FREQ_BINS, TIME_FRAMES

    # --- Sanity with pretrained model ---
    torch.manual_seed(0)
    x0 = torch.randn(4, 2, FREQ_BINS, TIME_FRAMES, device=device)
    p  = torch.tensor([60, 62, 64, 67], dtype=torch.long, device=device)

    out_euler    = sub.euler_sample(model, x0.clone(), p, n_steps=25)
    out_heun     = sub.heun_sample( model, x0.clone(), p, n_steps=25, guidance_scale=1.0)
    out_heun_gs1 = sub.heun_sample( model, x0.clone(), p, n_steps=25, guidance_scale=1.0)
    out_heun_gs6 = sub.heun_sample( model, x0.clone(), p, n_steps=25, guidance_scale=6.0)

    assert out_heun.shape == (4, 2, FREQ_BINS, TIME_FRAMES), \
        f"Wrong shape: {out_heun.shape}"
    assert out_heun.isfinite().all(), "Heun output contains NaN/Inf"
    assert not torch.allclose(out_heun, out_euler, atol=1e-3), \
        "heun_sample must differ from euler_sample"
    assert torch.allclose(out_heun, out_heun_gs1, atol=1e-5), \
        "heun_sample should be deterministic"
    assert not torch.allclose(out_heun_gs6, out_heun, atol=1e-3), \
        "heun_sample with guidance_scale=6 should differ from guidance_scale=1"

    # --- Analytical: v=t → Heun is exact (trapezoidal rule), x_final = x0 - 0.5 ---
    # For any n_steps, Heun gives EXACTLY x0 - 0.5.
    # Euler gives x0 - 0.5 - 1/(2*n_steps), so with n_steps=5 it gives x0 - 0.6.
    # This catches students who implement Euler inside heun_sample.
    x0s = torch.ones(2, 2, 8, 8)
    ps  = torch.zeros(2, dtype=torch.long)
    lin = _LinearTimeModel()

    out_heun_lin = sub.heun_sample(lin, x0s.clone(), ps, n_steps=5, guidance_scale=1.0)
    assert torch.allclose(out_heun_lin, x0s - 0.5, atol=1e-5), \
        "heun_sample with v=t should give x_final = x0 - 0.5 exactly for any n_steps. " \
        "Check that you use the corrector step (k1+k2)/2, not just k1."

    out_euler_lin = sub.euler_sample(lin, x0s.clone(), ps, n_steps=5)
    assert not torch.allclose(out_euler_lin, x0s - 0.5, atol=1e-4), \
        "Euler with v=t is not exact — sanity check on test setup failed"

    # --- Analytical: CFG formula through Heun ---
    # With CFGTestModel(0.2, 0.1) and gs=2 → v_cfg=0.3 (constant) → x_final = x0 - 0.3
    cfg_dummy = _CFGTestModel(c_cond=0.2, c_null=0.1)
    out_heun_cfg = sub.heun_sample(cfg_dummy, x0s.clone(), ps, n_steps=8, guidance_scale=2.0)
    assert torch.allclose(out_heun_cfg, x0s - 0.3, atol=1e-5), \
        "heun_sample with CFG (gs=2, c_cond=0.2, c_null=0.1): expected x_final = x0 - 0.3"


def test_q3b(sub, model, device):
    """Q3b: rk4_sample — sanity + RK4 exactness check with quadratic velocity model."""
    from dataset import FREQ_BINS, TIME_FRAMES

    # --- Sanity with pretrained model ---
    torch.manual_seed(0)
    x0 = torch.randn(4, 2, FREQ_BINS, TIME_FRAMES, device=device)
    p  = torch.tensor([60, 62, 64, 67], dtype=torch.long, device=device)

    out_rk4  = sub.rk4_sample(model, x0.clone(), p, n_steps=12, guidance_scale=1.0)
    out_heun = sub.heun_sample(model, x0.clone(), p, n_steps=25, guidance_scale=1.0)

    assert out_rk4.shape == (4, 2, FREQ_BINS, TIME_FRAMES), \
        f"Wrong shape: {out_rk4.shape}"
    assert out_rk4.isfinite().all(), "RK4 output contains NaN/Inf"
    assert not torch.allclose(out_rk4, out_heun, atol=1e-3), \
        "RK4 should differ from Heun"

    # --- Analytical: v=t² → RK4 (Simpson's rule) is exact, x_final = x0 - 1/3 ---
    # RK4 is exact for polynomials up to degree 3, so ∫₁⁰ t² dt = -1/3 → x0 - 1/3.
    # Heun (trapezoidal rule) gives x0 ≈ x0 - 0.344 (N=4), which differs by ~0.010.
    # This catches students who implement Heun's method inside rk4_sample.
    x0s  = torch.ones(2, 2, 8, 8)
    ps   = torch.zeros(2, dtype=torch.long)
    quad = _QuadTimeModel()

    out_rk4_quad = sub.rk4_sample(quad, x0s.clone(), ps, n_steps=4, guidance_scale=1.0)
    assert torch.allclose(out_rk4_quad, x0s - (1.0 / 3), atol=1e-5), \
        "rk4_sample with v=t² should give x_final = x0 - 1/3 exactly (Simpson's rule). " \
        "Check that your k2/k3 evaluations use t - dt/2 and that weights are (1,2,2,1)/6."

    out_heun_quad = sub.heun_sample(quad, x0s.clone(), ps, n_steps=4, guidance_scale=1.0)
    assert not torch.allclose(out_heun_quad, x0s - (1.0 / 3), atol=1e-4), \
        "Heun with v=t² should not give exactly x0 - 1/3 — sanity check on test setup failed"


def test_q4a_sample_timesteps(sub, device):
    """Q4a: sample_timesteps — shape, range, and distribution checks."""
    torch.manual_seed(0)

    # Shape and range — uniform
    t_unif = sub.sample_timesteps(64, device, 'uniform')
    assert t_unif.shape == (64,), \
        f"Expected shape (64,), got {t_unif.shape}"
    assert (t_unif >= 0).all() and (t_unif <= 1).all(), \
        "Uniform t values must all be in [0, 1]"

    # Shape and range — logit_normal
    t_logit = sub.sample_timesteps(64, device, 'logit_normal')
    assert t_logit.shape == (64,), \
        f"Expected shape (64,), got {t_logit.shape}"
    assert (t_logit >= 0).all() and (t_logit <= 1).all(), \
        "Logit-normal t values must all be in [0, 1]"

    # Distribution: uniform should be spread; logit_normal more concentrated near 0.5
    torch.manual_seed(7)
    t_u = sub.sample_timesteps(10000, device, 'uniform')
    t_l = sub.sample_timesteps(10000, device, 'logit_normal')

    mean_dev_u = (t_u - 0.5).abs().mean().item()
    mean_dev_l = (t_l - 0.5).abs().mean().item()

    # Uniform U[0,1]: E[|t-0.5|] ≈ 0.25; check it's in a reasonable range
    assert 0.20 < mean_dev_u < 0.30, \
        f"Uniform sampling mean|t-0.5| should be ≈ 0.25, got {mean_dev_u:.4f}. " \
        "Check that you're using torch.rand, not torch.sigmoid(torch.randn(...))"

    # Logit-normal: concentrated near 0.5, mean|t-0.5| ≈ 0.17
    assert mean_dev_l < mean_dev_u - 0.03, \
        f"Logit-normal mean|t-0.5|={mean_dev_l:.4f} should be meaningfully smaller " \
        f"than uniform mean|t-0.5|={mean_dev_u:.4f}. " \
        "Check that you're using torch.sigmoid(torch.randn(...))"


def test_q4b_flow_loss(sub, model, device):
    """Q4b: flow_loss — return type, interpolation formula, CFG dropout."""
    B = 4
    # Use simple tensors (constant x_data makes t=0/t=1 checks clean)
    x_data = torch.ones(B, 2, 8, 8, device=device) * 2.0
    pitch  = torch.zeros(B, dtype=torch.long, device=device)
    t      = torch.full((B,), 0.5, device=device)

    rec = _RecordingModel().to(device)

    # --- Return type: scalar differentiable tensor ---
    rec.train()
    loss = sub.flow_loss(rec, x_data, pitch, t, p_uncond=0.0)

    assert loss.shape == (), \
        f"flow_loss must return a scalar tensor (shape ()), got shape {loss.shape}. " \
        "Do not call .item() — the caller needs to call .backward() on this tensor."
    assert loss.grad_fn is not None, \
        "flow_loss must return a differentiable tensor. " \
        "Ensure your loss is computed via F.mse_loss (not .item()) and the model is in train mode."
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 20, f"Loss {loss.item():.2f} is unexpectedly large"

    # --- Interpolation at t=0: x_t should equal x_data ---
    t_zero = torch.zeros(B, device=device)
    _ = sub.flow_loss(rec, x_data, pitch, t_zero, p_uncond=0.0)
    assert torch.allclose(rec.last_xt, x_data, atol=1e-5), \
        "At t=0, the interpolated x_t should equal x_data exactly. " \
        "Check your formula: x_t = (1-t)*x_data + t*x_noise"

    # --- Interpolation at t=1: x_t should equal noise (not x_data) ---
    t_one = torch.ones(B, device=device)
    _ = sub.flow_loss(rec, x_data, pitch, t_one, p_uncond=0.0)
    max_diff = (rec.last_xt - x_data).abs().max().item()
    assert max_diff > 0.3, \
        f"At t=1, x_t should equal the sampled noise (far from x_data=2.0). " \
        f"Got max|x_t - x_data|={max_diff:.4f}. Check interpolation direction."

    # --- CFG dropout: p_uncond=1.0 → all pitches should become NULL_PITCH ---
    from model import NULL_PITCH
    _ = sub.flow_loss(rec, x_data, pitch, t, p_uncond=1.0)
    assert (rec.last_pitch == NULL_PITCH).all(), \
        "With p_uncond=1.0, every pitch in the batch should be replaced with NULL_PITCH"


def test_q4_reproducibility(sub, submission_dir, device):
    """Q4: Load checkpoint + noises, re-run sampler, verify output matches submission."""

    npz_path = os.path.join(submission_dir, "submission_q4", "submission.npz")
    ckpt_path = os.path.join(submission_dir, "model_ft_q4.pt")

    assert os.path.exists(npz_path), f"Missing {npz_path}"
    assert os.path.exists(ckpt_path), f"Missing {ckpt_path}"

    data = np.load(npz_path, allow_pickle=True)
    submitted_samples = np.array(data["samples"], dtype=np.float32)
    noises            = np.array(data["noises"], dtype=np.float32)
    pitches           = np.array(data["pitches"], dtype=np.int64)
    guidance_scale    = float(data["guidance_scale"])
    n_steps           = int(data["n_steps"])
    sampler_name      = str(data["sampler"])

    # Load student's fine-tuned model (wrapped for diffusion convention)
    from model import load_flow_model as _load
    model_ft, _ = _load(ckpt_path, device=device)

    # Map sampler name to function
    sampler_map = {
        "euler": sub.euler_sample,
        "cfg":   sub.cfg_sample,
        "heun":  sub.heun_sample,
        "rk4":   sub.rk4_sample,
    }
    assert sampler_name in sampler_map, \
        f"Unknown sampler '{sampler_name}'. Must be one of {list(sampler_map.keys())}"

    sampler_fn = sampler_map[sampler_name]

    # Convert to device tensors
    noises_t  = torch.from_numpy(noises).to(device)
    pitches_t = torch.from_numpy(pitches).long().to(device)
    submitted = torch.from_numpy(submitted_samples).to(device)

    # Re-generate a subset of samples and compare.
    # Use loose tolerance (0.15) to account for GPU floating-point non-determinism
    # across different hardware — 50-step ODE integration accumulates small diffs.
    # This is still tight enough to catch wrong implementations (which differ by >1.0).
    REPRO_ATOL = 0.15
    indices = list(range(0, min(100, len(noises)), 10))  # 10 evenly spaced
    max_diff_seen = 0.0
    for idx in indices:
        x0 = noises_t[idx:idx+1]
        p  = pitches_t[idx:idx+1]

        with torch.no_grad():
            if sampler_name == "euler":
                out = sampler_fn(model_ft, x0.clone(), p, n_steps=n_steps)
            else:
                out = sampler_fn(model_ft, x0.clone(), p, n_steps=n_steps,
                                 guidance_scale=guidance_scale)

        expected = submitted[idx:idx+1]
        diff = (out - expected).abs().max().item()
        max_diff_seen = max(max_diff_seen, diff)
        assert diff < REPRO_ATOL, \
            f"Sample {idx} does not match submitted output (max diff={diff:.4f}, tol={REPRO_ATOL}). " \
            f"Ensure your sampler is deterministic."


def test_q5_beat_baseline(sub, submission_dir, device):
    """Q5 (bonus): Check if student's samples beat the baseline FD or pitch accuracy."""
    import librosa
    from dataset import spec_to_audio, SR, HOP_LENGTH, FREQ_BINS, TIME_FRAMES
    from scipy.linalg import sqrtm
    from sklearn.decomposition import PCA

    npz_path = os.path.join(submission_dir, "submission_q5", "submission.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No Q5 submission found at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    samples = torch.from_numpy(data["samples"])
    pitches = data["pitches"]

    # --- Pitch class accuracy ---
    correct = 0
    total = 0
    for i in range(len(samples)):
        audio = spec_to_audio(samples[i]).numpy()
        chroma = librosa.feature.chroma_stft(y=audio, sr=SR, hop_length=HOP_LENGTH)
        pred_class = int(chroma.sum(axis=1).argmax())
        true_class = int(pitches[i]) % 12
        if pred_class == true_class:
            correct += 1
        total += 1

    pitch_acc = correct / total if total > 0 else 0

    # --- FD computation (against pretrained keyboard baseline) ---
    model_base, ckpt = load_pretrained_model(device)
    from model import NULL_PITCH

    torch.manual_seed(0)
    base_pitches = torch.tensor(
        [(48 + i % 36) for i in range(100)], dtype=torch.long, device=device)
    base_noise = torch.randn(100, 2, FREQ_BINS, TIME_FRAMES, device=device)

    base_samples = []
    with torch.no_grad():
        for i in range(0, 100, 16):
            out = sub.heun_sample(model_base, base_noise[i:i+16].clone(),
                                  base_pitches[i:i+16], n_steps=25,
                                  guidance_scale=6.0)
            base_samples.append(out.cpu())
    base_samples = torch.cat(base_samples)

    def compute_fd(real, gen, pca_dim=64):
        real_flat = real.reshape(len(real), -1).numpy().astype(np.float64)
        gen_flat  = gen.reshape(len(gen), -1).numpy().astype(np.float64)
        d = min(pca_dim, len(real_flat) - 1)
        pca = PCA(n_components=d, whiten=False)
        pca.fit(real_flat)
        rp = pca.transform(real_flat)
        gp = pca.transform(gen_flat)
        mu1, s1 = rp.mean(0), np.cov(rp.T)
        mu2, s2 = gp.mean(0), np.cov(gp.T)
        s1 += np.eye(d) * 1e-6
        s2 += np.eye(d) * 1e-6
        diff = mu1 - mu2
        covmean = sqrtm(s1 @ s2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))

    fd = compute_fd(base_samples, samples)

    BASELINE_FD = 354
    BASELINE_PITCH_ACC = 0.79

    beats_fd    = fd < BASELINE_FD
    beats_pitch = pitch_acc > BASELINE_PITCH_ACC

    print(f"         Student FD={fd:.1f} (baseline={BASELINE_FD}), "
          f"pitch_acc={pitch_acc*100:.1f}% (baseline={BASELINE_PITCH_ACC*100:.0f}%)")

    assert beats_fd or beats_pitch, \
        f"Did not beat baseline on either metric. " \
        f"FD={fd:.1f} (need <{BASELINE_FD}), pitch_acc={pitch_acc*100:.1f}% (need >{BASELINE_PITCH_ACC*100:.0f}%)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autograder for Flow Matching Assignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--submission_dir", default=".",
                        help="Directory containing assignment.py and submission files")
    args = parser.parse_args()

    submission_dir = os.path.abspath(args.submission_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure we can import dataset.py and model.py from this repo
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Also add submission dir so assignment.py can import dataset/model
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)

    print(f"Device: {device}")
    print(f"Submission: {submission_dir}")
    print()

    # Load student submission
    sub = load_submission(submission_dir)

    # Load pretrained model
    model, ckpt = load_pretrained_model(device)

    results = {}

    # ── Q1: euler_sample ──────────────────────────────────────────────────────
    run_test("Q1: euler_sample",
             lambda: test_q1(sub, model, device), results)

    # ── Q2a: naive_scale_sample ───────────────────────────────────────────────
    run_test("Q2a: naive_scale_sample",
             lambda: test_q2a(sub, model, device), results)

    # ── Q2b: cfg_sample ──────────────────────────────────────────────────────
    run_test("Q2b: cfg_sample",
             lambda: test_q2b(sub, model, device), results)

    # ── Q3a: heun_sample ─────────────────────────────────────────────────────
    run_test("Q3a: heun_sample",
             lambda: test_q3a(sub, model, device), results)

    # ── Q3b: rk4_sample ──────────────────────────────────────────────────────
    run_test("Q3b: rk4_sample",
             lambda: test_q3b(sub, model, device), results)

    # ── Q4a: sample_timesteps ────────────────────────────────────────────────
    run_test("Q4a: sample_timesteps",
             lambda: test_q4a_sample_timesteps(sub, device), results)

    # ── Q4b: flow_loss ────────────────────────────────────────────────────────
    run_test("Q4b: flow_loss",
             lambda: test_q4b_flow_loss(sub, model, device), results)

    # ── Q4: reproducibility ──────────────────────────────────────────────────
    run_test("Q4: reproducibility",
             lambda: test_q4_reproducibility(sub, submission_dir, device), results)

    # ── Q5: beat baseline (bonus) ────────────────────────────────────────────
    q5_npz = os.path.join(submission_dir, "submission_q5", "submission.npz")
    if os.path.exists(q5_npz):
        run_test("Q5: beat_baseline",
                 lambda: test_q5_beat_baseline(sub, submission_dir, device), results)
    else:
        results["Q5: beat_baseline"] = "SKIP"
        print(f"  {SKIP}  Q5: beat_baseline  [bonus — no submission found]")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("  AUTOGRADER RESULTS")
    print("=" * 56)

    total_earned = 0
    total_possible = 0

    for name, status in results.items():
        pts = POINTS[name]
        if status == "PASS":
            earned = pts
            icon = "\u2713"
        elif status == "SKIP":
            earned = 0
            icon = "-"
        else:
            earned = 0
            icon = "\u2717"

        is_bonus = name.startswith("Q5")
        if not is_bonus:
            total_possible += pts

        total_earned += earned
        bonus_tag = " (bonus)" if is_bonus else ""
        print(f"  {icon} {name}: {earned}/{pts}{bonus_tag}")

    print("=" * 56)
    print(f"  Score: {total_earned}/{total_possible}"
          + (f" + bonus" if results.get("Q5: beat_baseline") == "PASS" else ""))
    print("=" * 56)

    return total_earned


if __name__ == "__main__":
    main()
