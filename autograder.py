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
import torch.nn.functional as F

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(os.path.dirname(__file__), "pretrained_keyboard.pt")

POINTS = {
    "Q1: euler_sample":        20,
    "Q2a: naive_scale_sample": 10,
    "Q2b: cfg_sample":         20,
    "Q3a: heun_sample":        20,
    "Q3b: rk4_sample":         10,
    "Q4: train_step":          10,
    "Q4: reproducibility":     10,
    "Q5: beat_baseline":       10,
}

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_submission(submission_dir: str):
    """Import assignment.py from the submission directory as a module."""
    assignment_path = os.path.join(submission_dir, "assignment.py")
    if not os.path.exists(assignment_path):
        print(f"ERROR: {assignment_path} not found")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("assignment", assignment_path)
    module = importlib.util.module_from_spec(spec)
    # Suppress print output from notebook conversion artifacts
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
    """Q1: euler_sample — shape, not identity, finite, reasonable std."""
    from dataset import FREQ_BINS, TIME_FRAMES

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


def test_q2a(sub, model, device):
    """Q2a: naive_scale_sample — scale=1 matches Euler, scale=2 differs."""
    from dataset import FREQ_BINS, TIME_FRAMES

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


def test_q2b(sub, model, device):
    """Q2b: cfg_sample — gs=1 matches Euler, gs=6 differs, differs from naive."""
    from dataset import FREQ_BINS, TIME_FRAMES

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


def test_q3a(sub, model, device):
    """Q3a: heun_sample — shape, finite, differs from Euler, deterministic, CFG."""
    from dataset import FREQ_BINS, TIME_FRAMES

    torch.manual_seed(0)
    x0 = torch.randn(4, 2, FREQ_BINS, TIME_FRAMES, device=device)
    p  = torch.tensor([60, 62, 64, 67], dtype=torch.long, device=device)

    out_euler    = sub.euler_sample(model, x0.clone(), p, n_steps=25)
    out_heun     = sub.heun_sample(model, x0.clone(), p, n_steps=25, guidance_scale=1.0)
    out_heun_gs1 = sub.heun_sample(model, x0.clone(), p, n_steps=25, guidance_scale=1.0)
    out_heun_gs6 = sub.heun_sample(model, x0.clone(), p, n_steps=25, guidance_scale=6.0)

    assert out_heun.shape == (4, 2, FREQ_BINS, TIME_FRAMES), \
        f"Wrong shape: {out_heun.shape}"
    assert out_heun.isfinite().all(), "Heun output contains NaN/Inf"
    assert not torch.allclose(out_heun, out_euler, atol=1e-3), \
        "heun_sample must differ from euler_sample"
    assert torch.allclose(out_heun, out_heun_gs1, atol=1e-5), \
        "heun_sample should be deterministic"
    assert not torch.allclose(out_heun_gs6, out_heun, atol=1e-3), \
        "heun_sample with guidance_scale=6 should differ from guidance_scale=1"


def test_q3b(sub, model, device):
    """Q3b: rk4_sample — shape, finite, differs from Heun."""
    from dataset import FREQ_BINS, TIME_FRAMES

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


def test_q4_train_step(sub, model, device):
    """Q4: train_step — returns scalar loss, parameters change."""
    from dataset import FREQ_BINS, TIME_FRAMES

    torch.manual_seed(99)
    x_data = torch.randn(8, 2, FREQ_BINS, TIME_FRAMES, device=device) * 0.5
    p      = torch.randint(0, 128, (8,), dtype=torch.long, device=device)

    model_ag = copy.deepcopy(model)
    opt_ag   = torch.optim.AdamW(model_ag.parameters(), lr=1e-4)
    params_before = [p_.data.clone() for p_ in model_ag.parameters()]

    model_ag.train()
    loss_val = sub.train_step(model_ag, opt_ag, x_data, p, p_uncond=0.1, t_sample='uniform')
    model_ag.eval()

    # Return type
    is_scalar = (torch.is_tensor(loss_val) and loss_val.ndim == 0) or isinstance(loss_val, float)
    assert is_scalar, f"train_step must return a scalar, got {type(loss_val)}"
    loss_float = float(loss_val.item() if torch.is_tensor(loss_val) else loss_val)

    # Range
    assert 0 < loss_float < 10, f"Loss {loss_float:.4f} outside expected range (0, 10)"

    # Parameters changed
    changed = any(not torch.allclose(pb, pa)
                  for pb, pa in zip(params_before, model_ag.parameters()))
    assert changed, "Model parameters did not change — did you call optimizer.step()?"


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

    # Re-generate a subset of samples and compare
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
    # Generate baseline samples for comparison
    model_base, ckpt = load_pretrained_model(device)
    from model import NULL_PITCH

    torch.manual_seed(0)
    base_pitches = torch.tensor(
        [(48 + i % 36) for i in range(100)], dtype=torch.long, device=device)
    base_noise = torch.randn(100, 2, FREQ_BINS, TIME_FRAMES, device=device)

    # Use student's heun_sample with baseline model for reference
    base_samples = []
    with torch.no_grad():
        for i in range(0, 100, 16):
            out = sub.heun_sample(model_base, base_noise[i:i+16].clone(),
                                  base_pitches[i:i+16], n_steps=25,
                                  guidance_scale=6.0)
            base_samples.append(out.cpu())
    base_samples = torch.cat(base_samples)

    # Compute FD in PCA space
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

    # Baseline thresholds
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

    # ── Q4: train_step ────────────────────────────────────────────────────────
    run_test("Q4: train_step",
             lambda: test_q4_train_step(sub, model, device), results)

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

        # Q5 is bonus, don't count toward total possible
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
