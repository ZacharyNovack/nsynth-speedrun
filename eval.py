"""
eval.py — Sample from a trained flow model and compute evaluation metrics

Usage
-----
    python eval.py                              # FD + pitch accuracy on trained model
    python eval.py --random_baseline            # same metrics on untrained model
    python eval.py --skip_fd                    # only run pitch class accuracy (faster)

Metrics
-------
1. Frechet Distance (FD) in PCA-projected spectrogram space.
   Lower = generated distribution is closer to real.
   Formula: FD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·sqrt(Σ_r·Σ_g))

2. Pitch Class Accuracy.
   For each of the 12 pitch classes, generates samples, converts to audio,
   computes a chromagram, and checks whether the dominant pitch class matches
   the conditioning pitch. Note: pitch CLASSES (0-11, ignoring octave), so
   C4 and C5 are both class 0.
   Random baseline ≈ 1/12 = 8.3%.
"""

import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from tqdm import tqdm

from dataset import wav_to_spec, get_pitch, spec_to_audio, SR, HOP_LENGTH
from model import NULL_PITCH, build_model_from_config

# ── Defaults ───────────────────────────────────────────────────────────────────
TEST_DIR    = "/graft3/datasets/znovack/nsynth/nsynth-test/audio/"
CHECKPOINT  = "flow_model.pt"
N_EVAL      = 2_000   # samples for FD (bigger → better estimate)
N_STEPS     = 50      # Euler integration steps
PCA_DIM     = 128     # must be << N_EVAL / 10
N_PER_CLASS = 30      # samples per pitch class for pitch accuracy
SEED        = 0
# ──────────────────────────────────────────────────────────────────────────────

PITCH_CLASS_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


@torch.no_grad()
def euler_sample(
    model,
    pitches: torch.Tensor,
    freq_bins: int,
    time_frames: int,
    n_steps: int = N_STEPS,
    device: str = "cpu",
    batch_size: int = 128,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Euler ODE integration: noise → data, conditioned on pitches.

    x_{t+dt} = x_t  +  v · dt

    With guidance_scale > 1.0 (CFG):
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
    where v_uncond uses NULL_PITCH (128).

    Timesteps go from t=0 (pure noise) to t≈1 (data), matching the training
    distribution where t ~ Uniform(0, 1) and x_t = (1-t)*x0 + t*x1.

    Returns
    -------
    samples : (N, 2, freq_bins, time_frames)  on CPU
    """
    model.eval()
    use_cfg = (guidance_scale != 1.0)
    all_samples = []
    dt = 1.0 / n_steps
    n_samples = len(pitches)

    for start in tqdm(range(0, n_samples, batch_size), desc="Sampling", leave=False):
        end = min(start + batch_size, n_samples)
        p   = pitches[start:end].to(device)
        B   = len(p)
        x   = torch.randn(B, 2, freq_bins, time_frames, device=device)
        # Euler steps: t = 0/n, 1/n, ..., (n-1)/n
        for i in range(n_steps):
            t      = torch.full((B,), i / n_steps, device=device)
            v_cond = model(x, t, p)
            if use_cfg:
                p_null   = torch.full_like(p, NULL_PITCH)
                v_uncond = model(x, t, p_null)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            x = x + v * dt
        all_samples.append(x.cpu())

    return torch.cat(all_samples, dim=0)


# ── Frechet Distance ───────────────────────────────────────────────────────────

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    diff    = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fd(real_specs: np.ndarray, gen_specs: np.ndarray, pca_dim: int = PCA_DIM) -> float:
    n_min = min(len(real_specs), len(gen_specs))
    d = min(pca_dim, n_min - 1)

    real_flat = real_specs.reshape(len(real_specs), -1).astype(np.float64)
    gen_flat  = gen_specs.reshape(len(gen_specs),  -1).astype(np.float64)

    pca = PCA(n_components=d, whiten=False)
    pca.fit(real_flat)
    real_proj = pca.transform(real_flat)
    gen_proj  = pca.transform(gen_flat)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {d}D explains {explained*100:.1f}% of real-data variance")

    mu1, sigma1 = real_proj.mean(0), np.cov(real_proj.T)
    mu2, sigma2 = gen_proj.mean(0),  np.cov(gen_proj.T)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


# ── Pitch Class Accuracy ───────────────────────────────────────────────────────

def dominant_pitch_class(audio: np.ndarray, sr: int = SR, hop: int = HOP_LENGTH) -> int:
    """
    Compute the most-present pitch class in an audio clip via chromagram.

    Returns an integer 0-11  (0=C, 1=C#, ..., 11=B).
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop)
    # chroma: (12, T) — energy per pitch class per frame
    # Sum energy across time and pick the dominant class
    return int(chroma.sum(axis=1).argmax())


def compute_pitch_class_accuracy(
    model,
    freq_bins: int,
    time_frames: int,
    n_steps: int = N_STEPS,
    n_per_class: int = N_PER_CLASS,
    device: str = "cpu",
) -> dict:
    """
    For each of the 12 pitch classes, generate samples at a representative MIDI
    pitch (class + 60, i.e. one octave around middle C), convert to audio via
    ISTFT, and check if the dominant pitch class matches.

    Returns
    -------
    results : dict with keys 'per_class' (list of 12 accs) and 'mean_acc'
    """
    per_class_acc = []

    print("  Pitch class accuracy (12 classes, one octave around C4):")
    for pc in range(12):
        midi    = 60 + pc          # C4=60, C#4=61, ..., B4=71
        pitches = torch.full((n_per_class,), midi, dtype=torch.long)
        specs   = euler_sample(model, pitches, freq_bins, time_frames,
                               n_steps=n_steps, device=device)

        correct = 0
        for spec in specs:
            audio = spec_to_audio(spec.cpu()).numpy()
            pred  = dominant_pitch_class(audio)
            if pred == pc:
                correct += 1

        acc = correct / n_per_class
        per_class_acc.append(acc)
        marker = "✓" if acc > 1/12 else "·"
        print(f"    {marker} {PITCH_CLASS_NAMES[pc]:2s} (MIDI {midi}): {acc*100:.0f}%")

    mean_acc = np.mean(per_class_acc)
    print(f"  Mean pitch class accuracy: {mean_acc*100:.1f}%  (random baseline: 8.3%)")
    return {"per_class": per_class_acc, "mean_acc": float(mean_acc)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_test_specs_with_pitches(test_dir, n, seed=SEED):
    files = list(Path(test_dir).glob("*.wav"))
    rng = random.Random(seed)
    rng.shuffle(files)
    files = files[:n]

    specs, pitches = [], []
    for f in tqdm(files, desc="Loading test spectrograms", leave=False):
        try:
            specs.append(wav_to_spec(f))
            pitches.append(get_pitch(f))
        except Exception:
            pass

    return torch.stack(specs), torch.tensor(pitches, dtype=torch.long)


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    model = build_model_from_config(cfg).to(device)

    if args.random_baseline:
        print("*** RANDOM BASELINE — weights NOT loaded (randomly initialized) ***\n")
    else:
        model.load_state_dict(ckpt["model_state"])
        trained_with_cfg = cfg.get("p_uncond", 0) > 0
        if not trained_with_cfg:
            print("Note: model was trained without CFG dropout (p_uncond=0).\n")

    model.eval()
    label = "random baseline" if args.random_baseline else "trained model"
    print(f"Model: {ckpt['n_params']:,} params  |  {label}")
    print(f"Spec shape: (2, {cfg['freq_bins']}, {cfg['time_frames']})\n")

    # ── 1. Frechet Distance ────────────────────────────────────────────────────
    fd_results = {}   # guidance_scale → fd value
    if not args.skip_fd:
        guidance_scales = args.guidance_scales
        print(f"[1/2] Frechet Distance  ({args.n_eval} samples, {args.n_steps} steps, "
              f"guidance scales: {guidance_scales})")

        # Load real specs once; reuse for all guidance scales
        real_specs, pitches = load_test_specs_with_pitches(
            args.test_dir, args.n_eval, seed=args.seed
        )

        for gs in guidance_scales:
            label_gs = f"scale={gs:.1f}"
            if gs != 1.0:
                trained_with_cfg = cfg.get("p_uncond", 0) > 0
                if not trained_with_cfg:
                    print(f"  [{label_gs}] WARNING: model not trained with CFG "
                          f"(p_uncond=0) — results may be unreliable")
            print(f"  Sampling with guidance_scale={gs:.1f} …")
            gen_specs = euler_sample(
                model, pitches, cfg["freq_bins"], cfg["time_frames"],
                n_steps=args.n_steps, device=device, guidance_scale=gs,
            )
            fd = compute_fd(real_specs.numpy(), gen_specs.numpy(), pca_dim=args.pca_dim)
            fd_results[gs] = fd
            print(f"  [{label_gs}] FD={fd:.2f}  gen_std={gen_specs.std():.3f}  "
                  f"real_std={real_specs.std():.3f}")
        print()

    # ── 2. Pitch Class Accuracy ────────────────────────────────────────────────
    print(f"[2/2] Pitch Class Accuracy  ({args.n_per_class} samples/class, {args.n_steps} steps)")
    pitch_results = compute_pitch_class_accuracy(
        model, cfg["freq_bins"], cfg["time_frames"],
        n_steps=args.n_steps, n_per_class=args.n_per_class, device=device,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  {label}")
    if fd_results:
        print(f"  Frechet Distance (PCA-{args.pca_dim}):")
        for gs, fd in fd_results.items():
            tag = f" [guidance={gs:.1f}]"
            print(f"    {tag:>18s}  FD = {fd:.2f}")
    print(f"  Pitch class accuracy:     {pitch_results['mean_acc']*100:.1f}%")
    print(f"  (random baseline:  FD≈15352, pitch acc≈8.3%)")
    print(f"{'─'*52}")

    return fd_results, pitch_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate flow model: Frechet Distance + pitch class accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",      default=CHECKPOINT)
    p.add_argument("--test_dir",        default=TEST_DIR)
    p.add_argument("--n_eval",          type=int,   default=N_EVAL)
    p.add_argument("--n_steps",         type=int,   default=N_STEPS)
    p.add_argument("--pca_dim",         type=int,   default=PCA_DIM)
    p.add_argument("--n_per_class",     type=int,   default=N_PER_CLASS,
                   help="Samples per pitch class for pitch accuracy metric")
    p.add_argument("--seed",            type=int,   default=SEED)
    p.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0, 3.0, 6.0],
                   help="CFG guidance scales to evaluate FD at (1.0 = no CFG)")
    p.add_argument("--random_baseline", action="store_true",
                   help="Evaluate randomly initialized model as a baseline")
    p.add_argument("--skip_fd",         action="store_true",
                   help="Skip FD (slow) and only run pitch class accuracy")
    args = p.parse_args()

    evaluate(args)
