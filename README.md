# NSynth Flow Matching Speedrun

A minimal pitch-conditioned flow matching model that generates musical instrument sounds, trained on the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth). Designed as a class assignment to introduce flow matching, conditional generation, and audio synthesis in a tractable, single-GPU codebase.

Think of it as **"MNIST for audio generation"**: pitch conditioning provides the label, and success is achieving a lower Fréchet Distance than the random baseline.

---

## What is Flow Matching?

Flow matching learns a vector field that transports samples from noise to data along straight-line paths:

```
x_t = (1 - t) · x₀  +  t · x₁       (interpolation)
v_target = x₁ - x₀                   (constant velocity along the path)
loss = MSE( model(xₜ, t, pitch), v_target )
```

At inference, we integrate this vector field from `t=0` (pure noise) to `t=1` (data) using Euler steps.

---

## Data

NSynth is a dataset of ~300k 4-second monophonic instrument notes at 16 kHz. We use 0.5-second chunks converted to complex spectrograms:

```
STFT(n_fft=256, hop=128)  →  shape (2, 129, 63)
                                     │    │   └── time frames
                                     │    └────── frequency bins (n_fft/2 + 1)
                                     └─────────── channels (real, imaginary)
```

**Normalization**: power-law normalization on the complex STFT magnitude, applied per-file.
**Crop**: fixed first 0.5 s (avoids silent tails that dominate random crops).
**Filter**: `--instrument_filter keyboard` restricts training to keyboard sounds (the easiest single-family task).

Filename format: `{family}_{source}_{preset}-{pitch}-{velocity}.wav`
→ e.g. `keyboard_electronic_098-100-075.wav` → MIDI pitch **100**

---

## Quick Start (Colab)

Open `nsynth_colab.ipynb` in Google Colab with a **T4 GPU** runtime. The notebook handles everything: installs, data download, training, evaluation, and audio playback.

**Expected results on T4 (~50–60 min):**

| Metric | Value |
|---|---|
| FD @ guidance=1 | ~800–1000 |
| FD @ guidance=6 | ~280–350 |
| Pitch class accuracy | ~80–95% |
| Random baseline FD | ~15 000 |
| Random baseline pitch acc | ~8.3% |

---

## Local Setup

```bash
conda activate audio_tools   # torch 2.6+cu124, torchaudio, librosa, sklearn, scipy

# Or install from requirements.txt (torch/torchaudio must be installed separately):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Data paths are passed as arguments; defaults point to the server paths:
```
Train: /graft3/datasets/znovack/nsynth/nsynth-train/audio/
Test:  /graft3/datasets/znovack/nsynth/nsynth-test/audio/
```

---

## File Structure

| File | Purpose |
|---|---|
| `dataset.py` | `NSynthSpecDataset`, `wav_to_spec`, `spec_to_audio`, `normalize_complex_powerlaw` |
| `model.py` | `TinyFlowNet` (88k), `UNet2DFlowNet` (213k), `DiTFlowNet` (221k), `build_model_from_config` |
| `train.py` | OT-CFM training loop; supports Muon/AdamW, bf16, logit-normal timesteps, CFG dropout |
| `eval.py` | Euler sampling → Fréchet Distance (PCA-128) + pitch class accuracy |
| `infer.py` | Euler sampling → WAV files with optional Classifier-Free Guidance |
| `tests.py` | Unit tests for dataset, model, sampler, and CFG |
| `nsynth_colab.ipynb` | End-to-end Colab notebook (install → data → train → eval → listen) |
| `requirements.txt` | Python dependencies (excludes torch/torchaudio) |

---

## Usage

All scripts accept `CUDA_VISIBLE_DEVICES=N` and use the GPU automatically:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py [flags]
```

### 1. Train

**Winning recipe** (UNet + Muon + bf16, ~20 min on L40S, ~55 min on T4):

```bash
python train.py \
  --model_type unet \
  --instrument_filter keyboard \
  --t_sample logit_normal \
  --max_files 5000 \
  --optimizer muon --lr 0.02 \
  --batch_size 128 --epochs 500 \
  --bf16 \
  --save_path flow_model.pt
```

**Fast recipe** (trades ~5% pitch accuracy for 25% less time):

```bash
python train.py \
  --model_type unet \
  --instrument_filter keyboard \
  --t_sample logit_normal \
  --max_files 5000 \
  --optimizer muon --lr 0.04 \
  --batch_size 128 --epochs 300 \
  --bf16
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--model_type` | `tiny` | Architecture: `tiny` (88k), `unet` (213k), `dit` (221k) |
| `--max_files` | 5000 | Training subset size (5k ≈ fast, 50k ≈ thorough) |
| `--epochs` | 30 | Training epochs |
| `--batch_size` | 64 | Mini-batch size |
| `--optimizer` | `adamw` | `adamw` or `muon` (Muon needs `--lr 0.02`) |
| `--lr` | 3e-4 | Learning rate (AdamW: 1e-3–3e-3; Muon: 0.02) |
| `--t_sample` | `uniform` | `uniform` or `logit_normal` (SD3/Flux trick — faster convergence) |
| `--instrument_filter` | None | Train on one family only, e.g. `keyboard` |
| `--p_uncond` | 0.1 | CFG dropout probability (0 = disabled) |
| `--unet_hidden` | 32 | UNet base channel width (32=213k, 24=125k, 16=62k) |
| `--bf16` | off | bfloat16 autocast; free −15% on A100/L40S/H100; falls back on T4 |
| `--save_path` | `flow_model.pt` | Checkpoint output path |

> **Loss curve note**: flow matching loss can look flat even as sample quality improves, because it is an expectation over all noise levels. Use `eval.py` to track actual generation quality.

### 2. Evaluate

```bash
# Full eval: FD at multiple guidance scales + pitch accuracy
python eval.py --checkpoint flow_model.pt --guidance_scales 1 6

# Random (untrained) model baseline
python eval.py --checkpoint flow_model.pt --random_baseline

# Skip FD (slow) — only run pitch class accuracy
python eval.py --checkpoint flow_model.pt --skip_fd
```

**Metrics:**

- **Fréchet Distance (FD)**: 2-Wasserstein distance between PCA-projected real and generated spectrogram distributions. Lower is better. Evaluated at multiple CFG guidance scales.
- **Pitch Class Accuracy**: generates samples at all 12 pitch classes (C4–B4) and checks if the dominant pitch class in the chromagram matches. Random baseline ≈ 8.3%.

### 3. Generate Audio

```bash
# All 12 pitch classes in one octave (C4–B4), 2 samples each, CFG=6
python infer.py --checkpoint flow_model.pt \
  --pitches 60 61 62 63 64 65 66 67 68 69 70 71 \
  --n_per_pitch 2 --guidance_scale 6.0

# C major scale, more samples
python infer.py --pitches 60 62 64 65 67 69 71 72 --n_per_pitch 4 --guidance_scale 6.0

# Higher quality (more ODE steps, slower)
python infer.py --pitches 60 --n_steps 200
```

Output WAVs land in `samples/` at 16 kHz. To listen in a notebook:
```python
import IPython.display as ipd
ipd.Audio("samples/pitch_060_C4_cfg6_sample_0.wav")
```

### 4. Run Tests

```bash
python tests.py      # 16 tests covering dataset, model, sampler, CFG
python tests.py -v   # verbose
```

---

## Results

All runs use `UNet2DFlowNet` (213k params), batch=128, logit-normal timestep sampling, keyboard instrument filter, 5k training files.

### Best recipe comparison

| Recipe | FD@6 | Pitch acc | Time (L40S) | Time (T4 est) |
|---|---|---|---|---|
| Full baseline (51k files, 300ep, AdamW) | 288 | 94.4% | ~120 min | ~336 min |
| **★ Winning recipe (Exp V)** | **269** | **95.0%** | **16.8 min** | **~47 min** |
| Fast recipe (Exp Q, 300ep) | 304 | 89.4% | 14.8 min | ~41 min |
| Small model h=24 (125k, Exp S) | 322 | 81.7% | 20.0 min | ~56 min |
| Random baseline | ~15 000 | ~8.3% | — | — |

**Winning recipe**: `--optimizer muon --lr 0.02 --epochs 500 --bf16` → **beats the 51k full baseline on both metrics in 16.8 min.**

### Optimizer sweep (all ~19 500 steps, 5k keyboard)

| Exp | Setting | FD@6 | Pitch acc |
|---|---|---|---|
| G | AdamW lr=1e-3 | 310 | 81.4% |
| I | AdamW lr=3e-3 | 301 | 92.5% |
| **K** | **Muon lr=0.02** | **267** | **95.0%** |
| L | AdamW bs=64, 250ep | 362 | 73.6% |
| M | AdamW bs=256, 1000ep | 289 | 85.8% |

### Key insights

- **Muon optimizer** is the biggest single win: FD 310→267, pitch 81%→95%, same training time
- **logit-normal timestep sampling** (SD3/Flux trick): faster convergence vs. uniform — same loss at fewer steps
- **CFG (scale=6)** is essential: pushes FD from ~900 → 267
- **bf16 autocast**: −15% wall-clock time, zero quality loss on L40S/A100 (falls back to fp32 on T4)
- **More epochs > more data** at a fixed step budget
- **UNet > DiT** at this parameter scale

---

## Classifier-Free Guidance (CFG)

CFG amplifies the conditioning signal by running the model twice per step:

```
v = v_uncond + guidance_scale × (v_cond − v_uncond)
```

**Requirements**: the model must be trained with `--p_uncond > 0` (randomly masks pitch labels during training so the model learns the unconditional distribution).

**Guidance scale guide:**
- `1.0` — standard sampling (CFG off)
- `3–6` — strong pitch adherence, good quality (recommended)
- `> 10` — usually degrades quality (over-saturation)

---

## Model Architectures

Three architectures are provided, all with the same interface `model(x_t, t, pitch) → v`:

| Architecture | Params | Flag | Notes |
|---|---|---|---|
| `TinyFlowNet` | 88k | `--model_type tiny` | Flat ResBlocks; fastest to train; good for quick experiments |
| `UNet2DFlowNet` | 213k | `--model_type unet` | 2-level encoder-decoder; **best quality**; use `--unet_hidden` to resize |
| `DiTFlowNet` | 221k | `--model_type dit` | Diffusion Transformer; underperforms UNet at this scale |

UNet parameter scaling via `--unet_hidden`:

| `--unet_hidden` | Params | FD@6 | Pitch acc |
|---|---|---|---|
| 32 (default) | 213k | 267 | 95.0% |
| 24 | 125k | 322 | 81.7% |
| 16 | 62k | 413 | 55.8% |

---

## Ideas for Extending

1. **Better ODE solver** — replace Euler with Heun's method (2nd-order, same number of function evaluations)
2. **Minibatch OT pairing** — pair each noise sample with its nearest training sample to straighten trajectories
3. **Velocity conditioning** — add the velocity value from the filename as a second conditioning signal
4. **Instrument family conditioning** — add family/source embeddings alongside pitch
5. **Mel spectrogram** — try log-magnitude mel filterbank instead of raw complex STFT
6. **Larger model** — scale `--unet_hidden 48` or `--unet_hidden 64` (~500k params)
7. **Stochastic sampler** — add noise during inference (DDPM-style) and compare to deterministic Euler
8. **Compare metrics** — does lower FD always correlate with better pitch accuracy? With perceptual quality?
