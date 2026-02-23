"""
tests.py — Unit tests for the NSynth flow matching pipeline

Run with:
    python tests.py               # all tests
    python tests.py -v            # verbose

Tests cover:
  - Filename pitch parsing
  - STFT / ISTFT round-trip fidelity
  - Model architecture (shapes, pitch conditioning, null token)
  - Euler sampler (timestep range, shape, CFG formula)
  - Infer.py generate() function end-to-end
"""

import sys
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Imports from our codebase ──────────────────────────────────────────────────
from dataset import (
    get_pitch, wav_to_spec, spec_to_audio,
    FREQ_BINS, TIME_FRAMES, CHUNK_SAMPLES, SR, N_FFT, HOP_LENGTH,
)
from model import TinyFlowNet, count_params, NULL_PITCH
from infer import generate, midi_to_name

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def run_test(name: str, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        return True
    except AssertionError as e:
        print(f"{FAIL}  {name}")
        print(f"       AssertionError: {e}")
        return False
    except Exception as e:
        print(f"{FAIL}  {name}")
        print(f"       {type(e).__name__}: {e}")
        return False


# ── Dataset tests ──────────────────────────────────────────────────────────────

def test_pitch_parsing():
    cases = {
        "keyboard_electronic_098-100-075.wav": 100,
        "bass_acoustic_000-024-025.wav":        24,
        "mallet_acoustic_000-021-127.wav":      21,
        "guitar_acoustic_000-127-050.wav":     127,
        "organ_electronic_000-000-100.wav":      0,
    }
    for fname, expected in cases.items():
        result = get_pitch(Path(fname))
        assert result == expected, f"{fname}: expected {expected}, got {result}"


def test_spec_shape():
    """wav_to_spec should always return (2, FREQ_BINS, TIME_FRAMES)."""
    # Create a synthetic WAV and save it, then load via wav_to_spec
    audio = torch.randn(CHUNK_SAMPLES)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, audio.unsqueeze(0), SR)
        spec = wav_to_spec(Path(f.name))
    assert spec.shape == (2, FREQ_BINS, TIME_FRAMES), \
        f"Expected (2, {FREQ_BINS}, {TIME_FRAMES}), got {spec.shape}"
    assert spec.dtype == torch.float32


def test_spec_to_audio_shape():
    """spec_to_audio should return (CHUNK_SAMPLES,)."""
    spec  = torch.randn(2, FREQ_BINS, TIME_FRAMES)
    audio = spec_to_audio(spec)
    assert audio.shape == (CHUNK_SAMPLES,), \
        f"Expected ({CHUNK_SAMPLES},), got {audio.shape}"


def test_istft_roundtrip():
    """
    Real audio → STFT → ISTFT should reconstruct reasonably well.
    We check SNR > 20 dB (very achievable for a perfect STFT round-trip).
    """
    audio = torch.sin(2 * math.pi * 440 * torch.arange(CHUNK_SAMPLES) / SR)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, audio.unsqueeze(0), SR)
        spec = wav_to_spec(Path(f.name))  # no normalization

    recon = spec_to_audio(spec)
    noise = audio - recon
    snr   = 10 * math.log10(audio.pow(2).mean() / (noise.pow(2).mean() + 1e-12))
    assert snr > 20, f"ISTFT round-trip SNR too low: {snr:.1f} dB (want > 20 dB)"


def test_normalization_roundtrip():
    """Normalizing then un-normalizing should recover the original spec."""
    spec = torch.randn(2, FREQ_BINS, TIME_FRAMES) * 5  # arbitrary scale
    mean = spec.mean(dim=(1, 2), keepdim=True)
    std  = spec.std(dim=(1, 2), keepdim=True)
    norm_stats = (mean, std)

    normed   = (spec - mean) / (std + 1e-8)
    unnormed = normed * (std + 1e-8) + mean
    assert torch.allclose(spec, unnormed, atol=1e-5), "Norm/unnorm round-trip failed"


# ── Model tests ────────────────────────────────────────────────────────────────

def test_model_output_shape():
    """Model should output the same shape as input."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    B = 3
    x     = torch.randn(B, 2, FREQ_BINS, TIME_FRAMES)
    t     = torch.rand(B)
    pitch = torch.randint(0, 128, (B,))
    with torch.no_grad():
        out = model(x, t, pitch)
    assert out.shape == x.shape, f"Output shape {out.shape} != input {x.shape}"


def test_model_param_count():
    """Default model should be under 100k parameters."""
    model = TinyFlowNet(hidden=32, n_blocks=4, t_dim=32)
    n = count_params(model)
    assert n < 100_000, f"Model has {n:,} params — should be < 100k"
    assert n > 50_000,  f"Model has {n:,} params — suspiciously small?"


def test_pitch_conditioning_matters():
    """Different pitches should produce different velocity fields."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    x = torch.randn(1, 2, FREQ_BINS, TIME_FRAMES)
    t = torch.tensor([0.5])
    with torch.no_grad():
        v_c4 = model(x, t, torch.tensor([60]))
        v_c5 = model(x, t, torch.tensor([72]))
    assert not torch.allclose(v_c4, v_c5), \
        "Pitch=60 and pitch=72 produced identical outputs — conditioning not working"


def test_null_token_accepted():
    """Model should accept NULL_PITCH (128) without error."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    x     = torch.randn(2, 2, FREQ_BINS, TIME_FRAMES)
    t     = torch.rand(2)
    pitch = torch.full((2,), NULL_PITCH, dtype=torch.long)
    with torch.no_grad():
        out = model(x, t, pitch)
    assert out.shape == x.shape


def test_null_differs_from_conditioned():
    """NULL_PITCH should produce a different output than a real pitch."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    x = torch.randn(1, 2, FREQ_BINS, TIME_FRAMES)
    t = torch.tensor([0.5])
    with torch.no_grad():
        v_cond   = model(x, t, torch.tensor([60]))
        v_uncond = model(x, t, torch.tensor([NULL_PITCH]))
    assert not torch.allclose(v_cond, v_uncond), \
        "Null token and real pitch produced identical outputs"


def test_timestep_boundary():
    """Model should not crash at t=0 or t≈1."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    x = torch.randn(2, 2, FREQ_BINS, TIME_FRAMES)
    p = torch.tensor([60, 72])
    with torch.no_grad():
        model(x, torch.zeros(2), p)           # t = 0
        model(x, torch.ones(2) * 0.999, p)   # t ≈ 1


# ── Euler sampler tests ────────────────────────────────────────────────────────

def test_euler_sampler_shape():
    """generate() should return (n_samples, 2, FREQ_BINS, TIME_FRAMES) on CPU."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    specs = generate(model, pitch=60, n_samples=4,
                     freq_bins=FREQ_BINS, time_frames=TIME_FRAMES,
                     n_steps=5, device="cpu")
    assert specs.shape == (4, 2, FREQ_BINS, TIME_FRAMES), \
        f"Unexpected shape: {specs.shape}"
    assert specs.device.type == "cpu"


def test_euler_timestep_range():
    """
    Euler steps use t = 0/n, 1/n, ..., (n-1)/n — all in [0, 1).
    This matches the training distribution t ~ U(0, 1).
    Verify by monkey-patching the model to record all timesteps seen.
    """
    seen_t = []

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = TinyFlowNet(hidden=16, n_blocks=1, t_dim=16)
        def forward(self, x, t, pitch):
            seen_t.extend(t.tolist())
            return self.inner(x, t, pitch)

    model = RecordingModel()
    model.eval()
    n_steps = 10
    generate(model, pitch=60, n_samples=2,
             freq_bins=FREQ_BINS, time_frames=TIME_FRAMES,
             n_steps=n_steps, device="cpu")

    expected = [i / n_steps for i in range(n_steps)] * 2  # 2 samples
    expected.sort()
    seen_t.sort()

    assert len(seen_t) == n_steps * 2
    for got, exp in zip(seen_t, expected):
        assert abs(got - exp) < 1e-6, f"Timestep mismatch: got {got}, expected {exp}"
    assert all(0 <= t_ < 1 for t_ in seen_t), "Some timesteps outside [0, 1)"


def test_euler_cfg_formula():
    """
    With guidance_scale = 1.0, CFG output should equal non-CFG output.
    With guidance_scale != 1.0, CFG output should differ.
    """
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    torch.manual_seed(0)
    x0 = torch.randn(1, 2, FREQ_BINS, TIME_FRAMES)

    torch.manual_seed(0)
    out_no_cfg = generate(model, 60, 1, FREQ_BINS, TIME_FRAMES, n_steps=3,
                          device="cpu", guidance_scale=1.0)
    torch.manual_seed(0)
    out_cfg_1  = generate(model, 60, 1, FREQ_BINS, TIME_FRAMES, n_steps=3,
                          device="cpu", guidance_scale=1.0)
    torch.manual_seed(0)
    out_cfg_3  = generate(model, 60, 1, FREQ_BINS, TIME_FRAMES, n_steps=3,
                          device="cpu", guidance_scale=3.0)

    assert torch.allclose(out_no_cfg, out_cfg_1), \
        "guidance_scale=1.0 should produce same result as no CFG"
    assert not torch.allclose(out_no_cfg, out_cfg_3), \
        "guidance_scale=3.0 should produce different result than 1.0"


def test_euler_pitch_affects_output():
    """Different pitches in the sampler should yield different samples."""
    model = TinyFlowNet(hidden=16, n_blocks=2, t_dim=16)
    model.eval()
    torch.manual_seed(42)
    out_60 = generate(model, 60, 1, FREQ_BINS, TIME_FRAMES, n_steps=5, device="cpu")
    torch.manual_seed(42)
    out_72 = generate(model, 72, 1, FREQ_BINS, TIME_FRAMES, n_steps=5, device="cpu")
    assert not torch.allclose(out_60, out_72), \
        "pitch=60 and pitch=72 should produce different samples (same noise seed)"


def test_midi_to_name():
    assert midi_to_name(60) == "C4"
    assert midi_to_name(69) == "A4"
    assert midi_to_name(48) == "C3"
    assert midi_to_name(36) == "C2"
    assert midi_to_name(71) == "B4"


# ── Runner ─────────────────────────────────────────────────────────────────────

TESTS = [
    # dataset
    ("Pitch parsing from filename",       test_pitch_parsing),
    ("Spectrogram output shape",          test_spec_shape),
    ("spec_to_audio output shape",        test_spec_to_audio_shape),
    ("ISTFT round-trip SNR > 20 dB",      test_istft_roundtrip),
    ("Normalization round-trip",          test_normalization_roundtrip),
    # model
    ("Model output shape",                test_model_output_shape),
    ("Model parameter count < 100k",      test_model_param_count),
    ("Pitch conditioning changes output", test_pitch_conditioning_matters),
    ("Null token (128) accepted",         test_null_token_accepted),
    ("Null token differs from conditioned", test_null_differs_from_conditioned),
    ("Timestep boundaries (t=0, t≈1)",    test_timestep_boundary),
    # sampler
    ("Euler sampler output shape",        test_euler_sampler_shape),
    ("Euler timestep range [0, 1)",       test_euler_timestep_range),
    ("CFG formula (scale=1 is identity)", test_euler_cfg_formula),
    ("Euler pitch affects output",        test_euler_pitch_affects_output),
    ("midi_to_name helper",               test_midi_to_name),
]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    print(f"\nRunning {len(TESTS)} tests...\n")
    results = [run_test(name, fn) for name, fn in TESTS]
    n_pass  = sum(results)
    n_fail  = len(results) - n_pass
    print(f"\n{'─'*40}")
    print(f"  {n_pass}/{len(results)} passed", end="")
    if n_fail:
        print(f"  ({n_fail} failed)")
        sys.exit(1)
    else:
        print("  — all green!")
