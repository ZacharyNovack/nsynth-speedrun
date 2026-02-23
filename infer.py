"""
infer.py — Interactive inference: generate audio samples at specified pitches

Usage
-----
    # Basic: 3 samples at C2, C3, C4, C5, C6
    python infer.py

    # Specific pitches, more samples
    python infer.py --pitches 40 52 64 76 88 --n_per_pitch 4

    # With Classifier-Free Guidance (requires model trained with --p_uncond > 0)
    python infer.py --pitches 60 72 --guidance_scale 3.0

    # More ODE steps for higher quality
    python infer.py --pitches 60 --n_steps 200

Output
------
    samples/
        pitch_060_C4_sample_0.wav
        pitch_060_C4_sample_1.wav
        ...

MIDI pitch reference (C4 = 60):
    36=C2  48=C3  60=C4  72=C5  84=C6

Classifier-Free Guidance (CFG)
-------------------------------
CFG amplifies the pitch conditioning signal:
    v = v_uncond + guidance_scale * (v_cond - v_uncond)

guidance_scale = 1.0  →  standard sampling (no guidance)
guidance_scale = 3–7  →  stronger pitch adherence, less diversity
guidance_scale > 10   →  over-saturated / artifacts

IMPORTANT: CFG only works meaningfully if the model was trained with
--p_uncond > 0 (so it learned the unconditional distribution).
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from dataset import FREQ_BINS, TIME_FRAMES, SR, spec_to_audio
from model import NULL_PITCH, build_model_from_config

# ── Defaults ───────────────────────────────────────────────────────────────────
CHECKPOINT      = "flow_model.pt"
OUT_DIR         = "samples/"
N_PER_PITCH     = 3
N_STEPS         = 100     # more steps → cleaner audio (try 20–200)
PITCHES         = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]  # all 12 pitch classes, C4–B4
GUIDANCE_SCALE  = 1.0     # 1.0 = no CFG; try 3–7 with a CFG-trained model
# ──────────────────────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@torch.no_grad()
def generate(
    model,
    pitch: int,
    n_samples: int,
    freq_bins: int,
    time_frames: int,
    n_steps: int,
    device: str,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Euler ODE integration from noise → data at a given MIDI pitch.

    Timesteps: t = 0/n, 1/n, ..., (n-1)/n  (matches training: t ~ U(0,1))
    Update:    x_{t+dt} = x_t + model(x_t, t, pitch) * dt

    With guidance_scale > 1 (CFG):
        v = v_uncond + scale * (v_cond - v_uncond)
    where v_uncond uses pitch index NULL_PITCH (128).

    Returns
    -------
    specs : (n_samples, 2, freq_bins, time_frames)  on CPU
    """
    model.eval()
    dt = 1.0 / n_steps
    x  = torch.randn(n_samples, 2, freq_bins, time_frames, device=device)
    p  = torch.full((n_samples,), pitch,      dtype=torch.long, device=device)
    p_null = torch.full((n_samples,), NULL_PITCH, dtype=torch.long, device=device)
    use_cfg = (guidance_scale != 1.0)

    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)
        v_cond = model(x, t, p)

        if use_cfg:
            v_uncond = model(x, t, p_null)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        x = x + v * dt

    return x.cpu()  # (n_samples, 2, F, T)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    model = build_model_from_config(cfg).to(device)
    if args.random_weights:
        print(f"*** RANDOM WEIGHTS — model NOT loaded (randomly initialized) ***")
    else:
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded model ({ckpt['n_params']:,} params, "
              f"type={cfg.get('model_type','tiny')}) from {args.checkpoint}")
    model.eval()

    # CFG sanity check
    if args.guidance_scale != 1.0:
        trained_p_uncond = cfg.get("p_uncond", 0)
        if trained_p_uncond == 0:
            print(f"\nWARNING: guidance_scale={args.guidance_scale} but this model was "
                  f"trained with p_uncond=0.\nCFG will have unpredictable effects. "
                  f"Retrain with --p_uncond 0.1 for proper CFG support.\n")
        else:
            print(f"CFG enabled: guidance_scale={args.guidance_scale} "
                  f"(model trained with p_uncond={trained_p_uncond})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate per pitch ─────────────────────────────────────────────────────
    all_paths = []
    for pitch in args.pitches:
        name = midi_to_name(pitch)
        print(f"\nGenerating {args.n_per_pitch}× pitch {pitch} ({name}) …")

        specs = generate(
            model, pitch, args.n_per_pitch,
            cfg["freq_bins"], cfg["time_frames"],
            n_steps=args.n_steps, device=device,
            guidance_scale=args.guidance_scale,
        )  # (n_per_pitch, 2, F, T)

        for i, spec in enumerate(specs):
            audio = spec_to_audio(spec)                       # (CHUNK_SAMPLES,)
            audio = audio / (audio.abs().max() + 1e-8)       # peak normalize to [-1, 1]
            audio = audio.unsqueeze(0)                        # (1, T) for torchaudio

            cfg_tag  = f"_cfg{args.guidance_scale:.0f}" if args.guidance_scale != 1.0 else ""
            rand_tag = "_random" if args.random_weights else ""
            fname = out_dir / f"pitch_{pitch:03d}_{name}{cfg_tag}{rand_tag}_sample_{i}.wav"
            torchaudio.save(str(fname), audio, SR)
            all_paths.append(fname)
            print(f"  Saved: {fname}")

    print(f"\nDone. {len(all_paths)} files written to {out_dir}/")
    print("\nTo listen in Python:")
    print("  import IPython.display as ipd")
    print(f"  ipd.Audio('{all_paths[0]}')")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate NSynth audio samples at given MIDI pitches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",type=str,     default=CHECKPOINT)
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--pitches",        type=int, nargs="+", default=PITCHES,
                   help="MIDI pitches to generate (0-127)")
    p.add_argument("--n_per_pitch",    type=int, default=N_PER_PITCH)
    p.add_argument("--n_steps",        type=int, default=N_STEPS,
                   help="Euler ODE steps — more=cleaner, slower")
    p.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE,
                   help="CFG scale: 1.0=off, 3–7=strong. Needs model trained with --p_uncond > 0")
    p.add_argument("--random_weights", action="store_true",
                   help="Skip loading model weights — use random init for baseline comparison")
    args = p.parse_args()

    run(args)
