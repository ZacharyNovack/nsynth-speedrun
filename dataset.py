"""
dataset.py — NSynth spectrogram dataset

Loads 0.5-second chunks of NSynth audio and converts them to complex spectrograms.
Each item is ((2, FREQ_BINS, TIME_FRAMES), pitch) where pitch is a MIDI integer 0-127.

Filename format: {family}_{source}_{id}-{pitch:03d}-{velocity:03d}.wav
  e.g. keyboard_electronic_098-100-075.wav  →  pitch = 100

Normalization
-------------
We use power-law magnitude compression on the raw STFT:

    X_norm = β · |X|^α · exp(j·∠X)      (default α=0.5, β=1.0)

This is sqrt-magnitude compression — a standard technique in audio generation
that compresses the wide dynamic range without needing pre-computed dataset stats.
The inverse is exact: given X_norm, recover X via |X_norm/β|^(1/α) · exp(j·∠X_norm).

With n_fft=256, hop=128, and 0.5s at 16kHz:
    FREQ_BINS   = 129   (n_fft // 2 + 1)
    TIME_FRAMES = 63    (chunk_samples // hop_length + 1)
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Audio / STFT constants ─────────────────────────────────────────────────────
SR             = 16_000
CHUNK_DURATION = 0.5
CHUNK_SAMPLES  = int(SR * CHUNK_DURATION)  # 8 000
N_FFT          = 256
HOP_LENGTH     = 128
FREQ_BINS      = N_FFT // 2 + 1            # 129
TIME_FRAMES    = CHUNK_SAMPLES // HOP_LENGTH + 1  # 63

# ── Power-law compression parameters ──────────────────────────────────────────
ALPHA_RESCALE = 0.5   # magnitude exponent (0.5 = sqrt compression)
BETA_RESCALE  = 1.0   # scale factor (1.0 = no rescaling)


# ── Core normalization helpers ─────────────────────────────────────────────────

def normalize_complex_powerlaw(
    stft_complex: torch.Tensor,
    alpha: float = ALPHA_RESCALE,
    beta: float  = BETA_RESCALE,
) -> torch.Tensor:
    """
    Power-law magnitude compression, preserving phase.

    Transforms each STFT bin:  X → β · |X|^α · exp(j·∠X)

    Parameters
    ----------
    stft_complex : complex tensor (..., FREQ_BINS, TIME_FRAMES)
    """
    mag   = stft_complex.abs().clamp(min=1e-9)
    phase = torch.angle(stft_complex)
    mag_c = beta * mag.pow(alpha)
    return torch.complex(mag_c * torch.cos(phase), mag_c * torch.sin(phase))


def denormalize_complex_powerlaw(
    stft_norm: torch.Tensor,
    alpha: float = ALPHA_RESCALE,
    beta: float  = BETA_RESCALE,
) -> torch.Tensor:
    """Invert normalize_complex_powerlaw exactly: X_norm → original STFT complex."""
    mag_c = stft_norm.abs().clamp(min=1e-9)
    phase = torch.angle(stft_norm)
    mag   = (mag_c / beta).pow(1.0 / alpha)
    return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))


# ── Public pipeline ────────────────────────────────────────────────────────────

def get_pitch(path: Path) -> int:
    """Parse the MIDI pitch (0-127) from an NSynth filename.

    e.g. keyboard_electronic_098-100-075.wav  →  100
    """
    return int(path.stem.split("-")[-2])


def wav_to_spec(path: Path) -> torch.Tensor:
    """
    Load one WAV file, grab a random 0.5-second chunk, compute its STFT,
    and apply power-law magnitude compression.

    No pre-computed dataset statistics are needed.

    Returns
    -------
    spec : (2, FREQ_BINS, TIME_FRAMES)  float32
        Channel 0 = real part, Channel 1 = imaginary part of the compressed STFT.
    """
    audio, sr = torchaudio.load(str(path))
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    audio = audio[0]  # mono → (N,)

    # Always take the first CHUNK_SAMPLES: NSynth notes decay into silence
    # quickly, so a random crop would mostly yield silence and bias training.
    chunk = audio[:CHUNK_SAMPLES]
    if len(chunk) < CHUNK_SAMPLES:
        chunk = F.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

    # STFT + power-law compression
    window = torch.hann_window(N_FFT)
    stft = torch.stft(
        chunk, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window=window, return_complex=True, center=True,
    )  # complex (FREQ_BINS, T)
    stft = normalize_complex_powerlaw(stft)

    # Split complex → two real channels
    spec = torch.stack([stft.real, stft.imag], dim=0)  # (2, F, T)

    # Ensure exact time dimension (numerical STFT edge cases)
    if spec.shape[2] > TIME_FRAMES:
        spec = spec[:, :, :TIME_FRAMES]
    elif spec.shape[2] < TIME_FRAMES:
        spec = F.pad(spec, (0, TIME_FRAMES - spec.shape[2]))

    return spec  # (2, 129, 63)


def spec_to_audio(spec: torch.Tensor) -> torch.Tensor:
    """
    Convert a power-law-compressed (2, FREQ_BINS, TIME_FRAMES) spectrogram
    back to a waveform via the inverse power law and ISTFT.

    Returns
    -------
    audio : (CHUNK_SAMPLES,)  float32
    """
    stft_norm = torch.complex(spec[0], spec[1])
    stft = denormalize_complex_powerlaw(stft_norm)
    window = torch.hann_window(N_FFT, device=spec.device)
    audio = torch.istft(
        stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window=window, center=True, length=CHUNK_SAMPLES,
    )
    return audio  # (CHUNK_SAMPLES,)


# ── Dataset ────────────────────────────────────────────────────────────────────

class NSynthSpecDataset(Dataset):
    """
    Dataset of (spectrogram, pitch) pairs from NSynth WAV files.

    Spectrograms use power-law magnitude compression (no pre-computed stats needed).

    Parameters
    ----------
    audio_dir         : path to a directory full of .wav files
    max_files         : cap the number of files (useful for fast experiments)
    instrument_filter : if set, only keep files whose stem starts with this prefix.
                        Examples:
                          "keyboard_synthetic"      → all keyboard synth samples
                          "keyboard_synthetic_099"  → one specific instrument
                          "guitar_acoustic"         → all acoustic guitars
    cache             : if True (default), cache spectrograms in RAM after first load.
                        Epoch 1 reads from disk; all subsequent epochs hit RAM.
                        At ~65 KB/spec, 5 000 files ≈ 325 MB — fits easily in Colab.
    """

    def __init__(
        self,
        audio_dir: str,
        max_files: int = None,
        instrument_filter: str = None,
        cache: bool = True,
    ):
        files = sorted(Path(audio_dir).glob("*.wav"))

        if instrument_filter:
            files = [f for f in files if f.stem.startswith(instrument_filter)]
            print(f"Instrument filter '{instrument_filter}': {len(files)} files match")

        if max_files and len(files) > max_files:
            rng = random.Random(42)
            rng.shuffle(files)
            files = files[:max_files]

        self.files = files
        self._cache: list = [None] * len(files) if cache else None
        print(f"NSynthSpecDataset: {len(self.files)} files  (RAM cache: {'on' if cache else 'off'})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]

        path  = self.files[idx]
        spec  = wav_to_spec(path)                                    # (2, F, T)
        pitch = torch.tensor(get_pitch(path), dtype=torch.long)     # scalar
        item  = (spec, pitch)

        if self._cache is not None:
            self._cache[idx] = item

        return item
