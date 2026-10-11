"""Dataset for the masker experiments.

One dataset item is one simulated room version of one utterance, e.g.
``1234-5678-0001_v03``. For each item we return the dry speech, its
transcript and the paths of the three room impulse responses written by
``simulate_rooms.py``. The RIRs themselves are loaded later, in
``utils/acoustics.py``, so that a broken file only costs us one example.
"""

import os
import re

import numpy as np
import soundfile as sf
import torch


def load_audio_mono(path, sample_rate):
    """Read a wav as mono float32 at `sample_rate`, clipped to [-1, 1]."""
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        # Linear interpolation. Our data is already at 16 kHz, so this branch
        # is only a safety net and not a proper resampler.
        new_len = int(round(len(audio) * sample_rate / sr))
        if new_len <= 0:
            raise ValueError(f"Cannot resample {path} from {sr} Hz to {sample_rate} Hz.")
        x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    return torch.from_numpy(audio)


def strip_version_suffix(stem):
    """``sentence_001_v03`` -> ``sentence_001``."""
    return re.sub(r"_v\d+$", "", stem)


class SpeechDataset(torch.utils.data.Dataset):
    """
    wav_dir    : simulated mixtures, one ``<utt>_vNN.wav`` per room version.
                 Only the file names are used, to enumerate the versions.
    speech_dir : dry (anechoic) speech, ``<utt>.wav``
    txt_dir    : transcripts, ``<utt>.txt`` (may be None when no ASR is used)
    rir_dir    : RIR files, ``<utt>_vNN_rir{1,2,3}_*.npz``. Defaults to wav_dir,
                 which is where simulate_rooms.py puts them.
    """

    def __init__(self, wav_dir, txt_dir, sample_rate, speech_dir, rir_dir=None):
        if speech_dir is None:
            raise ValueError("speech_dir (dry speech) is required.")

        self.wav_dir = wav_dir
        self.txt_dir = txt_dir
        self.speech_dir = speech_dir
        self.rir_dir = rir_dir if rir_dir else wav_dir
        self.sample_rate = int(sample_rate)

        self.wav_files = sorted(
            f for f in os.listdir(self.wav_dir) if f.lower().endswith(".wav")
        )
        if not self.wav_files:
            raise RuntimeError(f"No wav files found in {self.wav_dir}")

    def __len__(self):
        return len(self.wav_files)

    def rir_paths(self, versioned_stem):
        # We do not check that the files exist here. The acoustic forward pass
        # logs missing or unreadable RIRs and skips those examples.
        return {
            "rir1": os.path.join(self.rir_dir, f"{versioned_stem}_rir1_spk_to_ngm.npz"),
            "rir2": os.path.join(self.rir_dir, f"{versioned_stem}_rir2_spk_to_listener.npz"),
            "rir3": os.path.join(self.rir_dir, f"{versioned_stem}_rir3_ngm_to_listener.npz"),
            "sample_id": versioned_stem,
        }

    def __getitem__(self, idx):
        wav_name = self.wav_files[idx]
        versioned_stem = os.path.splitext(wav_name)[0]
        base_stem = strip_version_suffix(versioned_stem)

        # Dry speech and transcript are shared by all room versions.
        dry_path = os.path.join(self.speech_dir, base_stem + ".wav")
        if not os.path.exists(dry_path):
            raise FileNotFoundError(f"Dry speech not found for {wav_name}: {dry_path}")

        dry_speech = load_audio_mono(dry_path, self.sample_rate)
        if dry_speech.numel() == 0:
            raise ValueError(f"Dry speech is empty: {dry_path}")

        text = ""
        if self.txt_dir is not None:
            txt_path = os.path.join(self.txt_dir, base_stem + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

        return dry_speech, int(dry_speech.numel()), text, self.rir_paths(versioned_stem)


def collate_fn(batch):
    """
    Returns
        dry_speech : [B, 1, T_max], zero padded
        lengths    : [B] number of valid samples in each row
        texts      : list of B transcripts
        rir_paths  : list of B dicts with keys rir1, rir2, rir3, sample_id
    """
    dry_speeches, lengths, texts, rir_paths = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    dry_speech = torch.zeros((len(batch), 1, int(lengths.max())), dtype=torch.float32)
    for i, speech in enumerate(dry_speeches):
        dry_speech[i, 0, : speech.numel()] = speech.float()

    return dry_speech, lengths, list(texts), list(rir_paths)
