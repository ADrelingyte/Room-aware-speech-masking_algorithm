"""Simulate the rooms and write the training data.

    python simulate_rooms.py --config configs/simulate_example.yaml

For every dry speech file, and for each of `n_versions_per_input` random
rooms, this writes into `output_dir`:

    <utt>_vNN.wav                        4 channels: left ear, right ear,
                                         reference mic 1 m from the talker,
                                         reference mic 1 m from the listener
    <utt>_vNN_rir1_spk_to_ngm.npz        talker -> NGM microphone
    <utt>_vNN_rir2_spk_to_listener.npz   talker -> listener ears
    <utt>_vNN_rir3_ngm_to_listener.npz   NGM loudspeaker -> listener ears
    <utt>_vNN.yaml                       geometry and levels of that room

plus, if enabled, SpeechOnly/, PreNGM/ and OnlyNoise/ with the intermediate
renders. Training only needs the file names of the wavs and the three RIR
files (see utils/dataset.py).

What is randomised for each version: room size and RT60, listener position,
talker position (within a distance range from the listener) and orientation,
babble source positions, and the NGM position. The NGM is either a single
device on the listener's table (mode "table_listener") or a ceiling grid
(mode "grid").

Levels:
  - Babble is scaled so that speech / babble, both measured at the ears
    after room rendering, equals the babble SNR.
  - The NGM noise in the mixture (speech-shaped or white) is scaled against
    the speech + babble mix at the ears. This noise is only part of the
    rendered mixture; the learned masker replaces it during training.

Files that already exist are skipped, so an interrupted job can simply be
started again. Files that fail are listed in simulation_failures.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm

import numpy as np
import librosa
import soundfile as sf
import yaml
import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectionVector, Cardioid


# =========================
# Channel layout
# =========================
CH_L = 0
CH_R = 1
CH_SPK_1M = 2
CH_LIS_1M = 3


# =========================
# Utilities
# =========================
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def cartesian_to_spherical(direction_xyz: np.ndarray) -> Tuple[float, float]:
    x, y, z = direction_xyz.tolist()
    azimuth = np.arctan2(y, x)
    r_xy = np.linalg.norm([x, y])
    colatitude = np.arctan2(r_xy, z)
    return float(azimuth), float(colatitude)


def clamp_inside_room(pos_xyz: np.ndarray, room_dim: Sequence[float], margin: float = 0.3) -> np.ndarray:
    pos = np.asarray(pos_xyz, dtype=float)
    rd = np.asarray(room_dim, dtype=float)
    return np.minimum(np.maximum(pos, [margin, margin, margin]), rd - margin)


def shaped_noise_like(x: np.ndarray) -> np.ndarray:
    """SSN-like: randomize phase with same magnitude spectrum."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N <= 2:
        n = np.random.randn(N).astype(float)
        return n / (np.sqrt(np.mean(n * n)) + 1e-12)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag)))
    n = np.fft.irfft(mag * phase, n=N)
    return (n / (np.sqrt(np.mean(n * n)) + 1e-12)).astype(float)


def white_noise(N: int) -> np.ndarray:
    n = np.random.randn(N).astype(float)
    return n / (np.sqrt(np.mean(n * n)) + 1e-12)


def get_babble_segment(babble_audio: np.ndarray, needed_len: int) -> np.ndarray:
    """Loop + random chunks to length."""
    babble_audio = np.asarray(babble_audio, dtype=float)
    if len(babble_audio) == 0:
        return np.zeros(needed_len, dtype=float)
    seg, total = [], 0
    while total < needed_len:
        start = random.randint(0, max(len(babble_audio) - 1, 0))
        slen = min(len(babble_audio) - start, needed_len - total)
        if slen <= 0:
            break
        seg.append(babble_audio[start : start + slen])
        total += slen
    if not seg:
        return np.zeros(needed_len, dtype=float)
    out = np.concatenate(seg)[:needed_len]
    if len(out) < needed_len:
        out = np.pad(out, (0, needed_len - len(out)))
    return out


def ears_rms(sig4: np.ndarray) -> float:
    rL = rms(sig4[:, CH_L])
    rR = rms(sig4[:, CH_R])
    return float(np.sqrt((rL * rL + rR * rR) / 2.0))


def write_room_3d_html(meta: Dict, html_path: Union[str, Path]) -> None:
    # plotly is only needed for this optional figure
    import plotly.graph_objects as go

    room_dim = np.asarray(meta["room_dim"], float)
    W, L, H = room_dim.tolist()

    mic_positions = np.asarray(meta.get("mic_positions", []), float)
    speaker = np.asarray(meta.get("speaker_position", []), float)
    babble_positions = np.asarray(meta.get("babble_positions", []), float)
    ngm_positions = np.asarray(meta.get("ngm_positions", []), float)

    fig = go.Figure()

    edges = [
        ([0, W], [0, 0], [0, 0]), ([0, W], [L, L], [0, 0]),
        ([0, 0], [0, L], [0, 0]), ([W, W], [0, L], [0, 0]),
        ([0, W], [0, 0], [H, H]), ([0, W], [L, L], [H, H]),
        ([0, 0], [0, L], [H, H]), ([W, W], [0, L], [H, H]),
        ([0, 0], [0, 0], [0, H]), ([W, W], [0, 0], [0, H]),
        ([0, 0], [L, L], [0, H]), ([W, W], [L, L], [0, H]),
    ]
    for x, y, z in edges:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="gray"), showlegend=False))

    if mic_positions.size:
        fig.add_trace(go.Scatter3d(
            x=mic_positions[:, 0], y=mic_positions[:, 1], z=mic_positions[:, 2],
            mode="markers+text", marker=dict(size=5, color="blue"),
            text=["Mic"] * len(mic_positions), textposition="top center", name="Microphones",
        ))

    fig.add_trace(go.Scatter3d(
        x=[speaker[0]], y=[speaker[1]], z=[speaker[2]],
        mode="markers+text", marker=dict(size=6, color="green"),
        text=["Speaker"], textposition="top center", name="Speaker",
    ))

    if babble_positions.size:
        fig.add_trace(go.Scatter3d(
            x=babble_positions[:, 0], y=babble_positions[:, 1], z=babble_positions[:, 2],
            mode="markers", marker=dict(size=4, color="red"), name="Babble",
        ))

    if ngm_positions.size:
        fig.add_trace(go.Scatter3d(
            x=ngm_positions[:, 0], y=ngm_positions[:, 1], z=ngm_positions[:, 2],
            mode="markers", marker=dict(size=5, color="orange", symbol="diamond"), name="NGM",
        ))

    fig.update_layout(
        scene=dict(xaxis_title="W (m)", yaxis_title="L (m)", zaxis_title="H (m)", aspectmode="data"),
        title=f"3D Room Layout — {Path(html_path).stem}",
        margin=dict(l=0, r=0, b=0, t=30),
    )
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))


# =========================
# Configs
# =========================
@dataclass
class RoomConfig:
    use_random_dims: bool = False
    dim: Optional[Tuple[float, float, float]] = None
    W_range: Tuple[float, float] = (6.0, 12.0)
    L_range: Tuple[float, float] = (6.0, 12.0)
    H_range: Tuple[float, float] = (2.4, 4.0)
    use_random_rt60: bool = True
    rt60_value: float = 0.4
    rt60_range: Tuple[float, float] = (0.2, 0.8)
    absorption_range: Tuple[float, float] = (0.2, 0.6)
    max_order: int = 3
    max_tries: int = 500


@dataclass
class MicConfig:
    mic_positions: Optional[np.ndarray] = None
    mic_center_height_range: Tuple[float, float] = (1.0, 1.4)
    binaural_offset_m: float = 0.10


@dataclass
class SpeakerConfig:
    distance_range_m: Tuple[float, float] = (0.66, 1.57)
    height_range: Tuple[float, float] = (1.4, 1.7)
    orientation_mode: str = "random"


@dataclass
class BabbleConfig:
    enabled: bool = True
    babble_paths: Optional[List[str]] = None
    n_babble: int = 8
    snr_db: Optional[float] = 0.0
    snr_db_range: Tuple[float, float] = (-6.0, 6.0)
    min_dist_m: float = 0.0
    min_dist_factor_vs_spk_mic: float = 2.0
    height_range: Tuple[float, float] = (1.2, 1.7)


@dataclass
class NGMConfig:
    enabled: bool = False
    mode: str = "grid"
    noise_type: str = "SSN"
    snr_db: Optional[float] = 0.0
    snr_db_range: Tuple[float, float] = (-6.0, 6.0)
    spacing_factor: float = 1.5
    min_wall_dist: float = 0.6
    jitter_max: float = 0.6
    table_dist_to_listener_range: Tuple[float, float] = (0.25, 0.60)
    table_height_range: Tuple[float, float] = (0.70, 0.95)
    table_margin: float = 0.3


@dataclass
class RIRConfig:
    # RIR1: Speaker -> NGM       (NN input path)
    rir1_spk_to_ngm: bool = True
    # RIR2: Speaker -> Listener  (direct acoustic path, bypasses NN)
    rir2_spk_to_listener: bool = True
    # RIR3: NGM -> Listener      (NN output -> listener ears)
    rir3_ngm_to_listener: bool = True
    write_bundle_all: bool = False
    rir3_ngm_to_listener_combined: bool = False


@dataclass
class OutputConfig:
    write_yaml: bool = True
    write_rirs: bool = True
    write_onlynoise: bool = True
    recursive_input: bool = False
    write_pre_ngm_mix: bool = True
    write_speech_only: bool = True   # save speech render before babble+NGM
    write_3d_viz: bool = False


# =========================
# Room sampling
# =========================
def sample_room(cfg: RoomConfig) -> Tuple[Tuple[float, float, float], float, float]:
    if not cfg.use_random_dims:
        if cfg.dim is None:
            raise ValueError("RoomConfig.dim must be provided when use_random_dims=False.")
        W, L, H = map(float, cfg.dim)
    else:
        W = random.uniform(*cfg.W_range)
        L = random.uniform(*cfg.L_range)
        H = random.uniform(*cfg.H_range)

    rt60 = float(random.uniform(*cfg.rt60_range) if cfg.use_random_rt60 else cfg.rt60_value)
    V = W * L * H
    S = 2.0 * (W * L + W * H + L * H)
    alpha = 0.161 * V / (S * rt60 + 1e-12)
    alpha = float(np.clip(alpha, cfg.absorption_range[0], cfg.absorption_range[1]))
    return (W, L, H), rt60, alpha


# =========================
# Mic placement
# =========================
def sample_mics(room_dim: Sequence[float], mic_cfg: MicConfig) -> Tuple[np.ndarray, np.ndarray]:
    if mic_cfg.mic_positions is not None:
        mic_positions = np.asarray(mic_cfg.mic_positions, float)
        if mic_positions.ndim != 2 or mic_positions.shape[1] != 3:
            raise ValueError(f"mic_positions must have shape (n_mics,3), got {mic_positions.shape}")
        mic_center = np.mean(mic_positions, axis=0)
        return mic_positions, mic_center

    W, L, _ = room_dim
    mic_center = np.array([
        random.uniform(W * 0.3, W * 0.7),
        random.uniform(L * 0.3, L * 0.7),
        random.uniform(*mic_cfg.mic_center_height_range),
    ], dtype=float)

    off = float(mic_cfg.binaural_offset_m)
    mic_positions = np.array([
        [mic_center[0] - off, mic_center[1], mic_center[2]],
        [mic_center[0] + off, mic_center[1], mic_center[2]],
    ], dtype=float)
    return mic_positions, mic_center


# =========================
# Speaker placement + directivity
# =========================
def sample_speaker(
    room_dim: Sequence[float],
    mic_center: np.ndarray,
    spk_cfg: SpeakerConfig,
) -> Tuple[np.ndarray, float, str, Cardioid, np.ndarray]:
    W, L, _ = room_dim
    speaker_pos = None
    speaker_mic_distance = None

    for _ in range(4000):
        x = random.uniform(1.0, W - 1.0)
        y = random.uniform(1.0, L - 1.0)
        d = float(np.linalg.norm([x - mic_center[0], y - mic_center[1]]))
        if spk_cfg.distance_range_m[0] <= d <= spk_cfg.distance_range_m[1]:
            speaker_pos = np.array([x, y, random.uniform(*spk_cfg.height_range)], dtype=float)
            speaker_mic_distance = d
            break

    if speaker_pos is None:
        speaker_pos = np.array([W * 0.5, L * 0.5, random.uniform(*spk_cfg.height_range)], dtype=float)
        speaker_mic_distance = float(np.linalg.norm(speaker_pos[:2] - mic_center[:2]))

    delta = mic_center - speaker_pos
    if spk_cfg.orientation_mode == "random":
        orientation_type = random.choice(["face_each_other", "face_away", "side_facing"])
    else:
        orientation_type = spk_cfg.orientation_mode

    if orientation_type == "face_each_other":
        speaker_dir = delta
    elif orientation_type == "face_away":
        speaker_dir = -delta
    else:
        speaker_dir = np.array([-delta[1], delta[0], 0.0], dtype=float)

    speaker_dir_unit = speaker_dir / (np.linalg.norm(speaker_dir) + 1e-12)
    az, col = cartesian_to_spherical(speaker_dir_unit)
    speaker_directivity = Cardioid(orientation=DirectionVector(az, col), gain=1.0)
    return speaker_pos, speaker_mic_distance, orientation_type, speaker_directivity, speaker_dir_unit


# =========================
# Babble placement
# =========================
def sample_babble_positions(
    room_dim: Sequence[float],
    mic_center: np.ndarray,
    speaker_mic_distance: float,
    bab_cfg: BabbleConfig,
) -> List[List[float]]:
    W, L, _ = room_dim
    out: List[List[float]] = []
    min_d = max(float(bab_cfg.min_dist_m), float(bab_cfg.min_dist_factor_vs_spk_mic) * float(speaker_mic_distance))
    for _ in range(int(bab_cfg.n_babble)):
        for _try in range(8000):
            bx = random.uniform(1.0, W - 1.0)
            by = random.uniform(1.0, L - 1.0)
            d = float(np.linalg.norm([bx - mic_center[0], by - mic_center[1]]))
            if d >= min_d:
                bz = random.uniform(*bab_cfg.height_range)
                out.append([bx, by, bz])
                break
        else:
            out.append([W - 1.0, L - 1.0, random.uniform(*bab_cfg.height_range)])
    return out


# =========================
# NGM placement
# =========================
def generate_ngm_grid_fixed(
    room_dim, ceiling_height, spacing_factor=1.5, min_wall_dist=0.6, avoid_xyz=None, jitter_max=0.6
):
    Lx, Ly, _ = room_dim
    spacing = max(0.5, float(ceiling_height) * float(spacing_factor))

    def axis_coords_fixed(L):
        usable = L - 2 * min_wall_dist
        n = int(np.floor(usable / spacing)) + 1
        while n > 0:
            total_span = (n - 1) * spacing
            margin = (L - total_span) / 2.0
            if min_wall_dist <= margin <= spacing / 2.0:
                return margin + np.arange(n) * spacing
            n -= 1
        return np.array([L / 2.0], dtype=float)

    xs = axis_coords_fixed(Lx)
    ys = axis_coords_fixed(Ly)
    z = float(ceiling_height) - 0.1
    pts = [[x, y, z] for x in xs for y in ys]

    if avoid_xyz and len(avoid_xyz) > 0:
        valid_min_x, valid_max_x = min_wall_dist, Lx - min_wall_dist
        valid_min_y, valid_max_y = min_wall_dist, Ly - min_wall_dist
        nudged = []
        for (x, y, _) in pts:
            xx, yy = x, y
            for ax, ay, _ in avoid_xyz:
                if np.hypot(xx - ax, yy - ay) < min_wall_dist:
                    dx = np.clip(xx - ax, -jitter_max, jitter_max)
                    dy = np.clip(yy - ay, -jitter_max, jitter_max)
                    if dx == 0:
                        dx = np.sign(valid_max_x / 2 - xx) * min(0.1, jitter_max)
                    if dy == 0:
                        dy = np.sign(valid_max_y / 2 - yy) * min(0.1, jitter_max)
                    xx = float(np.clip(xx + dx, valid_min_x, valid_max_x))
                    yy = float(np.clip(yy + dy, valid_min_y, valid_max_y))
            nudged.append([xx, yy, z])
        pts = nudged
    return np.array(pts, dtype=float)


def place_single_ngm_on_listener_table(
    listener_xyz: np.ndarray,
    room_dim: Sequence[float],
    dist_range: Tuple[float, float],
    height_range: Tuple[float, float],
    margin: float = 0.3,
) -> np.ndarray:
    d = random.uniform(*dist_range)
    theta = random.uniform(0, 2 * np.pi)
    dx, dy = d * np.cos(theta), d * np.sin(theta)
    z = random.uniform(*height_range)
    pos = np.array([listener_xyz[0] + dx, listener_xyz[1] + dy, z], dtype=float)
    return clamp_inside_room(pos, room_dim, margin=margin)


# =========================
# RIR computation
# =========================
def wall_material(absorption: float) -> "pra.Material":
    """Wall material for a given `absorption` value.

    The data of the paper was generated with the old ``absorption=`` argument
    of ``pra.ShoeBox``, which pyroomacoustics treated as an amplitude
    coefficient and converted to an energy absorption of 1 - (1 - a)^2. That
    argument has since been removed, so the same conversion is done here to
    keep the simulated rooms identical across library versions.
    """
    return pra.Material(1.0 - (1.0 - float(absorption)) ** 2)


def _pack_rir_numeric(room_rir: List[List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    n_mics = len(room_rir)
    n_src = len(room_rir[0]) if n_mics > 0 else 0
    lengths = np.zeros((n_mics, n_src), dtype=np.int32)
    Lmax = 0
    for mi in range(n_mics):
        for si in range(n_src):
            h = np.asarray(room_rir[mi][si], dtype=np.float32).ravel()
            lengths[mi, si] = int(h.shape[0])
            if h.shape[0] > Lmax:
                Lmax = int(h.shape[0])
    h_padded = np.zeros((n_mics, n_src, Lmax), dtype=np.float32)
    for mi in range(n_mics):
        for si in range(n_src):
            h = np.asarray(room_rir[mi][si], dtype=np.float32).ravel()
            L = h.shape[0]
            if L > 0:
                h_padded[mi, si, :L] = h
    return h_padded, lengths


def compute_rir_sources_to_mics_numeric(
    room_dim: Sequence[float],
    fs: int,
    absorption: float,
    max_order: int,
    source_positions: Sequence[Sequence[float]],
    mic_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    room = pra.ShoeBox(room_dim, fs=fs, materials=wall_material(absorption), max_order=max_order)
    mic_positions = np.asarray(mic_positions, float)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions.T, fs))
    for pos in source_positions:
        room.add_source(np.asarray(pos, float).tolist(), signal=None)
    room.compute_rir()
    return _pack_rir_numeric(room.rir)


def _sum_ngm_rirs_per_mic(h: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_mics, n_src, Lmax = h.shape
    maxlen_per_mic = np.max(lengths, axis=1) if n_src > 0 else np.zeros((n_mics,), dtype=np.int32)
    Lsum = int(np.max(maxlen_per_mic)) if n_mics > 0 else 0
    h_sum = np.zeros((n_mics, 1, Lsum), dtype=np.float32)
    len_sum = maxlen_per_mic.reshape(n_mics, 1).astype(np.int32)
    for mi in range(n_mics):
        Lm = int(maxlen_per_mic[mi])
        if Lm <= 0:
            continue
        acc = np.zeros((Lm,), dtype=np.float32)
        for si in range(n_src):
            Li = int(lengths[mi, si])
            if Li > 0:
                acc[: min(Lm, Li)] += h[mi, si, : min(Lm, Li)]
        h_sum[mi, 0, :Lm] = acc
    return h_sum, len_sum


# =========================
# Audio loading
# =========================
def load_audio_mono(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    x, fs = librosa.load(str(path), sr=None, mono=True)
    x = x.astype(float)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x, int(fs)


def load_babble_pool(babble_paths: List[str], target_sr: int) -> List[np.ndarray]:
    pool = []
    for p in babble_paths:
        b, fs = librosa.load(p, sr=None, mono=True)
        b = b.astype(float)
        if int(fs) != int(target_sr):
            b = librosa.resample(b, orig_sr=int(fs), target_sr=int(target_sr))
        b = b / (np.max(np.abs(b)) + 1e-9)
        pool.append(b)
    return pool


# =========================
# Simulation primitive
# =========================
def simulate_room_with_sources(
    room_dim: Sequence[float],
    sr: int,
    absorption: float,
    max_order: int,
    mic_positions_all4: np.ndarray,
    sources: List[Tuple[Sequence[float], np.ndarray, Optional[Cardioid]]],
) -> np.ndarray:
    if not sources:
        raise ValueError("simulate_room_with_sources: sources list is empty.")
    room = pra.ShoeBox(room_dim, fs=sr, materials=wall_material(absorption), max_order=max_order)
    room.add_microphone_array(pra.MicrophoneArray(np.asarray(mic_positions_all4, float).T, room.fs))
    for pos, sig, direct in sources:
        if direct is None:
            room.add_source(list(map(float, pos)), signal=sig)
        else:
            room.add_source(list(map(float, pos)), signal=sig, directivity=direct)
    room.simulate()
    N = len(sources[0][1])
    return room.mic_array.signals[:, :N].T


def scale_noise_to_target_snr_vs_mix(
    mix4: np.ndarray, noise4: np.ndarray, target_snr_db: float
) -> Tuple[np.ndarray, float]:
    mix_r = ears_rms(mix4)
    n0 = ears_rms(noise4)
    desired_noise_r = mix_r / (10 ** (float(target_snr_db) / 20.0))
    k = desired_noise_r / (n0 + 1e-12)
    noise_scaled = noise4 * k
    actual = 20 * np.log10(mix_r / (ears_rms(noise_scaled) + 1e-12))
    return noise_scaled, float(actual)


# =========================
# Main public functions
# =========================
def simulate_one_wav(
    input_wav: Union[str, Path],
    output_dir: Union[str, Path],
    out_stem: Optional[str] = None,
    room_cfg: Optional[RoomConfig] = None,
    mic_cfg: Optional[MicConfig] = None,
    spk_cfg: Optional[SpeakerConfig] = None,
    bab_cfg: Optional[BabbleConfig] = None,
    ngm_cfg: Optional[NGMConfig] = None,
    rir_cfg: Optional[RIRConfig] = None,
    out_cfg: Optional[OutputConfig] = None,
    seed: Optional[int] = None,
) -> Dict:
    input_wav = Path(input_wav)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    room_cfg = room_cfg or RoomConfig()
    mic_cfg  = mic_cfg  or MicConfig()
    spk_cfg  = spk_cfg  or SpeakerConfig()
    bab_cfg  = bab_cfg  or BabbleConfig()
    ngm_cfg  = ngm_cfg  or NGMConfig()
    rir_cfg  = rir_cfg  or RIRConfig()
    out_cfg  = out_cfg  or OutputConfig()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load speech (dry)
    speech, sr = load_audio_mono(input_wav)
    N = len(speech)

    stem = out_stem if out_stem is not None else input_wav.stem

    # Skip if already done
    out_wav_check = output_dir / f"{stem}.wav"
    if out_wav_check.exists():
        print(f"[SKIP] {stem} already exists, skipping.", flush=True)
        return {"skipped": True, "stem": stem}

    # Sample room + mics + speaker
    room_dim, rt60, absorption = sample_room(room_cfg)
    mic_positions, mic_center = sample_mics(room_dim, mic_cfg)
    speaker_pos, speaker_mic_distance, orientation_type, speaker_directivity, speaker_dir_unit = sample_speaker(
        room_dim, mic_center, spk_cfg
    )

    # Reference points at 1m
    listener_dir_unit = -speaker_dir_unit
    spk_ref_1m = clamp_inside_room(speaker_pos + speaker_dir_unit * 1.0, room_dim)
    lis_ref_1m = clamp_inside_room(mic_center + listener_dir_unit * 1.0, room_dim)

    # Mic array: [L_ear, R_ear, spk_ref_1m, lis_ref_1m]
    mic_positions_all4 = np.vstack([mic_positions, spk_ref_1m.reshape(1, 3), lis_ref_1m.reshape(1, 3)])

    # ── SPEECH ONLY room render (no babble, no NGM) ──────────────────────────
    speech_render = simulate_room_with_sources(
        room_dim=room_dim, sr=sr, absorption=absorption, max_order=room_cfg.max_order,
        mic_positions_all4=mic_positions_all4,
        sources=[(speaker_pos.tolist(), speech, speaker_directivity)],
    )

    # Save speech-only render (dry path, before babble or NGM)
    out_speech_only_wav = None
    if out_cfg.write_speech_only:
        speech_only_dir = output_dir / "SpeechOnly"
        speech_only_dir.mkdir(parents=True, exist_ok=True)
        out_speech_only_wav = speech_only_dir / f"{stem}.wav"
        sf.write(out_speech_only_wav, speech_render, sr)

    # ── BABBLE render ────────────────────────────────────────────────────────
    babble_positions: List[List[float]] = []
    babble_snr_db: Optional[float] = None
    babble_render_scaled = np.zeros_like(speech_render)

    if bab_cfg.enabled:
        if not bab_cfg.babble_paths:
            raise ValueError("Babble enabled but babble_paths is empty.")
        babble_positions = sample_babble_positions(room_dim, mic_center, speaker_mic_distance, bab_cfg)
        babble_pool = load_babble_pool(bab_cfg.babble_paths, sr)

        if bab_cfg.snr_db is None:
            babble_snr_db = round(random.uniform(*bab_cfg.snr_db_range), 2)
        else:
            babble_snr_db = float(bab_cfg.snr_db)

        babble_sources: List[Tuple[Sequence[float], np.ndarray, Optional[Cardioid]]] = []
        for i, bpos in enumerate(babble_positions):
            b = babble_pool[i % len(babble_pool)]
            bseg = get_babble_segment(b, N)
            bseg = bseg / (rms(bseg) + 1e-12)
            babble_sources.append((bpos, bseg, None))

        babble_render0 = simulate_room_with_sources(
            room_dim=room_dim, sr=sr, absorption=absorption, max_order=room_cfg.max_order,
            mic_positions_all4=mic_positions_all4, sources=babble_sources,
        )
        speech_ears = ears_rms(speech_render)
        babble_ears0 = ears_rms(babble_render0)
        desired_babble_ears = speech_ears / (10 ** (babble_snr_db / 20.0))
        k_babble = desired_babble_ears / (babble_ears0 + 1e-12)
        babble_render_scaled = babble_render0 * k_babble

    # Mix speech + babble (pre-NGM)
    mix4_pre_ngm = speech_render + babble_render_scaled
    mix4 = mix4_pre_ngm.copy()

    # Save pre-NGM mix (speech + babble, before NGM noise)
    out_pre_ngm_wav = None
    if out_cfg.write_pre_ngm_mix:
        pre_dir = output_dir / "PreNGM"
        pre_dir.mkdir(parents=True, exist_ok=True)
        out_pre_ngm_wav = pre_dir / f"{stem}.wav"
        sf.write(out_pre_ngm_wav, mix4_pre_ngm, sr)

    # ── NGM ──────────────────────────────────────────────────────────────────
    ngm_positions = np.zeros((0, 3), dtype=float)
    ngm_actual_snr = None
    ngm_target_snr_db: Optional[float] = None
    noise_scaled = None

    if ngm_cfg.enabled:
        mode = ngm_cfg.mode.strip().lower()
        if mode == "grid":
            avoid = [np.asarray(speaker_pos, float), np.asarray(mic_center, float)]
            ngm_positions = generate_ngm_grid_fixed(
                room_dim=room_dim, ceiling_height=float(room_dim[2]),
                spacing_factor=float(ngm_cfg.spacing_factor),
                min_wall_dist=float(ngm_cfg.min_wall_dist),
                avoid_xyz=avoid, jitter_max=float(ngm_cfg.jitter_max),
            )
        elif mode == "table_listener":
            pos = place_single_ngm_on_listener_table(
                listener_xyz=mic_center, room_dim=room_dim,
                dist_range=ngm_cfg.table_dist_to_listener_range,
                height_range=ngm_cfg.table_height_range, margin=ngm_cfg.table_margin,
            )
            ngm_positions = pos.reshape(1, 3)
        else:
            raise ValueError("NGMConfig.mode must be 'grid' or 'table_listener'.")

        noise_type = ngm_cfg.noise_type.strip().upper()
        if noise_type not in ("SSN", "WN"):
            raise ValueError("NGMConfig.noise_type must be 'SSN' or 'WN'.")

        if ngm_cfg.snr_db is None:
            ngm_target_snr_db = round(random.uniform(*ngm_cfg.snr_db_range), 2)
        else:
            ngm_target_snr_db = float(ngm_cfg.snr_db)

        ngm_sources: List[Tuple[Sequence[float], np.ndarray, Optional[Cardioid]]] = []
        for idx, pos in enumerate(np.asarray(ngm_positions, float)):
            # With a seed, every NGM gets its own reproducible noise. Without
            # one, the generators are left alone: re-seeding them with a fixed
            # value here would make every later room of the run identical.
            if seed is not None:
                seed_val = int(seed) + 104729 * idx
                np.random.seed(seed_val)
                random.seed(seed_val)
            noise_sig = shaped_noise_like(speech) if noise_type == "SSN" else white_noise(N)
            aim_vec = np.array([0.0, 0.0, -1.0], dtype=float)
            az, col = cartesian_to_spherical(aim_vec)
            ngm_directivity = Cardioid(orientation=DirectionVector(az, col))
            gain = 10 ** (np.random.uniform(-1, 1) / 20.0)
            ngm_sources.append((pos.tolist(), gain * noise_sig, ngm_directivity))

        noise_render = simulate_room_with_sources(
            room_dim=room_dim, sr=sr, absorption=absorption, max_order=room_cfg.max_order,
            mic_positions_all4=mic_positions_all4, sources=ngm_sources,
        )
        noise_scaled, ngm_actual_snr = scale_noise_to_target_snr_vs_mix(mix4_pre_ngm, noise_render, ngm_target_snr_db)
        mix4 = mix4_pre_ngm + noise_scaled

    # ── Save final mix (speech + babble + NGM) ───────────────────────────────
    out_wav = output_dir / f"{stem}.wav"
    sf.write(out_wav, mix4, sr)

    out_noise_wav = None
    if out_cfg.write_onlynoise and (noise_scaled is not None):
        only_dir = output_dir / "OnlyNoise"
        only_dir.mkdir(parents=True, exist_ok=True)
        out_noise_wav = only_dir / f"{stem}.wav"
        sf.write(out_noise_wav, noise_scaled, sr)

    # ── RIR saving ───────────────────────────────────────────────────────────
    # RIR1: Speaker -> NGM       (NN input path)
    # RIR2: Speaker -> Listener  (direct acoustic path)
    # RIR3: NGM -> Listener      (NN output -> listener ears)
    rir_files: Dict[str, str] = {}
    bundle_dict: Dict[str, np.ndarray] = {}

    if out_cfg.write_rirs:
        ears = np.asarray(mic_positions, float)

        # RIR1: Speaker -> NGM
        if rir_cfg.rir1_spk_to_ngm and ngm_cfg.enabled and len(ngm_positions) > 0:
            h, lengths = compute_rir_sources_to_mics_numeric(
                room_dim=room_dim, fs=sr, absorption=absorption, max_order=room_cfg.max_order,
                source_positions=[speaker_pos.tolist()],
                mic_positions=np.asarray(ngm_positions, float),
            )
            fp = output_dir / f"{stem}_rir1_spk_to_ngm.npz"
            np.savez(fp, h=h, lengths=lengths, fs=np.int32(sr))
            rir_files["rir1_spk_to_ngm"] = str(fp.resolve())
            bundle_dict["rir1_spk_to_ngm_h"] = h
            bundle_dict["rir1_spk_to_ngm_lengths"] = lengths

        # RIR2: Speaker -> Listener
        if rir_cfg.rir2_spk_to_listener:
            h, lengths = compute_rir_sources_to_mics_numeric(
                room_dim=room_dim, fs=sr, absorption=absorption, max_order=room_cfg.max_order,
                source_positions=[speaker_pos.tolist()],
                mic_positions=ears,
            )
            fp = output_dir / f"{stem}_rir2_spk_to_listener.npz"
            np.savez(fp, h=h, lengths=lengths, fs=np.int32(sr))
            rir_files["rir2_spk_to_listener"] = str(fp.resolve())
            bundle_dict["rir2_spk_to_listener_h"] = h
            bundle_dict["rir2_spk_to_listener_lengths"] = lengths

        # RIR3: NGM -> Listener
        if rir_cfg.rir3_ngm_to_listener and ngm_cfg.enabled and len(ngm_positions) > 0:
            h, lengths = compute_rir_sources_to_mics_numeric(
                room_dim=room_dim, fs=sr, absorption=absorption, max_order=room_cfg.max_order,
                source_positions=[p.tolist() for p in np.asarray(ngm_positions, float)],
                mic_positions=ears,
            )
            fp = output_dir / f"{stem}_rir3_ngm_to_listener.npz"
            np.savez(fp, h=h, lengths=lengths, fs=np.int32(sr))
            rir_files["rir3_ngm_to_listener"] = str(fp.resolve())
            bundle_dict["rir3_ngm_to_listener_h"] = h
            bundle_dict["rir3_ngm_to_listener_lengths"] = lengths

            if bool(rir_cfg.rir3_ngm_to_listener_combined):
                h_sum, len_sum = _sum_ngm_rirs_per_mic(h, lengths)
                fp2 = output_dir / f"{stem}_rir3_ngm_to_listener_combined.npz"
                np.savez(fp2, h=h_sum, lengths=len_sum, fs=np.int32(sr))
                rir_files["rir3_ngm_to_listener_combined"] = str(fp2.resolve())
                bundle_dict["rir3_ngm_to_listener_combined_h"] = h_sum
                bundle_dict["rir3_ngm_to_listener_combined_lengths"] = len_sum

        if bool(rir_cfg.write_bundle_all) and bundle_dict:
            fp = output_dir / f"{stem}_rirs_all.npz"
            bundle_dict["fs"] = np.int32(sr)
            np.savez(fp, **bundle_dict)
            rir_files["bundle_all"] = str(fp.resolve())

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "input_file":             str(input_wav.resolve()),
        "output_file":            str(out_wav.resolve()),
        "speech_only_file":       str(out_speech_only_wav.resolve()) if out_speech_only_wav else None,
        "pre_ngm_output_file":    str(out_pre_ngm_wav.resolve())     if out_pre_ngm_wav     else None,
        "noise_only_file":        str(out_noise_wav.resolve())        if out_noise_wav        else None,
        "sr":                     int(sr),
        "n_samples":              int(N),
        "room_dim":               tuple(map(float, room_dim)),
        "rt60":                   float(rt60),
        "absorption_used":        float(absorption),
        "max_order":              int(room_cfg.max_order),
        "mic_positions":          np.asarray(mic_positions, float).tolist(),
        "mic_center":             np.asarray(mic_center, float).tolist(),
        "speaker_position":       np.asarray(speaker_pos, float).tolist(),
        "speaker_mic_distance":   float(speaker_mic_distance),
        "speaker_orientation_type": str(orientation_type),
        "speaker_ref_mic_1m":     np.asarray(spk_ref_1m, float).tolist(),
        "listener_ref_mic_1m":    np.asarray(lis_ref_1m, float).tolist(),
        "babble_enabled":         bool(bab_cfg.enabled),
        "babble_snr_db":          babble_snr_db,
        "babble_positions":       babble_positions,
        "ngm_enabled":            bool(ngm_cfg.enabled),
        "ngm_mode":               str(ngm_cfg.mode)       if ngm_cfg.enabled else None,
        "ngm_noise_type":         str(ngm_cfg.noise_type) if ngm_cfg.enabled else None,
        "ngm_target_snr_db":      float(ngm_target_snr_db) if ngm_target_snr_db is not None else None,
        "ngm_actual_snr_db":      float(ngm_actual_snr)    if ngm_actual_snr    is not None else None,
        "ngm_positions":          np.asarray(ngm_positions, float).tolist() if ngm_cfg.enabled else [],
        "rir_files":              rir_files,
    }

    viz_path = None
    if out_cfg.write_3d_viz:
        viz_dir = output_dir / "3D_Visualizations"
        viz_path = viz_dir / f"{stem}_3d.html"
        try:
            write_room_3d_html(meta, viz_path)
        except Exception as e:
            print(f"[WARN] 3D figure not written for {stem}: {e}", flush=True)
            meta["viz_error"] = str(e)
    meta["viz_3d_html"] = str(viz_path.resolve()) if viz_path else None

    if out_cfg.write_yaml:
        out_yaml = output_dir / f"{stem}.yaml"
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(convert_numpy(meta), f, sort_keys=False)

    return meta


def simulate_batch(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    n_versions_per_input: int = 1,
    room_cfg: Optional[RoomConfig] = None,
    mic_cfg: Optional[MicConfig] = None,
    spk_cfg: Optional[SpeakerConfig] = None,
    bab_cfg: Optional[BabbleConfig] = None,
    ngm_cfg: Optional[NGMConfig] = None,
    rir_cfg: Optional[RIRConfig] = None,
    out_cfg: Optional[OutputConfig] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    room_cfg = room_cfg or RoomConfig()
    mic_cfg  = mic_cfg  or MicConfig()
    spk_cfg  = spk_cfg  or SpeakerConfig()
    bab_cfg  = bab_cfg  or BabbleConfig()
    ngm_cfg  = ngm_cfg  or NGMConfig()
    rir_cfg  = rir_cfg  or RIRConfig()
    out_cfg  = out_cfg  or OutputConfig()

    metas: List[Dict] = []

    if input_path.is_file():
        wavs = [input_path]
    elif input_path.is_dir():
        wavs = sorted(input_path.rglob("*.wav")) if out_cfg.recursive_input else sorted(input_path.glob("*.wav"))
    else:
        raise ValueError(f"input_path must be a file or directory (got {input_path}).")

    for i, wav in enumerate(tqdm(wavs, desc="Simulating WAV files", unit="file", dynamic_ncols=True, ascii=True)):
        if wav.suffix.lower() != ".wav":
            continue

        version_iter = range(int(n_versions_per_input))
        if int(n_versions_per_input) > 1:
            version_iter = tqdm(
                version_iter, desc=f"Versions for {wav.stem}",
                unit="version", leave=False, dynamic_ncols=True, ascii=True,
            )

        for v in version_iter:
            per_seed = None if seed is None else int(seed + 100000 * i + v)
            stem_out = f"{wav.stem}_v{v+1:02d}" if n_versions_per_input > 1 else wav.stem

            try:
                meta = simulate_one_wav(
                    input_wav=wav, output_dir=output_dir, out_stem=stem_out,
                    room_cfg=room_cfg, mic_cfg=mic_cfg, spk_cfg=spk_cfg,
                    bab_cfg=bab_cfg, ngm_cfg=ngm_cfg, rir_cfg=rir_cfg,
                    out_cfg=out_cfg, seed=per_seed,
                )
                metas.append(meta)
            except Exception as e:
                # Keep going, but say so: a silent failure here turns into a
                # missing RIR file during training.
                print(f"[FAILED] {stem_out}: {type(e).__name__}: {e}", flush=True)
                metas.append({
                    "input_file": str(wav),
                    "stem": stem_out,
                    "version": v + 1,
                    "error": f"{type(e).__name__}: {e}",
                })

    return metas


# =========================
# Command line
# =========================
CONFIG_SECTIONS = {
    "room": RoomConfig,
    "mics": MicConfig,
    "speaker": SpeakerConfig,
    "babble": BabbleConfig,
    "ngm": NGMConfig,
    "rirs": RIRConfig,
    "output": OutputConfig,
}


def config_from_yaml(cls, section: Optional[dict]):
    """Build one of the config dataclasses from a yaml section.

    Keys that are not given keep the default of the dataclass. Unknown keys
    are an error, so a typo in the yaml cannot go unnoticed.
    """
    section = dict(section or {})
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} for {cls.__name__}. Known keys: {sorted(known)}")

    for key, value in section.items():
        if key == "mic_positions" and value is not None:
            section[key] = np.asarray(value, dtype=float)
        elif isinstance(value, list) and key != "babble_paths":
            section[key] = tuple(value)          # ranges are written as [low, high]
    return cls(**section)


def write_failure_report(failures: List[Dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "input_file", "version", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate rooms and write mixtures + RIRs.")
    parser.add_argument("--config", required=True, help="Path to the simulation yaml.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    unknown = sorted(set(cfg) - set(CONFIG_SECTIONS) - {"input_path", "output_dir", "n_versions_per_input", "seed"})
    if unknown:
        raise ValueError(f"Unknown section(s) in {args.config}: {unknown}")
    for key in ("input_path", "output_dir"):
        if not cfg.get(key):
            raise ValueError(f"{key} is missing in {args.config}")

    sections = {name: config_from_yaml(cls, cfg.get(name)) for name, cls in CONFIG_SECTIONS.items()}
    output_dir = Path(cfg["output_dir"])

    metas = simulate_batch(
        input_path=cfg["input_path"],
        output_dir=output_dir,
        n_versions_per_input=int(cfg.get("n_versions_per_input", 1)),
        room_cfg=sections["room"],
        mic_cfg=sections["mics"],
        spk_cfg=sections["speaker"],
        bab_cfg=sections["babble"],
        ngm_cfg=sections["ngm"],
        rir_cfg=sections["rirs"],
        out_cfg=sections["output"],
        seed=cfg.get("seed"),
    )

    failures = [m for m in metas if "error" in m]
    skipped = sum(1 for m in metas if m.get("skipped"))
    done = len(metas) - len(failures) - skipped
    print(f"Simulated {done}, skipped {skipped} (already there), failed {len(failures)}.")

    if failures:
        report = output_dir / "simulation_failures.csv"
        write_failure_report(failures, report)
        print(f"Failed items are listed in {report}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
