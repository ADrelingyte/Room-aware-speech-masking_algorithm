"""Differentiable acoustic path between the talker, the masker device and
the listener.

    dry speech --RIR1--> speech at NGM mic --Wave-U-Net--> masker --delay--RIR3--> masker at listener
    dry speech --RIR2--------------------------------------------------------> speech at listener

    RIR1 : talker -> NGM microphone      (what the network hears)
    RIR2 : talker -> listener            (the speech we want to conceal)
    RIR3 : NGM loudspeaker -> listener   (how the masker reaches the listener)

The masker (the concealer) is scaled so that the mixture-to-concealer ratio
(MCR) equals the target value. The MCR is measured *at the listener*: it is
the energy of the reverberant speech (dry speech * RIR2) over the energy of
the masker after RIR3, in dB. The listener signal is then
speech_at_listener + masker_at_listener.

With ``acoustic_mode="dry"`` the three RIRs are replaced by a unit impulse,
which gives the anechoic ablation.

Only the first microphone (left ear) of RIR2/RIR3 and the first NGM of
RIR1/RIR3 are used.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from utils.run_logging import log_rir_warning

ACOUSTIC_MODES = ("rir", "dry")
IDENTITY_RIR = np.array([1.0], dtype=np.float32)


@dataclass
class AcousticBatch:
    """Signals on the Wave-U-Net output time axis, each [B, T_out]."""
    speech_at_listener: torch.Tensor
    raw_masker_at_ngm: torch.Tensor
    masker_at_listener: torch.Tensor
    listener_signal: torch.Tensor
    valid_lengths: torch.Tensor
    texts: list


def load_rir(npz_path, mic_idx=0, src_idx=0):
    """Read one impulse response from a file written by simulate_rooms.py.

    The file holds `h` [n_mics, n_sources, max_len] and `lengths` [n_mics, n_sources].
    """
    if not npz_path or not os.path.exists(npz_path):
        raise FileNotFoundError(f"RIR file does not exist: {npz_path!r}")
    with np.load(npz_path) as data:
        if "h" not in data or "lengths" not in data:
            raise KeyError(f"{npz_path!r} must contain the arrays 'h' and 'lengths'.")
        length = int(data["lengths"][mic_idx, src_idx])
        if length <= 0:
            raise ValueError(f"{npz_path!r} has an empty RIR (length {length}).")
        return data["h"][mic_idx, src_idx, :length].astype(np.float32)


def convolve_with_rir(signal, rir):
    """Causal convolution of a [B, 1, T] signal with a numpy RIR.

    The output is truncated to T samples, so the reverberation tail that would
    run past the end of the window is dropped.
    """
    if signal.dim() != 3 or signal.shape[1] != 1:
        raise ValueError(f"Expected [B,1,T], got {tuple(signal.shape)}")
    rir = torch.from_numpy(rir).to(device=signal.device, dtype=signal.dtype)
    # conv1d computes a correlation, hence the flip.
    kernel = rir.flip(0).view(1, 1, -1)
    full = F.conv1d(signal, kernel, padding=kernel.shape[-1] - 1)
    return full[..., : signal.shape[-1]]


def apply_causal_delay(signal, delay_samples):
    """Delay a [B, 1, T] signal by prepending zeros; the length stays T.

    Models the processing latency between the network output and the NGM
    loudspeaker. Samples pushed past the end of the window are lost.
    """
    delay_samples = int(delay_samples)
    if delay_samples < 0:
        raise ValueError(f"Delay must be >= 0 samples, got {delay_samples}")
    if delay_samples == 0:
        return signal

    total = signal.shape[-1]
    if delay_samples >= total:
        return torch.zeros_like(signal)
    zeros = torch.zeros_like(signal[..., :delay_samples])
    return torch.cat([zeros, signal[..., : total - delay_samples]], dim=-1)


def crop_or_pad(dry_speech, lengths, n_samples, random_crop):
    """Bring every utterance to exactly `n_samples` (the Wave-U-Net input size).

    Long utterances are cropped, at a random position during training and from
    the start otherwise, so validation and test are deterministic. Short ones
    are zero padded. Returns the [B, 1, n_samples] batch and the number of real
    (non padded) samples per row.
    """
    batch_size, _, available = dry_speech.shape
    chunks, new_lengths = [], []

    for i in range(batch_size):
        valid = int(torch.clamp(lengths[i], min=0, max=available).item())
        if valid > n_samples:
            start = 0
            if random_crop:
                start = int(torch.randint(0, valid - n_samples + 1, (1,), device=dry_speech.device).item())
            chunk = dry_speech[i:i + 1, :, start:start + n_samples]
            new_lengths.append(n_samples)
        else:
            chunk = F.pad(dry_speech[i:i + 1, :, :valid], (0, n_samples - valid))
            new_lengths.append(valid)
        chunks.append(chunk)

    new_lengths = torch.tensor(new_lengths, dtype=torch.long, device=dry_speech.device)
    return torch.cat(chunks, dim=0), new_lengths


def length_mask(valid_lengths, total_samples):
    """[B, T] float mask, 1 on real samples and 0 on padding."""
    valid_lengths = torch.clamp(valid_lengths, min=0, max=total_samples).long()
    t = torch.arange(total_samples, device=valid_lengths.device)[None, :]
    return (t < valid_lengths[:, None]).float()


def mcr_db_per_example(speech, masker, valid_lengths):
    mask = length_mask(valid_lengths, speech.shape[-1])
    speech_energy = torch.sum(speech ** 2 * mask, dim=1) + 1e-12
    masker_energy = torch.sum(masker ** 2 * mask, dim=1) + 1e-12
    return 10.0 * torch.log10(speech_energy / masker_energy)


def scale_masker_to_target_mcr(speech, masker, valid_lengths, target_mcr_db):
    """Rescale each masker so that speech/masker energy matches the target MCR.

    Energies are measured on the valid samples only, and the padded part of
    the masker is set to zero. The gain depends on the network output, so the
    gradient also flows through it: the network controls the shape of the
    masker, never its level.
    """
    mask = length_mask(valid_lengths, speech.shape[-1])
    speech_energy = torch.sum(speech ** 2 * mask, dim=1) + 1e-12
    masker_energy = torch.sum(masker ** 2 * mask, dim=1) + 1e-12
    wanted_energy = speech_energy / (10.0 ** (float(target_mcr_db) / 10.0))
    gain = torch.sqrt(wanted_energy / masker_energy)
    return masker * gain[:, None] * mask


def warn_if_mcr_mismatch(measured_mcr_db, target_mcr_db, where="", tolerance_db=0.05):
    if (torch.abs(measured_mcr_db - float(target_mcr_db)) > tolerance_db).any():
        print(
            f"[WARN MCR]{where} target={target_mcr_db} dB, "
            f"measured={measured_mcr_db.detach().cpu().tolist()}",
            flush=True,
        )


def check_rir_paths(rir_paths, batch_size):
    """The collate function gives one dict per example; make sure that is what we got."""
    if not isinstance(rir_paths, (list, tuple)) or len(rir_paths) != batch_size:
        raise ValueError(f"Expected a list of {batch_size} RIR dicts, one per example.")
    for item in rir_paths:
        if not isinstance(item, dict):
            raise TypeError(f"Each RIR entry must be a dict, got {type(item).__name__}.")
    return list(rir_paths)


def forward_acoustic_system(
    model,
    dry_speech,
    lengths,
    texts,
    rir_paths,
    target_mcr_db,
    random_crop,
    acoustic_mode="rir",
    delay_samples=0,
    stage="forward",
):
    """Run one batch through the whole chain and return an AcousticBatch.

    Examples whose RIR files are missing or unreadable are logged and left
    out, so the returned batch can be smaller than the input batch. Returns
    None when nothing usable is left.
    """
    if acoustic_mode not in ACOUSTIC_MODES:
        raise ValueError(f"acoustic_mode must be one of {ACOUSTIC_MODES}, got {acoustic_mode!r}")

    # The network uses valid convolutions: it needs input_size samples and
    # predicts only the centre part [out_start, out_end) of that window.
    in_samples = int(model.input_size)
    out_start = int(model.shapes["output_start_frame"])
    out_end = int(model.shapes["output_end_frame"])
    out_samples = out_end - out_start

    dry_speech, lengths = crop_or_pad(dry_speech, lengths, in_samples, random_crop)
    rir_paths = check_rir_paths(rir_paths, dry_speech.shape[0])

    speech_at_ngm, speech_at_listener = [], []
    rir3_list, kept_lengths, kept_texts = [], [], []

    for i, paths in enumerate(rir_paths):
        if acoustic_mode == "dry":
            rir1 = rir2 = rir3 = IDENTITY_RIR
        else:
            missing = [k for k in ("rir1", "rir2", "rir3") if not paths.get(k)]
            if missing:
                log_rir_warning(stage, i, texts[i], paths, f"no path given for {missing}")
                continue
            try:
                rir1 = load_rir(paths["rir1"])
                rir2 = load_rir(paths["rir2"])
                rir3 = load_rir(paths["rir3"])
            except Exception as exc:
                log_rir_warning(stage, i, texts[i], paths, f"RIR load failed: {type(exc).__name__}: {exc}")
                continue

        one = dry_speech[i:i + 1]
        speech_at_ngm.append(convolve_with_rir(one, rir1))
        speech_at_listener.append(convolve_with_rir(one, rir2))
        rir3_list.append(rir3)
        kept_lengths.append(lengths[i])
        kept_texts.append(texts[i])

    if not speech_at_ngm:
        return None

    speech_at_ngm = torch.cat(speech_at_ngm, dim=0)
    speech_at_listener = torch.cat(speech_at_listener, dim=0)
    kept_lengths = torch.stack(kept_lengths).long()

    raw_masker = model(speech_at_ngm)                          # [B, 1, T_out]
    if raw_masker.dim() != 3 or raw_masker.shape[1] != 1 or raw_masker.shape[-1] != out_samples:
        raise RuntimeError(
            f"Wave-U-Net returned {tuple(raw_masker.shape)}, expected [B, 1, {out_samples}]."
        )

    # The latency sits between the network and the loudspeaker, so only the
    # masker is delayed. The talker's own path to the listener is untouched.
    delayed_masker = apply_causal_delay(raw_masker, delay_samples)

    masker_at_listener = torch.cat(
        [convolve_with_rir(delayed_masker[i:i + 1], rir3) for i, rir3 in enumerate(rir3_list)],
        dim=0,
    ).squeeze(1)

    # Put the listener speech on the same time axis as the network output.
    speech_at_listener = speech_at_listener[..., out_start:out_end].squeeze(1)
    raw_masker = raw_masker.squeeze(1)
    valid_lengths = torch.clamp(kept_lengths - out_start, min=0, max=out_samples)

    # Drop examples with nothing left in the output window (very short
    # utterances) or with silent speech: the MCR is undefined for those.
    mask = length_mask(valid_lengths, out_samples)
    speech_energy = torch.sum(speech_at_listener ** 2 * mask, dim=1)
    keep = (valid_lengths >= 1) & (speech_energy > 1e-10)
    if not keep.any():
        return None
    keep_list = keep.cpu().tolist()

    speech_at_listener = speech_at_listener[keep]
    valid_lengths = valid_lengths[keep]
    masker_at_listener = scale_masker_to_target_mcr(
        speech_at_listener, masker_at_listener[keep], valid_lengths, target_mcr_db
    )

    return AcousticBatch(
        speech_at_listener=speech_at_listener,
        raw_masker_at_ngm=raw_masker[keep],
        masker_at_listener=masker_at_listener,
        listener_signal=speech_at_listener + masker_at_listener,
        valid_lengths=valid_lengths,
        texts=[t for t, k in zip(kept_texts, keep_list) if k],
    )
