"""Small helpers for writing run outputs: json/jsonl/csv, audio samples,
file-name tags and the log of skipped RIR examples."""

import csv
import json
import os
import time

import numpy as np
import soundfile as sf


def write_json(path, obj):
    """Write through a temporary file so a killed job never leaves half a json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def float_to_tag(value):
    """5.0 -> '5', 14.28 -> '14p28', -3 -> 'm3'. Used in directory names."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def sanitize_name(name):
    name = os.path.splitext(os.path.basename(str(name)))[0]
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name).strip("._")
    return cleaned or "sample"


def unique_prefix(prefix, used):
    """Avoid overwriting when two items would produce the same file name."""
    count = used.get(prefix, 0)
    used[prefix] = count + 1
    return prefix if count == 0 else f"{prefix}_dup{count:03d}"


# ---------------------------------------------------------------------------
# Skipped RIR examples
# ---------------------------------------------------------------------------
# A few simulated rooms end up without one of the three RIR files. Those
# examples are skipped rather than stopping a long run, and each of them is
# written once to a jsonl file so the skips can be counted afterwards.

_rir_log_path = None
_rir_already_logged = set()


def configure_rir_warning_log(path):
    global _rir_log_path, _rir_already_logged
    _rir_log_path = path
    _rir_already_logged = set()
    os.makedirs(os.path.dirname(path), exist_ok=True)


def log_rir_warning(stage, sample_index, text, paths, reason):
    paths = paths if isinstance(paths, dict) else {}
    sample_id = paths.get("sample_id") or f"batch_item_{sample_index}"

    if _rir_log_path is None:
        print(f"[WARN RIR] {stage}: skipped {sample_id}: {reason}", flush=True)
        return

    key = (sample_id, paths.get("rir1"), paths.get("rir2"), paths.get("rir3"))
    if key in _rir_already_logged:
        return
    _rir_already_logged.add(key)

    append_jsonl(_rir_log_path, {
        "type": "rir_skip",
        "time": timestamp(),
        "stage": stage,
        "sample_id": sample_id,
        "text": text or "",
        "rir_paths": {k: paths.get(k) for k in ("rir1", "rir2", "rir3")},
        "reason": reason,
    })
    print(f"[WARN RIR] skipped {sample_id}: {reason}", flush=True)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def save_audio_triplet(speech, listener, masker, base_name, epoch, length,
                       sample_rate, output_dir, tag=""):
    """Save speech / masked speech / masker at the listener for one example."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{sanitize_name(base_name)}_epoch{epoch:03d}" + (f"_{tag}" if tag else "")

    def to_numpy(x):
        return x[:length].detach().cpu().numpy()

    sf.write(os.path.join(output_dir, f"{prefix}_speech_at_listener.wav"), to_numpy(speech), sample_rate)
    sf.write(os.path.join(output_dir, f"{prefix}_masked_listener.wav"), to_numpy(listener), sample_rate)
    sf.write(os.path.join(output_dir, f"{prefix}_masker_at_listener.wav"), to_numpy(masker), sample_rate)


def tb_audio(signal, length):
    """TensorBoard wants audio as [batch, samples, channels] in [-1, 1]."""
    signal = np.clip(signal[:length].astype(np.float32), -1.0, 1.0)
    return signal[None, :, None]
