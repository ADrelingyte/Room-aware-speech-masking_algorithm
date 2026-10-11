"""Train the Wave-U-Net masker, or generate maskers from a trained checkpoint.

    python train.py --config configs/train_example.yaml --output-dir runs/ctc_stoi_rir

What is trained is chosen in the config:

    training.loss           ctc_stoi | stoi
    training.acoustic_mode  rir | dry

One run is trained for every combination of target MCR (and of lambda_stoi
when the loss is ctc_stoi). The best run is chosen on the validation set and
evaluated once on the test set.

With ``generation.only: true`` nothing is trained: the checkpoint given in
``generation.ckpt`` is used to write the masker signals of one split to disk.
"""

import argparse
import json
import math
import os
import random

import numpy as np
import soundfile as sf
import tensorflow as tf            # only used to write TensorBoard summaries
import torch
import yaml
from torch.utils.data import DataLoader
from torch_stoi import NegSTOILoss
from tqdm.auto import tqdm

from model.waveunet import Waveunet
from utils.acoustics import ACOUSTIC_MODES, forward_acoustic_system
from utils.dataset import SpeechDataset, collate_fn
from utils.losses import LOSS_TYPES, batch_objective, is_better, metric_should_maximize
from utils.run_logging import (
    append_jsonl,
    choose_seed,
    configure_rir_warning_log,
    load_seeds,
    save_seeds,
    float_to_tag,
    sanitize_name,
    save_audio_triplet,
    tb_audio,
    timestamp,
    unique_prefix,
    write_csv,
    write_json,
)
from utils.spectrograms import tf_log_spectrograms
from utils.splits import grouped_train_val_test_split, save_split_manifests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EPOCH_METRICS = ("val_total", "val_ctc", "val_stoi", "val_mse")
GRID_METRICS = ("best_val_total", "best_val_ctc", "best_val_stoi", "best_val_mse")
METRIC_NAMES = ("total", "ctc", "mse", "mcr_db", "stoi")

# The STFTs of the STOI and ASR losses see a new signal length for almost
# every batch, and the cuFFT plan cache keeps one plan per length. Turning
# the cache off stops the GPU memory from growing during training.
if torch.cuda.is_available():
    torch.backends.cuda.cufft_plan_cache.max_size = 0


def get(cfg, section, key, default=None):
    """cfg[section][key] with a default. A key left empty in the yaml counts as missing."""
    value = (cfg.get(section) or {}).get(key)
    return default if value is None else value


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_waveunet(cfg, device):
    return Waveunet(
        num_inputs=1,
        num_channels=list(get(cfg, "model", "channels", [8, 16, 32, 64, 128])),
        num_outputs=1,
        instruments=["noise"],
        kernel_size=int(get(cfg, "model", "kernel_size", 11)),
        target_output_size=int(get(cfg, "model", "output_samples", 112000)),
        conv_type="gn",
        res="fixed",
        separate=False,
        depth=1,
        strides=2,
    ).to(device)


def build_asr(cfg, device):
    # Imported here so that STOI-only runs do not need NeMo installed.
    from utils.frozen_asr import DEFAULT_ASR_MODEL, FrozenASR
    asr = FrozenASR(
        device=device,
        sample_rate=int(get(cfg, "dataset", "sample_rate", 16000)),
        model_name=str(get(cfg, "training", "asr_model", DEFAULT_ASR_MODEL)),
    )
    return asr.eval()


def load_checkpoint(path, device):
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Our checkpoints also store the config dict and the metrics, not only
    # tensors, so they cannot be read with weights_only=True.
    return torch.load(path, map_location=device, weights_only=False)


def stored_mcr(checkpoint):
    """Target MCR of a checkpoint. Older checkpoints store it as target_snr_db."""
    return checkpoint.get("target_mcr_db", checkpoint.get("target_snr_db"))


def stored_seed(checkpoint, name):
    """A seed saved in a checkpoint, or in the config it was trained with."""
    if not checkpoint:
        return None
    if checkpoint.get(name) is not None:
        return checkpoint[name]
    section, key = {"split_seed": ("training", "split_seed"), "rooms_seed": ("rooms", "seed")}[name]
    return ((checkpoint.get("cfg") or {}).get(section) or {}).get(key)


def read_settings(cfg):
    """Collect and check the options that decide what kind of run this is."""
    loss_type = str(get(cfg, "training", "loss", "ctc_stoi")).lower()
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"training.loss must be one of {LOSS_TYPES}, got {loss_type!r}")

    acoustic_mode = str(get(cfg, "training", "acoustic_mode", "rir")).lower()
    if acoustic_mode not in ACOUSTIC_MODES:
        raise ValueError(f"training.acoustic_mode must be one of {ACOUSTIC_MODES}, got {acoustic_mode!r}")

    # The ASR model is expensive, so with the STOI loss it is only loaded if
    # the CTC value is wanted as an extra metric.
    use_asr = loss_type == "ctc_stoi" or bool(get(cfg, "training", "track_ctc", False))

    sample_rate = int(get(cfg, "dataset", "sample_rate", 16000))
    delay_ms = float(get(cfg, "training", "deployment_delay_ms", 0.0))
    if delay_ms < 0:
        raise ValueError("training.deployment_delay_ms must be >= 0.")

    epoch_metric = str(get(cfg, "training", "epoch_selection_metric", "val_total"))
    if epoch_metric not in EPOCH_METRICS:
        raise ValueError(f"training.epoch_selection_metric must be one of {EPOCH_METRICS}.")

    grid_metric = str(get(cfg, "training", "selection_metric", "best_val_total"))
    grid_metric = grid_metric.replace("final_val_", "best_val_")      # name used in older configs
    if grid_metric not in GRID_METRICS:
        raise ValueError(f"training.selection_metric must be one of {GRID_METRICS}.")

    if not use_asr and ("ctc" in epoch_metric or "ctc" in grid_metric):
        raise ValueError("Selecting on CTC needs the ASR model: set training.track_ctc: true.")

    return {
        "loss_type": loss_type,
        "acoustic_mode": acoustic_mode,
        "use_asr": use_asr,
        "sample_rate": sample_rate,
        "delay_ms": delay_ms,
        "delay_samples": int(round(sample_rate * delay_ms / 1000.0)),
        "epoch_metric": epoch_metric,
        "grid_metric": grid_metric,
    }


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


class MetricAverager:
    """Accumulates the metrics of batch_objective() over an epoch."""

    def __init__(self):
        self.values = {name: [] for name in METRIC_NAMES}

    def add(self, metrics):
        self.values["total"].append(float(metrics["loss"].item()))
        if metrics["ctc_per_example"] is not None:
            self.values["ctc"].extend(metrics["ctc_per_example"].detach().cpu().tolist())
        self.values["mse"].extend(metrics["mse_per_example"].detach().cpu().tolist())
        self.values["mcr_db"].extend(metrics["mcr_per_example"].detach().cpu().tolist())
        if not torch.isnan(metrics["stoi"]):
            self.values["stoi"].append(float(metrics["stoi"].item()))

    def result(self):
        return {name: mean_or_nan(values) for name, values in self.values.items()}


def no_transcript(texts):
    return all(not t.strip() for t in texts)


@torch.no_grad()
def evaluate(model, asr, stoi_criterion, loader, device, settings, target_mcr_db,
             lambda_stoi, acoustic_mode=None):
    """Average loss and metrics over a loader, without random cropping."""
    acoustic_mode = acoustic_mode or settings["acoustic_mode"]
    model.eval()
    averager = MetricAverager()

    for dry_speech, lengths, texts, rir_paths in loader:
        if asr is not None and no_transcript(texts):
            continue

        batch = forward_acoustic_system(
            model, dry_speech.to(device), lengths.to(device), texts, rir_paths,
            target_mcr_db=target_mcr_db,
            random_crop=False,
            acoustic_mode=acoustic_mode,
            delay_samples=settings["delay_samples"],
            stage=f"evaluation_{acoustic_mode}",
        )
        if batch is None:
            continue

        metrics = batch_objective(
            batch, asr, stoi_criterion, settings["loss_type"], lambda_stoi, target_mcr_db, where=" [eval]"
        )
        if metrics is not None:
            averager.add(metrics)

    return averager.result()


def check_first_batch(loader, name, acoustic_mode):
    """Catch an empty loader or missing RIR files before the ASR model is loaded."""
    batch = next(iter(loader), None)
    if batch is None:
        raise RuntimeError(f"The {name} loader is empty.")
    _, _, _, rir_paths = batch

    if acoustic_mode == "dry":
        print(f"[CHECK] {name}: first batch OK ({len(rir_paths)} examples, dry mode).", flush=True)
        return
    complete = sum(
        all(os.path.exists(paths[k]) for k in ("rir1", "rir2", "rir3")) for paths in rir_paths
    )
    print(f"[CHECK] {name}: {complete}/{len(rir_paths)} examples of the first batch have their 3 RIR files.", flush=True)


def train_one_run(cfg, settings, train_loader, val_loader, device, target_mcr_db,
                  lambda_stoi, output_root, fixed_val_name, run_seed):
    """Train one model for one (MCR, lambda) setting. Returns a summary dict."""
    set_all_seeds(run_seed)
    if getattr(train_loader, "generator", None) is not None:
        train_loader.generator.manual_seed(run_seed)

    loss_type = settings["loss_type"]
    acoustic_mode = settings["acoustic_mode"]
    sample_rate = settings["sample_rate"]
    delay_ms, delay_samples = settings["delay_ms"], settings["delay_samples"]
    epoch_metric = settings["epoch_metric"]
    target_mcr_db, lambda_stoi = float(target_mcr_db), float(lambda_stoi)

    mcr_tag = float_to_tag(target_mcr_db)
    run_name = f"mcr_{mcr_tag}dB"
    if loss_type == "ctc_stoi":
        run_name += f"_lambda_{float_to_tag(lambda_stoi)}"
    run_name += f"_delay_{float_to_tag(delay_ms)}ms"

    run_dir = os.path.join(output_root, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    sample_dir = os.path.join(run_dir, "samples")
    for directory in (checkpoint_dir, log_dir, sample_dir):
        os.makedirs(directory, exist_ok=True)

    epochs = int(get(cfg, "training", "epochs", 10))
    grad_clip_norm = float(get(cfg, "training", "grad_clip_norm", 1.0))
    log_every = int(get(cfg, "training", "json_step_log_every", 20))
    save_audio_each_epoch = bool(get(cfg, "training", "save_audio_each_epoch", False))
    save_audio_final_epoch = bool(get(cfg, "training", "save_audio_final_epoch", True))
    tb_log_audio = bool(get(cfg, "training", "tb_log_audio", False))

    model = build_waveunet(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(get(cfg, "training", "lr", 1e-4)))

    # ------------------------------------------------------------------
    # Resume. `epochs` is the last epoch number, not a number of extra epochs:
    # a checkpoint from epoch 5 with epochs: 20 continues at 6 and stops at 20.
    # ------------------------------------------------------------------
    start_epoch, global_step = 1, 0
    resume_path = get(cfg, "training", "resume_checkpoint", None)
    resume_state = None
    if resume_path:
        print(f"[RESUME] Loading {resume_path}", flush=True)
        resume_state = load_checkpoint(resume_path, device)
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optim_state_dict"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        global_step = int(resume_state.get("global_step", 0))

        # Refuse to continue a checkpoint that was trained in another condition.
        # Checkpoints made before a field existed simply skip that check.
        for key, current in (
            ("target_mcr_db", target_mcr_db),
            ("lambda_stoi", lambda_stoi if loss_type == "ctc_stoi" else None),
            ("deployment_delay_ms", delay_ms),
            ("loss", loss_type),
            ("acoustic_mode", acoustic_mode),
        ):
            saved = stored_mcr(resume_state) if key == "target_mcr_db" else resume_state.get(key)
            if saved is None or current is None:
                continue
            same = saved == current if isinstance(current, str) else abs(float(saved) - current) < 1e-6
            if not same:
                raise ValueError(f"Checkpoint has {key}={saved}, but this run uses {key}={current}.")

        if start_epoch > epochs:
            raise ValueError(
                f"The checkpoint is already at epoch {start_epoch - 1}; "
                f"raise training.epochs (now {epochs}) to continue."
            )
        print(f"[RESUME] Continuing at epoch {start_epoch}.", flush=True)

    asr = build_asr(cfg, device) if settings["use_asr"] else None
    stoi_criterion = NegSTOILoss(sample_rate=sample_rate).to(device)

    objective = (
        f"-CTC + {lambda_stoi} * STOI(masked, speech at listener)"
        if loss_type == "ctc_stoi" else "STOI(masked, speech at listener)"
    )
    print(
        f"[RUN] {run_name} | acoustic mode: {acoustic_mode} | MCR {target_mcr_db} dB | "
        f"delay {delay_ms} ms ({delay_samples} samples) | epochs selected on {epoch_metric}",
        flush=True,
    )
    print(f"[RUN] Minimising {objective}", flush=True)

    tb_writer = tf.summary.create_file_writer(os.path.join(log_dir, "tensorboard"))
    step_log_path = os.path.join(log_dir, "steps.jsonl")
    metrics_path = os.path.join(log_dir, "metrics.json")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")

    history = {f"{split}_{name}": [] for name in METRIC_NAMES for split in ("train", "val")}
    best_epoch, best_value, best_metrics = None, None, None

    if resume_state is not None:
        # Reload the curves of the interrupted run and find its best epoch again.
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                previous = json.load(f)
            for key in history:
                if isinstance(previous.get(key), list):
                    history[key] = previous[key][: start_epoch - 1]
        for epoch_number, value in enumerate(history[epoch_metric], start=1):
            if is_better(value, best_value, epoch_metric):
                best_epoch, best_value = epoch_number, value
                best_metrics = {name: history[f"val_{name}"][epoch_number - 1] for name in METRIC_NAMES}
        if best_epoch is None and resume_state.get("epoch_selection_value") is not None:
            best_epoch = start_epoch - 1
            best_value = float(resume_state["epoch_selection_value"])
            best_metrics = dict(resume_state.get("val_metrics") or {})

    run_description = {
        "loss": loss_type,
        "acoustic_mode": acoustic_mode,
        "target_mcr_db": target_mcr_db,
        "lambda_stoi": lambda_stoi,
        "deployment_delay_ms": delay_ms,
        "deployment_delay_samples": delay_samples,
        "run_seed": run_seed,
        "split_seed": settings.get("split_seed"),
        "rooms_seed": settings.get("rooms_seed"),
        "epoch_selection_metric": epoch_metric,
    }
    append_jsonl(step_log_path, {
        "type": "meta",
        "time": timestamp(),
        **run_description,
        "objective": objective,
        "waveunet_input_samples": int(model.input_size),
        "waveunet_output_samples": int(model.output_size),
        "resume_checkpoint": resume_path,
        "start_epoch": start_epoch,
    })

    # The same validation example is saved after every epoch, so the samples
    # can be compared across epochs by ear.
    fixed_val_batch = next(iter(val_loader), None)
    if fixed_val_batch is None:
        raise RuntimeError("The validation loader is empty.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        averager = MetricAverager()
        progress = tqdm(train_loader, desc=f"[{run_name}] epoch {epoch}/{epochs}", dynamic_ncols=True)

        for batch_index, (dry_speech, lengths, texts, rir_paths) in enumerate(progress):
            global_step += 1
            if asr is not None and no_transcript(texts):
                continue

            batch = forward_acoustic_system(
                model, dry_speech.to(device), lengths.to(device), texts, rir_paths,
                target_mcr_db=target_mcr_db,
                random_crop=True,
                acoustic_mode=acoustic_mode,
                delay_samples=delay_samples,
                stage=f"train_epoch_{epoch}",
            )
            if batch is None:
                continue

            metrics = batch_objective(
                batch, asr, stoi_criterion, loss_type, lambda_stoi, target_mcr_db,
                where=f" [train epoch={epoch} batch={batch_index}]",
            )
            if metrics is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            averager.add(metrics)
            step = {
                "total": float(metrics["loss"].item()),
                "ctc": float(metrics["ctc_mean"].item()),
                "stoi": float(metrics["stoi"].item()),
                "mse": float(metrics["mse_per_example"].mean().item()),
                "mcr_db": float(metrics["mcr_per_example"].mean().item()),
            }

            with tb_writer.as_default():
                for name, value in step.items():
                    if not math.isnan(value):
                        tf.summary.scalar(f"step/train_{name}", value, step=global_step)

            if log_every > 0 and global_step % log_every == 0:
                append_jsonl(step_log_path, {
                    "type": "train_step",
                    "time": timestamp(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "examples_used": len(batch.texts),
                    **step,
                })

            progress.set_postfix(total=step["total"], ctc=step["ctc"], stoi=step["stoi"], mcr_db=step["mcr_db"])

        train_metrics = averager.result()
        val_metrics = evaluate(
            model, asr, stoi_criterion, val_loader, device, settings, target_mcr_db, lambda_stoi
        )
        for name in METRIC_NAMES:
            history[f"train_{name}"].append(train_metrics[name])
            history[f"val_{name}"].append(val_metrics[name])

        candidate = val_metrics[epoch_metric.replace("val_", "")]
        is_best = is_better(candidate, best_value, epoch_metric)

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "cfg": cfg,
            **run_description,
            "epoch_selection_value": candidate,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))
        if is_best:
            best_epoch, best_value, best_metrics = epoch, candidate, dict(val_metrics)
            torch.save(checkpoint, best_checkpoint_path)

        write_json(metrics_path, {
            **run_description,
            "best_epoch": best_epoch,
            "best_epoch_metric_value": best_value,
            **history,
        })

        with tb_writer.as_default():
            for split, split_metrics in (("train", train_metrics), ("val", val_metrics)):
                for name, value in split_metrics.items():
                    if not math.isnan(value):
                        tf.summary.scalar(f"epoch/{split}_{name}", value, step=epoch)

        append_jsonl(step_log_path, {
            "type": "epoch_summary",
            "time": timestamp(),
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": val_metrics,
            "is_best_epoch": bool(is_best),
        })
        print(
            f"[EPOCH {epoch:03d}] train total={train_metrics['total']:.6f} ctc={train_metrics['ctc']:.6f} "
            f"stoi={train_metrics['stoi']:.4f} | val total={val_metrics['total']:.6f} "
            f"ctc={val_metrics['ctc']:.6f} stoi={val_metrics['stoi']:.4f} | "
            f"MCR at listener={val_metrics['mcr_db']:.3f} dB{' [BEST]' if is_best else ''}",
            flush=True,
        )

        # Listening sample from the fixed validation example.
        save_wavs = save_audio_each_epoch or (save_audio_final_epoch and epoch == epochs)
        if save_wavs or tb_log_audio:
            model.eval()
            dry_speech, lengths, texts, rir_paths = fixed_val_batch
            with torch.no_grad():
                fixed = forward_acoustic_system(
                    model, dry_speech[:1].to(device), lengths[:1].to(device), texts[:1], rir_paths[:1],
                    target_mcr_db=target_mcr_db,
                    random_crop=False,
                    acoustic_mode=acoustic_mode,
                    delay_samples=delay_samples,
                    stage=f"fixed_validation_epoch_{epoch}",
                )
            # At most 7 seconds, which is the Wave-U-Net output window anyway.
            length = 0 if fixed is None else min(int(fixed.valid_lengths[0]), sample_rate * 7)
            if length > 1:
                tag = f"{acoustic_mode}_mcr_{mcr_tag}dB"
                if save_wavs:
                    save_audio_triplet(
                        fixed.speech_at_listener[0], fixed.listener_signal[0], fixed.masker_at_listener[0],
                        fixed_val_name, epoch, length, sample_rate, sample_dir, tag=tag,
                    )
                if tb_log_audio:
                    speech_np = fixed.speech_at_listener[0, :length].cpu().numpy().astype(np.float32)
                    listener_np = fixed.listener_signal[0, :length].cpu().numpy().astype(np.float32)
                    masker_np = fixed.masker_at_listener[0, :length].cpu().numpy().astype(np.float32)
                    tf_log_spectrograms(
                        tb_writer, f"spec/{fixed_val_name}_{tag}", speech_np, listener_np, sample_rate, epoch
                    )
                    with tb_writer.as_default():
                        tf.summary.audio("audio/speech_at_listener", tb_audio(speech_np, length), sample_rate, step=epoch)
                        tf.summary.audio("audio/masked_listener", tb_audio(listener_np, length), sample_rate, step=epoch)
                        tf.summary.audio("audio/masker_at_listener", tb_audio(masker_np, length), sample_rate, step=epoch)

        tb_writer.flush()

    tb_writer.close()
    if best_epoch is None or not best_metrics:
        raise RuntimeError("No epoch produced a usable validation value for this run.")

    return {
        **run_description,
        "best_epoch": best_epoch,
        "best_epoch_metric_value": float(best_value),
        **{f"best_val_{name}": float(best_metrics[name]) for name in METRIC_NAMES},
        "best_checkpoint": best_checkpoint_path,
        "output_dir": run_dir,
    }


@torch.no_grad()
def generate_maskers(model, loader, dataset, dataset_indices, device, settings, output_dir,
                     target_mcr_db, acoustic_mode, delay_samples, limit_files=0,
                     save_masked=True, save_clean=True, use_original_names=True, filename_tag=""):
    """Write the masker at the listener (and optionally speech / masked speech)
    for every item of a loader.

    Examples are processed one at a time, so an example that gets skipped can
    never shift the correspondence between audio and file names.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    sample_rate = settings["sample_rate"]
    tag = f"_{sanitize_name(filename_tag)}" if filename_tag else ""
    used_prefixes = {}
    seen, saved = 0, 0
    progress = tqdm(total=limit_files or len(dataset_indices), desc="Generating maskers", unit="file", dynamic_ncols=True)

    for dry_speech, lengths, texts, rir_paths in loader:
        for i in range(dry_speech.shape[0]):
            dataset_index = dataset_indices[seen]
            seen += 1

            one = forward_acoustic_system(
                model, dry_speech[i:i + 1].to(device), lengths[i:i + 1].to(device),
                [texts[i]], [rir_paths[i]],
                target_mcr_db=target_mcr_db,
                random_crop=False,
                acoustic_mode=acoustic_mode,
                delay_samples=delay_samples,
                stage="generation",
            )
            name = dataset.wav_files[dataset_index]
            if one is None or int(one.valid_lengths[0]) <= 1:
                print(f"[WARN GEN] {name}: nothing to generate, skipped.", flush=True)
                continue

            n = int(one.valid_lengths[0])
            if use_original_names:
                prefix = unique_prefix(sanitize_name(name) + tag, used_prefixes)
            else:
                prefix = f"sample_{saved:08d}{tag}"

            def write(suffix, signal):
                sf.write(
                    os.path.join(output_dir, f"{prefix}_{suffix}.wav"),
                    signal[0, :n].cpu().numpy().astype(np.float32),
                    sample_rate,
                )

            write("masker_at_listener", one.masker_at_listener)
            if save_clean:
                write("speech_at_listener", one.speech_at_listener)
            if save_masked:
                write("masked_listener", one.listener_signal)

            saved += 1
            progress.update(1)
            if limit_files and saved >= limit_files:
                progress.close()
                return saved

    progress.close()
    return saved


def best_run_of(results, metric):
    usable = [r for r in results if not math.isnan(r.get(metric, float("nan")))]
    if not usable:
        return None
    pick = max if metric_should_maximize(metric) else min
    return pick(usable, key=lambda r: r[metric])


def pareto_frontier(results):
    """Runs that no other run at the same MCR beats on both CTC (higher is
    better for us) and STOI (lower is better)."""
    usable = [
        r for r in results
        if not math.isnan(r["best_val_ctc"]) and not math.isnan(r["best_val_stoi"])
    ]
    frontier = []
    for run in usable:
        dominated = any(
            other is not run
            and other["target_mcr_db"] == run["target_mcr_db"]
            and other["best_val_ctc"] >= run["best_val_ctc"]
            and other["best_val_stoi"] <= run["best_val_stoi"]
            and (other["best_val_ctc"] > run["best_val_ctc"] or other["best_val_stoi"] < run["best_val_stoi"])
            for other in usable
        )
        if not dominated:
            frontier.append(run)
    return frontier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the yaml config.")
    parser.add_argument("--output-dir", default=None,
                        help="Where runs are written. Default: runs/<loss>_<acoustic_mode>")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    settings = read_settings(cfg)
    sample_rate = settings["sample_rate"]

    output_root = args.output_dir or os.path.join(
        PROJECT_ROOT, "runs", f"{settings['loss_type']}_{settings['acoustic_mode']}"
    )
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    configure_rir_warning_log(os.path.join(output_root, "rir_warnings.jsonl"))

    requested_device = str(get(cfg, "training", "device", "cuda"))
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] device={device}, sample rate={sample_rate}, output={output_root}", flush=True)

    val_split = float(get(cfg, "training", "val_split", 0.05))
    test_split = float(get(cfg, "training", "test_split", 0.05))
    if val_split <= 0 or test_split <= 0 or val_split + test_split >= 1:
        raise ValueError("Need 0 < val_split, 0 < test_split and val_split + test_split < 1.")

    # Generation reads the split of the run that produced the checkpoint, so
    # the checkpoint is loaded before the split is made.
    gen_cfg = cfg.get("generation") or {}
    gen_checkpoint = None
    if gen_cfg.get("only"):
        if not gen_cfg.get("ckpt"):
            raise ValueError("generation.only: true needs generation.ckpt")
        gen_checkpoint = load_checkpoint(gen_cfg["ckpt"], device)

    saved_seeds = load_seeds(output_root)
    split_seed = choose_seed(
        "split_seed", get(cfg, "training", "split_seed"),
        stored_seed(gen_checkpoint, "split_seed"), saved_seeds.get("split_seed"),
    )
    run_seed = choose_seed("run_seed", get(cfg, "training", "run_seed"))
    settings["split_seed"] = split_seed
    save_seeds(output_root, {"split_seed": split_seed, "run_seed": run_seed})

    # Configs written before the rename call this target_snr_db_list.
    mcr_values = get(cfg, "training", "target_mcr_db_list", get(cfg, "training", "target_snr_db_list", [5.0]))
    target_mcr_db_list = [float(v) for v in mcr_values]
    if settings["loss_type"] == "ctc_stoi":
        lambda_stoi_list = [float(v) for v in get(cfg, "training", "lambda_stoi_list", [1.0])]
    else:
        lambda_stoi_list = [1.0]          # lambda has no effect on the STOI-only loss
    if settings["grid_metric"] == "best_val_total" and len(lambda_stoi_list) > 1:
        raise ValueError(
            "The total loss is on a different scale for every lambda_stoi, so it cannot be used "
            "to choose between them. Use best_val_ctc or best_val_stoi as selection_metric."
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = SpeechDataset(
        wav_dir=get(cfg, "dataset", "wav_dir"),
        txt_dir=get(cfg, "dataset", "txt_dir"),
        sample_rate=sample_rate,
        speech_dir=get(cfg, "dataset", "speech_dir"),
        rir_dir=get(cfg, "dataset", "rir_dir"),
    )
    train_set, val_set, test_set, split_info = grouped_train_val_test_split(
        dataset, val_split, test_split, split_seed,
        split_by=get(cfg, "training", "split_by", "utterance"),
        speaker_regex=get(cfg, "training", "speaker_regex", None),
    )
    save_split_manifests(
        dataset, split_info,
        os.path.join(output_root, "dataset_split.json"),
        os.path.join(output_root, "dataset_split.csv"),
    )
    counts = split_info["counts"]
    if "speakers" in counts:
        print(
            f"[SPLIT] by speaker: train {counts['train_speakers']}, val {counts['validation_speakers']}, "
            f"test {counts['test_speakers']} of {counts['speakers']} speakers",
            flush=True,
        )
    print(
        f"[SPLIT] train {counts['train_items']} items / {counts['train_utterances']} utterances, "
        f"val {counts['validation_items']} / {counts['validation_utterances']}, "
        f"test {counts['test_items']} / {counts['test_utterances']}",
        flush=True,
    )

    num_workers = int(get(cfg, "training", "num_workers", 0))
    loader_kwargs = dict(
        batch_size=int(get(cfg, "training", "batch_size", 4)),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=bool(get(cfg, "training", "pin_memory", False)),
        persistent_workers=bool(get(cfg, "training", "persistent_workers", False)) and num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, generator=torch.Generator().manual_seed(run_seed), **loader_kwargs
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    for loader, name in ((train_loader, "train"), (val_loader, "validation"), (test_loader, "test")):
        check_first_batch(loader, name, settings["acoustic_mode"])

    # ------------------------------------------------------------------
    # Generation only
    # ------------------------------------------------------------------
    if gen_checkpoint is not None:
        checkpoint = gen_checkpoint
        model = build_waveunet(cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])

        split = str(gen_cfg.get("split") or "test").lower()
        subsets = {"train": train_set, "val": val_set, "test": test_set}
        if split == "all":
            gen_set, indices = dataset, list(range(len(dataset)))
        elif split in subsets:
            gen_set, indices = subsets[split], list(subsets[split].indices)
        else:
            raise ValueError("generation.split must be train, val, test or all")

        # Unless the config says otherwise, generate in the condition the
        # checkpoint was trained in.
        def from_cfg_or_checkpoint(key, fallback):
            if gen_cfg.get(key) is not None:
                return gen_cfg[key]
            return checkpoint.get(key, fallback)

        gen_delay_ms = float(from_cfg_or_checkpoint("deployment_delay_ms", settings["delay_ms"]))
        gen_mcr_db = gen_cfg.get("target_mcr_db")
        if gen_mcr_db is None:
            gen_mcr_db = stored_mcr(checkpoint)
        gen_mcr_db = float(target_mcr_db_list[0] if gen_mcr_db is None else gen_mcr_db)
        if gen_delay_ms < 0:
            raise ValueError("generation.deployment_delay_ms must be >= 0.")
        acoustic_mode = str(gen_cfg.get("acoustic_mode") or "rir").lower()
        out_dir = gen_cfg.get("out_dir") or os.path.join(output_root, "generated")

        n_saved = generate_maskers(
            model,
            DataLoader(gen_set, shuffle=False, **loader_kwargs),
            dataset, indices, device, settings,
            output_dir=out_dir,
            target_mcr_db=gen_mcr_db,
            acoustic_mode=acoustic_mode,
            delay_samples=int(round(sample_rate * gen_delay_ms / 1000.0)),
            limit_files=int(gen_cfg.get("limit_files") or 0),
            save_masked=bool(gen_cfg.get("save_masked", True)),
            save_clean=bool(gen_cfg.get("save_clean", True)),
            use_original_names=bool(gen_cfg.get("use_original_names", True)),
            filename_tag=gen_cfg.get("filename_tag") or "",
        )
        print(f"[GEN] {n_saved} items of the {split} split written to {out_dir}", flush=True)
        return

    # ------------------------------------------------------------------
    # Training grid
    # ------------------------------------------------------------------
    fixed_val_name = sanitize_name(dataset.wav_files[val_set.indices[0]])
    grid = [(mcr, lam) for mcr in target_mcr_db_list for lam in lambda_stoi_list]
    results = []

    for run_index, (target_mcr_db, lambda_stoi) in enumerate(grid, start=1):
        print(f"[GRID] run {run_index}/{len(grid)}: MCR={target_mcr_db} dB, lambda_stoi={lambda_stoi}", flush=True)
        results.append(train_one_run(
            cfg, settings, train_loader, val_loader, device,
            target_mcr_db, lambda_stoi, output_root, fixed_val_name, run_seed,
        ))
        # Rewritten after every run, so an interrupted grid still leaves a table.
        write_json(os.path.join(output_root, "grid_search_results.json"), {"results": results})
        write_csv(results, os.path.join(output_root, "grid_search_results.csv"))
        write_csv(pareto_frontier(results), os.path.join(output_root, "validation_pareto_frontier.csv"))

    best_run = best_run_of(results, settings["grid_metric"])
    if best_run is None:
        raise RuntimeError(f"No run has a valid {settings['grid_metric']}.")
    write_json(os.path.join(output_root, "best_run.json"), best_run)

    # ------------------------------------------------------------------
    # Test set: evaluated once, with the run chosen on validation
    # ------------------------------------------------------------------
    print(f"[TEST] Evaluating {best_run['best_checkpoint']} on the test set", flush=True)
    model = build_waveunet(cfg, device)
    model.load_state_dict(load_checkpoint(best_run["best_checkpoint"], device)["model_state_dict"])
    asr = build_asr(cfg, device) if settings["use_asr"] else None
    stoi_criterion = NegSTOILoss(sample_rate=sample_rate).to(device)

    # A model trained without reverberation is also tested through the RIRs,
    # which is the condition it would face once deployed.
    test_modes = ["rir", "dry"] if settings["acoustic_mode"] == "dry" else ["rir"]
    test_metrics = {
        mode: evaluate(
            model, asr, stoi_criterion, test_loader, device, settings,
            best_run["target_mcr_db"], best_run["lambda_stoi"], acoustic_mode=mode,
        )
        for mode in test_modes
    }
    write_json(os.path.join(output_root, "final_test_metrics.json"), {
        "selected_run": best_run,
        "selection_metric": settings["grid_metric"],
        "test_metrics": test_metrics,
    })
    print(json.dumps({"best_run": best_run, "test_metrics": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
