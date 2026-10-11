"""End-to-end check on synthetic data, on CPU, in a couple of minutes.

    python tests/smoke_test.py

It makes a few seconds of fake "speech", simulates small rooms for it,
trains a tiny masker for one epoch with the STOI loss (so NeMo is not
needed), and generates maskers from the checkpoint. Nothing about the
result is meaningful; the point is that every stage runs and writes the
files the next stage expects.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000


def run(*args):
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def fake_speech(rng, seconds):
    """Noise with a slow amplitude modulation, roughly speech-like in level."""
    t = np.arange(int(seconds * SR)) / SR
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28)))
    return (0.1 * envelope * rng.standard_normal(t.size)).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        dry_wav, dry_txt = os.path.join(tmp, "dry", "wav"), os.path.join(tmp, "dry", "txt")
        simulated = os.path.join(tmp, "simulated")
        os.makedirs(dry_wav)
        os.makedirs(dry_txt)

        for i in range(6):
            sf.write(os.path.join(dry_wav, f"utt{i:02d}.wav"), fake_speech(rng, 1.5), SR)
            with open(os.path.join(dry_txt, f"utt{i:02d}.txt"), "w") as f:
                f.write("hello world")
        babble = os.path.join(tmp, "babble.wav")
        sf.write(babble, fake_speech(rng, 10.0), SR)

        # 1) rooms
        sim_cfg = {
            "input_path": dry_wav,
            "output_dir": simulated,
            "n_versions_per_input": 2,
            "seed": 1,
            "room": {"use_random_dims": True, "W_range": [4, 6], "L_range": [4, 6], "H_range": [2.5, 3],
                     "max_order": 2},
            "speaker": {"distance_range_m": [1.0, 1.5]},
            "babble": {"enabled": True, "babble_paths": [babble], "n_babble": 2, "snr_db": 0.0},
            "ngm": {"enabled": True, "mode": "table_listener", "snr_db": 0.0},
            "output": {"write_3d_viz": False},
        }
        sim_yaml = os.path.join(tmp, "simulate.yaml")
        with open(sim_yaml, "w") as f:
            yaml.safe_dump(sim_cfg, f)
        run("simulate_rooms.py", "--config", sim_yaml)

        n_rirs = len([f for f in os.listdir(simulated) if f.endswith(".npz")])
        assert n_rirs == 6 * 2 * 3, f"expected 36 RIR files, found {n_rirs}"

        # 2) training, once through the rooms and once dry
        train_cfg = {
            "dataset": {"wav_dir": simulated, "rir_dir": simulated, "speech_dir": dry_wav,
                        "txt_dir": dry_txt, "sample_rate": SR},
            "model": {"channels": [8, 16, 32], "kernel_size": 5, "output_samples": 8000},
            "training": {"device": "cpu", "loss": "stoi", "acoustic_mode": "rir",
                         "target_mcr_db_list": [5], "deployment_delay_ms": 2.0,
                         "epochs": 1, "batch_size": 2, "val_split": 0.2, "test_split": 0.2,
                         "save_audio_each_epoch": True, "tb_log_audio": True},
        }
        for mode in ("rir", "dry"):
            train_cfg["training"]["acoustic_mode"] = mode
            train_yaml = os.path.join(tmp, f"train_{mode}.yaml")
            with open(train_yaml, "w") as f:
                yaml.safe_dump(train_cfg, f)
            run_dir = os.path.join(tmp, "runs", mode)
            run("train.py", "--config", train_yaml, "--output-dir", run_dir)
            assert os.path.exists(os.path.join(run_dir, "final_test_metrics.json"))

        # 3) maskers from the checkpoint trained through the rooms
        checkpoint = os.path.join(tmp, "runs", "rir", "mcr_5dB_delay_2ms", "checkpoints", "best.pt")
        generated = os.path.join(tmp, "generated")
        train_cfg["training"]["acoustic_mode"] = "rir"
        train_cfg["generation"] = {"only": True, "ckpt": checkpoint, "out_dir": generated,
                                   "split": "test", "filename_tag": None}
        gen_yaml = os.path.join(tmp, "generate.yaml")
        with open(gen_yaml, "w") as f:
            yaml.safe_dump(train_cfg, f)
        run("train.py", "--config", gen_yaml, "--output-dir", os.path.join(tmp, "runs", "rir"))

        wavs = sorted(os.listdir(generated))
        assert wavs and not any("None" in w for w in wavs), wavs
        print(f"\nGenerated {len(wavs)} files, e.g. {wavs[0]}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
