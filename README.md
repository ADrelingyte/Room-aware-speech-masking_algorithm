# Neural speech masker

Code for training a Wave-U-Net that generates an additive masking signal to
reduce the intelligibility of a nearby talker for a listener in the same room.

The network listens to the talker through a microphone on a small device (the
NGM, placed on the listener's table) and drives the loudspeaker of that
device. The masker is trained through simulated room acoustics so that, at the
listener's position and at a fixed mixture-to-concealer ratio (MCR), speech becomes hard
to understand for an ASR model (CTC loss) and according to an intelligibility
metric (STOI).

```
dry speech --RIR1--> NGM microphone --Wave-U-Net--> masker --delay--RIR3--> listener
dry speech --RIR2-------------------------------------------------------> listener
```

- RIR1: talker to NGM microphone
- RIR2: talker to listener
- RIR3: NGM loudspeaker to listener

## Installation

Python 3.10 or newer.

```bash
git clone <this repository>
cd <this repository>
python -m venv .venv && source .venv/bin/activate
# install PyTorch for your CUDA version first, see https://pytorch.org
pip install -r requirements.txt
```

`nemo_toolkit[asr]` is only needed for the CTC loss. The first run downloads
the pretrained `stt_en_fastconformer_ctc_xxlarge` model.

To check the installation without any data or GPU (about two minutes on CPU):

```bash
python tests/smoke_test.py
```

## Data layout

You provide dry (anechoic) speech at 16 kHz with one transcript per file. We
used a 10-hour subset of LibriSpeech.

```
dry/
  wav/1234-5678-0001.wav
  txt/1234-5678-0001.txt        # plain text transcript
```

Step 1 below turns this into the simulated set:

```
simulated/
  1234-5678-0001_v01.wav                        # 4-channel mixture for that room
  1234-5678-0001_v01_rir1_spk_to_ngm.npz
  1234-5678-0001_v01_rir2_spk_to_listener.npz
  1234-5678-0001_v01_rir3_ngm_to_listener.npz
  1234-5678-0001_v01.yaml                       # room geometry and levels
  1234-5678-0001_v02.wav
  ...
```

Each `.npz` holds `h` (`[n_mics, n_sources, max_len]`), `lengths` and `fs`.
Training uses the first microphone (left ear) and the first NGM.

### Babble files

The simulation can add background babble if necessary for your experiments (other people talking) to the
rendered mixtures. A new repository will be added eventually providing a .exe file for room simulation for easier use. 

```yaml
babble:
  enabled: true
  babble_paths:
    - "/path/to/office_noise_1.wav"
    - "/path/to/office_noise_2.wav"
  n_babble: 8
  snr_db_range: [-6.0, 6.0]
```

## Usage

### 1. Simulate the rooms

```bash
cp configs/simulate_example.yaml configs/local_simulate.yaml   # then set the paths
python simulate_rooms.py --config configs/local_simulate.yaml
```

Existing files are skipped, so an interrupted job can be restarted. Items that
fail are printed and listed in `simulation_failures.csv` in the output folder,
and the script then exits with a non-zero status.

### 2. Train

```bash
cp configs/train_example.yaml configs/local_train.yaml         # then set the paths
python train.py --config configs/local_train.yaml --output-dir runs/ctc_stoi_rir
```

The experiment is selected in the config:

| `training.loss` | `training.acoustic_mode` | Experiment |
|---|---|---|
| `ctc_stoi` | `rir` | main system: minimise −CTC + λ·STOI through the room responses |
| `stoi` | `rir` | STOI-only loss |
| `stoi` | `dry` | ablation: trained without room acoustics, tested with them |

One model is trained for every value in `target_mcr_db_list` (and in
`lambda_stoi_list` for `ctc_stoi`). The output folder then contains:

```
runs/ctc_stoi_rir/
  dataset_split.json / .csv        which utterance went to train / validation / test
  grid_search_results.csv          one line per run
  best_run.json                    run selected on the validation set
  final_test_metrics.json          that run evaluated once on the test set
  rir_warnings.jsonl               examples skipped because of a missing RIR file
  mcr_5dB_lambda_50_delay_0ms/
    checkpoints/epoch_XXX.pt, best.pt
    logs/metrics.json, steps.jsonl, tensorboard/
    samples/                       speech, masked speech and masker for one validation item
```

To continue an interrupted run, set `training.resume_checkpoint` to one of its
`epoch_XXX.pt` files and keep the same `--output-dir`. `training.epochs` is
the final epoch number, not a number of additional epochs.

### 3. Generate maskers from a checkpoint

In the same config set

```yaml
generation:
  only: true
  ckpt: "runs/ctc_stoi_rir/mcr_5dB_lambda_50_delay_0ms/checkpoints/best.pt"
  out_dir: "generated/ctc_stoi_rir"
  split: "test"
```

and run the same command:

```bash
python train.py --config configs/local_train.yaml --output-dir runs/ctc_stoi_rir
```

For every item of the split this writes `*_masker_at_listener.wav`,
`*_speech_at_listener.wav` and `*_masked_listener.wav`. Use the same
`val_split`, `test_split` and `split_by` as in training; the split seed is
read from the checkpoint, so the split is the same.


## Seeds

No seed is fixed in the code or in the example configs. Every seed left
empty (`training.split_seed`, `training.run_seed`, `rooms.seed`,
`baselines.seed`, and `seed` in the simulation config) is drawn at random at
start-up, printed as `[SEED] ... (random)`, and saved in `seeds.json` in the
output folder. Checkpoints also store the split and rooms seeds.

To repeat a run exactly, copy the values from `seeds.json` into the config.
Generation, `evaluate_baselines.py` and resumed runs take the split (and
rooms) seed from the checkpoint or from `seeds.json` when the config leaves
it empty, so they always work on the subsets of the training run.

## Processing delay

A real device needs some time between picking up the speech and playing the
masker. `training.deployment_delay_ms` puts that latency into the simulated
chain, between the network output and the NGM loudspeaker:

```
NGM microphone --Wave-U-Net--> masker --delay--> NGM loudspeaker --RIR3--> listener
```

Only the masker is delayed; the talker's own path to the listener (RIR2) is
not. The value is given in milliseconds and rounded to whole samples, so at
16 kHz `14.28` becomes 228 samples. The masker is shifted by that many samples
and the part pushed past the end of the window is dropped. `0.0` means no
delay.

**Training with a delay.** Set one value per run:

```yaml
training:
  deployment_delay_ms: 14.28
```

The delay is used in training, validation and the final test, is part of the
run folder name (`..._delay_14p28ms`) and is stored in every checkpoint.
Resuming a checkpoint with a different delay is refused.

**Comparing several delays.** The delay is a single value, not a list. Train
once per value with its own output folder, otherwise `best_run.json` and
`final_test_metrics.json` of one delay overwrite those of the other:

```bash
python train.py --config configs/delay_0ms.yaml  --output-dir runs/ctc_stoi_rir_delay_0ms
python train.py --config configs/delay_14ms.yaml --output-dir runs/ctc_stoi_rir_delay_14ms
```

**Generating with a delay.** By default the maskers are generated with the
delay stored in the checkpoint. To test a model at a delay it was not trained
for, override it:

```yaml
generation:
  only: true
  ckpt: "runs/ctc_stoi_rir_delay_0ms/mcr_5dB_lambda_50_delay_0ms/checkpoints/best.pt"
  deployment_delay_ms: 14.28
```

**Delay and look-ahead are separate.** The Wave-U-Net sees a short stretch of
future input for every output sample: half of (input size − output size),
which is 380 samples or 23.75 ms with the default model. `deployment_delay_ms`
does not remove or account for that look-ahead; it only adds latency after the
network.


## Repository

```
train.py               training, model selection, test, masker generation
evaluate_baselines.py  learned masker vs SSN / white noise on the test set
simulate_rooms.py      room simulation and RIR export (pyroomacoustics)
configs/               example configurations
model/                 Wave-U-Net
utils/
  dataset.py           dry speech + transcript + RIR paths
  acoustics.py         RIR convolution, delay, MCR scaling, full forward pass
  losses.py            CTC / STOI objectives and model-selection helpers
  frozen_asr.py        frozen NeMo CTC model with a differentiable front-end
  splits.py            train / validation / test split by speaker or utterance
  run_logging.py       json / csv / audio outputs
  spectrograms.py      TensorBoard spectrogram images
tests/smoke_test.py    end-to-end run on synthetic data
```

## Acknowledgements and licence

The Wave-U-Net implementation in `model/` is adapted from
[Wave-U-Net-Pytorch](https://github.com/f90/Wave-U-Net-Pytorch) by Daniel
Stoller (MIT licence). This repository is released under the MIT licence, see
`LICENSE`.
