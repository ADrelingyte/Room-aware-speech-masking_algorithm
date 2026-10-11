"""Train / validation / test split.

Every dry utterance exists in several simulated rooms (_v01, _v02, ...).
The split is done on the dry utterances, so all room versions of a sentence
land in the same subset and no sentence heard in training appears in
validation or test.
"""

import os

import torch
from torch.utils.data import Subset

from utils.dataset import strip_version_suffix
from utils.run_logging import write_csv, write_json


def base_utterance(filename):
    """``1234-5678-0001_v03.wav`` -> ``1234-5678-0001``"""
    return strip_version_suffix(os.path.splitext(os.path.basename(filename))[0])


def grouped_train_val_test_split(dataset, val_split, test_split, split_seed):
    """Returns (train_set, val_set, test_set, split_info)."""
    groups = {}
    for index, filename in enumerate(dataset.wav_files):
        groups.setdefault(base_utterance(filename), []).append(index)

    utterances = sorted(groups)
    n = len(utterances)
    if n < 3:
        raise ValueError(f"Need at least 3 different utterances to make three subsets, got {n}.")

    generator = torch.Generator().manual_seed(int(split_seed))
    order = torch.randperm(n, generator=generator).tolist()
    shuffled = [utterances[i] for i in order]

    n_val = max(1, int(n * float(val_split)))
    n_test = max(1, int(n * float(test_split)))
    if n - n_val - n_test <= 0:
        raise ValueError(f"No utterance left for training ({n} utterances, {n_val} val, {n_test} test).")

    subsets = {
        "validation": shuffled[:n_val],
        "test": shuffled[n_val:n_val + n_test],
        "train": shuffled[n_val + n_test:],
    }
    # Indices are sorted so the manifest is easy to read; the training
    # DataLoader does its own shuffling.
    indices = {
        name: sorted(i for utt in utts for i in groups[utt])
        for name, utts in subsets.items()
    }

    assigned = indices["train"] + indices["validation"] + indices["test"]
    if len(assigned) != len(dataset) or len(set(assigned)) != len(dataset):
        raise RuntimeError("The split does not cover the dataset exactly once.")

    split_info = {
        "split_method": "grouped_by_dry_utterance",
        "split_seed": int(split_seed),
        "requested_fractions": {
            "validation": float(val_split),
            "test": float(test_split),
            "train": float(1.0 - val_split - test_split),
        },
        "counts": {
            "utterances": n,
            "items": len(dataset),
            **{f"{name}_utterances": len(subsets[name]) for name in ("train", "validation", "test")},
            **{f"{name}_items": len(indices[name]) for name in ("train", "validation", "test")},
        },
        "splits": {
            name: {"base_utterances": sorted(subsets[name]), "indices": indices[name]}
            for name in ("train", "validation", "test")
        },
    }

    return (
        Subset(dataset, indices["train"]),
        Subset(dataset, indices["validation"]),
        Subset(dataset, indices["test"]),
        split_info,
    )


def save_split_manifests(dataset, split_info, json_path, csv_path):
    """Write the split as json (full description) and csv (one row per item)."""
    write_json(json_path, split_info)

    subset_of = {}
    for name, info in split_info["splits"].items():
        for index in info["indices"]:
            subset_of[index] = name

    rows = [
        {
            "dataset_index": index,
            "split": subset_of[index],
            "base_utterance": base_utterance(filename),
            "versioned_filename": filename,
        }
        for index, filename in enumerate(dataset.wav_files)
    ]
    write_csv(rows, csv_path)
