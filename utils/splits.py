"""Train / validation / test split.

Every dry utterance exists in several simulated rooms (_v01, _v02, ...), so
items are never split one by one. Two groupings are available
(``training.split_by``):

    utterance : all room versions of a sentence land in the same subset, so
                no sentence heard in training appears in validation or test.
    speaker   : all sentences of a talker land in the same subset, so the
                validation and test talkers are never heard in training.
                This also keeps all room versions of a sentence together.

Speaker split, with LibriSpeech in mind: file names are
``<speaker>-<chapter>-<utterance>``, e.g. ``1234-5678-0001``, and the speaker
is the first number. Another naming scheme can be handled with
``training.speaker_regex`` (the first group of the regex is the speaker).
"""

import os
import re

import torch
from torch.utils.data import Subset

from utils.dataset import strip_version_suffix
from utils.run_logging import write_csv, write_json

SPLIT_MODES = ("utterance", "speaker")
LIBRISPEECH_SPEAKER_REGEX = r"^(\d+)-\d+-\d+$"


def base_utterance(filename):
    """``1234-5678-0001_v03.wav`` -> ``1234-5678-0001``"""
    return strip_version_suffix(os.path.splitext(os.path.basename(filename))[0])


def speaker_of(utterance, pattern):
    match = re.match(pattern, utterance)
    if match is None or not match.groups():
        raise ValueError(
            f"Cannot read a speaker from {utterance!r} with speaker_regex {pattern.pattern!r}. "
            "LibriSpeech names look like 1234-5678-0001; for other corpora set "
            "training.speaker_regex, or use split_by: utterance."
        )
    return match.group(1)


def _assign_groups(groups, val_split, test_split, split_seed, by_item_count):
    """Shuffle the groups and deal them out to validation, test and train.

    by_item_count=False: the fractions are taken over the number of groups,
    as in the original utterance split (every utterance has the same number
    of room versions, so this is also a fraction of the items).

    by_item_count=True: groups are added to validation, then to test, until
    each holds its fraction of the *items*. Used for speakers, who have very
    different numbers of utterances, so that taking 10 % of the speakers does
    not end up as 3 % or 20 % of the data.
    """
    names = sorted(groups)
    n = len(names)
    generator = torch.Generator().manual_seed(int(split_seed))
    shuffled = [names[i] for i in torch.randperm(n, generator=generator).tolist()]

    if not by_item_count:
        n_val = max(1, int(n * float(val_split)))
        n_test = max(1, int(n * float(test_split)))
        return {
            "validation": shuffled[:n_val],
            "test": shuffled[n_val:n_val + n_test],
            "train": shuffled[n_val + n_test:],
        }

    total_items = sum(len(groups[g]) for g in names)
    assigned = {"validation": [], "test": [], "train": []}
    remaining = list(shuffled)
    for subset, fraction in (("validation", val_split), ("test", test_split)):
        target = float(fraction) * total_items
        count = 0
        # at least one group, and stop once the target is reached
        while remaining and (not assigned[subset] or count < target):
            group = remaining.pop(0)
            assigned[subset].append(group)
            count += len(groups[group])
    assigned["train"] = remaining
    return assigned


def grouped_train_val_test_split(dataset, val_split, test_split, split_seed,
                                 split_by="utterance", speaker_regex=None):
    """Returns (train_set, val_set, test_set, split_info)."""
    split_by = str(split_by).lower()
    if split_by not in SPLIT_MODES:
        raise ValueError(f"training.split_by must be one of {SPLIT_MODES}, got {split_by!r}")

    # utterance -> dataset indices of its room versions
    utterance_items = {}
    for index, filename in enumerate(dataset.wav_files):
        utterance_items.setdefault(base_utterance(filename), []).append(index)

    if split_by == "utterance":
        groups = utterance_items
        group_of_utterance = {utt: utt for utt in utterance_items}
    else:
        pattern = re.compile(speaker_regex or LIBRISPEECH_SPEAKER_REGEX)
        group_of_utterance = {utt: speaker_of(utt, pattern) for utt in utterance_items}
        groups = {}
        for utt, indices in utterance_items.items():
            groups.setdefault(group_of_utterance[utt], []).extend(indices)

    n_groups = len(groups)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 different {split_by}s to make three subsets, got {n_groups}.")

    subsets = _assign_groups(groups, val_split, test_split, split_seed, by_item_count=(split_by == "speaker"))
    if not subsets["train"]:
        raise ValueError(
            f"No {split_by} left for training ({n_groups} {split_by}s); lower val_split / test_split."
        )

    # Indices are sorted so the manifest is easy to read; the training
    # DataLoader does its own shuffling.
    indices = {
        name: sorted(i for group in members for i in groups[group])
        for name, members in subsets.items()
    }

    assigned = indices["train"] + indices["validation"] + indices["test"]
    if len(assigned) != len(dataset) or len(set(assigned)) != len(dataset):
        raise RuntimeError("The split does not cover the dataset exactly once.")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(subsets[a]) & set(subsets[b]):
            raise RuntimeError(f"The same {split_by} appears in {a} and {b}.")

    utterances_in = {
        name: sorted({base_utterance(dataset.wav_files[i]) for i in indices[name]})
        for name in subsets
    }
    split_info = {
        "split_method": f"grouped_by_{split_by}",
        "split_seed": int(split_seed),
        "requested_fractions": {
            "validation": float(val_split),
            "test": float(test_split),
            "train": float(1.0 - val_split - test_split),
        },
        "counts": {
            "utterances": len(utterance_items),
            "items": len(dataset),
            **{f"{name}_utterances": len(utterances_in[name]) for name in ("train", "validation", "test")},
            **{f"{name}_items": len(indices[name]) for name in ("train", "validation", "test")},
        },
        "splits": {
            name: {"base_utterances": utterances_in[name], "indices": indices[name]}
            for name in ("train", "validation", "test")
        },
    }
    if split_by == "speaker":
        split_info["speaker_regex"] = pattern.pattern
        split_info["counts"]["speakers"] = n_groups
        for name in ("train", "validation", "test"):
            split_info["counts"][f"{name}_speakers"] = len(subsets[name])
            split_info["splits"][name]["speakers"] = sorted(subsets[name])
        split_info["speaker_of_utterance"] = group_of_utterance

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

    speaker_of_utterance = split_info.get("speaker_of_utterance", {})
    rows = []
    for index, filename in enumerate(dataset.wav_files):
        utterance = base_utterance(filename)
        row = {
            "dataset_index": index,
            "split": subset_of[index],
            "base_utterance": utterance,
            "versioned_filename": filename,
        }
        if speaker_of_utterance:
            row["speaker"] = speaker_of_utterance[utterance]
        rows.append(row)
    write_csv(rows, csv_path)
