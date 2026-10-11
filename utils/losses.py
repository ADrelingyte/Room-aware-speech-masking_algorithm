"""Training objectives and the metrics logged next to them.

Two objectives are available (``training.loss`` in the config):

    ctc_stoi : minimise  -CTC + lambda_stoi * STOI
    stoi     : minimise  STOI

CTC is the loss of a frozen ASR model on the masked signal, divided by the
number of target tokens, so maximising it makes the recogniser fail. STOI is
computed between the masked signal and the unmasked speech at the listener,
so minimising it lowers the predicted intelligibility.
"""

import math

import torch

from utils.acoustics import length_mask, mcr_db_per_example, warn_if_mcr_mismatch

LOSS_TYPES = ("ctc_stoi", "stoi")


def normalized_ctc_per_example(asr, listener_signal, valid_lengths, texts):
    """CTC loss per example divided by its number of target tokens. [B]

    Returns None when every transcript in the batch is empty.
    """
    ctc = asr(listener_signal, valid_lengths, texts, reduction="none")
    if ctc is None:
        return None

    n_tokens = [len(asr.asr.tokenizer.text_to_ids(t.strip())) for t in texts]
    n_tokens = torch.tensor(n_tokens, dtype=ctc.dtype, device=ctc.device)
    return ctc / torch.clamp(n_tokens, min=1.0)


def masked_mse_per_example(listener_signal, speech_at_listener, valid_lengths):
    """Mean squared masker amplitude on the valid samples. Diagnostic only."""
    mask = length_mask(valid_lengths, listener_signal.shape[-1])
    squared_error = (listener_signal - speech_at_listener) ** 2
    return torch.sum(squared_error * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-8)


def batch_objective(batch, asr, stoi_criterion, loss_type, lambda_stoi, target_mcr_db, where=""):
    """Compute the loss and the logged metrics for one AcousticBatch.

    `asr` may be None with loss_type="stoi"; the CTC metric is then NaN.
    Returns None when the CTC loss is needed but no transcript is available.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"training.loss must be one of {LOSS_TYPES}, got {loss_type!r}")

    ctc_per_example = None
    if asr is not None:
        ctc_per_example = normalized_ctc_per_example(
            asr, batch.listener_signal, batch.valid_lengths, batch.texts
        )
    if loss_type == "ctc_stoi" and ctc_per_example is None:
        return None

    # torch_stoi returns the negative STOI of each example.
    stoi = -stoi_criterion(batch.listener_signal, batch.speech_at_listener).mean()

    if ctc_per_example is not None:
        ctc_mean = ctc_per_example.mean()
    else:
        ctc_mean = torch.tensor(float("nan"), device=stoi.device)

    if loss_type == "ctc_stoi":
        # If STOI ever comes out NaN, keep training on the CTC term for this
        # batch instead of propagating NaN gradients into the network.
        loss = -ctc_mean + float(lambda_stoi) * torch.nan_to_num(stoi, nan=0.0)
    else:
        loss = stoi

    mcr = mcr_db_per_example(batch.speech_at_listener, batch.masker_at_listener, batch.valid_lengths)
    warn_if_mcr_mismatch(mcr, target_mcr_db, where=where)

    return {
        "loss": loss,
        "ctc_per_example": ctc_per_example,
        "ctc_mean": ctc_mean,
        "stoi": stoi,
        "mse_per_example": masked_mse_per_example(
            batch.listener_signal, batch.speech_at_listener, batch.valid_lengths
        ),
        "mcr_per_example": mcr,
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def metric_should_maximize(metric_name):
    """We want a high CTC (ASR fails) and a low STOI / total loss."""
    name = metric_name.lower()
    if "ctc" in name:
        return True
    if "stoi" in name or "total" in name or "mse" in name:
        return False
    raise ValueError(f"Do not know whether {metric_name!r} should be maximised or minimised.")


def is_better(candidate, best_so_far, metric_name):
    if candidate is None or math.isnan(candidate):
        return False
    if best_so_far is None or math.isnan(best_so_far):
        return True
    if metric_should_maximize(metric_name):
        return candidate > best_so_far
    return candidate < best_so_far
