"""Frozen NeMo CTC recogniser used as a differentiable loss.

The waveform goes through our own STFT + mel filterbank and then through the
pretrained NeMo encoder and decoder. NeMo's built-in preprocessor cannot be
used for this: it runs under ``torch.no_grad()``, so no gradient would reach
the masker network. The front-end below is therefore the only one supported.

Note that it is a plain log-mel spectrogram (no pre-emphasis, no per-feature
normalisation), which is not exactly what the NeMo model saw during its own
training.
"""

import numpy as np
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr

DEFAULT_ASR_MODEL = "stt_en_fastconformer_ctc_xxlarge"


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    if fmax is None:
        fmax = sr / 2.0

    n_freqs = n_fft // 2 + 1
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_edges) / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]

        # At low frequencies two edges can fall in the same FFT bin.
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)

        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)

    return fb


class FrozenASR(nn.Module):
    """waveform -> log-mel -> NeMo encoder/decoder -> CTC loss per example.

    The ASR weights are frozen; gradients only flow back to the waveform.
    """

    def __init__(
        self,
        device,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        fmin=0.0,
        fmax=None,
        model_name=DEFAULT_ASR_MODEL,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)

        print(f"[FrozenASR] Loading {model_name}", flush=True)
        self.asr = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name).to(self.device)
        self.asr.eval()
        for p in self.asr.parameters():
            p.requires_grad = False
        if getattr(self.asr, "spec_augmentation", None) is not None:
            self.asr.spec_augmentation = None

        self.blank_id = self._blank_id()
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction="none", zero_infinity=True)

        fb = mel_filterbank(int(sample_rate), self.n_fft, int(n_mels), fmin=fmin, fmax=fmax)
        self.register_buffer("mel_fb", torch.from_numpy(fb), persistent=False)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def _blank_id(self):
        # NeMo puts the CTC blank after the last vocabulary token.
        decoder = self.asr.decoder
        if hasattr(decoder, "num_classes_with_blank"):
            return int(decoder.num_classes_with_blank) - 1
        return len(decoder.vocabulary)

    def log_mel(self, wav):
        """[B, T] waveform -> [B, n_mels, frames] log-mel power spectrogram."""
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.real.pow(2) + stft.imag.pow(2)
        mel = torch.einsum("mf,bft->bmt", self.mel_fb.to(device=wav.device, dtype=wav.dtype), power)
        return torch.log(torch.clamp(mel, min=1e-10))

    def _frame_lengths(self, lengths_samples):
        # Number of STFT frames with center=True.
        frames = torch.floor_divide(lengths_samples.long(), self.hop_length) + 1
        return torch.clamp(frames, min=1)

    def _tokenize(self, texts):
        token_lists = [self.asr.tokenizer.text_to_ids(t.strip()) for t in texts]
        target_lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long, device=self.device)
        if int(target_lengths.max()) == 0:
            return None, None
        targets = torch.tensor(
            [tok for tokens in token_lists for tok in tokens], dtype=torch.long, device=self.device
        )
        return targets, target_lengths

    @staticmethod
    def _to_bt(waveforms):
        if waveforms.dim() == 3 and waveforms.size(1) == 1:
            waveforms = waveforms[:, 0, :]
        if waveforms.dim() != 2:
            raise ValueError(f"Expected waveforms as [B,T] or [B,1,T], got {tuple(waveforms.shape)}")
        return waveforms

    def forward(self, waveforms, lengths, texts, reduction="none"):
        """
        waveforms : [B, T] or [B, 1, T]
        lengths   : [B] valid samples per waveform
        texts     : list of B transcripts

        Returns the CTC loss per example [B] (or its mean / sum), or None if
        all transcripts are empty.
        """
        waveforms = self._to_bt(waveforms).to(self.device, dtype=torch.float32)
        lengths = lengths.to(self.device, dtype=torch.long)
        if len(texts) != waveforms.shape[0]:
            raise ValueError(f"Got {len(texts)} transcripts for a batch of {waveforms.shape[0]}.")

        targets, target_lengths = self._tokenize(texts)
        if targets is None:
            return None

        feats = self.log_mel(waveforms)
        enc_out, enc_len = self.asr.encoder(audio_signal=feats, length=self._frame_lengths(lengths))
        log_probs = self.asr.decoder(encoder_output=enc_out)            # [B, T', C]

        log_probs = log_probs.permute(1, 0, 2).contiguous()             # CTCLoss wants [T', B, C]
        input_lengths = torch.clamp(enc_len.long(), min=1, max=log_probs.shape[0])
        ctc = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        if reduction == "none":
            return ctc
        if reduction == "mean":
            return ctc.mean()
        if reduction == "sum":
            return ctc.sum()
        raise ValueError(f"Unknown reduction {reduction!r}")

    @torch.no_grad()
    def greedy_decode(self, waveforms):
        """Transcribe with NeMo's own pipeline. For WER checks and debugging,
        not used in training."""
        waveforms = self._to_bt(waveforms).to(self.device, dtype=torch.float32)
        lengths = torch.full((waveforms.shape[0],), waveforms.shape[1], device=self.device, dtype=torch.long)
        _, _, predictions = self.asr(input_signal=waveforms, input_signal_length=lengths)
        return self.asr.decoding.ctc_decoder_predictions_tensor(predictions)
