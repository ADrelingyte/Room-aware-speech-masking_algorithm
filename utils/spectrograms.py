"""Spectrogram images for TensorBoard."""
import numpy as np
import tensorflow as tf


def tf_spectrogram_db_image(
    audio_1d_np: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    top_db: float = 80.0,
) -> tf.Tensor:
    """
    Returns an image tensor [1, H, W, 1] (float32, 0..1) suitable for tf.summary.image.
    audio_1d_np: shape (T,), float32 recommended, in [-1, 1]
    """
    x = tf.convert_to_tensor(audio_1d_np, dtype=tf.float32)
    x = tf.clip_by_value(x, -1.0, 1.0)

    stft = tf.signal.stft(
        x,
        frame_length=win_length,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )  # [frames, fft_bins]

    mag = tf.abs(stft)  # [frames, fft_bins]
    # Convert to dB (log10)
    mag_db = 20.0 * tf.math.log(mag + 1e-10) / tf.math.log(tf.constant(10.0, tf.float32))

    # Normalize to [0, 1] using top_db dynamic range
    max_db = tf.reduce_max(mag_db)
    mag_db = tf.maximum(mag_db, max_db - top_db)
    img = (mag_db - (max_db - top_db)) / top_db  # [0..1]

    # Make it image-like: [H, W] where H=freq, W=time
    img = tf.transpose(img, perm=[1, 0])  # [fft_bins, frames]
    img = tf.expand_dims(img, axis=-1)    # [H, W, 1]
    img = tf.expand_dims(img, axis=0)     # [1, H, W, 1]
    return img


def tf_log_spectrograms(
    tb_writer,
    tag_prefix: str,
    clean_1d_np: np.ndarray,
    pred_1d_np: np.ndarray,
    sr: int,
    step: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    top_db: float = 80.0,
):
    """Log the dB spectrograms of the clean and of the masked signal."""
    clean_1d_np = clean_1d_np.astype(np.float32, copy=False)
    pred_1d_np = pred_1d_np.astype(np.float32, copy=False)

    clean_img = tf_spectrogram_db_image(clean_1d_np, sr, n_fft, hop_length, win_length, top_db)
    pred_img  = tf_spectrogram_db_image(pred_1d_np,  sr, n_fft, hop_length, win_length, top_db)

    with tb_writer.as_default():
        tf.summary.image(f"{tag_prefix}/clean_db", clean_img, step=step)
        tf.summary.image(f"{tag_prefix}/pred_db", pred_img, step=step)
