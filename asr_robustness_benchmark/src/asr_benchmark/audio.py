
"""Контролируемые преобразования аудиосигнала для стресс-тестирования ASR."""
import numpy as np
from scipy.signal import butter, sosfilt


def _safe_peak_normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def apply_condition(audio: np.ndarray, sample_rate: int, condition: str,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Возвращает копию аудио после выбранного преобразования."""
    x = np.asarray(audio, dtype=np.float32).copy()
    if x.ndim > 1:
        x = np.mean(x, axis=-1, dtype=np.float32)
    if condition == "clean":
        return _safe_peak_normalize(x)
    if condition == "noise":
        rng = rng or np.random.default_rng(42)
        # SNR около 20 dB: слабый, но измеримый шум.
        signal_rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        noise_rms = signal_rms / (10 ** (20 / 20))
        noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
        return _safe_peak_normalize(x + noise)
    if condition == "volume":
        return _safe_peak_normalize(x * 0.35)
    if condition == "bandlimit":
        if sample_rate <= 4000:
            return _safe_peak_normalize(x)
        high_cut = min(3400.0, sample_rate * 0.45)
        low_cut = min(300.0, high_cut * 0.25)
        sos = butter(4, [low_cut, high_cut], btype="bandpass",
                     fs=sample_rate, output="sos")
        return _safe_peak_normalize(sosfilt(sos, x).astype(np.float32))
    if condition == "resample":
        # Имитация узкополосного канала: 16 кГц -> 8 кГц -> 16 кГц.
        from scipy.signal import resample_poly
        from math import gcd
        target_rate = 8000
        common = gcd(sample_rate, target_rate)
        down = sample_rate // common
        up = target_rate // common
        reduced = resample_poly(x, up, down)
        common_back = gcd(target_rate, sample_rate)
        up_back = sample_rate // common_back
        down_back = target_rate // common_back
        restored = resample_poly(reduced, up_back, down_back)
        if len(restored) > len(x):
            restored = restored[:len(x)]
        elif len(restored) < len(x):
            restored = np.pad(restored, (0, len(x) - len(restored)))
        return _safe_peak_normalize(restored.astype(np.float32))
    raise ValueError(f"Неизвестное условие: {condition}")
