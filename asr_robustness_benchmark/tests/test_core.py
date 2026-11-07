
import numpy as np
from asr_benchmark.audio import apply_condition
from asr_benchmark.metrics import calculate_wer, calculate_cer, calculate_mtra


def test_clean_keeps_shape_and_range():
    x = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    y = apply_condition(x, 16000, "clean")
    assert y.shape == x.shape
    assert np.max(np.abs(y)) <= 1.0


def test_noise_is_reproducible():
    x = np.ones(1000, dtype=np.float32) * 0.1
    a = apply_condition(x, 16000, "noise", np.random.default_rng(7))
    b = apply_condition(x, 16000, "noise", np.random.default_rng(7))
    assert np.allclose(a, b)


def test_metrics_identical_text():
    assert calculate_wer("острый нефритический синдром", "острый нефритический синдром") == 0
    assert calculate_cer("туберкулёз", "туберкулёз") == 0


def test_mtra():
    score = calculate_mtra(
        "диагностика туберкулёза",
        "диагностика туберкулёза",
        ["туберкулёза", "GeneXpert"]
    )
    assert score == 1.0


def test_mtra_nan_without_terms_in_reference():
    assert np.isnan(calculate_mtra("обычная речь", "обычная речь", ["туберкулёз"]))
