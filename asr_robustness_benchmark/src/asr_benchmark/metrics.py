
"""Метрики качества распознавания."""
import re
import unicodedata
import numpy as np
from jiwer import wer as jiwer_wer, cer as jiwer_cer


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer_wer(ref, hyp))


def calculate_cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer_cer(ref, hyp))


def calculate_mtra(reference: str, hypothesis: str, terms: list[str]) -> float:
    """Доля терминальных вхождений из reference, найденных в hypothesis."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    normalized_terms = [normalize_text(t) for t in terms if normalize_text(t)]
    present = [term for term in normalized_terms if term in ref]
    if not present:
        return float("nan")
    found = sum(term in hyp for term in present)
    return found / len(present)


def safe_mean(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
