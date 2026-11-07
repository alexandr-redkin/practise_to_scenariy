
"""Загрузка тестовых подмножеств Google FLEURS."""
from datasets import load_dataset


def load_fleurs_test(language: str, limit: int = 10):
    dataset = load_dataset("google/fleurs", language, split="test")
    if limit and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset
