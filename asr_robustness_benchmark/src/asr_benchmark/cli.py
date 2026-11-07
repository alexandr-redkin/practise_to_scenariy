
"""Командная строка для запуска базового ASR robustness benchmark."""
import argparse
import json
import os
import platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import transformers
import datasets
from transformers import pipeline
from tqdm.auto import tqdm

from .audio import apply_condition
from .dataset import load_fleurs_test
from .metrics import calculate_wer, calculate_cer, calculate_mtra, safe_mean

CONDITIONS = ["clean", "noise", "volume", "bandlimit", "resample"]
LANGUAGE_NAMES = {"kk_kz": "kazakh", "ru_ru": "russian"}


def load_terms(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("terms", [])


def transcribe(asr, audio: np.ndarray, sample_rate: int, language: str) -> str:
    lang_name = LANGUAGE_NAMES[language]
    result = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        generate_kwargs={"task": "transcribe", "language": lang_name},
    )
    return result.get("text", "").strip()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    terms = load_terms(Path(args.glossary))
    device = 0 if torch.cuda.is_available() else -1

    print(f"Устройство: {'CUDA' if device == 0 else 'CPU'}")
    print("Загрузка ASR-моделей...")
    recognizers = {}
    for model_id in args.models:
        recognizers[model_id] = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )

    rows = []
    rng = np.random.default_rng(args.seed)

    for language in args.languages:
        print(f"\nЗагрузка FLEURS: {language}")
        dataset = load_fleurs_test(language, args.samples_per_language)
        for index, item in enumerate(tqdm(dataset, desc=language)):
            audio_item = item["audio"]
            audio = np.asarray(audio_item["array"], dtype=np.float32)
            sample_rate = int(audio_item["sampling_rate"])
            reference = item["transcription"]
            sample_id = str(item.get("id", index))

            for condition in CONDITIONS:
                transformed = apply_condition(audio, sample_rate, condition, rng)
                if args.save_audio:
                    audio_dir = output / "audio" / language / condition
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(audio_dir / f"{sample_id}.wav", transformed, sample_rate)

                for model_id, recognizer in recognizers.items():
                    try:
                        hypothesis = transcribe(
                            recognizer, transformed, sample_rate, language
                        )
                        error = None
                    except Exception as exc:
                        hypothesis = ""
                        error = f"{type(exc).__name__}: {exc}"

                    rows.append({
                        "language": language,
                        "sample_id": sample_id,
                        "model": model_id,
                        "condition": condition,
                        "sample_rate": sample_rate,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "wer": calculate_wer(reference, hypothesis),
                        "cer": calculate_cer(reference, hypothesis),
                        "mtra": calculate_mtra(reference, hypothesis, terms),
                        "error": error,
                    })

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")

    clean = predictions[predictions["condition"] == "clean"][
        ["language", "sample_id", "model", "wer"]
    ].rename(columns={"wer": "clean_wer"})
    merged = predictions.merge(clean, on=["language", "sample_id", "model"], how="left")
    merged["wer_delta"] = merged["wer"] - merged["clean_wer"]

    summary = (
        merged.groupby(["language", "model", "condition"], dropna=False)
        .agg(
            samples=("sample_id", "nunique"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_mtra=("mtra", "mean"),
            mean_wer_delta=("wer_delta", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "languages": args.languages,
        "models": args.models,
        "conditions": CONDITIONS,
        "samples_per_language": args.samples_per_language,
        "seed": args.seed,
        "device": "cuda" if device == 0 else "cpu",
        "note": "Acoustic stress tests; not a deepfake detector or a TTS/VC spoofing experiment."
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nГотово.")
    print(summary.to_string(index=False))
    print(f"\nФайлы сохранены в: {output.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Сравнение устойчивости двух ASR-моделей на FLEURS."
    )
    parser.add_argument("--languages", nargs="+", default=["kk_kz", "ru_ru"],
                        choices=["kk_kz", "ru_ru"])
    parser.add_argument("--samples-per-language", type=int, default=10,
                        help="Количество записей на язык; 0 = весь test split.")
    parser.add_argument("--models", nargs="+",
                        default=["openai/whisper-small", "openai/whisper-medium"])
    parser.add_argument("--output", default="results")
    parser.add_argument("--glossary", default="data/glossary/medical_terms.json")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
