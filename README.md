# Kazakh-Russian Medical Speech ASR Robustness Benchmark

A research software prototype for the paper "Investigation into the Robustness of Kazakh-Russian Medical Speech Recognition Systems Against Audio Spoofing".

The project continues the preliminary analysis of Google FLEURS (kk_kz, ru_ru) and implements a reproducible baseline experiment: comparing two multilingual ASR models on clean audio versus audio with controlled signal transformations.

> **Important Interpretation Constraint.** Adding noise, changing gain, filtering bandwidth, and resampling are controlled acoustic perturbations for an initial assessment of ASR robustness. They do not constitute full-fledged deepfake/voice conversion generation and do not prove the system's ability to detect spoofed speech. Such conclusions require separate sets of spoofed audio and dedicated spoofing detection models.

## 1. Relation to the Scientific Article

The project directly maps to the research material: FLEURS test subsets (kk_kz and ru_ru) are utilized as input data, audio fields and transcriptions serve as input signals and references, the medical glossary (`data/glossary/medical_terms.json`) powers the MTRA metric for terminology error analysis, and WER/CER calculations evaluate performance across models and audio conditions while accounting for the general-speech limitations of the corpus.

## 2. Features

- Two multilingual ASR baselines: Whisper Small and Whisper Medium (`openai/whisper-small`, `openai/whisper-medium`).
- Two languages: Kazakh and Russian.
- Conditions: clean, noise, volume, bandlimit, resample.
- Metrics: WER, CER, MTRA, and WER delta relative to the original recording.
- Export of detailed results and an aggregated summary table to CSV.
- Optional saving of processed audio files and unit tests for audio transformations and metrics.

## 3. Running in Google Colab

To install and run a pilot experiment:

```bash
!git clone [https://github.com/YOUR_USERNAME/asr-robustness-benchmark.git](https://github.com/YOUR_USERNAME/asr-robustness-benchmark.git)
%cd asr-robustness-benchmark
!pip install -r requirements.txt
!pip install -e .

!python -m asr_benchmark.cli \
  --languages kk_kz ru_ru \
  --samples-per-language 10 \
  --models openai/whisper-small openai/whisper-medium \
  --output results

4. Generated Outputs

The results/ folder will contain:

    predictions.csv — detailed results for each sample, model, and condition;

    summary.csv — aggregated WER, CER, MTRA, and WER delta;

    run_metadata.json — execution parameters and core library versions.

In predictions.csv:

    language — FLEURS language configuration;

    sample_id — recording identifier;

    model — ASR model used;

    condition — type of acoustic perturbation;

    reference — reference transcription;

    hypothesis — ASR output;

    wer, cer — recognition errors;

    mtra — medical term recognition accuracy, if at least one dictionary term is present in the reference.

If the test text contains no medical dictionary terms, MTRA is recorded as NaN rather than 0. This is important: the absence of medical terms in general FLEURS speech does not imply zero medical recognition accuracy.
5. Medical Dictionary

The file data/glossary/medical_terms.json contains sample terms from the paper. Exact normalized occurrences are considered for MTRA. In the future, the dictionary can be expanded with Kazakh equivalents, spelling variants, and pronunciations.

Important: Since the original FLEURS is a general speech corpus, MTRA may be unavailable for some or all test samples. For a comprehensive MTRA evaluation, specialized medical dictations with reference transcriptions should be prepared separately.
6. Methodology and Interpretation

The following variations are generated for each audio file:

    clean — original signal;

    noise — addition of weak Gaussian noise;

    volume — amplitude adjustment;

    bandlimit — frequency bandwidth limitation;

    resample — simulation of channel degradation via downsampling to 8 kHz and upsampling back to the original rate.

WER/CER are compared between the original recording and the transformed variants. wer_delta shows the difference between the WER of the transformed and clean signals: a positive value indicates degradation.

These transformations serve as reproducible acoustic robustness stress tests, but they do not replace evaluations on TTS, voice conversion, replay, and adversarial audio.
7. Reproducibility and Limitations

    The paper specifies 8,118 samples across two configurations and 1,631 test samples in total.

    The code uses only the test splits of kk_kz and ru_ru to prevent training models on test samples.

    Results depend on model versions, library versions, hardware, and transformation parameters.

    Whisper Medium may be too heavy for a free Colab environment.

    Before publishing results, specify the run date, library versions, models, sample count, and actual experimental parameters.

    Do not present a demonstration run on 10 samples as the outcome of a full research study.

8. Project Structure

    asr-robustness-benchmark/
├── README.md
├── requirements.txt
├── data/
│   └── glossary/
│       └── medical_terms.json
├── results/
│   └── .gitkeep
├── src/
│   └── asr_benchmark/
│       ├── __init__.py
│       ├── audio.py
│       ├── metrics.py
│       ├── dataset.py
│       └── cli.py
└── tests/
    └── test_core.py

9. Scientific Formulation of the Practical Part

"As a practical continuation of the experimental base analysis, a software prototype was implemented for the comparative evaluation of multilingual automatic speech recognition systems' robustness. The prototype enables loading test audio recordings from the Kazakh and Russian configurations of the Google FLEURS corpus, generating controlled input signal variants, performing recognition using two ASR models, and calculating WER, CER, and MTRA metrics. The obtained metrics allow for estimating transcription quality changes under acoustic perturbations. Furthermore, the applied transformations are treated as basic robustness stress tests and are not interpreted as full-fledged generation or detection of audio spoofing."

10. Licensing

Before publication, check the licensing terms of the used models and dataset. The project code can be distributed under the MIT License, for example; model and FLEURS licenses are regulated separately.
