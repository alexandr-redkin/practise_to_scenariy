
# Kazakh-Russian Medical Speech ASR Robustness Benchmark

A research software prototype for the paper "Investigation of the Robustness of Kazakh-Russian Medical Speech Recognition Systems to Audio Fakes".

The project continues the preparatory analysis of Google FLEURS (kk_kz, ru_ru) and implements a reproducible baseline experiment: comparing two multilingual ASR models on original audio and on audio with controlled signal transformations.

Important Interpretation Constraint. Adding noise, changing gain, filtering frequency bandwidth, and resampling are controlled acoustic disturbances for preliminary ASR robustness assessment. They do not constitute full-fledged deepfake/voice conversion generation and do not prove the system's ability to detect fake speech. Such a conclusion would require separate sets of fake audio and spoofing detection models.

## 1. Connection to the Research Paper

| Paper Material | Project Implementation |
|---|---|
| FLEURS `kk_kz` и `ru_ru` | Loading test samples for both languages |
| audio, transcription, language fields | Used as input signal, reference, and language label |
| Medical dictionary | `data/glossary/medical_terms.json` |
| Potential terminology errors | MTRA metric if terms are present in the reference text |
| WER и CER | Calculated for each model and each audio type |
| Corpus limitations | The report separately accounts for the fact that FLEURS contains general rather than medical speech |

## 2. Features

    Two multilingual ASR baselines: Whisper Small and Whisper Medium.

    Two languages: Kazakh and Russian.

    Conditions: clean, noise, volume, bandlimit, resample.

    Metrics: WER, CER, MTRA, and WER change relative to the original recording.

    Exporting detailed results and the aggregated table to CSV.

    Optional saving of processed audio files.

    Unit tests for audio transformations and metrics.

Default models:

    openai/whisper-small

    openai/whisper-medium

Running Medium will require more RAM/VRAM. Under resource constraints, you can replace it with another compatible multilingual Whisper model via the command-line parameter.

## 3. Running in Google Colab
### Step 1. Installation

```bash
!git clone https://github.com/alexandr-redkin/practise_to_scenariy.git
%cd asr-robustness-benchmark
!pip install -r requirements.txt
!pip install -e .
```

### Step 2. Running a Small Pilot Experiment

```bash
!python -m asr_benchmark.cli \
  --languages kk_kz ru_ru \
  --samples-per-language 10 \
  --models openai/whisper-small openai/whisper-medium \
  --output results
```

For preliminary testing, it is best to start with 5–10 recordings per language. After checking the environment, you can increase the sample size. A full test on all 1,631 recordings may require significant time and computing resources.

### Step 3. Running on a Larger Number of Recordings

```bash
!python -m asr_benchmark.cli \
  --languages kk_kz ru_ru \
  --samples-per-language 100 \
  --models openai/whisper-small openai/whisper-medium \
  --output results
```

The parameter --samples-per-language 0 means using the entire test set. Do not run the full volume until a successful pilot test.

## 4. Output Artifacts

The results/ folder will contain:
    predictions.csv — detailed results for each recording, model, and condition;
    summary.csv — aggregated WER, CER, MTRA, and WER change;
    run_metadata.json — execution parameters and versions of major libraries.
In predictions.csv:
    language — FLEURS language configuration;
    sample_id — recording identifier;
    model — ASR model used;
    condition — type of audio disturbance;
    reference — reference transcription;
    hypothesis — ASR result;
    wer, cer — recognition errors;
    mtra — recognition accuracy of medical terms, if at least one dictionary term is present in the reference.

If there are no terms from the medical dictionary in the test text, MTRA is recorded as NaN rather than 0. This is important: the absence of medical terms in general FLEURS speech does not mean zero medical recognition accuracy.

## 5. Medical Glossary

The file data/glossary/medical_terms.json contains examples of terms from the paper. Exact normalized occurrences are considered for MTRA. In the future, the dictionary can be expanded with Kazakh equivalents, spelling variants, and pronunciations.

Important: Since the original FLEURS is a general speech corpus, MTRA may be unavailable for some or all test recordings. For a full MTRA evaluation, medical dictations with reference transcriptions should be prepared separately.

## 6. Methodology and Interpretation

Variants created for each audio file:

    clean — original signal;

    noise — adding low Gaussian noise;

    volume — amplitude change;

    bandlimit — frequency bandwidth limitation;

    resample — simulating quality degradation by lowering the sampling rate and restoring it to the original frequency.

WER/CER are compared for the original recording and the transformed variants. wer_delta shows the difference between the WER of the transformed and clean signals: a positive value indicates degradation.

These transformations are reproducible acoustic robustness stress tests, but not a replacement for testing on TTS, voice conversion, replay, and adversarial audio.
## 7. Reproducibility and Limitations

    The paper specifies 8,118 recordings across two configurations and 1,631 test recordings in total.

    The code uses only the test parts of kk_kz and ru_ru to avoid training models on test samples.

    Results depend on model versions, libraries, hardware, and transformation parameters.

    Whisper Medium may be too heavy for a free Colab environment.

    Before publishing results, specify the launch date, library versions, model, number of recordings, and actual experimental parameters.

    Do not present a 10-recording demo run as the result of a full study.

## 8. Project Structure

```text
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
```
## 9. Scientific Formulation of the Practical Part

"As a practical continuation of the experimental base analysis, a software prototype has been implemented for comparative evaluation of the robustness of multilingual automatic speech recognition systems. The prototype provides loading of test audio recordings of the Kazakh and Russian configurations of the Google FLEURS corpus, generation of controlled input signal variants, recognition using two ASR models, and calculation of WER, CER, and MTRA metrics. The obtained indicators make it possible to evaluate changes in transcription quality under acoustic disturbances. At the same time, the applied transformations are considered as basic robustness stress tests and are not interpreted as full-fledged generation or detection of audio fakes."
