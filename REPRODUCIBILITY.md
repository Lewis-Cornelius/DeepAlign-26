# Reproducibility Instructions

## Environment

Use Python `3.12`.

```bash
py -3.12 -m venv .venv312
.venv312\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,dl]"
```

Recommended smoke checks before any long run:

```bash
python -m dis_alignment.cli --help
python -m dis_alignment.model.train --dry-run
python -m pytest -q
```

## Optional Baselines

## Optional Transcription Features

The transcription-guided DeepAlign variants use Spotify Basic Pitch through the ONNX Runtime path. Install the runtime support first:

```bash
pip install -e ".[transcription]"
```

Then install Basic Pitch itself without pulling its TensorFlow dependency metadata, which currently targets an older TensorFlow range on Python 3.12:

```bash
pip install basic-pitch --no-deps
```

Expected import warnings about missing TensorFlow, CoreML, or TFLite are acceptable when using the ONNX Runtime path. The evaluator caches Basic Pitch `note`, `onset`, and `contour` outputs under `.cache/transcription/basic_pitch` by default.

### Synctoolbox / MrMsDTW

```bash
pip install -e ".[synctoolbox]"
```

On Windows, Synctoolbox may need an older pandas build chain. If installation fails, retry in the same Python 3.12 environment after pinning setuptools below 81 and reinstalling:

```bash
pip install "setuptools<81"
pip install -e ".[synctoolbox]" --no-build-isolation
```

### Matchmaker

Install Matchmaker from the official repository:

```bash
pip install git+https://github.com/pymatchmaker/matchmaker.git@v0.2.1
```

Expected extra system dependencies:

- PortAudio
- FluidSynth
- a working C/C++ build toolchain on Windows

If Matchmaker import or runtime setup fails, keep `matchmaker` out of `--methods` until those dependencies are available.

## Dataset Layouts

### SWD

The built-in downloader expects the public SWD archive:

```bash
deepalign prepare-swd data/
```

The evaluator resolves scores from:

- `01_RawData/score_musicxml`
- `01_RawData/score_midi`

and audio measure annotations from:

- `02_Annotations/ann_audio_measure`

### MazurkaBL-style Data

The Mazurka loader supports either:

1. `metadata.csv` / `mazurka_metadata.csv`
2. heuristic discovery of audio, annotation, and score files under the dataset root

Recommended metadata columns:

- `work_id`
- `performance_id`
- `audio_path`
- `annotation_path`
- `score_path`

## Training

```bash
deepalign train --config config/deepalign.yaml --swd-path data/Schubert_Winterreise_Dataset_v2-0
```

Outputs:

- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`
- `checkpoints/checkpoint_epoch*.pt`
- `checkpoints/training_history.json`

To validate the stack without SWD:

```bash
deepalign train --dry-run
```

To inspect for temporal collapse after training:

```bash
python scripts/check_collapse.py \
  --checkpoint checkpoints/best_model.pt \
  --swd-path data/Schubert_Winterreise_Dataset_v2-0
```

## Evaluation

### SWD

```bash
deepalign evaluate-swd data/Schubert_Winterreise_Dataset_v2-0 \
  --checkpoint checkpoints/best_model.pt \
  --methods chroma_dtw,mrmsdtw,deepalign,matchmaker \
  -o results/swd_evaluation.csv
```

To evaluate the transcription-fusion DeepAlign variant:

```bash
deepalign evaluate-swd data/Schubert_Winterreise_Dataset_v2-0 \
  --checkpoint checkpoints/best_model.pt \
  --methods deepalign \
  --deep-decode deepalign_transcription_fused \
  --transcription-cache-root .cache/transcription/basic_pitch \
  -o results/swd_deepalign_transcription_fused.csv
```

Use `--deep-decode deepalign_transcription_guided --refine-window-sec 8` for the coarse-to-fine guided variant.
Use `--deep-decode deepalign_transcription_fused_refined --score-refine-radius-sec 0.1` or `--deep-decode deepalign_score_guided_refined` only as experimental variants; compare them against `deepalign_transcription_fused` before promoting them into dissertation tables.

If `mrmsdtw` or `matchmaker` live in a separate environment, run them into separate CSVs and merge them afterwards:

```bash
deepalign merge-results \
  results/swd_chroma_deepalign.csv \
  results/swd_mrmsdtw.csv \
  results/swd_matchmaker.csv \
  -o results/swd_evaluation.csv
```

### Mazurka

```bash
deepalign evaluate-mazurka data/mazurka \
  --checkpoint checkpoints/best_model.pt \
  --methods chroma_dtw,mrmsdtw,deepalign,matchmaker \
  -o results/mazurka_evaluation.csv
```

If a dependency is missing, remove the corresponding method from `--methods`.

## Figures and Statistics

```bash
deepalign visualize results/swd_evaluation.csv -o figures/swd
deepalign analyze results/swd_evaluation.csv
```

Generated figure/table outputs include:

- error vs length
- runtime comparison
- MAE / AR boxplots
- MAE histogram
- success-criteria plot
- `summary.csv`
- `success_criteria.csv`

## Current Validation Status

This checkout has been validated with:

- `python -m dis_alignment.cli --help`
- `python -m dis_alignment.model.train --dry-run`
- `python -m pytest -q`

Datasets, checkpoints, and report artifacts still need to be produced locally because they are not committed in the repository.
