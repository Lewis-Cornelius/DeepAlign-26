# DeepAlign-26: Differentiable Audio Alignment for Dissertation Reproducibility

`DeepAlign-26` is the dissertation codebase for learning audio-to-audio alignment features with a CRNN encoder trained by Soft-DTW. The repository now supports a results-first workflow for:

- SWD training and evaluation
- SWD baseline comparison with `chroma_dtw`, `mrmsdtw`, optional `deepalign`, and optional `matchmaker`
- MazurkaBL-style robustness evaluation with the same result schema
- Figure generation, success-criteria summaries, and omnibus statistics

## What This Repo Is For

- Train DeepAlign-26 on SWD pairs.
- Evaluate a trained checkpoint against offline baselines on SWD.
- Stress test the same comparison workflow on MazurkaBL-style data.
- Generate CSV summaries, dissertation figures, and significance analysis.
- Keep the legacy MAESTRO benchmark path available for secondary experiments.

## Python Version

Use Python `3.12`. The repository is pinned to `>=3.12,<3.13` because the results workflow depends on the deep-learning stack and external alignment packages that are not reproducible on newer interpreters yet.

## Installation

```bash
# Create and activate a Python 3.12 environment
py -3.12 -m venv .venv312
.venv312\Scripts\activate

# Core project + tests + DeepAlign training
python -m pip install --upgrade pip
pip install -e ".[dev,dl]"
```

Optional baseline dependencies:

```bash
# Synctoolbox / MrMsDTW baseline
pip install -e ".[synctoolbox]"
```

Optional transcription features for research-backed DeepAlign decoding:

```bash
# ONNX Runtime support used by the Basic Pitch adapter
pip install -e ".[transcription]"

# Basic Pitch currently declares an old TensorFlow dependency range on Python 3.12.
# Install the package without its TensorFlow dependency and use the ONNX path.
pip install basic-pitch --no-deps
```

Matchmaker is not published on PyPI under its official package name. Install it from the official source repository in the same Python 3.12 environment:

```bash
pip install git+https://github.com/pymatchmaker/matchmaker.git@v0.2.1
```

Matchmaker also requires system-level audio dependencies such as PortAudio and FluidSynth. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for setup notes.

Console entrypoints:

- `deepalign`
- `dis-benchmark`

Both commands point at the same CLI. `deepalign` is the recommended public-facing alias.

## Recommended Workflow

```bash
# 0. Smoke checks before long runs
python -m dis_alignment.cli --help
python -m dis_alignment.model.train --dry-run
python -m pytest -q

# 1. Download / verify SWD
deepalign prepare-swd data/

# 2. Train DeepAlign-26
deepalign train --config config/deepalign.yaml --swd-path data/Schubert_Winterreise_Dataset_v2-0

# 3. Evaluate SWD with a unified multi-method entrypoint
deepalign evaluate-swd data/Schubert_Winterreise_Dataset_v2-0 \
  --checkpoint checkpoints/best_model.pt \
  --methods chroma_dtw,mrmsdtw,deepalign,matchmaker \
  -o results/swd_evaluation.csv

# 3a. Optional DeepAlign transcription-fusion variant
deepalign evaluate-swd data/Schubert_Winterreise_Dataset_v2-0 \
  --checkpoint checkpoints/best_model.pt \
  --methods deepalign \
  --deep-decode deepalign_transcription_fused \
  --transcription-cache-root .cache/transcription/basic_pitch \
  -o results/swd_deepalign_transcription_fused.csv

# 4. Evaluate MazurkaBL-style data with the same schema
deepalign evaluate-mazurka data/mazurka \
  --checkpoint checkpoints/best_model.pt \
  --methods chroma_dtw,mrmsdtw,deepalign,matchmaker \
  -o results/mazurka_evaluation.csv

# 5. Generate figures and summary tables
deepalign visualize results/swd_evaluation.csv -o figures/swd
deepalign visualize results/mazurka_evaluation.csv -o figures/mazurka

# 6. Run pairwise + omnibus statistics
deepalign analyze results/swd_evaluation.csv
deepalign analyze results/mazurka_evaluation.csv
```

If you need to split methods across environments, merge the per-method CSVs afterwards:

```bash
deepalign merge-results \
  results/swd_chroma_deepalign.csv \
  results/swd_mrmsdtw.csv \
  results/swd_matchmaker.csv \
  -o results/swd_evaluation.csv
```

`merge-results`, `visualize`, and `analyze` are DeepAlign-variant aware. Raw rows keep
`method=deepalign` and record the decoding variant in `deep_decode`; when a CSV contains
more than one DeepAlign variant, reporting labels such as
`deepalign:deepalign_transcription_fused` are created automatically so variants are not
deduplicated or averaged together by accident.

## Evaluation Outputs

The SWD and Mazurka evaluation commands write a consistent row schema, including:

- `dataset`
- `pair_id`
- `group_id`
- `piece_a_id`
- `piece_b_id`
- `method`
- `duration_s`
- `memory_mb`
- `mae`
- `median_ae`
- `ar_50ms`
- `ar_100ms`
- `ar_200ms`
- `runtime_s`
- `n_gt_points`
- `deep_decode`
- `pool_size`
- `deep_distance`
- fusion/refinement settings for DeepAlign transcription variants

The default success criteria checked by the CLI target the `deepalign` method:

- `MAE < 50 ms`
- `MAE < 20 ms`
- `AR@50ms > 98%`

## Current Dissertation Evidence

As of 2026-05-12, the strongest evidence package is the recovered
original-plan supervised route, not the older transcription-fused package. The
current dissertation-shaped result is:

- Full-SWD target viability: `results/teacher_target_20ms/teacher_results.csv`
- Anchor-calibrated teacher route: `18.9 ms` mean MAE, `98.4%` AR@50ms, all
  24 SWD pairs
- Best learned unconstrained student gate:
  `results/initial_plan_supervised_20ms/gate_8_lieder_false_dest_stage2_A4_debug_mae.csv`
- Best learned student metrics on the 8-lied gate: `112.0 ms` mean MAE,
  `31.2 ms` median AE, `69.2%` AR@50ms
- Best AR/p90 trade-off probe:
  `results/initial_plan_supervised_20ms/gate_8_lieder_false_dest_stage2_min1000_probe.csv`
  with `113.2 ms` mean MAE, `30.8 ms` median AE, `69.9%` AR@50ms
- Failure analysis:
  `results/failure_reports/false_dest_stage2_A4_debug_mae_8_lied_failure_report.csv`
- Decoder comparison:
  `results/selector/deepalign_decoder_comparison_fused_r050.csv`

The teacher route meets the original `<20 ms` / `>98%` target, establishing
that the metric is reachable under dense supervised timing. The learned
DeepAlign student does not yet meet that target under strict unconstrained
decoding. Its low median error but much higher mean MAE shows that the remaining
failure mode is global route excursions in repeated or ambiguous sections, not
uniformly poor local timing.

The old full-SWD transcription-fused bundle
(`results/swd_evaluation_transcription_fused_2026_04_25.csv`,
`367.0 ms` MAE, `41.5%` AR@50ms) is now historical evidence only. It should not
be promoted over the recovered supervised-route analysis above.

## Commands

### Datasets

- `deepalign prepare-swd`
- `deepalign verify-mazurka`

### Training

- `deepalign train`

### Evaluation

- `deepalign evaluate-swd`
- `deepalign evaluate-mazurka`
- `deepalign merge-results`
- `deepalign generate-teacher-paths`

Supported `--methods` values:

- `chroma_dtw`
- `mrmsdtw`
- `deepalign`
- `matchmaker`

`deepalign` requires `--checkpoint`. The other methods do not.

Supported `--deep-decode` values for `deepalign`:

- `unconstrained`
- `diagonal_band`
- `chroma_guided_band`
- `deepalign_coarse_to_fine`
- `deepalign_mrmsdtw_guided_refined`
- `deepalign_transcription_fused`
- `deepalign_transcription_fused_refined`
- `deepalign_transcription_guided`
- `deepalign_score_guided_refined`

The transcription decode modes cache Basic Pitch note/onset/contour outputs under `.cache/transcription/basic_pitch` by default. They keep the CSV method as `deepalign` and record the variant in `deep_decode`, so dissertation tables can treat them as DeepAlign variants rather than external baselines.
The refined/score-guided modes are experimental research variants; promote them only when a gate run beats `deepalign_transcription_fused` on full-SWD `AR@50ms`.

### Original-Plan Supervised 20 ms Route

The original dissertation plan was not a strict self-supervised audio-only
sprint. It allowed SWD annotation/score-derived timing artifacts to create weak
or dense alignment targets, then trained a learned DeepAlign encoder with
Soft-DTW and local timing supervision. The current implementation for that route
is:

- `scripts/run_initial_plan_supervised_20ms_sprint.ps1`
- `config/deepalign_initial_plan_supervised_20ms.yaml`

This route generates anchor-calibrated teacher paths for training, trains with
`teacher_path` crops, `PathDistillationLoss`, dense anchor supervision, anti-
collapse regularization, and a positive Soft-DTW weight. Final inference remains
learned DeepAlign features with unconstrained DTW at `--deep-hop 110` and
`--pool-size 1`.

Run the full target attempt with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_initial_plan_supervised_20ms_sprint.ps1 -Mode All -SwdPath data -Device cuda
```

The target gate for this route is full SWD `MAE < 20 ms` and `AR@50ms > 98%`,
with all 24 pairs present. If teacher generation or DeepAlign evaluation fails
that gate, the correct outcome is an honest failed target run, not a weakened
threshold.

### Strict Audio-Only Ablation

`scripts/run_initial_claim_strict_audio_only_sprint.ps1` is now an ablation and
claim-boundary stress test. It uses SWD audio only for training, forbids teacher
paths, anchors, score, transcription, guided bands, and non-`pool_size=1`
evaluation, and should not be presented as the original supervised Soft-DTW
target route.

Run the strict ablation sequence with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_initial_claim_strict_audio_only_sprint.ps1 -Mode All -SwdPath data -Device cuda
```

Its final gate is also full SWD `MAE < 20 ms` and `AR@50ms > 98%`, with all 24
pairs present. It is expected to be harder than the original-plan supervised
route because it removes the timing supervision that the plan relied on.

### Diagnostic Teacher Tracks

The audio-only SOTA sprint is driven by `config/deepalign_sota_audio_only_2026_05_09.yaml`
and `scripts/run_audio_only_sota_2026_05_09.ps1`. It first generates strict teacher-path
artifacts with:

```bash
deepalign generate-teacher-paths data \
  --output-dir results/teacher_audio_only_sota_2026_05_09 \
  --hop-length 110 \
  --trim-top-db 40 \
  --estimate-chroma-shift
```

The command writes per-pair `.npz` paths plus `teacher_results.csv` and
`oracle_results.csv`. Training uses those teacher paths only as supervision; final
DeepAlign evaluation remains audio-only and should be run with `--deep-hop 110` and
`--pool-size 1`.

The separate supervised fallback track is diagnostic-only for the initial claim.
It keeps final inference audio-only, but calibrates dense teacher paths with SWD
measure anchors for training, so it must not be reported as the headline
audio-only result:

```bash
deepalign generate-teacher-paths data \
  --output-dir results/teacher_anchor_calibrated_sota_2026_05_10 \
  --hop-length 110 \
  --trim-top-db 40 \
  --estimate-chroma-shift \
  --anchor-calibrate
```

Train that fallback with `config/deepalign_sota_anchor_calibrated_2026_05_10.yaml`
and keep its results separate from the strict audio-only teacher gate.

### Analysis

- `deepalign visualize`
- `deepalign analyze`

`deepalign analyze` now reports:

- Friedman / Nemenyi omnibus analysis when 3 or more methods are present
- Wilcoxon pairwise comparison for `--baseline` vs `--candidate`
- runtime scaling
- memory summaries
- failure analysis

### Legacy

- `deepalign legacy-maestro-benchmark /path/to/maestro-v3.0.0`

## Repository Structure

```text
dis-alignment/
|-- config/
|   |-- deepalign.yaml
|   `-- default.yaml
|-- scripts/
|   |-- check_collapse.py
|   |-- download_maestro.py
|   |-- download_swd.py
|   |-- evaluate_mazurka.py
|   `-- evaluate_swd.py
|-- src/dis_alignment/
|   |-- alignment/
|   |   |-- baseline_dtw.py
|   |   |-- matchmaker.py
|   |   `-- multiscale_dtw.py
|   |-- analysis/
|   |-- data/
|   |   |-- maestro.py
|   |   |-- mazurka.py
|   |   `-- swd.py
|   |-- evaluation/
|   |   |-- common.py
|   |   |-- mazurka.py
|   |   `-- swd.py
|   |-- features/
|   |-- model/
|   `-- cli.py
|-- tests/
|-- REPRODUCIBILITY.md
`-- TODO.md
```

## Reproducibility Notes

- The final dissertation evidence package under `results/final_dissertation_evidence_2026_05_12/`
  and `figures/final_dissertation_evidence_2026_05_12/` is committed intentionally.
- Other datasets, experiment checkpoints, ad hoc figures/result CSVs, logs, and caches are treated as local artifacts.
- The code path is implemented and validated in the Python 3.12 environment with:
  - `.venv312\Scripts\python.exe -m dis_alignment.cli --help`
  - `.venv312\Scripts\python.exe -m pytest -q`
- See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment and run instructions.

## References

- M. Cuturi and M. Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series," ICML, 2017.
- C. Weiss et al., "Schubert Winterreise Dataset: A Multimodal Scenario for Music Analysis," JOCCH, 2021.
- M. Muller et al., "Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization," JOSS, 2021.
- Spotify, "Basic Pitch: An open source MIDI converter from Spotify."
- J. Park et al., "Matchmaker: An Open-Source Library for Real-Time Piano Score Following and Systematic Evaluation," ISMIR, 2025.
