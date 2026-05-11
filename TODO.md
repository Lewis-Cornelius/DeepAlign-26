# Project Status & TODOs

Tracking progress for the DeepAlign-26 project.

**Current Date:** May 11, 2026
**Current Week:** 13 from the February 2, 2026 project start

## Current Best Evidence

- Best SWD result: `results/swd_evaluation_transcription_fused_2026_04_25.csv`
- Current figure/table bundle: `figures/swd_current_best_2026_04_27`
- Promoted method for write-up: `deepalign` with `deep_decode=deepalign_transcription_fused`
- Full SWD metrics: `MAE 367.0 ms`, `median AE 165.0 ms`, `AR@50ms 41.5%`
- Statistical result: paired MAE comparison against recorded `chroma_dtw` baseline shows `92.6%` mean improvement, Wilcoxon `p=0.0005`
- Important limitation: the original stretch target (`MAE < 20 ms`, `AR@50ms > 98%`) is not met.

## Completed

### Core System

- [x] Python 3.12 results environment validated
- [x] SWD dataset present locally under `data/`
- [x] SWD loader with shared measure annotation logic
- [x] DeepAlign encoder, Soft-DTW training, aligned-window recovery training, checkpointing, and inference
- [x] Cached CQT support for faster experiments
- [x] SWD evaluation with `chroma_dtw`, `mrmsdtw`, `deepalign`, and `matchmaker`
- [x] Basic Pitch transcription feature cache and transcription-fused DeepAlign decoding
- [x] MazurkaBL-style loader/evaluator implementation
- [x] Figure generation, success tables, Friedman/Nemenyi statistics, and pairwise statistics
- [x] Variant-aware result merging and reporting for multiple DeepAlign decoding modes

### Validation

- [x] `.venv312\Scripts\python.exe -m dis_alignment.cli --help`
- [x] `.venv312\Scripts\python.exe -m pytest -q`
- [x] Current full test suite: `141 passed`
- [x] Current SWD figures regenerated from the promoted evidence CSV

## Remaining Work

### Dissertation Evidence

- [ ] Decide whether `deepalign_transcription_fused` is final, or run one last clearly bounded full-SWD improvement attempt.
- [ ] If attempting the original target, use `scripts/run_initial_plan_supervised_20ms_sprint.ps1`, not the strict self-supervised ablation route.
- [ ] If final, do not keep tuning; write the dissertation around the recovered/improved method and its failure analysis.
- [ ] Archive the final CSVs, figures, checkpoint, config, and logs outside git because `results/`, `figures/`, `checkpoints/`, and `.cache/` are ignored.
- [ ] Record exact environment caveats: Basic Pitch is used through the ONNX path, and `synctoolbox` has dependency conflicts in the combined Python 3.12 environment.

### Robustness

- [ ] Run Mazurka evaluation only if suitable audio/annotation data is available.
- [ ] If Mazurka is not available in time, state it as planned robustness work rather than making unsupported claims.

### Write-Up

- [ ] Present the project as pairwise audio-to-audio alignment for performances of the same piece.
- [ ] Explain the novel contribution as the DeepAlign recovery path plus transcription-fused learned-feature decoding, not as a completely new SOTA method.
- [ ] Report that the system improves strongly over the recorded baselines but does not reach the original aspirational threshold.
- [ ] Include representative failure cases and explain why strict 50 ms alignment remains hard.
