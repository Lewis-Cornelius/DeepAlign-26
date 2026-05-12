# Project Status & TODOs

Tracking progress for the DeepAlign-26 project.

**Current Date:** May 12, 2026
**Current Week:** 13 from the February 2, 2026 project start

## Current Best Evidence

- Full-SWD target viability: `results/teacher_target_20ms/teacher_results.csv`
  reaches `MAE 18.9 ms`, `AR@50ms 98.4%`, all 24 SWD pairs.
- Best learned unconstrained student gate:
  `results/initial_plan_supervised_20ms/gate_8_lieder_false_dest_stage2_A4_debug_mae.csv`
  with `MAE 112.0 ms`, `median AE 31.2 ms`, `AR@50ms 69.2%`.
- Best learned AR/p90 trade-off probe:
  `results/initial_plan_supervised_20ms/gate_8_lieder_false_dest_stage2_min1000_probe.csv`
  with `MAE 113.2 ms`, `median AE 30.8 ms`, `AR@50ms 69.9%`.
- Failure evidence:
  `results/failure_reports/false_dest_stage2_A4_debug_mae_8_lied_failure_report.csv`
  shows second-scale route excursions on the hardest lieder despite good median timing.
- Decoder comparison:
  `results/selector/deepalign_decoder_comparison_fused_r050.csv` shows an oracle pair
  selector could reach `MAE 91.8 ms`, `AR@50ms 72.1%` on the 8-lied gate.
- Important limitation: the learned DeepAlign student does not meet the original
  stretch target (`MAE < 20 ms`, `AR@50ms > 98%`). The teacher route meets it,
  but the student has not absorbed the route reliably enough.

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
- [x] Original-plan supervised teacher route and strict teacher gate checker
- [x] False-destination mining and replay diagnostics
- [x] Coarse-to-fine DeepAlign decoder and selector diagnostics

### Validation

- [x] `.venv312\Scripts\python.exe -m dis_alignment.cli --help`
- [x] `.venv312\Scripts\python.exe -m pytest -q`
- [x] Current full test suite: `162 passed`, `20 warnings`
- [x] Current dissertation evidence CSVs regenerated for the recovered supervised route

## Remaining Work

### Dissertation Evidence

- [ ] Freeze the recovered supervised-route evidence and stop hyperparameter tuning unless a single bounded full-SWD run is required for the final table.
- [ ] If running a final learned-student full-SWD check, use the best false-destination checkpoint once and report the result honestly.
- [ ] Write the dissertation around the teacher/student gap, false-destination improvement, coarse-to-fine diagnostics, and route-excursion failure analysis.
- [ ] Archive the final CSVs, figures, checkpoint, config, and logs outside git because `results/`, `figures/`, `checkpoints/`, and `.cache/` are ignored.
- [ ] Record exact environment caveats: Basic Pitch is used through the ONNX path, and `synctoolbox` has dependency conflicts in the combined Python 3.12 environment.

### Robustness

- [ ] Run Mazurka evaluation only if suitable audio/annotation data is available.
- [ ] If Mazurka is not available in time, state it as planned robustness work rather than making unsupported claims.

### Write-Up

- [ ] Present the project as pairwise audio-to-audio alignment for performances of the same piece.
- [ ] Explain the novel contribution as a learned DeepAlign student trained from a supervised teacher route, plus diagnostics for why the student fails to absorb global route structure.
- [ ] Report that the supervised teacher route reaches the original target, but the learned unconstrained student does not.
- [ ] Include representative failure cases and explain why strict 50 ms alignment remains hard in repeated musical material.
