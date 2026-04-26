# Project Status & TODOs

Tracking progress for the DeepAlign-26 project (Differentiable DTW + Deep Learning for audio alignment).

**Current Date:** March 29, 2026  
**Current Week:** 8 of 12

## Completed

### Foundations and Core Model
- [x] Python 3.12 results environment documented and validated
- [x] SWD downloader and verifier
- [x] SWD data loader with pairing and measure-annotation logic
- [x] Feature extraction: CQT chroma and DLNCO
- [x] Baseline alignment algorithms: Global DTW and MrMsDTW wrapper
- [x] DeepAlign encoder, Soft-DTW loss, augmentation, training loop, inference
- [x] Dry-run training validation path
- [x] Unit tests for features, alignment, model, and public project surface

### Results Workflow Implementation
- [x] Unified SWD evaluator with `--methods`
- [x] Matchmaker adapter scaffold for score-following baselines
- [x] MazurkaBL-style dataset loader and evaluator
- [x] Shared result CSV schema across SWD and Mazurka evaluation paths
- [x] CLI commands: `evaluate-swd`, `evaluate-mazurka`, `visualize`, `analyze`
- [x] Friedman / Nemenyi omnibus statistics
- [x] Figure generation for boxplots, histograms, runtime scaling, and success criteria
- [x] Reproducibility instructions and updated README

### Validation
- [x] `python -m dis_alignment.cli --help`
- [x] `python -m dis_alignment.model.train --dry-run`
- [x] `python -m pytest -q`
- [x] Real SWD `chroma_dtw` smoke evaluation over 24 pairs

## Remaining Work

### Environment and External Baselines
- [ ] Install and validate Synctoolbox in a usable Python 3.12 baseline environment or keep it in a separate mergeable results env
- [ ] Install and validate official Matchmaker dependencies (PortAudio, FluidSynth, build tools)
- [ ] Confirm SWD score-following baseline runs end-to-end with real data

### SWD Results
- [ ] Download / verify the SWD dataset locally
- [ ] Train the final DeepAlign checkpoint on SWD
- [ ] Tune the Soft-DTW gamma annealing schedule on real SWD runs
- [ ] Run SWD comparison with `chroma_dtw`, `mrmsdtw`, `deepalign`, and `matchmaker`
- [ ] Verify Success Criterion 1: MAE < 50 ms
- [ ] Verify Success Criterion 2: MAE < 20 ms and AR@50ms > 98%

### Mazurka Robustness
- [ ] Prepare a MazurkaBL-style checkout with metadata, annotations, and score files
- [ ] Run Mazurka robustness evaluation with the same method set
- [ ] Inspect rubato-heavy failure cases and representative alignments

### Dissertation Outputs
- [ ] Generate final SWD and Mazurka CSVs
- [ ] Generate final figure directories from those CSVs
- [ ] Write up the final technical report around the produced results
- [ ] Package trained weights and final reproducibility bundle
