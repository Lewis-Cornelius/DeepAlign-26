# Project Status & TODOs

Tracking progress for the DeepAlign-26 project (Differentiable DTW + Deep Learning for audio alignment).

**Current Date:** March 10, 2026 · **Current Week:** 6 of 12

## Completed (Weeks 1–6)

### Phase 1: Foundations (Weeks 1–3)
- [x] Environment setup: Python 3.12, PyTorch, CUDA
- [x] SWD dataset downloader (`scripts/download_swd.py`)
- [x] SWD data loader with pairing/annotation logic (`data/swd.py`)
- [x] Feature extraction: CQT Chroma (`features/chroma.py`) and DLNCO (`features/dlnco.py`)
- [x] Baseline alignment algorithms: Global DTW, MrMsDTW (`alignment/`)
- [x] Evaluation metrics: MAE, AR, coverage (`evaluation/metrics.py`)
- [x] Benchmark runner (`evaluation/runner.py`)
- [x] CLI interface (`cli.py`)
- [x] Unit tests for features and alignment (`tests/test_features.py`, `tests/test_alignment.py`)
- [x] Config files (`config/default.yaml`, `config/deepalign.yaml`)

### Phase 2: Core Work (Weeks 4–6)
- [x] Dual-stream CRNN encoder architecture (`model/encoder.py`)
- [x] Soft-DTW loss with γ annealing scheduler (`model/soft_dtw_loss.py`)
- [x] PyTorch Dataset with CQT extraction + variable-length collation (`model/dataset.py`)
- [x] Full training loop: AdamW, CosineAnnealingLR, mixed precision, gradient clipping, checkpointing (`model/train.py`)
- [x] Dry-run validation mode for training pipeline (`model/train.py --dry-run`)
- [x] Inference pipeline: checkpoint loading + deep features + DTW alignment (`model/inference.py`)
- [x] Data augmentation: time-stretch ±20%, pitch shift, additive noise, SpecAugment (`model/augmentation.py`)
- [x] Unit tests for all DL components (`tests/test_model.py`)
- [x] PyTorch/torchaudio added to project dependencies (`pyproject.toml`)

## Remaining Work

### Phase 2 Continued (Week 7 — Buffer)
- [x] Actually train the model on SWD (download data → run `train.py`) — **50 epochs, 38 min, best val_loss ≈ 0**
- [ ] Tune the Soft-DTW smoothing parameter γ annealing schedule
- [ ] (Optional) Experiment with attention layers in the encoder

### Phase 3: Evaluation & Analysis (Weeks 8–10)
- [ ] Establish baseline MAE/AR floor using Matchmaker and Synctoolbox on SWD test set
- [ ] Evaluate trained DeepAlign model on SWD test data
- [ ] Verify **Success Criterion 1:** MAE < 50ms (comparable to hand-crafted Chroma)
- [ ] Verify **Success Criterion 2:** MAE < 20ms, AR > 98% at θ = 50ms
- [ ] Stress test on unseen **MazurkaBL dataset** (extreme rubato)
- [ ] Perform statistical validation (Friedman/Nemenyi significance tests)
- [ ] Generate visualizations: warping paths, error histograms

### Phase 4: Writing & Submission (Weeks 11–12)
- [ ] Synthesize findings into final technical report
- [ ] Clean up code documentation
- [ ] Prepare Reproducibility Instructions and GitHub repository
- [ ] Submit final deliverables: PDF Report, Code, Trained Weights
