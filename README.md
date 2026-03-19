# DeepAlign-26: Differentiable Audio Alignment for Dissertation Reproducibility

`DeepAlign-26` is the dissertation codebase for learning audio-to-audio alignment features with a CRNN encoder trained by Soft-DTW. The repository is centered on the Schubert Winterreise Dataset (SWD), with classical DTW and MrMsDTW retained as baseline methods rather than as the main project identity.

## What This Repo Is For

- Train DeepAlign-26 on SWD pairs.
- Evaluate a trained checkpoint against hand-crafted chroma DTW baselines on SWD.
- Generate report-ready CSV summaries and figures.
- Keep a legacy MAESTRO / classical benchmark path available for secondary experiments.

The packaged workflow in this repository currently focuses on SWD preparation, DeepAlign training, SWD evaluation, and downstream analysis. MazurkaBL stress testing and external baselines such as Matchmaker remain part of the broader dissertation plan, but they are not the primary automated path in this checkout.

## Installation

```bash
# Core DeepAlign workflow
pip install -e ".[dev,dl]"

# Add optional synctoolbox-backed baseline support
pip install -e ".[dev,dl,synctoolbox]"
```

Console entrypoints:

- `deepalign`
- `dis-benchmark`

Both commands point at the same CLI. `deepalign` is the recommended public-facing alias.

## Recommended Workflow

```bash
# 1. Download / verify SWD
deepalign prepare-swd data/

# 2. Train DeepAlign-26
deepalign train --config config/deepalign.yaml --swd-path data/Schubert_Winterreise_Dataset_v2-0

# 3. Evaluate DeepAlign on SWD
deepalign evaluate-swd data/Schubert_Winterreise_Dataset_v2-0 \
  --checkpoint checkpoints/best_model.pt \
  -o results/swd_evaluation.csv

# 4. Generate figures
deepalign visualize results/swd_evaluation.csv -o figures/

# 5. Analyze metrics / significance
deepalign analyze results/swd_evaluation.csv
```

## Training Notes

- `config/deepalign.yaml` is the main reproducibility config for the DeepAlign model.
- `python -m dis_alignment.model.train --dry-run` validates the training stack on synthetic data without SWD.
- The best checkpoint is saved as `checkpoints/best_model.pt`, alongside periodic checkpoints and `training_history.json`.

## Evaluation Outputs

The SWD evaluation path produces:

- `results/swd_evaluation.csv`: per-pair metrics for `chroma_dtw` and optional `deepalign`
- `figures/`: summary plots such as MAE boxplots and runtime/error scaling
- success-criterion checks for the dissertation targets:
  - `MAE < 50ms`
  - `MAE < 20ms`
  - `AR@50ms > 98%`

## Repository Structure

```text
dis-alignment/
├── config/
│   ├── deepalign.yaml          # Main SWD training config
│   └── default.yaml            # Baseline / legacy experiment defaults
├── scripts/
│   ├── download_swd.py         # Thin wrapper for SWD setup
│   ├── evaluate_swd.py         # Thin wrapper for package SWD evaluation
│   └── check_collapse.py       # Inspect a checkpoint for temporal collapse
├── src/dis_alignment/
│   ├── data/                   # SWD + legacy MAESTRO dataset access
│   ├── features/               # Chroma / DLNCO baseline features
│   ├── alignment/              # Global DTW / MrMsDTW baselines
│   ├── model/                  # DeepAlign encoder, loss, inference, training
│   ├── evaluation/             # SWD evaluation helpers and metrics
│   ├── analysis/               # Statistics and dissertation figures
│   └── cli.py                  # DeepAlign-first CLI
└── tests/                      # Smoke, baseline, and model tests
```

## Legacy Baselines

The older MAESTRO benchmarking path still exists for comparison work, but it is intentionally demoted behind the DeepAlign workflow:

```bash
deepalign legacy-maestro-benchmark /path/to/maestro-v3.0.0
```

Use this only for legacy / secondary experiments. The main repo story is SWD + DeepAlign-26.

## References

- M. Cuturi and M. Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series," ICML, 2017.
- C. Weiß et al., "Schubert Winterreise Dataset: A Multimodal Scenario for Music Analysis," JOCCH, 2021.
- M. Müller et al., "Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization," JOSS, 2021.
