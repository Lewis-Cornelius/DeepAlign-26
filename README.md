# Dis-Alignment: Audio-to-Score Alignment Benchmarking

A Python framework for comparative evaluation of multiscale DTW optimization strategies for audio-to-score alignment, with a focus on the **MrMsDTW** algorithm.

## Features

- 🎵 **Feature Extraction**: CQT-based Chroma and DLNCO (Decay-Locked Note Cue Onsets)
- 📊 **Alignment Algorithms**: Global DTW, MrMsDTW, and Sakoe-Chiba constrained variants
- 📏 **Evaluation Metrics**: MAE, Alignment Rate, Coverage, and percentile errors
- 📈 **Visualization**: Publication-quality figures for dissertation
- 📉 **Statistical Analysis**: Significance testing and failure analysis

## Installation

```bash
# Clone the repository
cd c:\code\Dis

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from dis_alignment import extract_chroma_cqt, align_global_dtw, align_mrmsdtw
import librosa

# Load audio
audio, sr = librosa.load("recording.wav", sr=22050)
score_audio, _ = librosa.load("score_rendition.wav", sr=22050)

# Extract features
features_audio = extract_chroma_cqt(audio, sr=sr)
features_score = extract_chroma_cqt(score_audio, sr=sr)

# Align with different algorithms
result_dtw = align_global_dtw(features_audio, features_score)
result_mrmsdtw = align_mrmsdtw(features_audio, features_score)

print(f"Global DTW: {result_dtw.runtime_seconds:.2f}s")
print(f"MrMsDTW: {result_mrmsdtw.runtime_seconds:.2f}s")
```

## CLI Usage

```bash
# Run benchmark on MAESTRO dataset
dis-benchmark benchmark /path/to/maestro-v3.0.0 --split test --limit 10

# Generate visualization figures
dis-benchmark visualize results/benchmark_results.csv -o figures/

# Run statistical analysis
dis-benchmark analyze results/benchmark_results.csv --baseline global_dtw --candidate mrmsdtw

# Extract features from audio file
dis-benchmark extract-features recording.wav -o features.png
```

## Project Structure

```
dis-alignment/
├── src/dis_alignment/
│   ├── features/        # Feature extraction (chroma, dlnco)
│   ├── alignment/       # DTW algorithms (baseline, multiscale)
│   ├── data/           # Dataset loaders (MAESTRO)
│   ├── evaluation/     # Metrics and benchmarking
│   ├── analysis/       # Visualization and statistics
│   └── cli.py          # Command-line interface
├── config/             # Configuration files
├── tests/              # Unit tests
└── scripts/            # Utility scripts
```

## References

- M. Müller et al., "Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization," JOSS, 2021.
- T. Prätzlich et al., "Memory-Restricted Multiscale Dynamic Time Warping," ICASSP, 2016.
- S. Ewert and M. Müller, "High Resolution Audio Synchronization using Chroma Onset Features," ICASSP, 2009.

## License

MIT License
