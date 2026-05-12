# Reproducibility Instructions

This repository is now frozen around the dissertation evidence package committed
on 2026-05-12:

- `results/final_dissertation_evidence_2026_05_12/`
- `figures/final_dissertation_evidence_2026_05_12/`

The package is for reporting and verification, not further model selection. The
core result is that the supervised teacher route reaches the original `<20 ms`
/ `>98% AR@50` target, while the learned unconstrained DeepAlign student does
not yet absorb that route reliably.

## Environment

Use Python `3.12`.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,dl]"
```

For the committed evidence package, the basic `dev,dl` install is enough to run
tests and inspect CSVs/checkpoints. Install optional extras only if rerunning
teacher generation, transcription variants, MrMsDTW, or Matchmaker.

Recommended validation checks:

```powershell
.\.venv312\Scripts\python.exe -m dis_alignment.cli --help
.\.venv312\Scripts\python.exe -m pytest -q
```

Most recent local validation before freezing:

- `162 passed, 20 warnings`

## Frozen Evidence Package

Primary files:

- `results/final_dissertation_evidence_2026_05_12/summary_table.md`
- `results/final_dissertation_evidence_2026_05_12/summary_table.csv`
- `results/final_dissertation_evidence_2026_05_12/MANIFEST.json`
- `figures/final_dissertation_evidence_2026_05_12/evidence_metric_comparison.png`
- `figures/final_dissertation_evidence_2026_05_12/per_lied_failure_metrics.png`
- `figures/final_dissertation_evidence_2026_05_12/event_error_boxplot.png`
- `figures/final_dissertation_evidence_2026_05_12/route_excursion_case_study.png`

The package includes the selected CSVs, configs, and three selected checkpoints:

- `best_student_debug_mae.pt`
- `best_ar_p90_balanced.pt`
- `local_transfer_balanced.pt`

`MANIFEST.json` records SHA-256 checksums for the copied evidence artifacts.

Main dissertation table:

| Evidence | Pairs | MAE | Median AE | AR@50 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Teacher route | 24 | 18.9 ms | 0.0 ms | 98.4% | Target is reachable with dense supervised timing. |
| Best learned student | 8 | 112.0 ms | 31.2 ms | 69.2% | Student learns local timing but not robust route control. |
| Best AR/p90 probe | 8 | 113.2 ms | 30.8 ms | 69.9% | Similar learned-decoder ceiling with slightly higher AR. |
| Coarse-to-fine fused | 8 | 116.9 ms | 28.4 ms | 67.4% | Guided decoder improves some pairs but hurts others. |
| Oracle decoder selector | 8 | 91.8 ms | n/a | 72.1% | Decoder complementarity helps but remains far from target. |

## Verify Checksums

Run this from the repository root:

```powershell
@'
import hashlib
import json
from pathlib import Path

root = Path(".")
manifest = json.loads(
    (root / "results/final_dissertation_evidence_2026_05_12/MANIFEST.json").read_text()
)
for item in manifest["artifacts"]:
    path = root / item["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise SystemExit(f"Checksum mismatch: {path}")
print(f"Verified {len(manifest['artifacts'])} artifacts.")
'@ | .\.venv312\Scripts\python.exe -
```

## Dataset Layout

The SWD dataset is expected under `data/`. The built-in downloader expects the
public SWD archive:

```powershell
deepalign prepare-swd data
```

The evaluator resolves SWD scores from:

- `01_RawData/score_musicxml`
- `01_RawData/score_midi`

and audio measure annotations from:

- `02_Annotations/ann_audio_measure`

## Rerunning Frozen Evidence

Do not use reruns for additional model selection. If a final learned-student
full-SWD row is required, run it once, record it, and do not tune against it.

### Teacher Target Viability

The committed teacher evidence is already copied to:

```text
results/final_dissertation_evidence_2026_05_12/csv/teacher_results.csv
```

To regenerate the teacher route locally:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_initial_plan_supervised_20ms_sprint.ps1 `
  -Mode Teacher `
  -SwdPath data `
  -Device cuda
```

This requires `synctoolbox`.

### Best Learned Student Gate

The frozen best learned-student gate is:

```text
results/final_dissertation_evidence_2026_05_12/csv/best_student_gate.csv
```

To rerun the same 8-lied gate from the committed checkpoint:

```powershell
deepalign evaluate-swd data `
  --checkpoint results/final_dissertation_evidence_2026_05_12/checkpoints/best_student_debug_mae.pt `
  --methods deepalign `
  --device cuda `
  --cache-root .cache/cqt `
  --deep-hop 110 `
  --pool-size 1 `
  --deep-distance sqeuclidean `
  --deep-decode unconstrained `
  --lied D911-02 `
  --lied D911-06 `
  --lied D911-07 `
  --lied D911-17 `
  --lied D911-18 `
  --lied D911-20 `
  --lied D911-22 `
  --lied D911-24 `
  -o results/reproduced_best_student_gate.csv
```

### Optional Full-SWD Learned-Student Check

Use this only as a reporting completeness run:

```powershell
deepalign evaluate-swd data `
  --checkpoint results/final_dissertation_evidence_2026_05_12/checkpoints/best_student_debug_mae.pt `
  --methods deepalign `
  --device cuda `
  --cache-root .cache/cqt `
  --deep-hop 110 `
  --pool-size 1 `
  --deep-distance sqeuclidean `
  --deep-decode unconstrained `
  -o results/final_learned_student_full_swd_once.csv
```

## Optional Dependencies

### Synctoolbox / MrMsDTW

Required for teacher generation and MrMsDTW baselines:

```powershell
pip install -e ".[synctoolbox]"
```

On Windows, if installation fails, retry after pinning setuptools below 81:

```powershell
pip install "setuptools<81"
pip install -e ".[synctoolbox]" --no-build-isolation
```

### Transcription Features

The transcription-guided DeepAlign variants use Basic Pitch through the ONNX
Runtime path.

```powershell
pip install -e ".[transcription]"
pip install basic-pitch --no-deps
```

Expected import warnings about missing TensorFlow, CoreML, or TFLite are
acceptable when using ONNX Runtime. The evaluator caches Basic Pitch outputs
under `.cache/transcription/basic_pitch`.

### Matchmaker

Install Matchmaker from the official repository:

```powershell
pip install git+https://github.com/pymatchmaker/matchmaker.git@v0.2.1
```

Expected extra system dependencies:

- PortAudio
- FluidSynth
- a working C/C++ build toolchain on Windows

If Matchmaker import or runtime setup fails, keep `matchmaker` out of
`--methods`.

## Local Artifacts

The final evidence package is committed intentionally. Other datasets,
experiment checkpoints, logs, caches, and ad hoc result folders remain local
artifacts and are excluded through `.git/info/exclude`.
