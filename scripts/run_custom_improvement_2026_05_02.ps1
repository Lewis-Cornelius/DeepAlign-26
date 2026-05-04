$ErrorActionPreference = "Stop"

$Root = "C:\code\dis-alignment"
$Python = Join-Path $Root ".venv312\Scripts\python.exe"
$RunId = "custom_improvement_2026_05_02"
$SwdPath = Join-Path $Root "data"
$OutDir = Join-Path $Root "results\$RunId"
$SummaryPath = Join-Path $Root "results\${RunId}_summary.csv"
$FigDir = Join-Path $Root "figures\$RunId"

$Stage1Checkpoint = Join-Path $Root "checkpoints\cycle1_stage1_local_precision_2026_04_19\best_model.pt"
$Stage2Checkpoint = Join-Path $Root "checkpoints\cycle1_stage2_long_context_2026_04_20\best_model.pt"
$Anchor010Checkpoint = Join-Path $Root "checkpoints\anchor_cycle_w010_2026_04_20\best_model_debug_ar50.pt"
$Anchor025Checkpoint = Join-Path $Root "checkpoints\anchor_cycle_w025_2026_04_20\best_model_debug_ar50.pt"

$ReferenceCurrentBest = Join-Path $Root "results\swd_deepalign_transcription_fused_full_2026_04_25.csv"
$ReferenceStage1 = Join-Path $Root "results\swd_deepalign_stage1_epoch3_full_2026_04_20.csv"
$ReferenceDeepRun = Join-Path $Root "results\swd_deep_run_ar_fused_2026_05_01.csv"

Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs", "results", "figures", "checkpoints", $OutDir, $FigDir | Out-Null

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$ContinueOnError
    )

    Write-Host ""
    Write-Host "===== $Name ====="
    Write-Host "Started: $(Get-Date -Format o)"
    try {
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
        Write-Host "Finished: $(Get-Date -Format o)"
        return $true
    }
    catch {
        Write-Host "FAILED: $($_.Exception.Message)"
        if (-not $ContinueOnError) {
            throw
        }
        return $false
    }
}

function Write-ImprovementSummary {
    $SummaryCode = @'
import csv
import sys
from pathlib import Path

import pandas as pd

out_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
reference_paths = [
    ("reference_current_best_apr25", Path(sys.argv[3])),
    ("reference_stage1_unconstrained_apr20", Path(sys.argv[4])),
    ("reference_failed_deep_run_may01", Path(sys.argv[5])),
]

rows = []
seen = set()
for label, path in reference_paths + [(path.stem, path) for path in sorted(out_dir.glob("*.csv"))]:
    if not path.exists() or path in seen:
        continue
    seen.add(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        rows.append({"label": label, "csv": str(path), "status": f"read_failed: {exc}"})
        continue
    if "method" in df.columns:
        df = df[df["method"].astype(str).eq("deepalign")].copy()
    if df.empty:
        rows.append({"label": label, "csv": str(path), "status": "no_deepalign_rows"})
        continue
    def mean_col(name):
        return float(df[name].mean()) if name in df.columns else float("nan")
    decode = ""
    if "deep_decode" in df.columns:
        values = sorted(str(v) for v in df["deep_decode"].dropna().unique())
        decode = "|".join(values[:3])
    rows.append(
        {
            "label": label,
            "status": "ok",
            "pairs": int(len(df)),
            "deep_decode": decode,
            "mae_ms": round(mean_col("mae") * 1000.0, 3),
            "median_ae_ms": round(mean_col("median_ae") * 1000.0, 3),
            "ar_50_pct": round(mean_col("ar_50ms") * 100.0, 3),
            "ar_100_pct": round(mean_col("ar_100ms") * 100.0, 3),
            "ar_200_pct": round(mean_col("ar_200ms") * 100.0, 3),
            "runtime_s": round(mean_col("runtime_s"), 3),
            "csv": str(path),
        }
    )

ok_rows = [row for row in rows if row.get("status") == "ok"]
bad_rows = [row for row in rows if row.get("status") != "ok"]
ok_rows.sort(key=lambda row: (-row["ar_50_pct"], row["mae_ms"]))
rows = ok_rows + bad_rows

fieldnames = [
    "label",
    "status",
    "pairs",
    "deep_decode",
    "mae_ms",
    "median_ae_ms",
    "ar_50_pct",
    "ar_100_pct",
    "ar_200_pct",
    "runtime_s",
    "csv",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary written: {summary_path}")
if ok_rows:
    best = ok_rows[0]
    print(
        "Current leader: "
        f"{best['label']} | AR@50={best['ar_50_pct']:.3f}% | "
        f"MAE={best['mae_ms']:.1f} ms | Median={best['median_ae_ms']:.1f} ms"
    )
    print("Top rows:")
    for row in ok_rows[:8]:
        print(
            f"  {row['label']}: AR@50={row['ar_50_pct']:.3f}% "
            f"MAE={row['mae_ms']:.1f}ms Median={row['median_ae_ms']:.1f}ms"
        )
'@
    $SummaryCode | & $Python - $OutDir $SummaryPath $ReferenceCurrentBest $ReferenceStage1 $ReferenceDeepRun
}

function Invoke-EvalCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Checkpoint,
        [string]$DeepDecode = "deepalign_transcription_fused",
        [double]$FusionDeepWeight = 1.0,
        [double]$FusionOnsetWeight = 1.0,
        [double]$FusionNoteWeight = 0.75,
        [double]$FusionDlncoWeight = 0.5,
        [double]$FusionChromaWeight = 0.25,
        [Nullable[double]]$ScoreRefineRadiusSec = $null
    )

    if (-not (Test-Path $Checkpoint)) {
        Write-Host ""
        Write-Host "===== Skipping $Label ====="
        Write-Host "Missing checkpoint: $Checkpoint"
        return
    }

    $OutputCsv = Join-Path $OutDir "$Label.csv"
    $Arguments = @(
        "-u", "-m", "dis_alignment.cli", "evaluate-swd", $SwdPath,
        "--checkpoint", $Checkpoint,
        "--methods", "deepalign",
        "--deep-decode", $DeepDecode,
        "--device", "cuda",
        "--cache-root", ".cache/cqt",
        "--transcription-cache-root", ".cache/transcription/basic_pitch",
        "--fusion-deep-weight", ([string]$FusionDeepWeight),
        "--fusion-onset-weight", ([string]$FusionOnsetWeight),
        "--fusion-note-weight", ([string]$FusionNoteWeight),
        "--fusion-dlnco-weight", ([string]$FusionDlncoWeight),
        "--fusion-chroma-weight", ([string]$FusionChromaWeight),
        "--output", $OutputCsv
    )
    if ($null -ne $ScoreRefineRadiusSec) {
        $Arguments += @("--score-refine-radius-sec", ([string]$ScoreRefineRadiusSec))
    }

    $Succeeded = Invoke-PythonStep "Evaluate $Label" $Arguments -ContinueOnError
    if ($Succeeded) {
        Write-ImprovementSummary
    }
}

function Invoke-TrainingCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$CheckpointDir
    )

    $Succeeded = Invoke-PythonStep "Train $Label" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $ConfigPath,
        "--swd-path", $SwdPath,
        "--output-dir", $CheckpointDir,
        "--device", "cuda",
        "--cache-spectrograms",
        "--no-augment"
    ) -ContinueOnError

    if (-not $Succeeded) {
        return
    }

    $CandidateCheckpoint = Join-Path $CheckpointDir "best_model_debug_ar50.pt"
    if (-not (Test-Path $CandidateCheckpoint)) {
        $CandidateCheckpoint = Join-Path $CheckpointDir "best_model.pt"
    }
    Invoke-EvalCandidate -Label "${Label}_default_fused" -Checkpoint $CandidateCheckpoint
}

Write-Host "===== Custom improvement run ====="
Write-Host "Run id: $RunId"
Write-Host "Started: $(Get-Date -Format o)"
Write-Host "Baseline to beat: AR@50 41.5%, MAE 367 ms from Apr 25 transcription-fused DeepAlign."

Write-ImprovementSummary

Invoke-EvalCandidate -Label "stage1_default_fused" -Checkpoint $Stage1Checkpoint
Invoke-EvalCandidate -Label "stage1_onset2_note05_dlnco05_chroma025_deep1" -Checkpoint $Stage1Checkpoint -FusionOnsetWeight 2.0 -FusionNoteWeight 0.5 -FusionDlncoWeight 0.5 -FusionChromaWeight 0.25 -FusionDeepWeight 1.0
Invoke-EvalCandidate -Label "stage1_onset2_note0_dlnco1_chroma0_deep1" -Checkpoint $Stage1Checkpoint -FusionOnsetWeight 2.0 -FusionNoteWeight 0.0 -FusionDlncoWeight 1.0 -FusionChromaWeight 0.0 -FusionDeepWeight 1.0
Invoke-EvalCandidate -Label "stage1_onset3_note025_dlnco1_chroma0_deep1" -Checkpoint $Stage1Checkpoint -FusionOnsetWeight 3.0 -FusionNoteWeight 0.25 -FusionDlncoWeight 1.0 -FusionChromaWeight 0.0 -FusionDeepWeight 1.0
Invoke-EvalCandidate -Label "stage1_deep05_onset2_note05_dlnco1_chroma025" -Checkpoint $Stage1Checkpoint -FusionDeepWeight 0.5 -FusionOnsetWeight 2.0 -FusionNoteWeight 0.5 -FusionDlncoWeight 1.0 -FusionChromaWeight 0.25
Invoke-EvalCandidate -Label "stage1_nodeep_onset2_note075_dlnco05_chroma025" -Checkpoint $Stage1Checkpoint -FusionDeepWeight 0.0 -FusionOnsetWeight 2.0 -FusionNoteWeight 0.75 -FusionDlncoWeight 0.5 -FusionChromaWeight 0.25

Invoke-EvalCandidate -Label "stage1_refined_r005" -Checkpoint $Stage1Checkpoint -DeepDecode "deepalign_transcription_fused_refined" -ScoreRefineRadiusSec 0.05
Invoke-EvalCandidate -Label "stage1_refined_r010" -Checkpoint $Stage1Checkpoint -DeepDecode "deepalign_transcription_fused_refined" -ScoreRefineRadiusSec 0.10
Invoke-EvalCandidate -Label "stage1_score_guided_r000" -Checkpoint $Stage1Checkpoint -DeepDecode "deepalign_score_guided_refined" -ScoreRefineRadiusSec 0.0
Invoke-EvalCandidate -Label "stage1_score_guided_r005" -Checkpoint $Stage1Checkpoint -DeepDecode "deepalign_score_guided_refined" -ScoreRefineRadiusSec 0.05
Invoke-EvalCandidate -Label "stage1_score_guided_r010" -Checkpoint $Stage1Checkpoint -DeepDecode "deepalign_score_guided_refined" -ScoreRefineRadiusSec 0.10

Invoke-EvalCandidate -Label "stage2_default_fused" -Checkpoint $Stage2Checkpoint
Invoke-EvalCandidate -Label "anchor_w010_default_fused" -Checkpoint $Anchor010Checkpoint
Invoke-EvalCandidate -Label "anchor_w025_default_fused" -Checkpoint $Anchor025Checkpoint

Invoke-TrainingCandidate -Label "gentle_lr1e5" -ConfigPath (Join-Path $Root "config\custom_runs\gentle_finetune_lr1e5_2026_05_02.yaml") -CheckpointDir (Join-Path $Root "checkpoints\custom_gentle_lr1e5_2026_05_02")
Invoke-TrainingCandidate -Label "gentle_lr3e5" -ConfigPath (Join-Path $Root "config\custom_runs\gentle_finetune_lr3e5_2026_05_02.yaml") -CheckpointDir (Join-Path $Root "checkpoints\custom_gentle_lr3e5_2026_05_02")

Write-ImprovementSummary

Write-Host ""
Write-Host "===== Custom improvement run complete ====="
Write-Host "Finished: $(Get-Date -Format o)"
Write-Host "Candidate CSVs: $OutDir"
Write-Host "Summary CSV: $SummaryPath"
