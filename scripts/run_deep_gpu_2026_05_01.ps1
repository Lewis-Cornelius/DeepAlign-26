$ErrorActionPreference = "Stop"

$Root = "C:\code\dis-alignment"
$Python = Join-Path $Root ".venv312\Scripts\python.exe"
$RunId = "deep_run_ar_fused_2026_05_01"
$Config = Join-Path $Root "config\deepalign_deep_run_2026_05_01.yaml"
$SwdPath = Join-Path $Root "data"
$CheckpointDir = Join-Path $Root "checkpoints\$RunId"
$DeepCsv = Join-Path $Root "results\swd_$RunId.csv"
$MergedCsv = Join-Path $Root "results\swd_${RunId}_merged_with_recorded_baselines.csv"
$FigureDir = Join-Path $Root "figures\$RunId"
$MergedFigureDir = Join-Path $Root "figures\${RunId}_merged"

Set-Location $Root
New-Item -ItemType Directory -Force -Path "logs", "results", "figures", "checkpoints" | Out-Null

function Run-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "===== $Name ====="
    Write-Host "Started: $(Get-Date -Format o)"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Host "Finished: $(Get-Date -Format o)"
}

Run-Step "GPU training" @(
    "-u", "-m", "dis_alignment.cli", "train",
    "--config", $Config,
    "--swd-path", $SwdPath,
    "--output-dir", $CheckpointDir,
    "--device", "cuda",
    "--cache-spectrograms",
    "--no-augment"
)

Run-Step "Full SWD DeepAlign transcription-fused evaluation" @(
    "-u", "-m", "dis_alignment.cli", "evaluate-swd", $SwdPath,
    "--checkpoint", (Join-Path $CheckpointDir "best_model_debug_ar50.pt"),
    "--methods", "deepalign",
    "--deep-decode", "deepalign_transcription_fused",
    "--device", "cuda",
    "--cache-root", ".cache/cqt",
    "--transcription-cache-root", ".cache/transcription/basic_pitch",
    "--output", $DeepCsv
)

Run-Step "Merge with recorded baseline comparison" @(
    "-u", "-m", "dis_alignment.cli", "merge-results",
    "results\swd_evaluation_2026_04_19.csv",
    $DeepCsv,
    "--output", $MergedCsv
)

Run-Step "Visualize DeepAlign-only result" @(
    "-u", "-m", "dis_alignment.cli", "visualize",
    $DeepCsv,
    "--output-dir", $FigureDir
)

Run-Step "Visualize merged comparison" @(
    "-u", "-m", "dis_alignment.cli", "visualize",
    $MergedCsv,
    "--output-dir", $MergedFigureDir
)

Run-Step "Analyze merged comparison" @(
    "-u", "-m", "dis_alignment.cli", "analyze",
    $MergedCsv,
    "--baseline", "chroma_dtw",
    "--candidate", "deepalign:deepalign_transcription_fused",
    "--metric", "mae"
)

Write-Host ""
Write-Host "===== Deep run complete ====="
Write-Host "Checkpoint dir: $CheckpointDir"
Write-Host "DeepAlign CSV: $DeepCsv"
Write-Host "Merged CSV: $MergedCsv"
Write-Host "Figures: $FigureDir"
Write-Host "Merged figures: $MergedFigureDir"
