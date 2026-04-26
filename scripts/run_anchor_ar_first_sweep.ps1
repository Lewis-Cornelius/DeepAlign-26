param(
    [string]$PythonExe = ".\.venv312\Scripts\python.exe",
    [string]$SwdPath = "data",
    [string]$BaseCheckpoint = "checkpoints\cycle1_stage1_local_precision_2026_04_19\best_model.pt",
    [string]$ConfigPath = "config\deepalign.yaml",
    [string]$CacheRoot = ".cache\cqt",
    [string]$Device = "cuda",
    [double[]]$AnchorWeights = @(0.05, 0.10, 0.25)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Format-AnchorTag {
    param([double]$Weight)

    return [int]([math]::Round($Weight * 100))
}

function Get-Mean {
    param(
        [object[]]$Rows,
        [string]$Property
    )

    if (-not $Rows -or $Rows.Count -eq 0) {
        return [double]::NaN
    }

    $values = foreach ($row in $Rows) {
        if ($null -ne $row.$Property -and $row.$Property -ne "") {
            [double]$row.$Property
        }
    }
    if (-not $values -or $values.Count -eq 0) {
        return [double]::NaN
    }
    return ($values | Measure-Object -Average).Average
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host "`n>>> $Label"
    Write-Host "    $PythonExe $($Arguments -join ' ')"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$stamp = (Get-Date).ToString("yyyy_MM_dd")
$summaryPath = Join-Path "results" "anchor_ar_first_sweep_summary_${stamp}.csv"
$summaryRows = @()

Write-Host "=== Anchor AR-first sweep ($stamp) ==="
Write-Host "Repo root: $repoRoot"
Write-Host "Base checkpoint: $BaseCheckpoint"
Write-Host "Weights: $($AnchorWeights -join ', ')"

foreach ($weight in $AnchorWeights) {
    $tag = "{0:D3}" -f (Format-AnchorTag -Weight $weight)
    $outputDir = Join-Path "checkpoints" "anchor_cycle_w${tag}_${stamp}"
    $resultCsv = Join-Path "results" "swd_anchor_cycle_w${tag}_full_${stamp}.csv"
    $bestCheckpoint = Join-Path $outputDir "best_model_debug_ar50.pt"

    Invoke-Step -Label "Train anchor pilot w=$weight" -Arguments @(
        "-m", "dis_alignment.cli", "train",
        "--config", $ConfigPath,
        "--swd-path", $SwdPath,
        "--output-dir", $outputDir,
        "--epochs", "5",
        "--batch-size", "4",
        "--max-length-sec", "15",
        "--device", $Device,
        "--resume-from", $BaseCheckpoint,
        "--selection-metric", "debug_ar50",
        "--segment-sampling", "aligned_measures",
        "--samples-per-epoch", "64",
        "--cache-spectrograms",
        "--cache-root", $CacheRoot,
        "--deep-decode", "unconstrained",
        "--start-gamma", "1.0",
        "--end-gamma", "0.1",
        "--anchor-loss-weight", "$weight",
        "--anchor-temperature", "0.1",
        "--anchor-min-anchor-gap", "1",
        "--no-augment"
    )

    if (-not (Test-Path $bestCheckpoint)) {
        throw "Expected checkpoint missing: $bestCheckpoint"
    }

    Invoke-Step -Label "Evaluate anchor pilot w=$weight" -Arguments @(
        "-m", "dis_alignment.cli", "evaluate-swd",
        $SwdPath,
        "--checkpoint", $bestCheckpoint,
        "--methods", "deepalign",
        "--output", $resultCsv,
        "--device", $Device,
        "--cache-root", $CacheRoot,
        "--deep-decode", "unconstrained",
        "--quiet"
    )

    if (-not (Test-Path $resultCsv)) {
        throw "Expected evaluation CSV missing: $resultCsv"
    }

    $rows = @(Import-Csv $resultCsv | Where-Object { $_.method -eq "deepalign" })
    $summaryRows += [pscustomobject]@{
        anchor_loss_weight = $weight
        checkpoint = $bestCheckpoint
        csv = $resultCsv
        pairs = $rows.Count
        mae_ms = [math]::Round((Get-Mean -Rows $rows -Property "mae") * 1000.0, 1)
        median_ae_ms = [math]::Round((Get-Mean -Rows $rows -Property "median_ae") * 1000.0, 1)
        ar_50ms_pct = [math]::Round((Get-Mean -Rows $rows -Property "ar_50ms") * 100.0, 1)
        ar_100ms_pct = [math]::Round((Get-Mean -Rows $rows -Property "ar_100ms") * 100.0, 1)
        ar_200ms_pct = [math]::Round((Get-Mean -Rows $rows -Property "ar_200ms") * 100.0, 1)
        runtime_s = [math]::Round((Get-Mean -Rows $rows -Property "runtime_s"), 2)
    }

    $latest = $summaryRows[-1]
    Write-Host ("Summary w={0}: MAE {1} ms | AR@50 {2}% | AR@100 {3}% | pairs {4}" -f `
        $latest.anchor_loss_weight, $latest.mae_ms, $latest.ar_50ms_pct, $latest.ar_100ms_pct, $latest.pairs)
}

$summaryRows |
    Sort-Object @{Expression = "ar_50ms_pct"; Descending = $true}, @{Expression = "mae_ms"; Descending = $false} |
    Export-Csv -Path $summaryPath -NoTypeInformation

$winner = Import-Csv $summaryPath | Select-Object -First 1
Write-Host "`n=== Sweep summary saved to $summaryPath ==="
Write-Host ("Winning weight: {0} | MAE {1} ms | AR@50 {2}% | CSV {3}" -f `
    $winner.anchor_loss_weight, $winner.mae_ms, $winner.ar_50ms_pct, $winner.csv)
