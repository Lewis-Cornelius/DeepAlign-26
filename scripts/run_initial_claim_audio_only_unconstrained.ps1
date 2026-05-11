param(
    [ValidateSet("All", "Bootstrap", "Teacher", "TeacherGate", "TeacherFull", "Pretrain", "Pseudo", "Train", "Gate", "Full", "Summary")]
    [string]$Mode = "All",
    [string]$SwdPath = "data",
    [string]$Config = "config/deepalign_initial_claim_pseudo_bootstrap.yaml",
    [string]$PretrainConfig = "config/deepalign_initial_claim_self_audio_pretrain.yaml",
    [string]$TeacherDir = "results/teacher_initial_claim_audio_only",
    [string]$TeacherGateDir = "results/teacher_initial_claim_audio_only_gate8",
    [string]$PseudoTeacherDir = "results/pseudo_initial_claim_audio_only",
    [string]$PretrainCheckpoint = "checkpoints/initial_claim_self_audio_pretrain/best_model.pt",
    [string]$Checkpoint = "checkpoints/initial_claim_pseudo_bootstrap/best_model_debug_ar50.pt",
    [string]$ResultsDir = "results/initial_claim_audio_only_unconstrained",
    [string]$Device = "cuda",
    [switch]$RequireSotaStretch
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv312\Scripts\python.exe"
$GateLieder = @(
    "D911-02",
    "D911-06",
    "D911-07",
    "D911-17",
    "D911-18",
    "D911-20",
    "D911-22",
    "D911-24"
)

function Invoke-CheckedPython {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "=== $Label ==="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-ResultGate {
    param(
        [string]$Label,
        [string]$CsvPath,
        [double]$MaxMaeMs,
        [double]$MinAr50Pct,
        [int]$ExpectedPairs,
        [switch]$RequireDeepalignUnconstrained,
        [string[]]$Lieder = @()
    )
    $Script = @"
import sys
import pandas as pd

csv_path = sys.argv[1]
max_mae_ms = float(sys.argv[2])
min_ar50_pct = float(sys.argv[3])
expected_pairs = int(sys.argv[4])
require_deepalign = sys.argv[5] == "1"
lieder = [item for item in sys.argv[6:] if item]

df = pd.read_csv(csv_path)
if lieder and "group_id" in df.columns:
    df = df[df["group_id"].astype(str).isin(lieder)].copy()
if require_deepalign:
    if "method" not in df.columns:
        raise SystemExit("Missing method column for DeepAlign gate")
    df = df[df["method"].astype(str).eq("deepalign")].copy()
    if df.empty:
        raise SystemExit("No deepalign rows found")
    if "deep_decode" not in df.columns or set(df["deep_decode"].astype(str)) != {"unconstrained"}:
        raise SystemExit("Headline gate requires deep_decode=unconstrained only")
    if "pool_size" in df.columns and set(df["pool_size"].astype(float)) != {1.0}:
        raise SystemExit("Headline gate requires pool_size=1")

pairs = len(df)
mae_ms = float(df["mae"].mean()) * 1000.0
ar50_pct = float(df["ar_50ms"].mean()) * 100.0
print(f"pairs={pairs} MAE={mae_ms:.3f} ms AR@50={ar50_pct:.3f}%")
if pairs != expected_pairs:
    raise SystemExit(f"Expected {expected_pairs} pairs, found {pairs}")
if not mae_ms < max_mae_ms:
    raise SystemExit(f"MAE gate failed: {mae_ms:.3f} ms is not below {max_mae_ms:.3f} ms")
if not ar50_pct > min_ar50_pct:
    raise SystemExit(f"AR@50 gate failed: {ar50_pct:.3f}% is not above {min_ar50_pct:.3f}%")
"@
    $Require = if ($RequireDeepalignUnconstrained) { "1" } else { "0" }
    $Arguments = @("-c", $Script, $CsvPath, "$MaxMaeMs", "$MinAr50Pct", "$ExpectedPairs", $Require)
    $Arguments += $Lieder
    Invoke-CheckedPython $Label $Arguments
}

function Invoke-TeacherGenerate {
    param(
        [string]$Label,
        [string]$OutputDir,
        [string[]]$Lieder = @(),
        [switch]$FailBelowGate
    )
    $Arguments = @(
        "-u", "-m", "dis_alignment.cli", "generate-teacher-paths", $SwdPath,
        "--output-dir", $OutputDir,
        "--hop-length", "110",
        "--backend", "mrmsdtw",
        "--memory-limit-mb", "500",
        "--chroma-weight", "1.0",
        "--dlnco-weight", "1.0",
        "--spectral-flux-weight", "0.5",
        "--trim-top-db", "40",
        "--estimate-chroma-shift",
        "--chroma-shift-max-frames", "1500",
        "--confidence-local-radius-frames", "24",
        "--confidence-exclusion-radius-frames", "2",
        "--min-ar50", "0.90",
        "--max-mae-ms", "50"
    )
    foreach ($Lied in $Lieder) {
        $Arguments += @("--lied", $Lied)
    }
    if (-not $FailBelowGate) {
        $Arguments += "--no-fail-below-gate"
    }
    Invoke-CheckedPython $Label $Arguments
}

function Invoke-TeacherGate {
    Invoke-TeacherGenerate `
        -Label "Gate 0/1: generate 8-lied audio-only teacher paths" `
        -OutputDir $TeacherGateDir `
        -Lieder $GateLieder `
        -FailBelowGate
    Invoke-ResultGate `
        -Label "Gate 0: oracle metric sanity" `
        -CsvPath (Join-Path $TeacherGateDir "oracle_results.csv") `
        -MaxMaeMs 0.001 `
        -MinAr50Pct 99.999 `
        -ExpectedPairs 8
    Invoke-ResultGate `
        -Label "Gate 1: 8-lied audio-only teacher quality" `
        -CsvPath (Join-Path $TeacherGateDir "teacher_results.csv") `
        -MaxMaeMs 50 `
        -MinAr50Pct 90 `
        -ExpectedPairs 8 `
        -Lieder $GateLieder
}

function Invoke-TeacherFull {
    Invoke-TeacherGenerate `
        -Label "Generate full audio-only teacher paths for training" `
        -OutputDir $TeacherDir
}

function Invoke-Pretrain {
    Invoke-CheckedPython "Stage 0: self-audio representation pretrain" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $PretrainConfig,
        "--swd-path", $SwdPath,
        "--device", $Device
    )
}

function Invoke-Pseudo {
    Invoke-CheckedPython "Mine audio-only coarse pseudo-teacher paths" @(
        "-u", "-m", "dis_alignment.cli", "mine-pseudo-teacher-paths", $SwdPath,
        "--checkpoint", $PretrainCheckpoint,
        "--output-dir", $PseudoTeacherDir,
        "--device", $Device,
        "--cache-root", ".cache/cqt",
        "--hop-length", "440",
        "--min-confidence", "0.55",
        "--min-gap-frames", "1",
        "--max-anchors", "4096",
        "--band-radius-sec", "-1",
        "--coarse-prior", "full",
        "--coarse-local-radius-sec", "2.0",
        "--coarse-distance", "cosine",
        "--use-coarse-path-only",
        "--min-anchors", "16"
    )
}

function Invoke-Train {
    Invoke-CheckedPython "Stage 1/2: train initial-claim audio-only DeepAlign" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Config,
        "--swd-path", $SwdPath,
        "--device", $Device
    )
}

function Invoke-Gate {
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
    $Arguments = @(
        "-u", "-m", "dis_alignment.cli", "evaluate-swd", $SwdPath,
        "--checkpoint", $Checkpoint,
        "--methods", "deepalign",
        "--device", $Device,
        "--cache-root", ".cache/cqt",
        "--deep-hop", "110",
        "--pool-size", "1",
        "--deep-distance", "sqeuclidean",
        "--deep-decode", "unconstrained",
        "-o", (Join-Path $ResultsDir "gate_8_lieder.csv")
    )
    foreach ($Lied in $GateLieder) {
        $Arguments += @("--lied", $Lied)
    }
    Invoke-CheckedPython "Gate 2: strict 8-lied unconstrained DeepAlign" $Arguments
    Invoke-ResultGate `
        -Label "Gate 2 check: <50 ms and >95% AR@50" `
        -CsvPath (Join-Path $ResultsDir "gate_8_lieder.csv") `
        -MaxMaeMs 50 `
        -MinAr50Pct 95 `
        -ExpectedPairs 8 `
        -RequireDeepalignUnconstrained
}

function Invoke-Full {
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
    Invoke-CheckedPython "Gate 3: strict full-SWD unconstrained DeepAlign" @(
        "-u", "-m", "dis_alignment.cli", "evaluate-swd", $SwdPath,
        "--checkpoint", $Checkpoint,
        "--methods", "deepalign",
        "--device", $Device,
        "--cache-root", ".cache/cqt",
        "--deep-hop", "110",
        "--pool-size", "1",
        "--deep-distance", "sqeuclidean",
        "--deep-decode", "unconstrained",
        "-o", (Join-Path $ResultsDir "full_swd.csv")
    )
    Invoke-ResultGate `
        -Label "Gate 3 check: <50 ms and >98% AR@50" `
        -CsvPath (Join-Path $ResultsDir "full_swd.csv") `
        -MaxMaeMs 50 `
        -MinAr50Pct 98 `
        -ExpectedPairs 24 `
        -RequireDeepalignUnconstrained
    if ($RequireSotaStretch) {
        Invoke-ResultGate `
            -Label "Gate 4 check: <20 ms and >98% AR@50" `
            -CsvPath (Join-Path $ResultsDir "full_swd.csv") `
            -MaxMaeMs 20 `
            -MinAr50Pct 98 `
            -ExpectedPairs 24 `
            -RequireDeepalignUnconstrained
    }
}

function Invoke-Summary {
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
    Invoke-CheckedPython "Summarize headline and diagnostic CSVs" @(
        "-u", "scripts/summarize_swd_results.py",
        "--results-dir", $ResultsDir,
        "--reference", "audio_teacher=$(Join-Path $TeacherDir 'teacher_results.csv')",
        "--output", (Join-Path $ResultsDir "summary.csv")
    )
}

if ($Mode -eq "All" -or $Mode -eq "Teacher") {
    Invoke-TeacherGate
    Invoke-TeacherFull
}
if ($Mode -eq "Bootstrap") {
    Invoke-Pretrain
    Invoke-Pseudo
    Invoke-Train
    Invoke-Gate
}
if ($Mode -eq "TeacherGate") {
    Invoke-TeacherGate
}
if ($Mode -eq "TeacherFull") {
    Invoke-TeacherFull
}
if ($Mode -eq "All" -or $Mode -eq "Pretrain") {
    Invoke-Pretrain
}
if ($Mode -eq "All" -or $Mode -eq "Pseudo") {
    Invoke-Pseudo
}
if ($Mode -eq "All" -or $Mode -eq "Train") {
    Invoke-Train
}
if ($Mode -eq "All" -or $Mode -eq "Gate") {
    Invoke-Gate
}
if ($Mode -eq "All" -or $Mode -eq "Full") {
    Invoke-Full
}
if ($Mode -eq "All" -or $Mode -eq "Summary") {
    Invoke-Summary
}
