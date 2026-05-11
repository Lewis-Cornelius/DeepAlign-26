param(
    [ValidateSet("All", "Teacher", "CalibratedTeacher", "Train", "Gate", "Full")]
    [string]$Mode = "All",
    [string]$SwdPath = "data",
    [string]$Config = "config/deepalign_sota_audio_only_2026_05_09.yaml",
    [string]$TeacherDir = "results/teacher_audio_only_sota_2026_05_09",
    [string]$CalibratedTeacherDir = "results/teacher_anchor_calibrated_sota_2026_05_10",
    [string]$Checkpoint = "checkpoints/sota_audio_only_2026_05_09/best_model_debug_ar50.pt",
    [string]$ResultsDir = "results/audio_only_sota_2026_05_09",
    [string]$Device = "cuda"
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

function Invoke-Teacher {
    Invoke-CheckedPython "Gate 0/1: generate and validate audio-only teacher" @(
        "-u", "-m", "dis_alignment.cli", "generate-teacher-paths", $SwdPath,
        "--output-dir", $TeacherDir,
        "--hop-length", "110",
        "--backend", "mrmsdtw",
        "--memory-limit-mb", "500",
        "--chroma-weight", "1.0",
        "--dlnco-weight", "1.0",
        "--spectral-flux-weight", "0.5",
        "--trim-top-db", "40",
        "--estimate-chroma-shift",
        "--chroma-shift-max-frames", "1500",
        "--min-ar50", "0.95"
    )
}

function Invoke-CalibratedTeacher {
    Invoke-CheckedPython "Supervised artifact: generate anchor-calibrated teacher paths" @(
        "-u", "-m", "dis_alignment.cli", "generate-teacher-paths", $SwdPath,
        "--output-dir", $CalibratedTeacherDir,
        "--hop-length", "110",
        "--backend", "mrmsdtw",
        "--memory-limit-mb", "500",
        "--chroma-weight", "1.0",
        "--dlnco-weight", "1.0",
        "--spectral-flux-weight", "0.5",
        "--trim-top-db", "40",
        "--estimate-chroma-shift",
        "--chroma-shift-max-frames", "1500",
        "--anchor-calibrate",
        "--min-ar50", "0.98"
    )
}

function Invoke-Train {
    Invoke-CheckedPython "Stage 1/2: train teacher-distilled audio-only DeepAlign" @(
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
    Invoke-CheckedPython "Gate 2: strict 8-lied DeepAlign promotion" $Arguments
}

function Invoke-Full {
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
    Invoke-CheckedPython "Gate 3: strict full-SWD DeepAlign result" @(
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
}

if ($Mode -eq "All" -or $Mode -eq "Teacher") {
    Invoke-Teacher
}
if ($Mode -eq "CalibratedTeacher") {
    Invoke-CalibratedTeacher
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
