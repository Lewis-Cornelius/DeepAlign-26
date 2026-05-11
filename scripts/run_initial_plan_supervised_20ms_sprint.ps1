param(
    [ValidateSet("All", "Teacher", "Train", "Gate", "Full", "Smoke")]
    [string]$Mode = "All",
    [string]$SwdPath = "data",
    [string]$Config = "config/deepalign_initial_plan_supervised_20ms.yaml",
    [string]$TeacherDir = "results/teacher_target_20ms",
    [string]$Checkpoint = "checkpoints/initial_plan_supervised_20ms/best_model_balanced.pt",
    [string]$ResultsDir = "results/initial_plan_supervised_20ms",
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
    param([string]$Label, [string[]]$Arguments)
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
        [int]$ExpectedPairs
    )
    Invoke-CheckedPython $Label @(
        "scripts/check_strict_result_gate.py",
        $CsvPath,
        "$MaxMaeMs",
        "$MinAr50Pct",
        "$ExpectedPairs"
    )
}

function Invoke-TeacherResultGate {
    param(
        [string]$Label,
        [string]$CsvPath,
        [double]$MaxMaeMs,
        [double]$MinAr50Pct,
        [int]$ExpectedPairs,
        [string]$Method
    )
    Invoke-CheckedPython $Label @(
        "scripts/check_teacher_result_gate.py",
        $CsvPath,
        "$MaxMaeMs",
        "$MinAr50Pct",
        "$ExpectedPairs",
        "--method",
        $Method
    )
}

function Invoke-StrictEval {
    param(
        [string]$Label,
        [string]$CheckpointPath,
        [string]$Output,
        [switch]$GateOnly
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $Output -Parent) | Out-Null
    $Arguments = @(
        "-u", "-m", "dis_alignment.cli", "evaluate-swd", $SwdPath,
        "--checkpoint", $CheckpointPath,
        "--methods", "deepalign",
        "--device", $Device,
        "--cache-root", ".cache/cqt",
        "--deep-hop", "110",
        "--pool-size", "1",
        "--deep-distance", "sqeuclidean",
        "--deep-decode", "unconstrained",
        "-o", $Output
    )
    if ($GateOnly) {
        foreach ($Lied in $GateLieder) {
            $Arguments += @("--lied", $Lied)
        }
    }
    Invoke-CheckedPython $Label $Arguments
}

function Invoke-Teacher {
    Invoke-CheckedPython "Generate anchor-calibrated 20 ms teacher targets" @(
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
        "--anchor-calibrate",
        "--min-ar50", "0.98",
        "--max-mae-ms", "20"
    )
    Invoke-TeacherResultGate `
        -Label "Teacher target gate: <20 ms and >98% AR@50" `
        -CsvPath (Join-Path $TeacherDir "teacher_results.csv") `
        -MaxMaeMs 20 `
        -MinAr50Pct 98 `
        -ExpectedPairs 24 `
        -Method "anchor_calibrated_audio_teacher"
}

function Invoke-Train {
    Invoke-CheckedPython "Train original-plan supervised Soft-DTW DeepAlign" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Config,
        "--swd-path", $SwdPath,
        "--device", $Device
    )
}

function Invoke-Gate {
    $Csv = Join-Path $ResultsDir "gate_8_lieder.csv"
    Invoke-StrictEval -Label "8-lied supervised target gate" -CheckpointPath $Checkpoint -Output $Csv -GateOnly
    Invoke-ResultGate -Label "8-lied gate: <50 ms and >95% AR@50" -CsvPath $Csv -MaxMaeMs 50 -MinAr50Pct 95 -ExpectedPairs 8
}

function Invoke-Full {
    $Csv = Join-Path $ResultsDir "full_24_pair_20ms.csv"
    Invoke-StrictEval -Label "Full SWD supervised target eval" -CheckpointPath $Checkpoint -Output $Csv
    Invoke-ResultGate -Label "Full target gate: <20 ms and >98% AR@50" -CsvPath $Csv -MaxMaeMs 20 -MinAr50Pct 98 -ExpectedPairs 24
}

function Invoke-Smoke {
    Invoke-CheckedPython "Smoke train original-plan supervised config" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Config,
        "--swd-path", $SwdPath,
        "--device", $Device,
        "--epochs", "1",
        "--samples-per-epoch", "8",
        "--output-dir", "checkpoints/initial_plan_supervised_20ms_smoke",
        "--alignment-eval-every-n-epochs", "0",
        "--soft-dtw-loss-weight", "0.0"
    )
}

switch ($Mode) {
    "Teacher" { Invoke-Teacher }
    "Train" { Invoke-Train }
    "Gate" { Invoke-Gate }
    "Full" { Invoke-Full }
    "Smoke" { Invoke-Smoke }
    "All" {
        Invoke-Teacher
        Invoke-Train
        Invoke-Gate
        Invoke-Full
    }
}
