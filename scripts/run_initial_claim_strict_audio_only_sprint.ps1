param(
    [ValidateSet("All", "Smoke", "Baseline", "Stage1", "Gate1", "Stage2", "Gate2", "SelfMine", "Stage3", "Full")]
    [string]$Mode = "All",
    [string]$SwdPath = "data",
    [string]$Stage1Config = "config/deepalign_initial_claim_strict_ssl_stage1_v2.yaml",
    [string]$Stage2Config = "config/deepalign_initial_claim_strict_cycle_stage2.yaml",
    [string]$Stage3Config = "config/deepalign_initial_claim_strict_selfmine_stage3_v2.yaml",
    [string]$BaselineCheckpoint = "checkpoints/initial_claim_self_audio_pretrain/best_model.pt",
    [string]$Stage1Checkpoint = "checkpoints/initial_claim_strict_ssl_stage1_v2/best_model.pt",
    [string]$Stage2Checkpoint = "checkpoints/initial_claim_strict_cycle_stage2/best_model.pt",
    [string]$Stage3Checkpoint = "checkpoints/initial_claim_strict_selfmine_stage3_v2/best_model.pt",
    [string]$SelfMinedDir = "results/self_mined_initial_claim_strict_cycle_stage2",
    [string]$ResultsDir = "results/initial_claim_strict_audio_only_v2",
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

function Invoke-StrictEval {
    param(
        [string]$Label,
        [string]$Checkpoint,
        [string]$Output,
        [switch]$GateOnly
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $Output -Parent) | Out-Null
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
        "-o", $Output
    )
    if ($GateOnly) {
        foreach ($Lied in $GateLieder) {
            $Arguments += @("--lied", $Lied)
        }
    }
    Invoke-CheckedPython $Label $Arguments
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

function Invoke-Baseline {
    Invoke-StrictEval `
        -Label "Gate 0: strict baseline 8-lied eval" `
        -Checkpoint $BaselineCheckpoint `
        -Output (Join-Path $ResultsDir "baseline_gate_8_lieder.csv") `
        -GateOnly
}

function Invoke-Stage1 {
    Invoke-CheckedPython "Stage 1 v2: strict same-audio ordered SSL with local timing tasks" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Stage1Config,
        "--swd-path", $SwdPath,
        "--device", $Device
    )
}

function Invoke-Gate1 {
    $Csv = Join-Path $ResultsDir "stage1_gate_8_lieder.csv"
    Invoke-StrictEval -Label "Gate 1: Stage 1 strict 8-lied eval" -Checkpoint $Stage1Checkpoint -Output $Csv -GateOnly
    Invoke-ResultGate -Label "Gate 1 threshold: <95 ms and >75% AR@50" -CsvPath $Csv -MaxMaeMs 95 -MinAr50Pct 75 -ExpectedPairs 8
}

function Invoke-Stage2 {
    Invoke-CheckedPython "Stage 2: strict same-lied cycle consistency from Stage 1 v2" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Stage2Config,
        "--swd-path", $SwdPath,
        "--device", $Device
    )
}

function Invoke-Gate2 {
    $Csv = Join-Path $ResultsDir "stage2_gate_8_lieder.csv"
    Invoke-StrictEval -Label "Gate 2: Stage 2 strict 8-lied eval" -Checkpoint $Stage2Checkpoint -Output $Csv -GateOnly
    Invoke-ResultGate -Label "Gate 2 threshold: <75 ms and >88% AR@50" -CsvPath $Csv -MaxMaeMs 75 -MinAr50Pct 88 -ExpectedPairs 8
}

function Invoke-SelfMine {
    Invoke-CheckedPython "Gate 3: mine paths from learned DeepAlign Stage 2" @(
        "-u", "-m", "dis_alignment.cli", "mine-self-paths", $SwdPath,
        "--checkpoint", $Stage2Checkpoint,
        "--output-dir", $SelfMinedDir,
        "--device", $Device,
        "--cache-root", ".cache/cqt",
        "--hop-length", "110",
        "--pool-size", "1",
        "--deep-distance", "sqeuclidean"
    )
}

function Invoke-Stage3 {
    Invoke-CheckedPython "Stage 3: strict learned self-mined refinement" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Stage3Config,
        "--swd-path", $SwdPath,
        "--device", $Device
    )
}

function Invoke-Full {
    $Csv = Join-Path $ResultsDir "full_24_pair_strict.csv"
    Invoke-StrictEval -Label "Gate 4: full 24-pair strict eval" -Checkpoint $Stage3Checkpoint -Output $Csv
    Invoke-ResultGate -Label "Gate 4 threshold: <20 ms and >98% AR@50" -CsvPath $Csv -MaxMaeMs 20 -MinAr50Pct 98 -ExpectedPairs 24
}

function Invoke-Smoke {
    Invoke-CheckedPython "Smoke: Stage 1 v2 one-epoch strict SSL" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Stage1Config,
        "--swd-path", $SwdPath,
        "--device", $Device,
        "--epochs", "1",
        "--samples-per-epoch", "8",
        "--output-dir", "checkpoints/initial_claim_strict_smoke_stage1_v2",
        "--alignment-eval-every-n-epochs", "0"
    )
    Invoke-CheckedPython "Smoke: Stage 2 one-epoch strict cycle training" @(
        "-u", "-m", "dis_alignment.cli", "train",
        "--config", $Stage2Config,
        "--swd-path", $SwdPath,
        "--device", $Device,
        "--epochs", "1",
        "--samples-per-epoch", "8",
        "--resume-from", "checkpoints/initial_claim_strict_smoke_stage1_v2/best_model.pt",
        "--output-dir", "checkpoints/initial_claim_strict_smoke_stage2_cycle",
        "--soft-dtw-loss-weight", "0.0",
        "--alignment-eval-every-n-epochs", "0"
    )
}

switch ($Mode) {
    "Smoke" { Invoke-Smoke }
    "Baseline" { Invoke-Baseline }
    "Stage1" { Invoke-Stage1 }
    "Gate1" { Invoke-Gate1 }
    "Stage2" { Invoke-Stage2 }
    "Gate2" { Invoke-Gate2 }
    "SelfMine" { Invoke-SelfMine }
    "Stage3" { Invoke-Stage3 }
    "Full" { Invoke-Full }
    "All" {
        Invoke-Baseline
        Invoke-Stage1
        Invoke-Gate1
        Invoke-Stage2
        Invoke-Gate2
        Invoke-SelfMine
        Invoke-Stage3
        Invoke-Full
    }
}
