param(
    [ValidateSet("Gate", "FullTop", "Summary")]
    [string]$Mode = "Gate",
    [string]$RunId = "model_sprint_2026_05_05",
    [int]$TopN = 3,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv312\Scripts\python.exe"
$SwdPath = Join-Path $Root "data"
$OutDir = Join-Path $Root "results\$RunId"
$SummaryPath = Join-Path $Root "results\${RunId}_summary.csv"
$Stage1Checkpoint = Join-Path $Root "checkpoints\cycle1_stage1_local_precision_2026_04_19\best_model.pt"
$ReferenceBest = Join-Path $Root "results\swd_deepalign_transcription_fused_full_2026_04_25.csv"
$ReferenceGate = Join-Path $Root "results\swd_deepalign_transcription_fused_8song_gate_2026_04_25.csv"
$Summarizer = Join-Path $Root "scripts\summarize_swd_results.py"

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

$Candidates = @(
    @{ Label = "stage1_pool2_default"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool1_default"; Pool = 1; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool3_default"; Pool = 3; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_onset075"; Pool = 2; Deep = 1.0; Onset = 0.75; Note = 0.75; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_onset125"; Pool = 2; Deep = 1.0; Onset = 1.25; Note = 0.75; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_note050"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.50; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_note100"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 1.00; Dlnco = 0.5; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_dlnco025"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.25; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_dlnco075"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.75; Chroma = 0.25; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_chroma000"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.5; Chroma = 0.00; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_chroma050"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.5; Chroma = 0.50; Distance = "sqeuclidean" },
    @{ Label = "stage1_pool2_cosine_default"; Pool = 2; Deep = 1.0; Onset = 1.0; Note = 0.75; Dlnco = 0.5; Chroma = 0.25; Distance = "cosine" }
)

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$StepName,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host "===== $StepName ====="
    Write-Host "Started: $(Get-Date -Format o)"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
    Write-Host "Finished: $(Get-Date -Format o)"
}

function Write-SprintSummary {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SummaryPath) | Out-Null
    Invoke-CheckedPython "Summarize sprint results" @(
        $Summarizer,
        "--reference", "reference_current_best_apr25=$ReferenceBest",
        "--reference", "reference_8song_gate_apr25=$ReferenceGate",
        "--results-dir", $OutDir,
        "--output", $SummaryPath
    )
}

function Find-Candidate {
    param([Parameter(Mandatory = $true)][string]$Label)
    foreach ($Candidate in $Candidates) {
        if ($Candidate.Label -eq $Label) {
            return $Candidate
        }
    }
    throw "Unknown candidate label: $Label"
}

function Invoke-EvalCandidate {
    param(
        [Parameter(Mandatory = $true)]$Candidate,
        [Parameter(Mandatory = $true)][ValidateSet("gate", "full")][string]$Scope
    )

    $OutputCsv = Join-Path $OutDir "$($Candidate.Label)_$Scope.csv"
    if ((Test-Path $OutputCsv) -and (-not $Force)) {
        Write-Host ""
        Write-Host "===== Skipping existing $($Candidate.Label)_$Scope ====="
        Write-Host "Use -Force to rerun: $OutputCsv"
        return
    }

    $Arguments = @(
        "-u", "-m", "dis_alignment.cli", "evaluate-swd", $SwdPath,
        "--checkpoint", $Stage1Checkpoint,
        "--methods", "deepalign",
        "--device", "cuda",
        "--cache-root", ".cache/cqt",
        "--pool-size", ([string]$Candidate.Pool),
        "--deep-distance", $Candidate.Distance,
        "--deep-decode", "deepalign_transcription_fused",
        "--transcription-cache-root", ".cache/transcription/basic_pitch",
        "--fusion-deep-weight", ([string]$Candidate.Deep),
        "--fusion-onset-weight", ([string]$Candidate.Onset),
        "--fusion-note-weight", ([string]$Candidate.Note),
        "--fusion-dlnco-weight", ([string]$Candidate.Dlnco),
        "--fusion-chroma-weight", ([string]$Candidate.Chroma),
        "--output", $OutputCsv
    )

    if ($Scope -eq "gate") {
        foreach ($Lied in $GateLieder) {
            $Arguments += @("--lied", $Lied)
        }
    }

    Invoke-CheckedPython "Evaluate $($Candidate.Label)_$Scope" $Arguments
    Write-SprintSummary
}

function Get-TopGateCandidateLabels {
    if (-not (Test-Path $SummaryPath)) {
        Write-SprintSummary
    }

    $Rows = Import-Csv $SummaryPath |
        Where-Object {
            $_.status -eq "ok" -and
            $_.label.EndsWith("_gate") -and
            $_.label -notlike "reference_*"
        } |
        Sort-Object @{ Expression = { [double]$_.ar_50_pct }; Descending = $true },
                    @{ Expression = { [double]$_.median_ae_ms }; Ascending = $true },
                    @{ Expression = { [double]$_.mae_ms }; Ascending = $true }

    return @($Rows | Select-Object -First $TopN | ForEach-Object {
        $_.label -replace "_gate$", ""
    })
}

Set-Location $Root
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $Python)) {
    throw "Project Python not found: $Python"
}
if (-not (Test-Path $Stage1Checkpoint)) {
    throw "Stage 1 checkpoint not found: $Stage1Checkpoint"
}

switch ($Mode) {
    "Gate" {
        foreach ($Candidate in $Candidates) {
            Invoke-EvalCandidate -Candidate $Candidate -Scope "gate"
        }
        Write-SprintSummary
    }
    "FullTop" {
        $TopLabels = Get-TopGateCandidateLabels
        if ($TopLabels.Count -eq 0) {
            throw "No gate candidates found. Run -Mode Gate first."
        }
        Write-Host "Full-SWD candidates: $($TopLabels -join ', ')"
        foreach ($Label in $TopLabels) {
            Invoke-EvalCandidate -Candidate (Find-Candidate $Label) -Scope "full"
        }
        Write-SprintSummary
    }
    "Summary" {
        Write-SprintSummary
    }
}
