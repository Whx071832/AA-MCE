<#
.SYNOPSIS
    Reproduce the AA-MCE paper: data preparation, every experiment in dependency
    order, then the tables and figures.

.DESCRIPTION
    Works in the repository root (the parent of this scripts folder), whatever
    the current directory is.

    * Data preparation (src.data.prepare) is idempotent: existing outputs under
      Data\processed\v1 are kept.
    * An experiment stage is skipped when outputs\experiments\<run>\completion.json
      exists. The validation-only ensemble selection is skipped when
      outputs\experiments\optimized_tagging_validation_v2\ensemble_selection.json
      exists.
    * A failed experiment stage is retried once (training runners restart with
      --overwrite). The script stops if the retry fails as well.
    * The reporting steps always run.
    * Logs are written to outputs\logs\<stage>.log.

.PARAMETER Python
    Python interpreter to use (default: python).

.PARAMETER SkipDataPrep
    Do not run src.data.prepare and src.data.validate_processed.

.PARAMETER DryRun
    Print the plan (which stages would run or be skipped) without running anything.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_paper.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$SkipDataPrep,
    [switch]$DryRun
)

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "outputs\logs"
$attempts = 2
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Logged {
    param([string]$Log, [string[]]$Arguments)
    $ErrorActionPreference = "Continue"
    $header = ">>> {0} {1}   [{2}]" -f $Python, ($Arguments -join " "), (Get-Date -Format s)
    Add-Content -Path $Log -Encoding UTF8 -Value $header
    & $Python @Arguments 2>&1 | ForEach-Object { "$_" } | Add-Content -Path $Log -Encoding UTF8
    return $LASTEXITCODE
}

function Invoke-Stage {
    param([hashtable]$Stage, [int]$MaxAttempts)
    $done = $null
    if ($Stage.ContainsKey("Done")) { $done = Join-Path $root $Stage.Done }
    if ($done -and (Test-Path $done)) {
        Write-Host ("[skip] {0}: {1} exists" -f $Stage.Name, $Stage.Done)
        return $true
    }
    if ($DryRun) {
        Write-Host ("[plan] {0}: {1} {2}" -f $Stage.Name, $Python, ($Stage.Args -join " "))
        return $true
    }
    $log = Join-Path $logDir ($Stage.Name + ".log")
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Host ("[run ] {0} (attempt {1}; log: {2})" -f $Stage.Name, $attempt, $log)
        $code = Invoke-Logged -Log $log -Arguments $Stage.Args
        if ($code -eq 0 -and (-not $done -or (Test-Path $done))) { return $true }
        Write-Host ("[fail] {0} attempt {1} (exit code {2})" -f $Stage.Name, $attempt, $code)
    }
    return $false
}

function New-Run {
    param([string]$Run, [string[]]$Arguments)
    return @{ Name = $Run; Done = "outputs\experiments\$Run\completion.json"; Args = $Arguments }
}

# 0. Data preparation: only the stages this paper needs (see docs/DATA.md).
$prepare = @(
    @{ Name = "prepare_data"; Args = @("-m", "src.data.prepare", "--project-root", ".", "--stages", "aa", "musicsem", "fma", "features", "tags", "align", "manifest") },
    @{ Name = "validate_data"; Args = @("-m", "src.data.validate_processed", "--project-root", ".") }
)

# 1-7. Experiments in dependency order (later runs read earlier ones).
$experiments = @(
    (New-Run "robust_multimodal_tagging_v3" @("-m", "src.experiments.robust_multimodal_tagging", "--project-root", ".", "--run-name", "robust_multimodal_tagging_v3", "--overwrite")),
    (New-Run "optimized_tagging_validation_v2" @("-m", "src.experiments.optimized_multimodal_tagging", "--project-root", ".", "--run-name", "optimized_tagging_validation_v2", "--stage", "sweep", "--overwrite")),
    @{ Name = "select_tagging_ensemble"; Done = "outputs\experiments\optimized_tagging_validation_v2\ensemble_selection.json"; Args = @("-m", "src.experiments.select_tagging_ensemble", "--project-root", ".", "--run-name", "optimized_tagging_validation_v2") },
    (New-Run "optimized_multimodal_tagging_final_v1" @("-m", "src.experiments.optimized_multimodal_tagging", "--project-root", ".", "--run-name", "optimized_multimodal_tagging_final_v1", "--stage", "final", "--overwrite")),
    (New-Run "advanced_fusion_baselines_v1" @("-m", "src.experiments.advanced_fusion_baselines", "--project-root", ".", "--run-name", "advanced_fusion_baselines_v1", "--overwrite")),
    (New-Run "full_feature_ablation_v1" @("-m", "src.experiments.full_feature_ablation", "--project-root", ".", "--run-name", "full_feature_ablation_v1", "--overwrite")),
    (New-Run "fma_cross_dataset_v1" @("-m", "src.experiments.fma_cross_dataset", "--project-root", ".", "--run-name", "fma_cross_dataset_v1", "--overwrite")),
    (New-Run "extended_modalities_v1" @("-m", "src.experiments.extended_modalities", "--project-root", ".", "--run-name", "extended_modalities_v1", "--overwrite")),
    (New-Run "availability_aware_ensemble_v1" @("-m", "src.experiments.availability_aware_ensemble", "--project-root", ".", "--run-name", "availability_aware_ensemble_v1", "--overwrite")),
    (New-Run "availability_aware_ensemble_5c_v1" @("-m", "src.experiments.availability_aware_ensemble", "--project-root", ".", "--section", "availability_aware_ensemble_5c", "--run-name", "availability_aware_ensemble_5c_v1", "--overwrite")),
    (New-Run "availability_aware_ensemble_fma_v1" @("-m", "src.experiments.availability_aware_campaigns", "--project-root", ".", "--dataset", "fma", "--run-name", "availability_aware_ensemble_fma_v1", "--overwrite")),
    (New-Run "availability_aware_ensemble_sixview_v1" @("-m", "src.experiments.availability_aware_campaigns", "--project-root", ".", "--dataset", "sixview", "--run-name", "availability_aware_ensemble_sixview_v1", "--overwrite")),
    (New-Run "posthoc_calibration_v1" @("-m", "src.experiments.posthoc_calibration", "--project-root", ".", "--run-name", "posthoc_calibration_v1")),
    (New-Run "robustness_checks_v1" @("-m", "src.experiments.robustness_checks", "--project-root", ".", "--run-name", "robustness_checks_v1")),
    (New-Run "paper_analysis_v1" @("-m", "src.experiments.paper_analysis", "--project-root", ".", "--run-name", "paper_analysis_v1")),
    (New-Run "paper_efficiency_v1" @("-m", "src.experiments.paper_efficiency", "--project-root", ".")),
    (New-Run "decision_layer_diagnostics_v1" @("-m", "src.experiments.decision_layer_diagnostics", "--project-root", ".", "--run-name", "decision_layer_diagnostics_v1"))
)

# 8. Tables and figures (always rebuilt; paths are relative to the repository root).
$reports = @(
    @{ Name = "paper_data"; Args = @("-m", "src.reporting.paper_data", "--out", "outputs/paper/data") },
    @{ Name = "paper_facts"; Args = @("-m", "src.reporting.paper_facts") },
    @{ Name = "paper_figures"; Args = @("-m", "src.reporting.paper_figures", "--data", "outputs/paper/data", "--out", "outputs/paper/figures") },
    @{ Name = "final_figures"; Args = @("-m", "src.reporting.final_figures", "--project-root", ".") },
    @{ Name = "final_tables"; Args = @("-m", "src.reporting.final_tables", "--project-root", ".") }
)

Push-Location $root
Write-Host "Repository root: $root"
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$ok = $true

if ($SkipDataPrep) {
    Write-Host "[skip] data preparation (-SkipDataPrep)"
} else {
    foreach ($stage in $prepare) {
        if (-not (Invoke-Stage $stage 1)) { $ok = $false; Write-Host "Stopped: $($stage.Name) failed."; break }
    }
}
if ($ok) {
    foreach ($stage in $experiments) {
        if (-not (Invoke-Stage $stage $attempts)) { $ok = $false; Write-Host "Stopped: $($stage.Name) did not complete."; break }
    }
}
if ($ok) {
    $failedReports = @()
    foreach ($stage in $reports) {
        if (-not (Invoke-Stage $stage 1)) { $failedReports += $stage.Name }
    }
    if ($failedReports.Count -gt 0) {
        $ok = $false
        Write-Host ("Reporting steps failed: {0}" -f ($failedReports -join ", "))
    }
}
Pop-Location

if ($ok) {
    Write-Host "Finished."
    exit 0
}
Write-Host "Not finished; see the logs in outputs\logs."
exit 1
