# Runs the ci-proto experiment suite with uv on Windows.
#
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1                 # laptop profile (CPU)
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunProfile full  # full protocol (GPU)
#
# Logs go to results\logs, the summary to results\SUMMARY_<profile>.md.
param([string]$RunProfile = "laptop")

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it with:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}
New-Item -ItemType Directory -Force -Path "results\logs" | Out-Null

Write-Host "=== uv sync (Python 3.12, ci-proto, Chronos-2) ==="
uv sync --extra chronos 2>&1 | Tee-Object -FilePath "results\logs\00_sync.log"
if ($LASTEXITCODE -ne 0) { Write-Host "uv sync failed - see results\logs\00_sync.log"; exit 1 }

Write-Host "=== datasets ==="
uv run --no-sync python scripts/get_data.py 2>&1 | Tee-Object -FilePath "results\logs\01_data.log"

Write-Host "=== tests ==="
uv run --no-sync pytest -q 2>&1 | Tee-Object -FilePath "results\logs\02_tests.log"

$experiments = @("e02_leakage", "e04_covariates", "e03_epf", "e06_scarcity", "e05_context",
                 "e07_ablations", "e08_efficiency", "e01_benchmarks")
foreach ($e in $experiments) {
    Write-Host "=== $e ($RunProfile) ==="
    $start = Get-Date
    uv run --no-sync python "experiments/$e.py" --profile $RunProfile 2>&1 | Tee-Object -FilePath "results\logs\$e.log"
    Write-Host ("    {0} finished in {1:N1} min (exit {2})" -f $e, ((Get-Date) - $start).TotalMinutes, $LASTEXITCODE)
}

Write-Host "=== analysis ==="
uv run --no-sync python analysis/report.py epf --results "results/E3_electricity_prices/$RunProfile" --reference "ci-proto[joint]" 2>&1 | Tee-Object -FilePath "results\logs\90_report_epf.log"
uv run --no-sync python analysis/report.py bench --results "results/E2_leakage_control/$RunProfile" --metric SQL --baseline SeasonalNaive 2>&1 | Tee-Object -FilePath "results\logs\91_report_bench.log"
uv run --no-sync python analysis/summarize.py --profile $RunProfile 2>&1 | Tee-Object -FilePath "results\logs\92_summary.log"
Write-Host "Done. Open results\SUMMARY_$RunProfile.md"
