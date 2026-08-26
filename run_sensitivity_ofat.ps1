$ErrorActionPreference = "Stop"

$SensitivityScenarios = @(0, 5, 11, 17, 23, 29)

# Optional validation before the formal run:
python .\scripts\15_run_sensitivity_analysis.py `
  --config .\configs\base.yaml `
  --sensitivity .\configs\sensitivity_ofat.yaml `
  --scenario-id 0 `
  --dry-run

if ($LASTEXITCODE -ne 0) {
    throw "OFAT sensitivity dry-run failed"
}

foreach ($ScenarioId in $SensitivityScenarios) {
    python .\scripts\15_run_sensitivity_analysis.py `
      --config .\configs\base.yaml `
      --sensitivity .\configs\sensitivity_ofat.yaml `
      --scenario-id $ScenarioId `
      --preposition-method proposed

    if ($LASTEXITCODE -ne 0) {
        throw "OFAT sensitivity analysis failed for scenario $ScenarioId"
    }
}

Write-Host "OFAT sensitivity analysis completed."
Write-Host "Combined results: outputs/metrics/sensitivity_ofat/sensitivity_ofat_all_scenarios.csv"
