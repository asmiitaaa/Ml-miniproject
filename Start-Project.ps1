param(
    [ValidateSet('Dashboard','Test','QuickTest','GUI','Smoke','Baselines','Evaluate','Doctor','Record')]
    [string]$Mode = 'Dashboard',
    [string]$Results = 'results\corridor_selected_evaluation',
    [string]$Config = 'configs\evaluation.corridor.json',
    [ValidateSet('fixed','actuated','dqn')][string]$Controller = 'dqn'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { & "$PSScriptRoot\Setup-Project.ps1" }
switch ($Mode) {
    'Dashboard' {
        if (-not (Test-Path -LiteralPath (Join-Path $Results 'runs.csv')) -and (Test-Path -LiteralPath 'report\training\corridor_results\runs.csv')) { $Results = 'report\training\corridor_results' }
        if (-not (Test-Path -LiteralPath (Join-Path $Results 'runs.csv')) -and (Test-Path -LiteralPath 'report\results\runs.csv')) { $Results = 'report\results' }
        & $projectPython -m streamlit run dashboard.py -- --results $Results
    }
    'Test' { & $projectPython -m pytest -q }
    'QuickTest' { & $projectPython scripts\quick_test.py --config $Config }
    'GUI' { & $projectPython scripts\preview_gui.py --config $Config --scenario peak --controllers $Controller --seconds 300 --every 15 --delay-ms 0 --pace-ms 100 }
    'Doctor' { & $projectPython -m traffic_eval doctor --config $Config }
    'Smoke' {
        & $projectPython scripts\prepare_scenarios.py --kind smoke
        if ($LASTEXITCODE -ne 0) { throw 'Scenario generation failed.' }
        $smokeOutput = 'results\smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
        & $projectPython -m traffic_eval run --config configs\evaluation.smoke.json --controllers fixed actuated --output $smokeOutput
    }
    'Baselines' {
        & $projectPython scripts\prepare_scenarios.py --kind corridor
        if ($LASTEXITCODE -ne 0) { throw 'Scenario generation failed.' }
        $baselineOutput = 'results\provisional_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
        & $projectPython -m traffic_eval run --config configs\evaluation.provisional.json --controllers fixed actuated --output $baselineOutput
    }
    'Evaluate' {
        $fullOutput = 'results\evaluation_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
        & $projectPython scripts\run_parallel_evaluation.py --config $Config --output $fullOutput --jobs 3
    }
    'Record' { & $projectPython scripts\record_demo.py --config $Config --scenario peak --controllers fixed actuated dqn }
}
if ($LASTEXITCODE -ne 0) { throw "Project command failed with exit code $LASTEXITCODE. See the message above." }
