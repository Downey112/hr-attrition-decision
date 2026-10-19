# Full pipeline: fetch -> survival tests -> Cox model -> decision layer -> Power BI extracts + PBIP -> tests. About 3 min.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = '.\.venv\Scripts\python.exe'
foreach ($s in '01_fetch_data', '02_survival_tests', '03_cox', '04_decision', '05_export_bi', '06_build_pbip') {
    Write-Host "== $s" -ForegroundColor Cyan
    & $py -W ignore "scripts\$s.py"
    if ($LASTEXITCODE) { throw "$s failed" }
}
& $py -m pytest -q tests
