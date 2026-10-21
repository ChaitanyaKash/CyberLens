param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $condaPython = Join-Path $env:USERPROFILE 'anaconda3/envs/NLP/python.exe'
    $taskPython = if ($env:CYBERLENS_PYTHON) {
        $env:CYBERLENS_PYTHON
    } elseif (Test-Path "$PSScriptRoot/.venv/Scripts/python.exe") {
        "$PSScriptRoot/.venv/Scripts/python.exe"
    } elseif (Test-Path $condaPython) {
        $condaPython
    } else { 'python' }
    & $taskPython -m cyberlens.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "CyberLens exited with code $LASTEXITCODE" }
} finally { Pop-Location }
