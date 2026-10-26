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
    & $taskPython -m streamlit run app_saas.py --server.port 8502
} finally { Pop-Location }
