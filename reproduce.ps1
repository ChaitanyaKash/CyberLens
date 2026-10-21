# Rebuild everything locally, in dependency order, and preserve console output.
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
    $env:OMP_NUM_THREADS = '6'
    $env:MKL_NUM_THREADS = '6'
    $transcriptPath = Join-Path $PSScriptRoot 'reports/reproduction.log'
    Start-Transcript -Path $transcriptPath -Force | Out-Null
    try {
        & $taskPython -m cyberlens.cli prepare
        if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
        foreach ($backend in 'baseline','word2vec','subword') {
            & $taskPython -m cyberlens.cli train --backend $backend
            if ($LASTEXITCODE -ne 0) { throw "$backend training failed" }
        }
        if (-not (Test-Path "$PSScriptRoot/.cache/pretrained/distilmbert/model.safetensors")) {
            & $taskPython -m cyberlens.download_encoder
            if ($LASTEXITCODE -ne 0) { throw 'Public encoder download failed' }
        }
        & $taskPython -m cyberlens.cli train --backend transformer --task call_scam --epochs 3
        if ($LASTEXITCODE -ne 0) { throw 'Call encoder training failed' }
        & $taskPython -m cyberlens.cli train --backend transformer --task complaint --epochs 1 --batch-size 32 --accumulation 1
        if ($LASTEXITCODE -ne 0) { throw 'Complaint encoder training failed' }
        & $taskPython -m cyberlens.report
        if ($LASTEXITCODE -ne 0) { throw 'Initial model selection/reporting failed' }
        & $taskPython -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
        & $taskPython -m cyberlens.challenge
        if ($LASTEXITCODE -ne 0) { throw 'Challenge evaluation failed' }
        & $taskPython -m cyberlens.report
        if ($LASTEXITCODE -ne 0) { throw 'Reporting failed' }
    } finally { Stop-Transcript | Out-Null }
} finally { Pop-Location }
