param(
    [ValidateSet('assembleDebug', ':app:testDebugUnitTest', ':app:recordRoborazziDebug', ':app:verifyRoborazziDebug', 'lintDebug')]
    [string[]]$Tasks = @('assembleDebug'),
    [string]$Python = '',
    [ValidateRange(1, 2)]
    [int]$Workers = 1,
    [switch]$RecordScreenshots,
    [switch]$ScreenshotSmoke
)
$ErrorActionPreference = 'Stop'
$androidRoot = Split-Path -Parent $PSScriptRoot
$previousJava = $env:JAVA_HOME
$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $androidRoot
    if (-not $Python) {
        $candidate = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'AppData\Local\Python\pythoncore-3.14-64\python.exe'
        $Python = if (Test-Path -LiteralPath $candidate) { $candidate } else { (Get-Command python -ErrorAction Stop).Source }
    }
    if (-not $env:JAVA_HOME) {
        $env:JAVA_HOME = Join-Path $env:ProgramFiles 'Android\Android Studio\jbr'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME 'bin\java.exe'))) {
        throw 'Set JAVA_HOME to a supported JDK before building.'
    }
    $gradleArguments = @('--no-daemon', "--max-workers=$Workers", '--console=plain') + $Tasks
    if ($RecordScreenshots) { $gradleArguments += '-Proborazzi.test.record=true' }
    if ($ScreenshotSmoke) { $gradleArguments += '-PscreenshotSmoke=true' }
    & $Python tools/thermal_guard.py -- cmd.exe /d /c gradlew.bat @gradleArguments
    if ($LASTEXITCODE -ne 0) { throw "Guarded build failed with exit code $LASTEXITCODE" }
} finally {
    $env:JAVA_HOME = $previousJava
    Set-Location -LiteralPath $previousLocation
}
