# Downloads the CyberLens RAG corpus. Run from the rag_corpus folder:  powershell -ExecutionPolicy Bypass -File .\download.ps1
# Tries the main URL, then any official mirror in alt_urls. Anything that still fails goes to failed_links.html for the browser.
$ErrorActionPreference = "Continue"
$items = Get-Content -Raw .\manifest.json | ConvertFrom-Json
New-Item -ItemType Directory -Force pdf,html,text | Out-Null
function Test-Good($it, $path) {
  if (-not (Test-Path $path)) { return $false }
  if ($it.type -eq "pdf") { return ([System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($path)[0..3]) -eq "%PDF") }
  $text = Get-Content -Raw $path -ErrorAction SilentlyContinue
  return ((Get-Item $path).Length -gt 5000) -and ($text -notmatch "testing whether you are a human")
}
$failed = @()
foreach ($it in $items) {
  $out = Join-Path $PWD $it.file
  if (($it.type -eq "txt" -and (Test-Path $out)) -or (Test-Good $it $out) -or (Test-Path (Join-Path $PWD "pdf\$($it.id).pdf"))) { Write-Host "skip  $($it.id)"; continue }
  $ok = $false
  foreach ($u in @($it.url) + @($it.alt_urls)) {
    if (-not $u) { continue }
    try { Invoke-WebRequest -Uri $u -OutFile $out -TimeoutSec 60 -UseBasicParsing } catch { }
    if (Test-Good $it $out) { $ok = $true; break }
    Remove-Item $out -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
  }
  if ($ok) { Write-Host "ok    $($it.id)" } else { Write-Host "FAIL  $($it.id)"; $failed += $it }
}
$failed | Select-Object id,file,url | Export-Csv -NoTypeInformation .\failed_downloads.csv
$links = ($failed | ForEach-Object { "<li><a href='$($_.url)' target='_blank'>$($_.title)</a> <small>($($_.type))</small></li>" }) -join "`n"
"<html><body style='font-family:sans-serif;max-width:900px;margin:2em auto'><h2>Click each link</h2><p>PDFs download to your Downloads folder. For web pages press Ctrl+P, Save as PDF, into Downloads. Then run import_from_downloads.ps1.</p><ol>$links</ol></body></html>" | Set-Content -Encoding UTF8 .\failed_links.html
Write-Host "`n$($items.Count - $failed.Count) of $($items.Count) ready. Open failed_links.html for the rest."
