# Moves PDFs you saved from the browser (in Downloads) into pdf\ with the manifest names.
$dl = Join-Path $env:USERPROFILE "Downloads"
$items = Get-Content -Raw .\manifest.json | ConvertFrom-Json
foreach ($it in $items) {
  if ($it.type -ne "pdf") { continue }
  $out = Join-Path $PWD $it.file
  if (Test-Path $out) { continue }
  $name = [System.Uri]::UnescapeDataString(($it.url -split "/")[-1])
  $src = Get-ChildItem $dl -File | Where-Object { $_.Name -ieq $name } | Select-Object -First 1
  if ($src) { Move-Item $src.FullName $out; Write-Host "moved $($it.id)" }
}
$missing = $items | Where-Object { -not (Test-Path (Join-Path $PWD $_.file)) -and -not (Test-Path (Join-Path $PWD "pdf\$($_.id).pdf")) }
Write-Host "`nStill missing: $($missing.Count)"; $missing | ForEach-Object { Write-Host "  $($_.id)  $($_.url)" }
Write-Host "Web pages saved as PDF: put them in pdf\ named <id>.pdf (ids listed above)."
