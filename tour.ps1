# Captures en jeu de plusieurs lieux, assemblees en une planche : .\tour.ps1 planche.png GRAINE "lat,lon,cap,altitude" ...
param([string]$Out, [int]$Seed, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Places)
$exe = "$env:LOCALAPPDATA\terra-target\release\terra-demo.exe"
$dir = Join-Path $env:TEMP 'terra-tour'
New-Item -ItemType Directory -Force $dir | Out-Null
$files = @()
foreach ($place in $Places) {
    $v = $place -split ','
    $file = Join-Path $dir ("p{0}.png" -f $files.Count)
    Remove-Item $file -ErrorAction SilentlyContinue
    $args2 = @('--screenshot', $file, '--seed', $Seed, '--pos', "`"$($v[0]),$($v[1])`"", '--yaw', $v[2], '--altitude', $v[3], '--pitch', $(if ($v.Count -gt 4) { $v[4] } else { '-6' }), '--size', '"960,540"', '--render-distance', '150')
    if ($v.Count -gt 5) { $args2 += @('--time', $v[5]) }
    $p = Start-Process -FilePath $exe -ArgumentList $args2 -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dir 'out.log')
    if (-not $p.WaitForExit(120000)) { $p.Kill(); "TIMEOUT $place" }
    if (Test-Path $file) { $files += $file }
}
Add-Type -AssemblyName System.Drawing
$cols = [Math]::Min(3, $files.Count)
$rows = [Math]::Ceiling($files.Count / $cols)
$sheet = New-Object System.Drawing.Bitmap (640 * $cols), (360 * $rows)
$g = [System.Drawing.Graphics]::FromImage($sheet)
for ($i = 0; $i -lt $files.Count; $i++) {
    $img = [System.Drawing.Image]::FromFile($files[$i])
    $g.DrawImage($img, 640 * ($i % $cols), 360 * [Math]::Floor($i / $cols), 640, 360)
    $img.Dispose()
}
$g.Dispose()
$sheet.Save((Join-Path (Get-Location) $Out))
$sheet.Dispose()
"planche : $Out ($($files.Count) vues)"
