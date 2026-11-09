# Compile puis enregistre une capture : .\shoot.ps1 sortie.png --seed 1 --time 0.42 ...
param([string]$Out, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$ErrorActionPreference = 'Continue'
& "$PSScriptRoot\cargo.ps1" build --release 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match '^(error|warning: unused)|-->' }
$exe = "$env:LOCALAPPDATA\terra-target\release\terra-demo.exe"
$log = [IO.Path]::ChangeExtension($Out, '.log')
$p = Start-Process -FilePath $exe -ArgumentList (@('--screenshot', $Out) + $Rest) -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
if (-not $p.WaitForExit(180000)) { $p.Kill(); 'TIMEOUT' }
Get-Content $log
Get-Content "$log.err" | Select-Object -First 15
