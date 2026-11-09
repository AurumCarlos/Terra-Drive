# Lance cargo avec Rust et MinGW dans le PATH, sans dependre du PATH du terminal courant.
# Exemple : .\cargo.ps1 run --release
$mingw = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter "BrechtSanders.WinLibs*" -ErrorAction SilentlyContinue |
    Select-Object -First 1 |
    ForEach-Object { Join-Path $_.FullName "mingw64\bin" }
$env:PATH = "$env:USERPROFILE\.cargo\bin;$mingw;$env:PATH"
# Dossier de build court : les chemins profonds depassent la limite de 260 caracteres de Windows.
$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\terra-target"
# PowerShell avale le "--" qui separe les options de cargo de celles du programme :
# on le remet devant le premier argument que cargo ne connait pas.
$cargoArgs = @($args)
if ($cargoArgs.Count -gt 1 -and ($cargoArgs[0] -eq "run" -or $cargoArgs[0] -eq "test") -and -not ($cargoArgs -contains "--")) {
    $flags = @("--release", "-r", "--lib", "--bins", "--tests", "--quiet", "-q", "--offline", "--locked", "--frozen", "--all-features", "--no-default-features", "-v", "--verbose")
    $withValue = @("--bin", "--test", "--example", "--features", "-F", "--profile", "-p", "--package", "--target", "-j", "--jobs", "--manifest-path")
    $i = 1
    while ($i -lt $cargoArgs.Count) {
        $a = [string]$cargoArgs[$i]
        if ($flags -contains $a) { $i += 1 }
        elseif ($withValue -contains $a) { $i += 2 }
        elseif ($cargoArgs[0] -eq "test" -and -not $a.StartsWith("-")) { $i += 1 }
        else { break }
    }
    if ($i -lt $cargoArgs.Count) {
        $cargoArgs = @($cargoArgs[0..($i - 1)]) + @("--") + @($cargoArgs[$i..($cargoArgs.Count - 1)])
    }
}
& cargo @cargoArgs
exit $LASTEXITCODE
