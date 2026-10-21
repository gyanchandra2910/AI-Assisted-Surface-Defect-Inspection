param([Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArgs)
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PATH = "$projectRoot\.deps\mingw64\bin;C:\msys64\mingw64\bin;$env:PATH"
Set-Location -LiteralPath $projectRoot
& "$projectRoot\build\bin\inspect.exe" @AppArgs
exit $LASTEXITCODE
