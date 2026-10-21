$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PATH = "$projectRoot\.deps\mingw64\bin;C:\msys64\mingw64\bin;$env:PATH"
& "$projectRoot\.deps\mingw64\bin\cmake.exe" -S $projectRoot -B "$projectRoot\build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$projectRoot/.deps/mingw64" -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe
if ($LASTEXITCODE) { throw 'CMake configuration failed' }
& "$projectRoot\.deps\mingw64\bin\cmake.exe" --build "$projectRoot\build" --parallel 2
if ($LASTEXITCODE) { throw 'Build failed' }
& "$projectRoot\.deps\mingw64\bin\ctest.exe" --test-dir "$projectRoot\build" --output-on-failure
if ($LASTEXITCODE) { throw 'Tests failed' }
