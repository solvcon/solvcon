#Requires -Version 5.1
#
# Warning: this script is still work in progress. Read carefully before using.
#
# Build solvcon (the _solvcon extension and, by default, the Qt pilot) on
# Windows against a scdv produced by
# contrib/dependency/windows/build-scdv-windows.ps1.  solvcon's Makefile and
# setup.py shell out to "make", and its module-copy target runs
# "python3-config"; neither exists on Windows.  This drives CMake directly
# through the win-rel / win-dbg presets in CMakePresets.json, adding only the
# scdv-specific cache variables, and places the module by hand.
#
# It finds a CMake >= 4.0.1 (solvcon's minimum; the VS 2022 Build Tools bundle
# only 3.31.6, so it falls back to VS 2026's 4.x) and compiles with the same
# toolset as the scdv's numpy -- select it with SCDV_VS_VERSION='[17.0,18.0)'
# for VS 2022 v143.  The scdv must already be built; activate it first or pass
# -ScdvBase <scdv dir>.
#
# Usage:
#   .\build.ps1
#       Configure (preset win-rel) and build _solvcon and the pilot against the
#       active scdv (or -ScdvBase), then place the module.
#   .\build.ps1 -ScdvBase <dir>       activate the scdv at <dir> first
#   .\build.ps1 -BuildType Debug      use the win-dbg preset
#   .\build.ps1 -NoQt                 build only _solvcon (BUILD_QT=OFF)
#   .\build.ps1 -Test                 then run "pytest tests\" headless
#   .\build.ps1 -Pilot                then launch the pilot GUI
#   .\build.ps1 -PilotTest            then run "pilot.exe --mode=pytest" headless
#   .\build.ps1 -Gtest                also build and run the C++ gtest suite
#   .\build.ps1 -Sanitize             build and run the gtest suite under
#                                     AddressSanitizer (implies -Gtest, -NoQt)
#   .\build.ps1 -Sanitize -PilotTest  instead build the pilot under
#                                     AddressSanitizer and run its pytest suite
#                                     (pilot.exe --mode=pytest) under it
#
# Overridable variables:
#   SCDV_VS_VERSION: vswhere -version range picking the VS whose cl/vcvars
#     compiles solvcon, e.g. "[17.0,18.0)" for VS 2022.  Match the scdv's numpy.
#   SCDV_VCVARS: full path to a vcvars64.bat (overrides SCDV_VS_VERSION).

[CmdletBinding()]
param(
    [string]$ScdvBase,
    [string]$Repo,
    [ValidateSet('Release', 'Debug')][string]$BuildType = 'Release',
    [switch]$NoQt,
    [switch]$Gtest,
    [switch]$Test,
    [switch]$Pilot,
    [switch]$PilotTest,
    [switch]$Sanitize,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "${What}: exited with code $LASTEXITCODE" }
}

if ($Help) {
    Get-Content -LiteralPath $PSCommandPath | Select-Object -Skip 1 |
        ForEach-Object { if ($_ -notmatch '^\s*$') { $_ -replace '^#\s?', '' } } |
        Select-Object -First 40
    exit 0
}

# The sanitizer build runs a solvcon-produced binary under AddressSanitizer.
# By default it is the C++ gtest binary (the Windows counterpart of the Linux
# "make gtest USE_SANITIZER=ON" job), built BUILD_QT=OFF to stay headless. With
# -PilotTest it is pilot.exe running its pytest suite, built BUILD_QT=ON so the
# Qt canvas code is instrumented. Either binary links the ASan runtime directly
# so ASan starts with the process; a stock python.exe loading the instrumented
# .pyd would start it too late and crash.
if ($Sanitize) {
    if ($Pilot) {
        throw '-Sanitize cannot drive a live -Pilot window; use -PilotTest ' +
            'to run the pilot pytest suite under AddressSanitizer'
    }
    if ($BuildType -eq 'Debug') {
        throw '-Sanitize needs the Release preset: MSVC ASan rejects the ' +
            "Debug /RTC1 runtime checks. Drop -BuildType Debug."
    }
    if (-not $PilotTest) {
        $Gtest = $true
        $NoQt = $true
    }
}

# This script lives at the repo root; build that checkout by default.
if (-not $Repo) { $Repo = $PSScriptRoot }
if (-not (Test-Path -LiteralPath (Join-Path $Repo 'CMakePresets.json'))) {
    throw "no CMakePresets.json under -Repo '$Repo'; point it at a solvcon checkout"
}
$preset = if ($BuildType -eq 'Debug') { 'win-dbg' } else { 'win-rel' }
$bld = Join-Path $Repo "build\$preset"
$solvconDir = Join-Path $Repo 'solvcon'

# --- MSVC environment -------------------------------------------------------

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vcvars = $env:SCDV_VCVARS
    if (-not $vcvars) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} `
            'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere)) {
            throw "vswhere.exe not found; install the Visual Studio Build Tools"
        }
        $sel = @('-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property', 'installationPath')
        if ($env:SCDV_VS_VERSION) {
            $sel = @('-version', $env:SCDV_VS_VERSION) + $sel
        } else {
            $sel = @('-latest') + $sel
        }
        $vsroot = & $vswhere @sel
        Assert-LastExit 'vswhere'
        if (-not $vsroot) { throw 'no Visual Studio with the VC++ toolset found' }
        $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    }
    if (-not (Test-Path -LiteralPath $vcvars)) {
        throw "vcvars64.bat not found at $vcvars"
    }
    Write-Host "importing MSVC environment from $vcvars"
    $out = & cmd.exe /c "`"$vcvars`" >nul && set"
    Assert-LastExit 'vcvars64.bat'
    foreach ($line in $out) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
}

# --- CMake >= 4.0.1 ---------------------------------------------------------

function Resolve-Cmake {
    # A CMake >= 4.0.1 (solvcon's minimum): the one on PATH if new enough (vcvars
    # may put VS 2022's 3.31.6 there), else the newest bundled with a VS.
    $candidates = @()
    $onpath = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($onpath) { $candidates += $onpath.Source }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} `
        'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $rel = 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
        $candidates += (& $vswhere -products '*' -sort -find $rel)
    }
    foreach ($c in $candidates) {
        if (-not $c -or -not (Test-Path -LiteralPath $c)) { continue }
        $line = (& $c --version | Select-Object -First 1)
        if ($line -match '(\d+\.\d+\.\d+)' -and
            [version]$Matches[1] -ge [version]'4.0.1') {
            return $c
        }
    }
    throw ('no CMake >= 4.0.1 found (solvcon requires it); install CMake 4.x ' +
        'or a Visual Studio with the C++ CMake component')
}

$cmake = Resolve-Cmake
Write-Host "using cmake: $cmake"

# --- Activate the scdv ------------------------------------------------------

if (-not $env:SCDV_USRDIR) {
    if (-not $ScdvBase) { $ScdvBase = $env:SCDV_BASE }
    if (-not $ScdvBase) {
        throw ('no active scdv: activate one (". <scdv>\Activate.ps1") or pass ' +
            '-ScdvBase <scdv dir>')
    }
    $activate = Join-Path $ScdvBase 'Activate.ps1'
    if (-not (Test-Path -LiteralPath $activate)) {
        throw "Activate.ps1 not found under -ScdvBase '$ScdvBase'"
    }
    Write-Host "activating scdv: $ScdvBase"
    . $activate
}
$usr = $env:SCDV_USRDIR
$py = Join-Path $usr 'python3.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw "scdv python3.exe not found at $py"
}

# --- Configure and build via the preset -------------------------------------

$pybind = & $py -m pybind11 --cmakedir
Assert-LastExit 'pybind11 --cmakedir'
# The preset holds the static knobs (generator, build type, BUILD_QT,
# USE_GOOGLETEST, BLA_VENDOR, output dirs); pass the scdv-specific cache
# variables on top.  -NoQt overrides the preset's BUILD_QT=ON.
$extra = @(
    "-DPYTHON_EXECUTABLE=$py",
    "-Dpybind11_path=$pybind",
    "-DCMAKE_PREFIX_PATH=$usr"
)
if ($NoQt) { $extra += '-DBUILD_QT=OFF' }
# -Gtest turns on the gtest binary (USE_GOOGLETEST is OFF in the preset), and
# -Sanitize (which implies -Gtest) builds it under AddressSanitizer. The gtest
# binary links the ASan runtime directly so ASan initializes at process start,
# unlike loading an instrumented .pyd into a stock python.exe.
if ($Gtest) { $extra += '-DUSE_GOOGLETEST=ON' }
if ($Sanitize) { $extra += '-DUSE_SANITIZER=ON' }

Push-Location $Repo
try {
    Write-Host "configuring solvcon (preset $preset, BUILD_QT=$(if ($NoQt) {'OFF'} else {'ON'})) ..."
    & $cmake --preset $preset @extra
    Assert-LastExit 'cmake configure'

    $targets = @('_solvcon')
    if (-not $NoQt) { $targets += 'pilot' }
    if ($Gtest) { $targets += 'test_nopython' }
    Write-Host "building targets: $($targets -join ', ')"
    & $cmake --build --preset $preset --target @targets
    Assert-LastExit 'cmake build'
} finally {
    Pop-Location
}

# _solvcon.pyd is a LIBRARY artifact (-> solvcon\, per the preset); pilot.exe is
# a RUNTIME artifact (-> the preset's binary dir, build\$preset).
# Copy the module to the repo root as the top-level _solvcon: solvcon.core
# imports it there first, and the tests' qualified type names assume that name.
$pyd = Get-ChildItem -LiteralPath $solvconDir -Filter '_solvcon*.pyd' |
    Select-Object -First 1
if (-not $pyd) { throw "no _solvcon*.pyd produced under $solvconDir" }
Copy-Item $pyd.FullName (Join-Path $Repo $pyd.Name) -Force
Write-Host "placed module: $(Join-Path $Repo $pyd.Name)"
if (-not $NoQt) {
    Write-Host "built pilot: $(Join-Path $bld 'pilot.exe')"
}

# --- Optional: run ----------------------------------------------------------

if ($Gtest -or $Test -or $PilotTest -or $Pilot) {
    $env:PYTHONPATH = $Repo
    # Headless runs default to offscreen; -Pilot keeps the native platform.
    if (($Test -or $PilotTest) -and -not $Pilot -and -not $env:QT_QPA_PLATFORM) {
        $env:QT_QPA_PLATFORM = 'offscreen'
    }
    if ($Sanitize -and -not $env:ASAN_OPTIONS) {
        # Hook the RTL allocators so the uninstrumented Qt and PySide6 DLLs
        # share ASan's heap (else a cross-module free reads as a false
        # mismatch); fail the run on the first report. No LSan on Windows.
        $env:ASAN_OPTIONS =
            'windows_hook_rtl_allocators=1:abort_on_error=1:detect_leaks=0'
    }
    $pilotExe = Join-Path $bld 'pilot.exe'
    Push-Location $Repo
    try {
        if ($Gtest) {
            Write-Host '=== gtest (test_nopython) ==='
            $gtestExe = Join-Path $bld 'test_nopython.exe'
            Write-Host "run: $gtestExe"
            & $gtestExe
            Assert-LastExit 'test_nopython'
        }
        if ($Test) {
            Write-Host '=== pytest tests ==='
            Write-Host "run: $py -m pytest tests"
            & $py -m pytest tests
            Assert-LastExit 'pytest'
        }
        if ($PilotTest) {
            if ($NoQt) { throw '-PilotTest requires the pilot (drop -NoQt)' }
            Write-Host '=== pilot in-binary test suite ==='
            Write-Host "run: $pilotExe --mode=pytest"
            & $pilotExe --mode=pytest
            Assert-LastExit 'pilot --mode=pytest'
        }
        if ($Pilot) {
            if ($NoQt) { throw '-Pilot requires the pilot (drop -NoQt)' }
            # Native platform so the window shows (clear a stray offscreen).
            if ($env:QT_QPA_PLATFORM -eq 'offscreen') {
                Remove-Item Env:QT_QPA_PLATFORM
            }
            Write-Host '=== launching pilot GUI (close the window to exit) ==='
            Write-Host "run: $pilotExe"
            # -NoNewWindow: pilot.exe reads this terminal's stdin (an embedded
            # Python console); a fresh console would hit EOF and quit at once.
            $proc = Start-Process -FilePath $pilotExe -WorkingDirectory $Repo `
                -NoNewWindow -PassThru
            for ($i = 0; $i -lt 40 -and -not $proc.HasExited; $i++) {
                Start-Sleep -Milliseconds 250
                $proc.Refresh()
                if ($proc.MainWindowHandle -ne 0) { break }
            }
            if ($proc.HasExited) {
                throw ("pilot exited before showing a window (code " +
                    "$($proc.ExitCode))")
            }
            Write-Host ("pilot window '$($proc.MainWindowTitle)' opened " +
                "(handle $($proc.MainWindowHandle)); close it to exit")
            $proc.WaitForExit()
            if ($proc.ExitCode -ne 0) {
                throw "pilot exited with code $($proc.ExitCode)"
            }
        }
    } finally {
        Pop-Location
    }
}

Write-Host 'done'

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
