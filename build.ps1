#Requires -Version 5.1
#
# Warning: this script is still work in progress. Read carefully before using.
#
# Build solvcon (the _solvcon extension and, by default, the Qt pilot) on
# Windows against a scdv produced by
# contrib/dependency/windows/build-scdv-windows.ps1.  solvcon's Makefile and
# its Unix development workflow require "make", which is not a standard
# Windows tool.  This drives CMake directly through the win-reldbg / win-rel /
# win-dbg presets in CMakePresets.json, adding only the scdv-specific cache
# variables.
# Those three path variables belong in CMakeUserPresets.json; pass -Preset
# <name> to build such a preset and the script leaves them to it.  Start from
# contrib/cmake/CMakeUserPresets.win-example.json: the other template inherits
# presets that are disabled on Windows.
#
# It finds a CMake >= 4.0.1 (solvcon's minimum; the VS 2022 Build Tools bundle
# only 3.31.6, so it falls back to VS 2026's 4.x) and compiles with the same
# toolset as the scdv's numpy -- select it with SCDV_VS_VERSION='[17.0,18.0)'
# for VS 2022 v143.  The scdv must already be built; activate it first or pass
# -ScdvBase <scdv dir>.
#
# Usage:
#   .\build.ps1
#       Configure (preset win-reldbg) and build _solvcon and the pilot against
#       the active scdv (or -ScdvBase), then place the module.
#   .\build.ps1 -ScdvBase <dir>       activate the scdv at <dir> first
#   .\build.ps1 -BuildType Release    use the win-rel preset, dropping the
#                                     debug symbols to compile faster
#   .\build.ps1 -BuildType Debug      use the win-dbg preset
#   .\build.ps1 -Preset local         use a preset from CMakeUserPresets.json,
#                                     which is expected to supply the
#                                     dependency-prefix paths itself (start
#                                     from CMakeUserPresets.win-example.json)
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
    [ValidateSet('RelWithDebInfo', 'Release', 'Debug')]
    [string]$BuildType = 'RelWithDebInfo',
    [string]$Preset,
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
        Select-Object -First 44
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
        throw '-Sanitize cannot use the Debug preset: MSVC ASan rejects the ' +
            "Debug /RTC1 runtime checks. Drop -BuildType Debug."
    }
    if ($Test) {
        throw '-Sanitize cannot run -Test through stock python.exe; use ' +
            '-Gtest or -PilotTest so AddressSanitizer starts with the process'
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
# A named preset is taken verbatim, so a preset from CMakeUserPresets.json can
# be built; -BuildType then selects nothing, since the named preset states its
# own build type.  PowerShell variable names are case-insensitive, so the
# resolved name cannot be spelled $preset next to the -Preset parameter.
if ($Preset) {
    if ($PSBoundParameters.ContainsKey('BuildType')) {
        throw '-Preset and -BuildType are exclusive: the named preset states its own build type'
    }
    $presetName = $Preset
} else {
    $presetName = switch ($BuildType) {
        'Debug'   { 'win-dbg' }
        'Release' { 'win-rel' }
        default   { 'win-reldbg' }
    }
}

function Get-ConfigurePresetBinaryDir {
    # The binary directory a configure preset states, resolved through
    # inherits.  Reading it beats recomputing "build\<preset>": the presets
    # are free to name their build trees, and a user preset that inherits one
    # of them gets its own.
    param([string]$Root, [string]$Name)
    $presets = @{}
    foreach ($file in @('CMakePresets.json', 'CMakeUserPresets.json')) {
        $path = Join-Path $Root $file
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $doc = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        if (-not $doc.PSObject.Properties['configurePresets']) { continue }
        foreach ($entry in $doc.configurePresets) { $presets[$entry.name] = $entry }
    }
    # Depth-first through inherits, nearest definition wins, as CMake resolves
    # it.
    function Find-BinaryDir {
        param([string]$Name)
        $entry = $presets[$Name]
        if (-not $entry) { return $null }
        if ($entry.PSObject.Properties['binaryDir']) { return $entry.binaryDir }
        if ($entry.PSObject.Properties['inherits']) {
            foreach ($parent in @($entry.inherits)) {
                $found = Find-BinaryDir $parent
                if ($found) { return $found }
            }
        }
        return $null
    }
    $raw = Find-BinaryDir $Name
    if (-not $raw) { throw "preset '$Name' states no binaryDir" }
    $raw = $raw.Replace('${sourceDir}', $Root).Replace('${presetName}', $Name)
    if ($raw -match '\$\{') { throw "unsupported macro in binaryDir '$raw'" }
    $raw = $raw.Replace('/', '\')
    if (-not [IO.Path]::IsPathRooted($raw)) {
        $raw = Join-Path $Root $raw
    }
    return [IO.Path]::GetFullPath($raw)
}

$bld = Get-ConfigurePresetBinaryDir $Repo $presetName
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
    $onpath = @(Get-Command cmake.exe -All -ErrorAction SilentlyContinue)
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

$py = $null
if (-not $Preset) {
    if (-not $env:SCDV_USRDIR) {
        if (-not $ScdvBase) { $ScdvBase = $env:SCDV_BASE }
        if (-not $ScdvBase) {
            throw ('no active scdv: activate one (". <scdv>\Activate.ps1") ' +
                'or pass -ScdvBase <scdv dir>')
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
}

# --- Configure and build via the preset -------------------------------------

# The preset holds the static knobs (generator, build type, BUILD_QT,
# USE_GOOGLETEST, BLA_VENDOR, output dirs); pass the scdv-specific cache
# variables on top.  -NoQt overrides the preset's BUILD_QT=ON.
$extra = @()
if ($Preset) {
    # A named preset comes from CMakeUserPresets.json, whose whole purpose is
    # to carry the paths of one machine's dependency prefix, so leave them to
    # it rather than overriding them from the command line.
    Write-Host "preset $presetName supplies the dependency-prefix paths"
} else {
    $pybind = & $py -m pybind11 --cmakedir
    Assert-LastExit 'pybind11 --cmakedir'
    $extra += @(
        "-DPYTHON_EXECUTABLE=$py",
        "-Dpybind11_path=$pybind",
        "-DCMAKE_PREFIX_PATH=$usr"
    )
}
if ($NoQt) { $extra += '-DBUILD_QT=OFF' }
# -Gtest adds the gtest binary to the target list below; the preset already
# states USE_GOOGLETEST, so nothing has to be turned on here. -Sanitize (which
# implies -Gtest) builds that binary under AddressSanitizer. It links the ASan
# runtime directly so ASan initializes at process start, unlike loading an
# instrumented .pyd into a stock python.exe.
if ($Sanitize) { $extra += '-DUSE_SANITIZER=ON' }

# Timestamps preserved from Python wheels can appear ahead of the local clock
# and make Ninja's automatic CMake regeneration loop.  Suppress it only for
# this scripted build, then regenerate the tree with the normal rule restored.
$configureExtra = @($extra) + '-DCMAKE_SUPPRESS_REGENERATION=ON'
$restoreExtra = @($extra) + '-DCMAKE_SUPPRESS_REGENERATION=OFF'
$cachePath = Join-Path $bld 'CMakeCache.txt'
$operationFailure = $null

Push-Location $Repo
try {
    Write-Host "configuring solvcon (preset $presetName, BUILD_QT=$(if ($NoQt) {'OFF'} else {'ON'})) ..."
    & $cmake --preset $presetName @configureExtra
    Assert-LastExit 'cmake configure'

    if (-not $py) {
        $cacheEntry = Get-Content -LiteralPath $cachePath |
            Select-String -Pattern '^PYTHON_EXECUTABLE:[^=]+=(.*)$' |
            Select-Object -First 1
        if (-not $cacheEntry) {
            throw "preset '$presetName' did not configure PYTHON_EXECUTABLE"
        }
        $py = $cacheEntry.Matches[0].Groups[1].Value
        if (-not (Test-Path -LiteralPath $py)) {
            throw "preset '$presetName' configured missing Python at $py"
        }
    }

    # The build presets carry the target lists: <preset> builds the module and
    # the pilot, <preset>-module the module alone, <preset>-gtest the C++ test
    # binary.
    $buildPresets = @()
    if ($NoQt) {
        $buildPresets += "$presetName-module"
    } else {
        $buildPresets += $presetName
    }
    if ($Gtest) { $buildPresets += "$presetName-gtest" }
    foreach ($buildPreset in $buildPresets) {
        Write-Host "building preset: $buildPreset"
        & $cmake --build --preset $buildPreset
        Assert-LastExit "cmake build --preset $buildPreset"
    }
} catch {
    $operationFailure = $_
} finally {
    # A tree left suppressed only inconveniences other tools, so warn rather
    # than fail a build that already succeeded.
    try {
        $suppressed = (Test-Path -LiteralPath $cachePath) -and
            (Select-String -LiteralPath $cachePath -Quiet `
                -Pattern '^CMAKE_SUPPRESS_REGENERATION:[^=]+=ON$')
        if ($suppressed) {
            Write-Host 'restoring automatic CMake regeneration ...'
            & $cmake --preset $presetName @restoreExtra
            if ($LASTEXITCODE -ne 0) {
                Write-Warning ('automatic CMake regeneration could not be ' +
                    "restored; reconfigure $bld before building again")
            }
        }
    } catch {
        Write-Warning "regeneration cleanup failed: $_"
    }
    Pop-Location
}
if ($operationFailure) { throw $operationFailure }

# The _solvcon_py target places the module at the repository root, where
# solvcon.core imports it.
$pyd = Get-ChildItem -LiteralPath $Repo -Filter '_solvcon*.pyd' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $pyd) { throw "no _solvcon*.pyd produced under $Repo" }
Write-Host "placed module: $($pyd.FullName)"
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
