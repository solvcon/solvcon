#Requires -Version 5.1
#
# Warning: this script is still work in progress. Read carefully before using.
#
# Build solvcon's runtime dependencies from source on Windows.  Windows sibling
# of contrib/dependency/build-scdv.sh: same BASE/PYTHON/NUMPY/QT sections
# (guarded by SCDVBUILD_*), build functions, download/log/timing helpers, and
# activate contract, in PowerShell + MSVC.
#
# Windows specifics (detailed at each build function, and in the devplan
# doc/source/devplan/win-devenv/index.md): no ./configure (CMake+Ninja+MSVC;
# openssl via perl Configure+nmake; CPython via PCbuild + PC\layout); DLLs are
# found by PATH, not LD/DYLD_LIBRARY_PATH; libclang is Qt's prebuilt vs2022_64
# 7z; the MSVC environment is imported from vcvars64.bat via vswhere.
#
# Prerequisites are not installed here; see -PrintPrereq.
#
# Toolchain: build the NUMPY/QT sections with a stable MSVC (VS 2022, v143;
# SCDV_VS_VERSION='[17.0,18.0)').  The VS 2026 preview (v145) miscompiles
# numpy's long double math, so "import numpy" aborts with OverflowError;
# CPython builds fine on either.  The VS Build Tools installer is machine-wide
# (admin); -PrintPrereq prints the elevated command.
#
# Usage:
#   .\build-scdv-windows.ps1
#       Build everything (BASE + PYTHON + NUMPY + QT).  The default is "build
#       all" when no SCDVBUILD_* env var is set.  To limit to a single section,
#       set one of SCDVBUILD_BASE/PYTHON/NUMPY/QT=1 and leave the others unset;
#       SCDVBUILD_ALL=1 works the same as the default invocation.  Before any
#       package is built the user is prompted to confirm; pass -NoConfirm to
#       skip.
#   .\build-scdv-windows.ps1 -WriteActivateOnly
#       Write only <SCDV_BASE>\Activate.ps1 and exit.  Useful for refreshing
#       the activation script for an already-built scdv without triggering any
#       build section.
#   .\build-scdv-windows.ps1 -Skip PKG[,PKG...]
#       Skip a package within whatever sections are otherwise selected.  PKG
#       can be one of: zlib openssl sqlite openblas mkl python pybind11 cython
#       numpy scipy qt pyside6.  Accepts a comma-separated list and may be
#       repeated.
#   .\build-scdv-windows.ps1 -NoConfirm
#       Skip the "Press Enter to start the build" prompt.  Use in
#       non-interactive runs (CI, scripts).
#   .\build-scdv-windows.ps1 -NoSharedDownload
#       Keep a per-scdv download directory instead of a junction into the
#       SCDV_SHARED_DLDIR cache.  Equivalent to SCDV_SHARED_DLDIR="none".
#   .\build-scdv-windows.ps1 -PrintPrefix
#       Print SCDV_PREFIX (the path prefix that SCDV_BASE is derived from) and
#       exit.  Nothing else is printed and no directories are created, so this
#       is safe to capture: $prefix = .\build-scdv-windows.ps1 -PrintPrefix.
#   .\build-scdv-windows.ps1 -PrintPrereq
#       Print the prerequisite install commands and manual notes, then exit.
#       Nothing is built and no directories are created.
#   .\build-scdv-windows.ps1 -PrintVsInstall
#       Print just the elevated command to install the VS 2022 Build Tools
#       (the v143 MSVC toolset), then exit.  Run it from an administrator
#       account (the installer is machine-wide).
#
# Overridable variables (environment variables; defaults resolved below):
#   Package versions:
#     PYTHON_VERSION: CPython release tag.
#     QT_MAJOR_VER: Qt major.minor version.
#     QT_SUB_VER: Qt patch version.
#     PYSIDE_VERSION: Qt for Python (pyside-setup) source release version.
#     LIBCLANG_VERSION: Qt prebuilt libclang version for shiboken.
#     PYTHON_SHA256: required when overriding PYTHON_VERSION.
#     LIBCLANG_SHA256: required when overriding LIBCLANG_VERSION.
#     QT_SHA256: required when overriding QT_MAJOR_VER or QT_SUB_VER.
#     PYSIDE_SHA256: required when overriding PYSIDE_VERSION.
#   Build settings:
#     SCDV_NP: Parallel build jobs.
#     SCDV_WIN_BLAS: LP64 BLAS/LAPACK provider for solvcon's matmul() /
#       EigenSystem: "mkl" (default, Intel oneAPI MKL, an optimized vendor
#       library for x86) or "openblas" (the official OpenBLAS prebuilt).
#     SCDV_PREFIX: Path prefix to SCDV_BASE.
#     SCDV_BASE: Base directory holding the install prefix, including Python
#       and Qt version.
#     SCDV_DLDIR: Directory for downloaded tarballs (real dir or junction,
#       depending on SCDV_SHARED_DLDIR).
#     SCDV_SHARED_DLDIR: Shared cache for downloaded tarballs across solvcon
#       development environments (scdvs).  Set to "none" (or pass
#       -NoSharedDownload) to keep a per-scdv download directory.
#     SCDV_FFMPEG_DIR: Prefix of an FFmpeg (holding include, lib, and bin).
#       Set it to build Qt Multimedia against the FFmpeg backend, which is
#       what encodes the pilot's MP4 movies; unset leaves the module out.
#   MSVC selection (see the toolchain note above and Import-VcVars):
#     SCDV_VS_VERSION: vswhere -version range choosing which Visual Studio
#       provides cl/vcvars, e.g. "[17.0,18.0)" for VS 2022.  Default: newest.
#     SCDV_VCVARS: full path to a vcvars64.bat, used verbatim (overrides
#       SCDV_VS_VERSION).
#     SCDV_MSVC_TOOLSET: PlatformToolset for CPython's PCbuild, e.g. "v143".
#       Default: auto-detected newest installed.

[CmdletBinding()]
param(
    [switch]$WriteActivateOnly,
    [switch]$PrintPrefix,
    [switch]$PrintPrereq,
    [switch]$PrintVsInstall,
    [switch]$NoConfirm,
    [switch]$NoSharedDownload,
    [string[]]$Skip,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Build in isolation from the per-user site (%APPDATA%\Python\...).  Every
# Python of the same version shares it, so a package already there (a user's
# own pytest/matplotlib/flake8) makes "pip install -U <pkg>" report "already
# satisfied" and skip installing into this scdv, leaving it incomplete.  The
# activate script sets this too, for use after the build.
$env:PYTHONNOUSERSITE = '1'

# Native (non-PowerShell) commands do not raise on a non-zero exit; StrictMode
# and $ErrorActionPreference only cover cmdlets.  Call this after every native
# invocation whose failure should abort the build, mirroring bash "set -e".
function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        throw "${What}: exited with code $LASTEXITCODE"
    }
}

function Get-EnvOrDefault {
    param([string]$Name, [string]$Default)
    $val = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($val)) { return $Default }
    return $val
}

$KnownPkgs = @(
    'zlib', 'openssl', 'sqlite', 'openblas', 'mkl', 'python', 'pybind11',
    'cython', 'numpy', 'scipy', 'qt', 'pyside6'
)

# Normalize -Skip (comma lists and repeats already flatten into $Skip) and warn
# on unknown names but accept them, matching the bash scdv_add_skip behavior.
$SkipList = @()
foreach ($raw in $Skip) {
    foreach ($name in ($raw -split ',')) {
        $n = $name.Trim()
        if ($n -eq '') { continue }
        if ($KnownPkgs -notcontains $n) {
            Write-Warning ("--skip '$n' is not a known package " +
                "(known: $($KnownPkgs -join ' '))")
        }
        $SkipList += $n
    }
}

function Test-Skip {
    param([string]$Pkg)
    return ($SkipList -contains $Pkg)
}

function Assert-FortranCompiler {
    foreach ($name in @('flang-new.exe', 'flang.exe', 'gfortran.exe')) {
        $compiler = Get-Command $name -ErrorAction SilentlyContinue
        if ($compiler) {
            Write-Host "using Fortran compiler: $($compiler.Source)"
            return
        }
    }
    throw ('no Fortran compiler found; install one with the command from ' +
        '-PrintPrereq, then open a new shell so it is on PATH')
}

if ($Help) {
    # Print the header comment block (lines 2..first blank), stripped of "#".
    Get-Content -LiteralPath $PSCommandPath |
        Select-Object -Skip 1 |
        ForEach-Object {
            if ($_ -match '^\s*$') { return }
            $_ -replace '^#\s?', ''
        } | Select-Object -First 90
    exit 0
}

# CPython release tag.
$PythonVersion = Get-EnvOrDefault 'PYTHON_VERSION' '3.14.7'
# Qt major.minor version.
$QtMajorVer = Get-EnvOrDefault 'QT_MAJOR_VER' '6.11'
# Qt patch version.
$QtSubVer = Get-EnvOrDefault 'QT_SUB_VER' '1'
# Qt for Python (pyside-setup) source release version.
$PysideVersion = Get-EnvOrDefault 'PYSIDE_VERSION' "$QtMajorVer.$QtSubVer"
# Qt prebuilt libclang version for shiboken (see Get-LibClang).  Pinned to the
# same 21.x line the macOS script uses; Qt's Windows prebuilt libclang for this
# release is published as a vs2022_64 7z.
$LibClangVersion = Get-EnvOrDefault 'LIBCLANG_VERSION' '21.1.2'

# SHA-256 of every version-bearing download.  The keys are literal on
# purpose: interpolating the version variables would make an overridden
# version match its own key and receive the default version's digest,
# which would make <NAME>_SHA256 unreachable.
$PinnedHashes = @{
    'Python-3.14.7.tar.xz' =
        '3b48dac8fb59f62eaa67ac83c1eb12bda1b7a08406dd286e252c11a66be27f81'
    'libclang-release_21.1.2-based-windows-vs2022_64.7z' =
        '76a4cd9eed084119515dae2f4271c47b02a7736d9ead07ced2d776b20aad51ab'
    'qt-6.11.1.tar.xz' =
        '252acef8c5ae68074d91cadba2ee4a83465051bbb970dd26e8f0daa0f3904e03'
    'pyside-setup-everywhere-src-6.11.1.tar.xz' =
        '6ffd9835bb0dd2c56f061d62f1616bb1707cfc0202b80e3165d6be087f3965e2'
}

function Get-PinnedHash {
    param([string]$Fn, [string]$EnvName)
    if ($PinnedHashes.ContainsKey($Fn)) { return $PinnedHashes[$Fn] }
    return [Environment]::GetEnvironmentVariable($EnvName)
}

function Show-VsInstall {
    # Print the elevated command to install the Visual Studio 2022 Build Tools
    # (the v143 MSVC toolset + Windows SDK).  This script never installs
    # anything; copy the output, review it, and run it from an administrator
    # account.  Exposed on its own via -PrintVsInstall and included in
    # -PrintPrereq.
    @'
# ---------------------------------------------------------------------------
# MSVC toolchain -- RUN FROM A SEPARATE ADMINISTRATOR ACCOUNT.
# ---------------------------------------------------------------------------
# The Visual Studio Build Tools installer is machine-wide (it elevates and
# writes to HKLM), so run the command below from an administrator account, not
# the unprivileged account that runs this build.  It installs the "Desktop
# development with C++" workload (VCTools): cl.exe, nmake, the Windows SDK, and
# the vcvars64.bat this script imports.  --includeRecommended adds the matching
# Windows SDK.  The command is a single line so it pastes cleanly into an
# elevated PowerShell or cmd:

winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Or, with the standalone bootstrapper (elevated):

vs_BuildTools.exe --passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended

# Why VS 2022 specifically: its toolset is v143, the one numpy's own official
# wheels are built with.  The VS 2026 preview toolset (v145, MSVC 14.51)
# miscompiles numpy's long double math and breaks "import numpy" (see the
# header toolchain note).  CPython builds fine on either.
#
# After installing, run these (in the build account's PowerShell) to target
# VS 2022 for the numpy/scipy/Qt builds without disturbing a coexisting VS
# 2026 -- both are honored:

$env:SCDV_VS_VERSION   = '[17.0,18.0)'   # pick VS 2022 for cl/vcvars
$env:SCDV_MSVC_TOOLSET = 'v143'          # PlatformToolset for CPython
'@
}

function Show-Prereq {
    # Print the prerequisite install commands and manual notes.  This script
    # never installs anything; copy the output, review it, and run it.
    #
    # winget is the primary channel.  The MSVC toolchain is called out
    # separately below because its installer is machine-wide.
    @'
# VS 2022's bundled CMake is older than solvcon's required 4.0.1, so install
# current CMake and Ninja separately.  curl.exe and tar ship with Windows;
# 7-Zip is auto-fetched (7zr) when absent.  Perl and NASM are only for building
# openssl (skip openssl to avoid them):
winget install --id Kitware.CMake -e --scope user
winget install --id Ninja-build.Ninja -e --scope user
winget install --id StrawberryPerl.StrawberryPerl -e
winget install --id NASM.NASM -e
'@
    Show-VsInstall
    @'

# scipy needs a Fortran compiler.  WinLibs supplies gfortran through winget:
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --scope user

# LLVM flang is also supported when it is already on PATH.  numpy/scipy need
# BLAS/LAPACK too; this script uses the scipy-openblas64 wheel and exports its
# pkg-config, so no separate OpenBLAS install is required.  See Build-Numpy /
# Build-Scipy for details.

# Qt Multimedia encodes the pilot's MP4, and the QT section builds it only
# where SCDV_FFMPEG_DIR names an FFmpeg.  Nothing to install: unpack a
# prebuilt LGPL shared package, which ships the MSVC import libraries the Qt
# build links against.  Qt 6.11 is tested against FFmpeg 7.1; 8.x compiles.
$env:SCDV_FFMPEG_DIR = 'C:\path\to\ffmpeg-lgpl-shared'
'@
}

if ($PrintVsInstall) {
    Show-VsInstall
    exit 0
}

if ($PrintPrereq) {
    Show-Prereq
    exit 0
}

# Default to a full BASE+PYTHON+NUMPY+QT build when the caller has not selected
# a specific section.  Setting any one of SCDVBUILD_* to "1" disables this
# default and runs only the explicitly selected sections.
$BuildAll = Get-EnvOrDefault 'SCDVBUILD_ALL' ''
$BuildBase = Get-EnvOrDefault 'SCDVBUILD_BASE' ''
$BuildPython = Get-EnvOrDefault 'SCDVBUILD_PYTHON' ''
$BuildNumpy = Get-EnvOrDefault 'SCDVBUILD_NUMPY' ''
$BuildQt = Get-EnvOrDefault 'SCDVBUILD_QT' ''
if (-not ($BuildAll + $BuildBase + $BuildPython + $BuildNumpy + $BuildQt)) {
    $BuildAll = '1'
}

# Parallel build jobs.
$ScdvNp = Get-EnvOrDefault 'SCDV_NP' "$($env:NUMBER_OF_PROCESSORS)"
if ([string]::IsNullOrEmpty($ScdvNp)) { $ScdvNp = '4' }
# Path prefix to SCDV_BASE.
$ScdvPrefix = Get-EnvOrDefault 'SCDV_PREFIX' (Join-Path $HOME 'var\scdv\windows')

# -PrintPrefix is honored as soon as SCDV_PREFIX is resolved so the caller can
# capture the value without triggering any side effect (mkdir, activate-write,
# build).
if ($PrintPrefix) {
    Write-Output "SCDV_PREFIX=$ScdvPrefix"
    exit 0
}

# Base directory holding the install prefix, including Python and Qt version.
$ScdvBase = Get-EnvOrDefault 'SCDV_BASE' `
    "$ScdvPrefix-py$PythonVersion-qt$QtMajorVer.$QtSubVer"
# Directory for downloaded tarballs.
$ScdvDlDir = Get-EnvOrDefault 'SCDV_DLDIR' (Join-Path $ScdvBase 'downloaded')
# Shared tarball cache; SCDV_DLDIR becomes a junction into it.  Opt out with
# -NoSharedDownload or SCDV_SHARED_DLDIR="none" ("-").  No empty-string opt-out
# as in the Unix scripts: a Windows child sees an env var set to "" as absent.
$ScdvSharedDlDir = [Environment]::GetEnvironmentVariable('SCDV_SHARED_DLDIR')
if ($null -eq $ScdvSharedDlDir) {
    $ScdvSharedDlDir = Join-Path $HOME 'var\scdv\downloaded'
}
if ($NoSharedDownload -or $ScdvSharedDlDir -in @('none', '-')) {
    $ScdvSharedDlDir = ''
}

# Directory for building from source.
$ScdvSrcDir = Join-Path $ScdvBase 'src'
# Install root (user-space).
$ScdvUsrDir = Join-Path $ScdvBase 'usr'

# LP64 BLAS/LAPACK provider solvcon links for SimpleArray::matmul() and
# EigenSystem: "mkl" (default) or "openblas".  Distinct from the ILP64
# scipy-openblas64 wheel numpy links (see Get-OpenblasPkgConfig).
$WinBlas = (Get-EnvOrDefault 'SCDV_WIN_BLAS' 'mkl').ToLower()
if ($WinBlas -notin @('openblas', 'mkl')) {
    throw "SCDV_WIN_BLAS='$WinBlas' is not valid (use 'openblas' or 'mkl')."
}

# Directories and the activation script are created below, after the
# confirmation prompt, so aborting it leaves no trace.

# PC\layout puts python.exe at the prefix root, pip scripts under Scripts\, and
# CMake-installed DLLs under bin\; those three go on PATH.
$Py = Join-Path $ScdvUsrDir 'python.exe'
$env:PATH = "$ScdvUsrDir;$(Join-Path $ScdvUsrDir 'Scripts');" +
            "$(Join-Path $ScdvUsrDir 'bin');$env:PATH"

# Use System32 bsdtar, not a GNU tar first on PATH (Git for Windows): GNU tar
# reads "C:\path" as an rsh "host:file" spec ("Cannot connect to C:").
$Tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (-not (Test-Path -LiteralPath $Tar)) { $Tar = 'tar.exe' }

# --- MSVC environment ------------------------------------------------------

function Import-VcVars {
    # Run vcvars64.bat (located via vswhere) and copy its variables back, so
    # cl/nmake/the SDK are available.  No-op if cl.exe is already on PATH.
    # Select the VS when several are installed: SCDV_VCVARS (a vcvars64.bat
    # path) or SCDV_VS_VERSION (a vswhere -version range, e.g. "[17.0,18.0)"
    # for the v143 toolset); default -latest.
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }

    $vcvars = $env:SCDV_VCVARS
    if (-not $vcvars) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} `
            'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere)) {
            throw ("vswhere.exe not found at $vswhere; install the Visual " +
                "Studio Build Tools with the 'Desktop development with C++' " +
                "workload (see -PrintPrereq)")
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
        if ([string]::IsNullOrEmpty($vsroot)) {
            throw ("no Visual Studio install with the VC++ toolset was found" +
                ($(if ($env:SCDV_VS_VERSION) {
                    " for SCDV_VS_VERSION=$env:SCDV_VS_VERSION" } else { "" })))
        }
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

function Get-MsvcToolset {
    # Newest installed MSVC platform toolset ("v143" for VS 2022, "v145" for VS
    # 2026); Build-Python passes it to PCbuild, which pins an older v143 default
    # that MSB8020-fails on a VS shipping only a newer one.  Override with
    # SCDV_MSVC_TOOLSET.
    if ($env:SCDV_MSVC_TOOLSET) { return $env:SCDV_MSVC_TOOLSET }
    if (-not $env:VSINSTALLDIR) { return '' }
    $base = Join-Path $env:VSINSTALLDIR 'MSBuild\Microsoft\VC'
    $names = @()
    foreach ($vc in Get-ChildItem -LiteralPath $base -Directory `
                        -ErrorAction SilentlyContinue) {
        $tsdir = Join-Path $vc.FullName 'Platforms\x64\PlatformToolsets'
        if (Test-Path -LiteralPath $tsdir) {
            $names += (Get-ChildItem -LiteralPath $tsdir -Directory |
                Where-Object { $_.Name -match '^v\d+$' } |
                Select-Object -ExpandProperty Name)
        }
    }
    if (-not $names) { return '' }
    return ($names | Sort-Object { [int]($_ -replace '\D', '') } |
        Select-Object -Last 1)
}

# --- Helpers (translated from devenv/scripts/func.d/build_utils) ------------

function Get-Download {
    # Download into SCDV_DLDIR, verify it, then atomically replace the cache.
    param([string]$Fn, [string]$Url, [string]$ExpectedHash)
    if ([string]::IsNullOrWhiteSpace($ExpectedHash)) {
        throw "no checksum configured for $Fn"
    }
    if ($ExpectedHash.Length -ne 64) {
        throw "invalid SHA-256 checksum for $Fn"
    }
    $expected = $ExpectedHash.ToLower()
    $loc = Join-Path $ScdvDlDir $Fn
    if (Test-Path -LiteralPath $loc) {
        $actual = (Get-FileHash -LiteralPath $loc `
            -Algorithm SHA256).Hash.ToLower()
        if ($actual -eq $expected) { return }
    }

    $temp = "$loc.$PID.part"
    try {
        Write-Host "Downloading $Url"
        curl.exe -fsSL -o $temp $Url
        Assert-LastExit "download $Fn"
        $actual = (Get-FileHash -LiteralPath $temp `
            -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $expected) {
            throw "$Fn SHA256 mismatch: expected $expected got $actual"
        }
        Move-Item -LiteralPath $temp -Destination $loc -Force
    } finally {
        if (Test-Path -LiteralPath $temp) {
            Remove-Item -LiteralPath $temp -Force
        }
    }
}

function Invoke-Logged {
    # Run a command (a scriptblock), tee stdout+stderr to $Log, abort on
    # non-zero exit (bash with_log).  Relax ErrorActionPreference around the
    # call: under 'Stop' the 2>&1 merge turns the tool's first stderr line (a
    # mere cmake/meson/cl warning) into a terminating error; rely on the exit
    # code via Assert-LastExit instead.
    param([string]$Log, [scriptblock]$Cmd)
    "run: $($Cmd.ToString().Trim())" | Tee-Object -FilePath $Log | Out-Host
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Cmd 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
    } finally {
        $ErrorActionPreference = $prev
    }
    Assert-LastExit "command logged to $Log"
}

function Expand-Source {
    # Extract an archive from SCDV_DLDIR into SCDV_SRCDIR, replacing any prior
    # unpack of the same destination.  bsdtar (the "tar" shipped with Windows
    # 10 1803+) reads .tar.gz and .tar.xz natively.
    param([string]$Fn, [string]$DestDir)
    Push-Location $ScdvSrcDir
    try {
        $dest = Join-Path $ScdvSrcDir $DestDir
        if (Test-Path -LiteralPath $dest) {
            Remove-Item -LiteralPath $dest -Recurse -Force
        }
        & $Tar xf (Join-Path $ScdvDlDir $Fn)
        Assert-LastExit "extract $Fn"
    } finally {
        Pop-Location
    }
}

# --- Timing: record wall-clock per build_<pkg> and print a summary on exit ---

$ScdvOverallStart = [DateTimeOffset]::UtcNow
$ScdvTimings = New-Object System.Collections.Generic.List[object]

function Format-Elapsed {
    param([int]$Seconds)
    if ($Seconds -ge 3600) {
        return ('{0}h{1:d2}m{2:d2}s' -f `
            [int]($Seconds / 3600), [int](($Seconds % 3600) / 60),
            ($Seconds % 60))
    } elseif ($Seconds -ge 60) {
        return ('{0}m{1:d2}s' -f [int]($Seconds / 60), ($Seconds % 60))
    }
    return "${Seconds}s"
}

function Invoke-Timed {
    # Run a build function, recording its wall-clock elapsed and status
    # (skipped vs ok).  A failure propagates (the caller's trap-equivalent
    # finally block records the partial time as FAIL).
    param([scriptblock]$Fn, [string]$Label)
    $start = [DateTimeOffset]::UtcNow
    & $Fn
    $elapsed = [int]([DateTimeOffset]::UtcNow - $start).TotalSeconds
    $status = if (Test-Skip $Label) { 'skipped' } else { 'ok' }
    $ScdvTimings.Add([pscustomobject]@{
        Package = $Label; Status = $status; Seconds = $elapsed
    })
}

function Write-Timings {
    param([int]$ExitCode)
    $overall = [int]([DateTimeOffset]::UtcNow - $ScdvOverallStart).TotalSeconds
    if ($ScdvTimings.Count -eq 0 -and $overall -lt 2) { return }
    Write-Host ''
    Write-Host '=== build timings ==='
    ('{0,-10} {1,-8} {2,10}' -f 'package', 'status', 'time') | Write-Host
    ('{0,-10} {1,-8} {2,10}' -f '----------', '--------', '----------') |
        Write-Host
    foreach ($t in $ScdvTimings) {
        ('{0,-10} {1,-8} {2,10}' -f `
            $t.Package, $t.Status, (Format-Elapsed $t.Seconds)) | Write-Host
    }
    ('{0,-10} {1,-8} {2,10}' -f '----------', '--------', '----------') |
        Write-Host
    ('{0,-10} {1,-8} {2,10} (exit {3})' -f `
        'TOTAL', '-', (Format-Elapsed $overall), $ExitCode) | Write-Host
}

# --- Activation script ------------------------------------------------------

function Write-Activate {
    # Write <SCDV_BASE>\Activate.ps1.  The activation script is self-locating
    # (reads its own path at dot-source time) so the scdv directory can be moved
    # without rewriting the file.  Dot-sourcing it prepends our bin directories
    # to PATH, points Qt at our plugins, and defines scdv_deactivate to restore
    # the original environment.  Windows resolves DLL dependencies through PATH,
    # so there is no LD_LIBRARY_PATH / DYLD_LIBRARY_PATH analog to manage.
    $target = Join-Path $ScdvBase 'Activate.ps1'
    $body = @'
# Activate this SCDV: dot-source this file (do not execute).
#
#   PS> . <path-to-this-file>\Activate.ps1
#   PS> scdv_deactivate   # to restore the original environment

if ($env:SCDV_BASE) {
    Write-Error ("SCDV '$(Split-Path -Leaf $env:SCDV_BASE)' is already " +
        "active.  Run scdv_deactivate first.")
    return
}

# Resolve the directory this file lives in as SCDV_BASE.
$env:SCDV_BASE = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:SCDV_USRDIR = Join-Path $env:SCDV_BASE 'usr'

# Snapshot the current environment so scdv_deactivate can restore it.
$env:_SCDV_OLD_PATH = $env:PATH
$global:_SCDV_HAD_QT_QPA_PLATFORM = $null -ne $env:QT_QPA_PLATFORM
$env:_SCDV_OLD_QT_QPA_PLATFORM = $env:QT_QPA_PLATFORM
$global:_SCDV_HAD_QT_PLUGIN_PATH = $null -ne $env:QT_PLUGIN_PATH
$env:_SCDV_OLD_QT_PLUGIN_PATH = $env:QT_PLUGIN_PATH
$global:_SCDV_HAD_CMAKE_PREFIX_PATH = $null -ne $env:CMAKE_PREFIX_PATH
$env:_SCDV_OLD_CMAKE_PREFIX_PATH = $env:CMAKE_PREFIX_PATH
$global:_SCDV_HAD_PYTHONNOUSERSITE = $null -ne $env:PYTHONNOUSERSITE
$env:_SCDV_OLD_PYTHONNOUSERSITE = $env:PYTHONNOUSERSITE
$global:_SCDV_HAD_SOLVCON_USE_MKL = $null -ne $env:SOLVCON_USE_MKL
$env:_SCDV_OLD_SOLVCON_USE_MKL = $env:SOLVCON_USE_MKL
$global:_SCDV_HAD_SOLVCON_DEPS_CACHE = $null -ne $env:SOLVCON_DEPS_CACHE
$env:_SCDV_OLD_SOLVCON_DEPS_CACHE = $env:SOLVCON_DEPS_CACHE

# Ignore the per-user site (%APPDATA%\Python\...); a stray pytest/numpy/PySide6
# there would shadow this scdv (every same-version Python shares it).
$env:PYTHONNOUSERSITE = '1'

$usr = $env:SCDV_USRDIR
$_sp = Join-Path $usr 'Lib\site-packages'
# pyside6/shiboken6 DLLs live in their site-packages dirs (not bin/); Windows
# searches PATH for an exe's DLLs, so the pilot needs them there.
$env:PATH = "$usr;$(Join-Path $usr 'Scripts');$(Join-Path $usr 'bin');" +
            "$(Join-Path $_sp 'PySide6');$(Join-Path $_sp 'shiboken6');" +
            $env:PATH
$env:QT_PLUGIN_PATH = Join-Path $usr 'plugins'

# So a downstream CMake build finds this scdv's Qt, pybind11, and BLAS/LAPACK.
if ($env:CMAKE_PREFIX_PATH) {
    $env:CMAKE_PREFIX_PATH = "$usr;$env:CMAKE_PREFIX_PATH"
} else {
    $env:CMAKE_PREFIX_PATH = $usr
}

# Point the build's dependency archive cache (see gtests/CMakeLists.txt) at
# this scdv's tarball directory, so the two hold one set of tarballs.  A value
# the caller already chose wins, and a missing directory is left to
# build-scdv-windows.ps1, which makes it a junction into the shared cache.
$_dl = Join-Path $env:SCDV_BASE 'downloaded'
if (-not $env:SOLVCON_DEPS_CACHE -and
        (Test-Path -LiteralPath $_dl -PathType Container)) {
    $env:SOLVCON_DEPS_CACHE = $_dl
}

# An MKL scdv defaults the solvcon build to MKL: SOLVCON_USE_MKL in
# cpp/solvcon/CMakeLists.txt honors this env var (overridable with -D).
if ('__SCDV_WIN_BLAS__' -eq 'mkl') {
    $env:SOLVCON_USE_MKL = 'ON'
}

# Headless CI wants the offscreen platform; leave an interactive desktop alone.
if (-not $env:QT_QPA_PLATFORM -and $env:CI) {
    $env:QT_QPA_PLATFORM = 'offscreen'
}

function global:scdv_deactivate {
    if (-not $env:SCDV_BASE) {
        Write-Error 'No active SCDV.'
        return
    }
    $env:PATH = $env:_SCDV_OLD_PATH
    if ($global:_SCDV_HAD_QT_QPA_PLATFORM) {
        $env:QT_QPA_PLATFORM = $env:_SCDV_OLD_QT_QPA_PLATFORM
    } else {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
    if ($global:_SCDV_HAD_QT_PLUGIN_PATH) {
        $env:QT_PLUGIN_PATH = $env:_SCDV_OLD_QT_PLUGIN_PATH
    } else {
        Remove-Item Env:QT_PLUGIN_PATH -ErrorAction SilentlyContinue
    }
    if ($global:_SCDV_HAD_CMAKE_PREFIX_PATH) {
        $env:CMAKE_PREFIX_PATH = $env:_SCDV_OLD_CMAKE_PREFIX_PATH
    } else {
        Remove-Item Env:CMAKE_PREFIX_PATH -ErrorAction SilentlyContinue
    }
    if ($global:_SCDV_HAD_PYTHONNOUSERSITE) {
        $env:PYTHONNOUSERSITE = $env:_SCDV_OLD_PYTHONNOUSERSITE
    } else {
        Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    }
    if ($global:_SCDV_HAD_SOLVCON_USE_MKL) {
        $env:SOLVCON_USE_MKL = $env:_SCDV_OLD_SOLVCON_USE_MKL
    } else {
        Remove-Item Env:SOLVCON_USE_MKL -ErrorAction SilentlyContinue
    }
    if ($global:_SCDV_HAD_SOLVCON_DEPS_CACHE) {
        $env:SOLVCON_DEPS_CACHE = $env:_SCDV_OLD_SOLVCON_DEPS_CACHE
    } else {
        Remove-Item Env:SOLVCON_DEPS_CACHE -ErrorAction SilentlyContinue
    }
    Remove-Item Env:_SCDV_OLD_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:_SCDV_OLD_QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    Remove-Item Env:_SCDV_OLD_QT_PLUGIN_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:_SCDV_OLD_CMAKE_PREFIX_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:_SCDV_OLD_PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    Remove-Item Env:_SCDV_OLD_SOLVCON_USE_MKL -ErrorAction SilentlyContinue
    Remove-Item Env:_SCDV_OLD_SOLVCON_DEPS_CACHE -ErrorAction SilentlyContinue
    Remove-Item Env:SCDV_BASE -ErrorAction SilentlyContinue
    Remove-Item Env:SCDV_USRDIR -ErrorAction SilentlyContinue
    Remove-Variable -Scope global -Name _SCDV_HAD_QT_QPA_PLATFORM `
        -ErrorAction SilentlyContinue
    Remove-Variable -Scope global -Name _SCDV_HAD_QT_PLUGIN_PATH `
        -ErrorAction SilentlyContinue
    Remove-Variable -Scope global -Name _SCDV_HAD_CMAKE_PREFIX_PATH `
        -ErrorAction SilentlyContinue
    Remove-Variable -Scope global -Name _SCDV_HAD_PYTHONNOUSERSITE `
        -ErrorAction SilentlyContinue
    Remove-Variable -Scope global -Name _SCDV_HAD_SOLVCON_USE_MKL `
        -ErrorAction SilentlyContinue
    Remove-Variable -Scope global -Name _SCDV_HAD_SOLVCON_DEPS_CACHE `
        -ErrorAction SilentlyContinue
    Remove-Item Function:scdv_deactivate -ErrorAction SilentlyContinue
}
'@
    # Bake the provider so an MKL scdv turns on SOLVCON_USE_MKL on activation.
    $body = $body -replace '__SCDV_WIN_BLAS__', $WinBlas
    Set-Content -LiteralPath $target -Value $body -Encoding ascii
    Write-Host "wrote activation script: $target"
}

# --- Startup banner ---------------------------------------------------------

Write-Host "SCDV_NP=$ScdvNp"
Write-Host "SCDV_PREFIX=$ScdvPrefix"
Write-Host "SCDV_BASE=$ScdvBase"
Write-Host "SCDV_DLDIR=$ScdvDlDir"
Write-Host "SCDV_SHARED_DLDIR=$ScdvSharedDlDir"
Write-Host "SCDV_SRCDIR=$ScdvSrcDir"
Write-Host "SCDV_USRDIR=$ScdvUsrDir"
Write-Host "SCDV_WIN_BLAS=$WinBlas"
Write-Host "PYTHON_VERSION=$PythonVersion"
if ($SkipList.Count -gt 0) {
    Write-Host "SCDV_SKIP_LIST=$($SkipList -join ' ')"
}
Write-Host 'ready to build'

# -WriteActivateOnly is its own explicit confirmation: create the scdv
# directory just enough to drop the activate file in, then exit.
if ($WriteActivateOnly) {
    New-Item -ItemType Directory -Force -Path $ScdvBase | Out-Null
    Write-Activate
    exit 0
}

if (($BuildAll -eq '1' -or $BuildNumpy -eq '1') -and
        -not (Test-Skip 'scipy')) {
    Assert-FortranCompiler
}

# If SCDV_SHARED_DLDIR is configured it must exist already; the script never
# auto-creates it.  Check before any filesystem change so a typo or unmounted
# cache fails immediately instead of half-creating the per-scdv tree.
if ($ScdvSharedDlDir -and -not (Test-Path -LiteralPath $ScdvSharedDlDir)) {
    throw ("SCDV_SHARED_DLDIR=$ScdvSharedDlDir does not exist; create it or " +
        "set SCDV_SHARED_DLDIR to the empty string to opt out.")
}

# Final confirmation before any package is built.  Skip with -NoConfirm in
# non-interactive contexts.  No directory or activation file is created until
# the user has confirmed, so aborting here leaves the filesystem untouched.
if (-not $NoConfirm) {
    Read-Host 'Press Enter to start the build, Ctrl-C to abort' | Out-Null
}

New-Item -ItemType Directory -Force -Path $ScdvUsrDir, $ScdvSrcDir | Out-Null
if ($ScdvSharedDlDir) {
    # Make SCDV_DLDIR a directory junction into the shared cache.  Its
    # existence was verified above.  Refuse to clobber a real directory at
    # SCDV_DLDIR to avoid losing data; the caller has to move it (or clear
    # SCDV_SHARED_DLDIR) first.
    $item = Get-Item -LiteralPath $ScdvDlDir -ErrorAction SilentlyContinue
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        if ($item.Target -ne $ScdvSharedDlDir) {
            Remove-Item -LiteralPath $ScdvDlDir -Force
            New-Item -ItemType Junction -Path $ScdvDlDir `
                -Target $ScdvSharedDlDir | Out-Null
        }
    } elseif ($item) {
        throw ("SCDV_DLDIR=$ScdvDlDir exists as a real directory but " +
            "SCDV_SHARED_DLDIR is set; move its contents to " +
            "$ScdvSharedDlDir and remove the directory, or clear " +
            "SCDV_SHARED_DLDIR.")
    } else {
        New-Item -ItemType Junction -Path $ScdvDlDir `
            -Target $ScdvSharedDlDir | Out-Null
    }
} else {
    New-Item -ItemType Directory -Force -Path $ScdvDlDir | Out-Null
}
Write-Activate

Import-VcVars

# --- Build functions (translated from devenv/scripts/build.d/<pkg>) ---------

function Build-Zlib {
    if (Test-Skip 'zlib') { Write-Host 'skip: zlib'; return }
    $ver = '1.3.1'; $full = "zlib-$ver"; $fn = "$full.tar.gz"
    Get-Download $fn `
        "https://github.com/madler/zlib/archive/refs/tags/v$ver.tar.gz" `
        '17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c'
    Expand-Source $fn $full
    $bld = Join-Path $ScdvSrcDir "$full\build"
    New-Item -ItemType Directory -Force -Path $bld | Out-Null
    Push-Location $bld
    try {
        Invoke-Logged 'cmake.log' {
            cmake -G Ninja -DCMAKE_BUILD_TYPE=Release `
                "-DCMAKE_INSTALL_PREFIX=$ScdvUsrDir" ..
        }
        Invoke-Logged 'build.log' { cmake --build . --parallel }
        Invoke-Logged 'install.log' { cmake --install . }
    } finally {
        Pop-Location
    }
}

function Build-Openssl {
    if (Test-Skip 'openssl') { Write-Host 'skip: openssl'; return }
    # openssl has no CMake build; its perl Configure emits an nmake makefile.
    # VC-WIN64A is the 64-bit MSVC target.  Needs Strawberry Perl and NASM on
    # PATH (see -PrintPrereq).  OpenSSL 3.5 is supported through April 2030.
    $ver = '3.5.8'; $full = "openssl-$ver"; $fn = "$full.tar.gz"
    $url = "https://github.com/openssl/openssl/releases/download/openssl-$ver/$fn"
    Get-Download $fn $url `
        'a8f84a39918ec6415ce765d9b429d313ba97b8143169c172e734b9514464f5b2'
    Expand-Source $fn $full
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        Invoke-Logged 'configure.log' {
            perl Configure VC-WIN64A "--prefix=$ScdvUsrDir" `
                "--openssldir=$(Join-Path $ScdvUsrDir 'share\ssl')"
        }
        Invoke-Logged 'build.log' { nmake }
        Invoke-Logged 'install.log' { nmake install_sw install_ssldirs }
    } finally {
        Pop-Location
    }
}

function Build-Sqlite {
    if (Test-Skip 'sqlite') { Write-Host 'skip: sqlite'; return }
    # Build the amalgamation into a DLL + import lib with cl directly; the
    # autoconf tarball's ./configure does not apply on MSVC.  Same 3.36.0
    # amalgamation the Unix scripts use.
    $ver = '3360000'; $full = "sqlite-autoconf-$ver"; $fn = "$full.tar.gz"
    Get-Download $fn "https://www.sqlite.org/2021/$fn" `
        'bd90c3eb96bee996206b83be7065c9ce19aef38c3f4fb53073ada0d0b69bbce3'
    Expand-Source $fn $full
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        # SQLITE_API=__declspec(dllexport): without it cl /LD exports nothing
        # and produces no import lib.  Quote the /D against PowerShell parsing.
        Invoke-Logged 'build.log' {
            cl /nologo /O2 /LD '/DSQLITE_API=__declspec(dllexport)' `
                /DSQLITE_ENABLE_FTS5 /DSQLITE_ENABLE_RTREE `
                sqlite3.c /Fe:sqlite3.dll /link /IMPLIB:sqlite3.lib
        }
        $bin = Join-Path $ScdvUsrDir 'bin'
        $lib = Join-Path $ScdvUsrDir 'lib'
        $inc = Join-Path $ScdvUsrDir 'include'
        foreach ($d in @($bin, $lib, $inc)) {
            New-Item -ItemType Directory -Force -Path $d | Out-Null
        }
        Copy-Item 'sqlite3.dll' $bin -Force
        Copy-Item 'sqlite3.lib' $lib -Force
        Copy-Item @('sqlite3.h', 'sqlite3ext.h') $inc -Force
    } finally {
        Pop-Location
    }
}

function Build-Openblas {
    if (Test-Skip 'openblas') { Write-Host 'skip: openblas'; return }
    # solvcon's optional CBLAS matmul / LAPACK EigenSystem want a standard LP64
    # OpenBLAS (libopenblas + unsuffixed cblas.h), distinct from the ILP64
    # scipy-openblas64 wheel numpy uses (64_-suffixed, 64-bit ints).  Building
    # LP64 OpenBLAS needs Fortran, so install the official prebuilt Windows
    # release (self-contained, with an MSVC import lib), like Ubuntu's
    # libopenblas-dev.
    $ver = '0.3.33'
    $fn = "OpenBLAS-$ver-x64.zip"
    Get-Download $fn `
        "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$ver/$fn" `
        '7ad797ef0c9a5c42e28903bf726eaaaade307dafe187ff0e923d90cd4002780c'
    $dest = Join-Path $ScdvSrcDir "OpenBLAS-$ver"
    if (Test-Path -LiteralPath $dest) {
        Remove-Item -LiteralPath $dest -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    & $Tar -xf (Join-Path $ScdvDlDir $fn) -C $dest
    Assert-LastExit "extract $fn"
    $bin = Join-Path $ScdvUsrDir 'bin'
    $lib = Join-Path $ScdvUsrDir 'lib'
    $inc = Join-Path $ScdvUsrDir 'include'
    foreach ($d in @($bin, $lib, $inc)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }
    Copy-Item (Join-Path $dest 'lib\libopenblas.lib') $lib -Force
    Copy-Item (Join-Path $dest 'bin\libopenblas.dll') $bin -Force
    # Also next to python.exe: Windows does not search PATH for a .pyd's
    # dependent DLLs (since Python 3.8), but it does search the interpreter's
    # own directory, so solvcon's _solvcon extension finds libopenblas.dll
    # there at import.
    Copy-Item (Join-Path $dest 'bin\libopenblas.dll') $ScdvUsrDir -Force
    Copy-Item (Join-Path $dest 'include\*.h') $inc -Force
}

function Build-MklBlas {
    if (Test-Skip 'mkl') { Write-Host 'skip: mkl'; return }
    # Intel oneAPI MKL, the optimized vendor BLAS/LAPACK for x86, from its
    # official wheels.  Lay the pieces out under the scdv usr tree the way
    # Build-Openblas does, so solvcon's CMake finds mkl_rt + mkl_cblas.h.
    & $Py -m pip install -U mkl mkl-devel mkl-include intel-openmp
    Assert-LastExit 'pip install mkl'

    $bin = Join-Path $ScdvUsrDir 'bin'
    $lib = Join-Path $ScdvUsrDir 'lib'
    $inc = Join-Path $ScdvUsrDir 'include'
    foreach ($d in @($bin, $lib, $inc)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }

    # The wheels install under <prefix>\Library\{bin,lib,include}; find the
    # pieces by name in a single recursive pass rather than hardcoding that
    # layout or rescanning the whole (Python-laden) tree per piece.
    $mkl = Get-ChildItem -LiteralPath $ScdvUsrDir -Recurse -File `
        -Include 'mkl_cblas.h', 'mkl_rt.lib', 'mkl_rt*.dll' `
        -ErrorAction SilentlyContinue
    $hdr = $mkl | Where-Object Name -EQ 'mkl_cblas.h' | Select-Object -First 1
    $implib = $mkl | Where-Object Name -EQ 'mkl_rt.lib' | Select-Object -First 1
    $rtDll = $mkl | Where-Object Name -Like 'mkl_rt*.dll' | Select-Object -First 1
    if (-not ($hdr -and $implib -and $rtDll)) {
        throw ('Build-MklBlas: MKL not found after installing wheels ' +
            "(header=$hdr implib=$implib runtime=$rtDll)")
    }
    Copy-Item (Join-Path $hdr.DirectoryName '*') $inc -Recurse -Force
    Copy-Item (Join-Path $implib.DirectoryName 'mkl_*.lib') $lib -Force

    # Runtime: mkl_rt dynamically loads the mkl_* kernel DLLs and libiomp5md,
    # all installed beside it; copy that dir's DLLs to bin\ and next to
    # python.exe (Windows does not search PATH for a .pyd's dependent DLLs since
    # 3.8 but always searches the interpreter's own directory).
    foreach ($d in Get-ChildItem (Join-Path $rtDll.DirectoryName '*.dll')) {
        Copy-Item $d.FullName $bin -Force
        Copy-Item $d.FullName $ScdvUsrDir -Force
    }
}

function Build-Python {
    if (Test-Skip 'python') { Write-Host 'skip: python'; return }
    # CPython is built by PCbuild\build.bat (not ./configure).  "-e" fetches its
    # own vendored externals (private zlib/openssl/sqlite/...), so the BASE libs
    # are only for Qt etc.  PC\layout then copies a deployable prefix.
    $ver = $PythonVersion; $full = "Python-$ver"; $fn = "$full.tar.xz"
    $hash = Get-PinnedHash $fn 'PYTHON_SHA256'
    Get-Download $fn "https://www.python.org/ftp/python/$ver/$fn" $hash
    Expand-Source $fn $full
    $src = Join-Path $ScdvSrcDir $full
    Push-Location $src
    try {
        # Retarget PCbuild to the installed toolset (CPython's pinned v143
        # MSB8020-fails on a VS shipping only a newer one) via the
        # PlatformToolset env var, which MSBuild imports as a property; a /p:
        # switch would not survive build.bat's "="-splitting cmd parsing.
        $toolset = Get-MsvcToolset
        if ($toolset) { $env:PlatformToolset = $toolset }
        $buildBat = Join-Path $src 'PCbuild\build.bat'
        try {
            Invoke-Logged 'build.log' { & $buildBat -e -p x64 -c Release }
        } finally {
            Remove-Item Env:PlatformToolset -ErrorAction SilentlyContinue
        }
        # PC\layout options are on/off toggles: --include-dev (headers +
        # python3xx.lib), --include-pip, --precompile.
        $built = Join-Path $src 'PCbuild\amd64\python.exe'
        Invoke-Logged 'layout.log' {
            & $built (Join-Path $src 'PC\layout') --copy $ScdvUsrDir `
                --precompile --include-dev --include-pip
        }
        # --include-dev omits the stable-ABI python3.lib / python3.dll, but
        # shiboken6 links the Limited API and its CMake FATAL_ERRORs without
        # python3.lib; copy both from the build.
        $amd64 = Join-Path $src 'PCbuild\amd64'
        Copy-Item (Join-Path $amd64 'python3.lib') `
            (Join-Path $ScdvUsrDir 'libs') -Force
        Copy-Item (Join-Path $amd64 'python3.dll') $ScdvUsrDir -Force
    } finally {
        Pop-Location
    }

    # python3.exe copy: the ecosystem and solvcon's CMake (find_program(python3)
    # and "python3 -c ..." probes) use the Unix name.
    Copy-Item (Join-Path $ScdvUsrDir 'python.exe') `
        (Join-Path $ScdvUsrDir 'python3.exe') -Force

    # A .pth (its "import" line runs at startup) adds DLL search directories:
    # Windows does not search PATH for a .pyd's dependencies (Python 3.8+), the
    # Unix LD path analog.  bin/ holds this scdv's Qt DLLs; _solvcon.pyd also
    # links pyside6.abi3.dll and shiboken6.abi3.dll directly, which live in
    # their own site-packages dirs, so register those too (they load before
    # PySide6's own __init__ runs its add_dll_directory).
    $pth = Join-Path $ScdvUsrDir 'Lib\site-packages\scdv_dll_directory.pth'
    $pthline = 'import os, sys, sysconfig; ' +
        '_sp = sysconfig.get_paths().get(''purelib'') or ''''; ' +
        '[os.add_dll_directory(_d) for _d in [' +
        'os.path.join(sys.prefix, ''bin''), ' +
        'os.path.join(_sp, ''PySide6''), ' +
        'os.path.join(_sp, ''shiboken6'')] if os.path.isdir(_d)]'
    Set-Content -LiteralPath $pth -Value $pthline -Encoding ascii

    # The layout ships pip via --include-pip; bring setuptools/wheel/pip current
    # so later --no-build-isolation installs resolve their build backends.
    & $Py -m pip install -U setuptools wheel pip
    Assert-LastExit 'pip bootstrap'
}

function Build-Pybind11 {
    if (Test-Skip 'pybind11') { Write-Host 'skip: pybind11'; return }
    $ver = '3.1.0'; $full = "pybind11-$ver"; $fn = "$full.tar.gz"
    Get-Download $fn `
        "https://github.com/pybind/pybind11/archive/refs/tags/v$ver.tar.gz" `
        'ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665'
    Expand-Source $fn $full
    $bld = Join-Path $ScdvSrcDir "$full\build"
    New-Item -ItemType Directory -Force -Path $bld | Out-Null
    Push-Location $bld
    try {
        Invoke-Logged 'cmake.log' {
            cmake -G Ninja -DCMAKE_BUILD_TYPE=Release `
                "-DPYTHON_EXECUTABLE:FILEPATH=$Py" `
                "-DCMAKE_INSTALL_PREFIX=$ScdvUsrDir" `
                -DPYBIND11_TEST=OFF ..
        }
        Invoke-Logged 'build.log' { cmake --build . --parallel }
        Invoke-Logged 'install.log' { cmake --install . }
    } finally {
        Pop-Location
    }
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        Invoke-Logged 'setup.log' { & $Py -m pip install . }
    } finally {
        Pop-Location
    }
}

function Build-Cython {
    if (Test-Skip 'cython') { Write-Host 'skip: cython'; return }
    $ver = '3.0.12'; $full = "cython-$ver"; $fn = "$full.tar.gz"
    Get-Download $fn `
        "https://github.com/cython/cython/archive/refs/tags/$ver.tar.gz" `
        'a156fff948c2013f2c8c398612c018e2b52314fdf0228af8fbdb5585e13699c2'
    Expand-Source $fn $full
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        Invoke-Logged 'install.log' { & $Py -m pip install . }
    } finally {
        Pop-Location
    }
}

function Get-OpenblasPkgConfig {
    # numpy/scipy BLAS: install the scipy-openblas64 wheel and export its
    # pkg-config dir (returned, for PKG_CONFIG_PATH).  meson needs a pkg-config
    # binary, which Windows lacks, so install the pkgconf wheel and point
    # PKG_CONFIG at it.
    & $Py -m pip install -U scipy-openblas64 pkgconf
    Assert-LastExit 'pip install scipy-openblas64 pkgconf'
    $pkgconf = & $Py -c `
        'import pkgconf; print(pkgconf.get_executable())'
    Assert-LastExit 'pkgconf.get_executable'
    if ($pkgconf -and (Test-Path -LiteralPath $pkgconf)) {
        $env:PKG_CONFIG = $pkgconf
    }
    # From-source numpy links the scipy-openblas64 DLL but does not vendor it;
    # copy it next to python.exe (always on the DLL search path), else "import
    # numpy" fails with "DLL load failed importing _multiarray_umath".
    $oblibdir = & $Py -c 'import scipy_openblas64 as o; print(o.get_lib_dir())'
    Assert-LastExit 'scipy_openblas64.get_lib_dir'
    foreach ($dll in Get-ChildItem -LiteralPath $oblibdir -Filter '*.dll') {
        Copy-Item $dll.FullName $ScdvUsrDir -Force
    }
    $pcdir = Join-Path $ScdvSrcDir 'openblas-pkgconfig'
    New-Item -ItemType Directory -Force -Path $pcdir | Out-Null
    $pc = & $Py -c `
        'import scipy_openblas64 as o; print(o.get_pkg_config())'
    Assert-LastExit 'scipy_openblas64.get_pkg_config'
    Set-Content -LiteralPath (Join-Path $pcdir 'scipy-openblas.pc') `
        -Value $pc -Encoding ascii
    return $pcdir
}

function Build-Numpy {
    if (Test-Skip 'numpy') { Write-Host 'skip: numpy'; return }
    $ver = '2.5.1'; $full = "numpy-$ver"; $fn = "$full.tar.gz"
    Get-Download $fn `
        "https://github.com/numpy/numpy/releases/download/v$ver/$fn" `
        'a48a113e6afea91f5608793bafa7ef2ad481fefbda87ec5069f483de61cb9fa3'
    Expand-Source $fn $full
    $pcdir = Get-OpenblasPkgConfig
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        $env:PKG_CONFIG_PATH = "$pcdir;$($env:PKG_CONFIG_PATH)"
        Invoke-Logged 'dependency.log' {
            & $Py -m pip install -r requirements\build_requirements.txt
        }
        Invoke-Logged 'install.log' {
            & $Py -m pip install . --no-build-isolation `
                --config-settings='setup-args=-Dblas=scipy-openblas' `
                --config-settings='setup-args=-Dlapack=scipy-openblas'
        }
    } finally {
        Pop-Location
    }
    # Import self-check; also the v145 canary (see header): on MSVC 14.51 this
    # aborts with "OverflowError: cannot convert longdouble infinity to
    # integer".  v143 builds and imports cleanly.
    & $Py -c 'import numpy as np ; np.show_config()'
    Assert-LastExit 'numpy.show_config'
}

function Build-Scipy {
    if (Test-Skip 'scipy') { Write-Host 'skip: scipy'; return }
    # scipy additionally needs a Fortran compiler.  meson auto-detects gfortran
    # or LLVM flang from PATH; install one of them (see -PrintPrereq).  This is
    # the least-exercised step on Windows and the most likely to need a local
    # tweak; the devplan tracks it.
    $ver = '1.17.1'; $full = "scipy-$ver"; $fn = "$full.tar.gz"
    Get-Download $fn `
        "https://github.com/scipy/scipy/releases/download/v$ver/$fn" `
        '95d8e012d8cb8816c226aef832200b1d45109ed4464303e997c5b13122b297c0'
    Expand-Source $fn $full
    $pcdir = Get-OpenblasPkgConfig
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        $env:PKG_CONFIG_PATH = "$pcdir;$($env:PKG_CONFIG_PATH)"
        Invoke-Logged 'dependency.log' {
            & $Py -m pip install -r requirements\build.txt
        }
        Invoke-Logged 'install.log' {
            & $Py -m pip install . --no-build-isolation `
                --config-settings='setup-args=-Dblas=scipy-openblas' `
                --config-settings='setup-args=-Dlapack=scipy-openblas'
        }
    } finally {
        Pop-Location
    }
}

function Get-LibClang {
    # Qt's prebuilt Windows libclang for shiboken, laid out as LLVM_INSTALL_DIR
    # under SCDV_SRCDIR\libclang.  It is a vs2022_64 .7z; Windows bsdtar cannot
    # read 7z, so use any 7-Zip-family tool (7z/7za/7zr).
    $sevenzip = $null
    foreach ($cand in '7z', '7za', '7zr') {
        $g = Get-Command "$cand.exe" -ErrorAction SilentlyContinue
        if ($g) { $sevenzip = $g.Source; break }
    }
    if (-not $sevenzip) {
        # None installed: fetch a pinned standalone 7zr.exe (no installer), so
        # this does not depend on the 7-Zip MSI.
        $sevenzipVersion = '26.02'
        $sevenzipFn = "7zr-$sevenzipVersion.exe"
        Get-Download $sevenzipFn `
            "https://github.com/ip7z/7zip/releases/download/$sevenzipVersion/7zr.exe" `
            '56b8cc9f4971cef253644fafe54063ed7fdca551d4dee0f8c6baa81b855acd72'
        $sevenzip = Join-Path $ScdvDlDir $sevenzipFn
    }
    $fn = "libclang-release_$LibClangVersion-based-windows-vs2022_64.7z"
    $hash = Get-PinnedHash $fn 'LIBCLANG_SHA256'
    Get-Download $fn `
        "https://download.qt.io/development_releases/prebuilt/libclang/$fn" `
        $hash
    $dest = Join-Path $ScdvSrcDir 'libclang'
    if (Test-Path -LiteralPath $dest) {
        Remove-Item -LiteralPath $dest -Recurse -Force
    }
    Push-Location $ScdvSrcDir
    try {
        & $sevenzip x -y (Join-Path $ScdvDlDir $fn) | Out-Null
        Assert-LastExit "extract $fn"
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dest 'bin\libclang.dll'))) {
        throw "Get-LibClang: libclang.dll missing under $dest after extract"
    }
}

function Build-Qt {
    if (Test-Skip 'qt') { Write-Host 'skip: qt'; return }
    $ver = "$QtMajorVer.$QtSubVer"
    $full = "qt-$ver"
    $pkgfolder = "qt-everywhere-src-$ver"
    $fn = "$full.tar.xz"
    $url = "https://download.qt.io/official_releases/qt/$QtMajorVer/$ver/" +
        "single/qt-everywhere-src-$ver.tar.xz"
    $hash = Get-PinnedHash $fn 'QT_SHA256'
    Get-Download $fn $url $hash
    $src = Join-Path $ScdvSrcDir $full
    if (Test-Path -LiteralPath $src) {
        Write-Host "Qt source already at $src; skipping extract"
    } else {
        Push-Location $ScdvSrcDir
        try {
            Write-Host 'Extracting Qt source (large; takes a few minutes) ...'
            & $Tar xf (Join-Path $ScdvDlDir $fn)
            Assert-LastExit "extract $fn"
            Rename-Item -LiteralPath (Join-Path $ScdvSrcDir $pkgfolder) `
                -NewName $full
        } finally {
            Pop-Location
        }
    }

    # Build Qt in a SHORT directory: its autogen (moc) paths under the deep scdv
    # prefix blow past the 260-char MAX_PATH ("Cannot create ...").  Only the
    # build tree moves; the install still lands in the prefix.  Override with
    # SCDV_QT_BUILDDIR.
    $bld = $env:SCDV_QT_BUILDDIR
    if (-not $bld) { $bld = Join-Path $env:SystemDrive 'tmp\scdvqtb' }
    if ($env:SCDV_QTNOCLEAN -ne '1') {
        if (Test-Path -LiteralPath $bld) {
            Remove-Item -LiteralPath $bld -Recurse -Force
        }
    }
    New-Item -ItemType Directory -Force -Path $bld | Out-Null
    # Modules solvcon does not need, mirroring the devenv defaults shared by the
    # macOS and Ubuntu scripts.  On Windows the native "windows" QPA plugin is
    # used, so there is no xcb/X11 feature to force on (that is Linux-only).
    $mods = @(
        'qtquicktimeline', 'qtgraphs', 'qt5compat', 'qtactiveqt',
        'qtcharts', 'qtcoap', 'qtconnectivity', 'qtdatavis3d',
        'qtwebsockets', 'qthttpserver', 'qttools', 'qtdoc', 'qtlottie',
        'qtmqtt', 'qtnetworkauth', 'qtopcua', 'qtserialport', 'qtlocation',
        'qtpositioning', 'qtquick3dphysics', 'qtremoteobjects', 'qtscxml',
        'qtsensors', 'qtserialbus', 'qtspeech', 'qttranslations',
        'qtvirtualkeyboard', 'qtwayland', 'qtwebchannel', 'qtwebengine',
        'qtwebview', 'qtquickeffectmaker', 'qtgrpc'
    )
    # Only qtmultimedia's FFmpeg backend encodes the pilot's MP4, so it is
    # built where SCDV_FFMPEG_DIR names an FFmpeg.  The native Windows
    # backend is no use: deprecated since Qt 6.10, and it takes no frames
    # from a QVideoFrameInput.  A prebuilt LGPL shared package serves, since
    # those ship the MSVC import libraries the build links against.
    $ffmpeg = $env:SCDV_FFMPEG_DIR
    if ($ffmpeg -and
        -not (Test-Path (Join-Path $ffmpeg 'include\libavcodec\avcodec.h'))) {
        Write-Warning ("SCDV_FFMPEG_DIR='$ffmpeg' holds no " +
                       'include\libavcodec\avcodec.h; building without ' +
                       'qtmultimedia.')
        $ffmpeg = $null
    }
    if (-not $ffmpeg) { $mods += 'qtmultimedia' }
    Push-Location $bld
    try {
        $cfg = @(
            'cmake',
            "-DCMAKE_INSTALL_PREFIX=$ScdvUsrDir",
            '-DCMAKE_BUILD_TYPE=Release'
        )
        foreach ($m in $mods) { $cfg += "-DBUILD_$m=OFF" }
        if ($ffmpeg) {
            $cfg += @("-DFFMPEG_DIR=$ffmpeg", '-DQT_DEPLOY_FFMPEG=ON')
        }
        $cfg += @(
            '-DQT_ALLOW_SYMLINK_IN_PATHS=ON',
            "-DCMAKE_PREFIX_PATH=$ScdvUsrDir",
            '-G', 'Ninja', '-S', $src, '-B', $bld
        )
        if ($env:DVQT_NOCONFIG -ne '1') {
            Invoke-Logged 'configure.log' { & $cfg[0] $cfg[1..($cfg.Count-1)] }
        }
        if ($env:DVQT_NOBUILD -ne '1') {
            Invoke-Logged 'build.log' { cmake --build $bld --parallel }
        }
        if ($env:DVQT_NOINSTALL -ne '1') {
            Invoke-Logged 'install.log' { cmake --install $bld }
        }
    } finally {
        Pop-Location
    }

    # Prune the stray build intermediate Qt installs under qml/Qt/test/...
    # (objects-Release\...); its deep path overflows MAX_PATH when PySide6 later
    # copies the qml tree (WinError 3).
    $qmldir = Join-Path $ScdvUsrDir 'qml'
    if (Test-Path -LiteralPath $qmldir) {
        Get-ChildItem -LiteralPath $qmldir -Directory -Recurse `
            -Filter 'objects-Release' -ErrorAction SilentlyContinue |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    }
}

function Build-Pyside6 {
    if (Test-Skip 'pyside6') { Write-Host 'skip: pyside6'; return }
    $ver = $PysideVersion
    $full = "pyside-setup-everywhere-src-$ver"
    $fn = "$full.tar.xz"
    $url = "https://download.qt.io/official_releases/QtForPython/pyside6/" +
        "PySide6-$ver-src/$fn"
    $hash = Get-PinnedHash $fn 'PYSIDE_SHA256'
    Get-Download $fn $url $hash
    Expand-Source $fn $full
    # PySide6's windows_desktop.py unconditionally copies plugins/designer,
    # which does not exist here (qttools is disabled), aborting with WinError 3.
    # Guard the copy on the source existing.
    $wd = Join-Path $ScdvSrcDir `
        "$full\build_scripts\platforms\windows_desktop.py"
    $orig = [System.IO.File]::ReadAllText($wd)
    if (-not $orig.Contains('_designer_src')) {
        $needle = "        if not is_pypy:`n" +
            "            copydir(`"{install_dir}/plugins/designer`","
        $repl = "        _designer_src = " +
            "`"{install_dir}/plugins/designer`".format(**_vars)`n" +
            "        if not is_pypy and os.path.isdir(_designer_src):`n" +
            "            copydir(`"{install_dir}/plugins/designer`","
        if ($orig.Contains($needle)) {
            [System.IO.File]::WriteAllText($wd, $orig.Replace($needle, $repl))
        } else {
            Write-Warning ('pyside designer-copy guard: pattern not found in ' +
                'windows_desktop.py; skipping (packaging may fail)')
        }
    }
    # Bindings solvcon needs: Core/Gui/Widgets (pilot) plus Svg (matplotlib's Qt
    # backend).  Faster than the full build, and avoids QtDesigner/QtUiTools
    # (qttools is disabled).  Override with SCDV_PYSIDE_MODULES.
    $subset = 'Core,Gui,Widgets,Svg'
    # shiboken binds the subset, so a built Qt Multimedia is not enough:
    # without Multimedia named here there is no PySide6.QtMultimedia.
    # Network comes with it, the bindings being generated against it.  The
    # installed Qt decides, not SCDV_FFMPEG_DIR, which parts company with it
    # when this section runs alone.
    if (Test-Path (Join-Path $ScdvUsrDir 'include\QtMultimedia')) {
        $subset += ',Network,Multimedia'
    }
    $modules = Get-EnvOrDefault 'SCDV_PYSIDE_MODULES' $subset
    Push-Location (Join-Path $ScdvSrcDir $full)
    try {
        # --make-spec=ninja: pyside-setup's _get_make returns a str for "make"
        # (vs Path), crashing a later .is_absolute().  --qtpaths -> the new Qt;
        # LLVM_INSTALL_DIR (from the caller) -> shiboken's libclang.
        Invoke-Logged 'install.log' {
            & $Py setup.py install `
                "--qtpaths=$env:QTPATHS" `
                "--module-subset=$modules" `
                --verbose-build `
                --ignore-git `
                --no-qt-tools `
                --enable-numpy-support `
                "--parallel=$ScdvNp" `
                --make-spec=ninja
        }
    } finally {
        Pop-Location
    }
}

# --- Section driver ---------------------------------------------------------

$exitCode = 0
try {
    # Base section.
    if ($BuildAll -eq '1' -or $BuildBase -eq '1') {
        Invoke-Timed { Build-Zlib } 'zlib'
        Invoke-Timed { Build-Openssl } 'openssl'
        Invoke-Timed { Build-Sqlite } 'sqlite'
        # OpenBLAS is a self-contained prebuilt download and belongs here.  The
        # MKL provider installs from wheels and so runs in the PYTHON section,
        # once the scdv pip exists.
        if ($WinBlas -eq 'openblas') {
            Invoke-Timed { Build-Openblas } 'openblas'
        }
    } else {
        Write-Host 'Set SCDVBUILD_ALL or SCDVBUILD_BASE to build BASE section'
    }

    # Python section.
    if ($BuildAll -eq '1' -or $BuildPython -eq '1') {
        Write-Host 'Python build uses PGO-free Release; expect several minutes'
        Invoke-Timed { Build-Python } 'python'
        # MKL builds here rather than BASE: its wheels need the scdv pip.
        if ($WinBlas -eq 'mkl') {
            Invoke-Timed { Build-MklBlas } 'mkl'
        }
        & $Py -m pip install -U flake8 autopep8 black pytest jsonschema certifi
        Assert-LastExit 'pip install python-tools'
        & $Py -m pip install -U sphinx myst-parser pydata-sphinx-theme `
            breathe sphinxcontrib-bibtex sphinxcontrib-mermaid
        Assert-LastExit 'pip install doc-tools'
        Invoke-Timed { Build-Pybind11 } 'pybind11'
    } else {
        Write-Host ('Set SCDVBUILD_ALL or SCDVBUILD_PYTHON to build PYTHON ' +
            'section')
    }

    # Numpy section.
    if ($BuildAll -eq '1' -or $BuildNumpy -eq '1') {
        Invoke-Timed { Build-Cython } 'cython'
        Invoke-Timed { Build-Numpy } 'numpy'
        Invoke-Timed { Build-Scipy } 'scipy'

        $certPath = & $Py -m certifi
        Assert-LastExit 'certifi'
        $env:SSL_CERT_FILE = $certPath
        $env:REQUESTS_CA_BUNDLE = $certPath
        & $Py -m pip install -U matplotlib
        Assert-LastExit 'pip install matplotlib'
    } else {
        Write-Host ('Set SCDVBUILD_ALL or SCDVBUILD_NUMPY to build NUMPY ' +
            'section')
    }

    # Qt section.
    if ($BuildAll -eq '1' -or $BuildQt -eq '1') {
        # libclang for shiboken.  Fetch Qt's prebuilt Windows libclang into the
        # scdv tree unless LLVM_INSTALL_DIR already points at one.
        if ([string]::IsNullOrEmpty($env:LLVM_INSTALL_DIR)) {
            Invoke-Timed { Get-LibClang } 'libclang'
            $env:LLVM_INSTALL_DIR = Join-Path $ScdvSrcDir 'libclang'
        }
        if (-not (Test-Path -LiteralPath $env:LLVM_INSTALL_DIR)) {
            throw ("LLVM_INSTALL_DIR='$env:LLVM_INSTALL_DIR' does not exist; " +
                "clear it to fetch Qt's prebuilt libclang, or point it at a " +
                "libclang install.")
        }

        Invoke-Timed { Build-Qt } 'qt'

        $env:QTPATHS = Get-EnvOrDefault 'QTPATHS' `
            (Join-Path $ScdvUsrDir 'bin\qtpaths6.exe')
        if (-not (Test-Path -LiteralPath $env:QTPATHS)) {
            throw "qtpaths6 not found at $env:QTPATHS; check the Qt build"
        }
        $env:PYSIDE_BUILD = '1'

        Invoke-Timed { Build-Pyside6 } 'pyside6'
    } else {
        Write-Host 'Set SCDVBUILD_ALL or SCDVBUILD_QT to build QT section'
    }
} catch {
    $exitCode = 1
    Write-Error $_
} finally {
    Write-Timings $exitCode
}

exit $exitCode

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
