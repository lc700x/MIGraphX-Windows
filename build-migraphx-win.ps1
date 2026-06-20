# ============================================================================
# Build MIGraphX from source on Windows (PowerShell)
# GPU + CK (Composable Kernel) + MLIR + ONNX + Python bindings
#
# Prerequisites:
#   - Windows 10/11 x64
#   - Python 3.12 with venv
#   - Git for Windows
#   - Visual Studio 2022 Build Tools (for Windows SDK rc.exe)
#   - Internet connection (clones repos, downloads deps)
#
# Usage:
#   .\build-migraphx-win.ps1 [-GpuTargets <target[]>]
#
# Examples:
#   .\build-migraphx-win.ps1                                  # default: gfx1103
#   .\build-migraphx-win.ps1 -GpuTargets gfx1103              # RDNA3 (Phoenix APU)
#   .\build-migraphx-win.ps1 -GpuTargets gfx1100              # RDNA3 (RX 7900 XTX)
#   .\build-migraphx-win.ps1 -GpuTargets gfx1030              # RDNA2 (RX 6800)
#   .\build-migraphx-win.ps1 -GpuTargets gfx942               # MI300
#   .\build-migraphx-win.ps1 -GpuTargets gfx1100,gfx1103      # multi-GPU (comma separated)
#   .\build-migraphx-win.ps1 -GpuTargets gfx1100,gfx1103,gfx942  # 3 targets
#   .\build-migraphx-win.ps1 -GpuTargets all                  # all common consumer + DC arches
#
# Common GPU architectures:
#   RDNA3:   gfx1100 (7900XTX) gfx1101 (7800XT) gfx1102 (7700XT) gfx1103 (780M/760M APU)
#   RDNA2:   gfx1030 (6800/6900) gfx1031 (6600) gfx1032 (6500)
#   CDNA3:   gfx942 (MI300X/MI300A)
#   CDNA2:   gfx90a (MI250X/MI210)
#   CDNA1:   gfx908 (MI100)
#
# WARNING: Each GPU target adds ~20-40 min of HIP kernel compilation time.
#
# The script is idempotent - safe to re-run after fixing errors.
# ============================================================================

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string[]]$GpuTargets = @("gfx1103")
)

$ErrorActionPreference = 'Stop'
# Native exes (git, cmake, clang) write progress to stderr. Under PS 7.4+ that
# stderr is promoted to a terminating error; disable so we judge by exit code.
$PSNativeCommandUseErrorActionPreference = $false

# Run git so its stderr progress ("Cloning into...") can never raise a
# terminating NativeCommandError (PS 5.1 ignores the pref above). Merge
# stderr->stdout under a temporarily-relaxed ErrorActionPreference; caller
# checks $LASTEXITCODE.
function Invoke-Git {
    param([string[]]$GitArgs)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & git @GitArgs 2>&1 | Write-Host } finally { $ErrorActionPreference = $old }
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

$AllTargets = @("gfx1201","gfx1200","gfx1153","gfx1152","gfx1151","gfx1150","gfx1103","gfx1102","gfx1101","gfx1100","gfx1036","gfx1035","gfx1034","gfx1033","gfx1032","gfx1031","gfx1030","gfx1012","gfx1011","gfx1010","gfx906","gfx90c","gfx900")

if ($GpuTargets.Count -eq 1 -and $GpuTargets[0] -eq "all") {
    $GpuTargetsStr = $AllTargets -join ";"
    $TargetsArray = $AllTargets
} else {
    $TargetsArray = $GpuTargets
    $GpuTargetsStr = $GpuTargets -join ";"
}

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $RootDir ".env"
$MigraphxSrc = Join-Path $RootDir "AMDMIGraphX"
$CkSrc = Join-Path $RootDir "composable_kernel"
$RocmlirSrc = Join-Path $RootDir "rocMLIR"
$CgetPrefix = Join-Path $MigraphxSrc "cget"
$InstallDir = Join-Path $MigraphxSrc "install"

if ($env:BUILD_JOBS) {
    $Jobs = $env:BUILD_JOBS
} else {
    $Jobs = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
}

# Pinned commits (from MIGraphX requirements.txt)
$CkCommit = "ad0db05b040bacda751c65c705261b8a0a7ed25d"
$RocmlirCommit = "364015202c7271708f6375f34eaf20c2a9c199a3"

# Auto-detect latest Windows SDK version
$WinSdkBase = "C:\Program Files (x86)\Windows Kits\10\bin"
if (Test-Path $WinSdkBase) {
    $LatestSdk = Get-ChildItem -Path $WinSdkBase -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending |
        Select-Object -First 1
    if ($LatestSdk) {
        $WinSdkBin = Join-Path $LatestSdk.FullName "x64"
    } else {
        Write-Error "No Windows SDK version directories found under $WinSdkBase"
        exit 1
    }
} else {
    Write-Error "Windows SDK base directory not found at $WinSdkBase"
    exit 1
}

Write-Host "============================================"
Write-Host "MIGraphX Windows Build Script (PowerShell)"
Write-Host "GPU targets: $GpuTargetsStr"
Write-Host "Root dir:    $RootDir"
Write-Host "Jobs:        $Jobs"
Write-Host "Win SDK:     $WinSdkBin"
Write-Host "============================================"

# ---------------------------------------------------------------------------
# Step 0: Validate Windows SDK (rc.exe needed for protobuf)
# ---------------------------------------------------------------------------
$RcExe = Join-Path $WinSdkBin "rc.exe"
if (-not (Test-Path $RcExe)) {
    Write-Error "rc.exe not found at $RcExe`nInstall Visual Studio Build Tools with Windows SDK component."
    exit 1
}

# ---------------------------------------------------------------------------
# Step 1: Create venv and install ROCm SDK + build tools
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Step 1: Python venv + ROCm SDK"

if (-not (Test-Path $VenvDir)) {
    & python -m venv $VenvDir
}

# Activate venv by prepending Scripts to PATH
$VenvScripts = Join-Path $VenvDir "Scripts"
$env:PATH = "$VenvScripts;$env:PATH"

& pip install --quiet cmake ninja

# Install ROCm SDK via pip (nightly)
$RocmPipArgs = @("install", "--quiet", "--pre", "rocm_sdk_core", "rocm_sdk_devel", "rocm_sdk_libraries")
foreach ($target in $TargetsArray) {
    $RocmPipArgs += "rocm_sdk_device_$target"
}
$RocmPipArgs += @("--index-url", "https://rocm.nightlies.amd.com/whl-multi-arch/",
                   "--extra-index-url", "https://pypi.org/simple/")
try {
    & pip @RocmPipArgs 2>$null
} catch {
    Write-Warning "Some ROCm pip packages may have failed. Check manually."
}

# Set tool paths
$Cmake = Join-Path $VenvScripts "cmake.exe"
$Ninja = Join-Path $VenvScripts "ninja.exe"
$RocmDevel = Join-Path $VenvDir "Lib\site-packages\_rocm_sdk_devel"
$RocmCore = Join-Path $VenvDir "Lib\site-packages\_rocm_sdk_core"
$RocmLibs = Join-Path $VenvDir "Lib\site-packages\_rocm_sdk_libraries"
$ClangCl = Join-Path $RocmDevel "lib\llvm\bin\clang-cl.exe"
$Clangxx = Join-Path $RocmDevel "lib\llvm\bin\clang++.exe"

if (-not (Test-Path $ClangCl)) {
    Write-Error "clang-cl.exe not found at $ClangCl`nROCm SDK devel package may not be installed correctly."
    exit 1
}

$env:PATH = "$VenvScripts;$(Join-Path $RocmDevel 'lib\llvm\bin');$(Join-Path $RocmDevel 'bin');$(Join-Path $RocmCore 'bin');$WinSdkBin;$env:PATH"

$cmakeVersion = & $Cmake --version | Select-Object -First 1
$ninjaVersion = & $Ninja --version
$clangVersion = & $ClangCl --version | Select-Object -First 1
Write-Host "  CMake:    $cmakeVersion"
Write-Host "  Ninja:    $ninjaVersion"
Write-Host "  Clang:    $clangVersion"

# ---------------------------------------------------------------------------
# Step 2: Clone source repos
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Step 2: Clone source repos"

Invoke-Git @('config','--global','core.longpaths','true')

if (-not (Test-Path $MigraphxSrc)) {
    Write-Host "  Cloning AMDMIGraphX..."
    Invoke-Git @('clone','https://github.com/ROCm/AMDMIGraphX.git',$MigraphxSrc)
} else {
    Write-Host "  AMDMIGraphX already exists, skipping clone."
}

if (-not (Test-Path $CkSrc)) {
    Write-Host "  Cloning composable_kernel at $CkCommit..."
    Invoke-Git @('clone','https://github.com/ROCm/composable_kernel.git',$CkSrc)
    Invoke-Git @('-C',$CkSrc,'checkout',$CkCommit)
} else {
    Write-Host "  composable_kernel already exists, skipping clone."
}

if (-not (Test-Path $RocmlirSrc)) {
    Write-Host "  Cloning rocMLIR at $RocmlirCommit..."
    Invoke-Git @('clone','https://github.com/ROCm/rocMLIR.git',$RocmlirSrc)
    Invoke-Git @('-C',$RocmlirSrc,'checkout',$RocmlirCommit)
    Invoke-Git @('-C',$RocmlirSrc,'submodule','update','--init','--recursive')
} else {
    Write-Host "  rocMLIR already exists, skipping clone."
}

# ---------------------------------------------------------------------------
# Step 3: Apply Windows patches
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Step 3: Apply Windows patches"

# --- Patch 3a: CK codegen CMakeLists.txt ---
# clang-cl needs /std:c++20 not -std=c++20, and compile options must come
# before add_embed_library so the embed lib gets C++20 too.
$CkCMake = Join-Path $CkSrc "codegen\CMakeLists.txt"
if (Test-Path $CkCMake) {
    $ckCmakeContent = Get-Content $CkCMake -Raw
    if ($ckCmakeContent -notmatch '/std:c\+\+20') {
        Write-Host "  Patching CK codegen CMakeLists.txt (C++20 for clang-cl)..."
        $patchBlock = @"

if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
    add_compile_options(/std:c++20)
else()
    add_compile_options(-std=c++20)
endif()
"@
        $ckCmakeContent = $ckCmakeContent -replace '(include\(Embed\))', "`$1$patchBlock"
        Set-Content -Path $CkCMake -Value $ckCmakeContent -NoNewline
    } else {
        Write-Host "  CK codegen CMakeLists.txt already patched."
    }
}

# --- Patch 3b: CK utils.hpp missing #include <string> ---
$CkUtils = Join-Path $CkSrc "codegen\include\ck\host\utils.hpp"
if (Test-Path $CkUtils) {
    $ckUtilsContent = Get-Content $CkUtils -Raw
    if ($ckUtilsContent -notmatch '#include <string>') {
        Write-Host "  Patching CK utils.hpp (add #include <string>)..."
        $ckUtilsContent = $ckUtilsContent -replace '(#include <cstdint>)', "`$1`n#include <string>"
        Set-Content -Path $CkUtils -Value $ckUtilsContent -NoNewline
    } else {
        Write-Host "  CK utils.hpp already patched."
    }
}

# --- Patch 3c: MIGraphX ck.hpp - remove #ifndef _WIN32 guard on env vars ---
$CkHpp = Join-Path $MigraphxSrc "src\targets\gpu\include\migraphx\gpu\ck.hpp"
if (Test-Path $CkHpp) {
    $ckHppContent = Get-Content $CkHpp -Raw
    if ($ckHppContent -match '#ifndef _WIN32') {
        Write-Host "  Patching MIGraphX ck.hpp (remove _WIN32 guard on env vars)..."
        # Remove the #ifndef _WIN32 line
        $ckHppContent = $ckHppContent -replace '#ifndef _WIN32\r?\n', ''
        # Remove the matching #endif that follows the MIGRAPHX_DECLARE_ENV_VAR block
        $ckHppContent = $ckHppContent -replace '(MIGRAPHX_DECLARE_ENV_VAR\(MIGRAPHX_TUNE_CK\);)\s*#endif', '$1'
        Set-Content -Path $CkHpp -Value $ckHppContent -NoNewline
    } else {
        Write-Host "  MIGraphX ck.hpp already patched."
    }
}

# --- Patch 3d: MIGraphX PythonModules.cmake - skip non-numeric py.exe entries ---
# `py -0p` may list Astral/uv standalone pythons as "-V:Astral/CPython3.13.13 <path>"
# which the version regex can't parse -> list index out of range. Only process
# entries shaped like "-V:3.12 <path>".
$PyMods = Join-Path $MigraphxSrc "cmake\PythonModules.cmake"
if (Test-Path $PyMods) {
    $pyModsLines = Get-Content $PyMods
    $oldLine = '        if(NOT _found_python MATCHES "^\\*[ \t]*")'
    $newLine = '        if(NOT _found_python MATCHES "^\\*[ \t]*" AND _found_python MATCHES "^-V:[0-9]")'
    if (($pyModsLines -contains $oldLine) -and -not ($pyModsLines -contains $newLine)) {
        Write-Host "  Patching PythonModules.cmake (skip non-numeric py entries)..."
        $pyModsLines = $pyModsLines | ForEach-Object { if ($_ -eq $oldLine) { $newLine } else { $_ } }
        Set-Content -Path $PyMods -Value $pyModsLines
    } else {
        Write-Host "  PythonModules.cmake already patched (or pattern changed)."
    }
}

Write-Host "  Patches applied."

# ---------------------------------------------------------------------------
# Step 4: Build C++ dependencies
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Step 4: Build C++ dependencies"

if (-not (Test-Path $CgetPrefix)) {
    New-Item -ItemType Directory -Path $CgetPrefix -Force | Out-Null
}

$CmakeGlobalArgs = @(
    "-G", "Ninja",
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$ClangCl",
    "-DCMAKE_CXX_COMPILER=$ClangCl",
    "-DCMAKE_INSTALL_PREFIX=$CgetPrefix",
    "-DCMAKE_PREFIX_PATH=$CgetPrefix;$RocmDevel;$RocmCore;$RocmLibs",
    "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
)

function Build-Dep {
    param(
        [string]$Name,
        [string]$SrcDir,
        [string[]]$ExtraArgs = @(),
        [string]$BuildTarget = ""
    )
    $BuildDir = Join-Path $SrcDir "build"
    $DoneMarker = Join-Path $BuildDir ".build_done"

    if (Test-Path $DoneMarker) {
        Write-Host "  [$Name] Already built, skipping."
        return
    }

    Write-Host "  [$Name] Configuring..."
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    }
    $configArgs = $CmakeGlobalArgs + $ExtraArgs + @("-B", $BuildDir, "-S", $SrcDir)
    & $Cmake @configArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Name] CMake configure failed" }

    Write-Host "  [$Name] Building..."
    # Build a specific target when given (e.g. just the library, skipping test
    # exes that don't compile on Windows), otherwise the default 'all'.
    if ($BuildTarget) {
        & $Cmake --build $BuildDir --parallel $Jobs --target $BuildTarget
    } else {
        & $Cmake --build $BuildDir --parallel $Jobs
    }
    if ($LASTEXITCODE -ne 0) { throw "[$Name] CMake build failed" }

    Write-Host "  [$Name] Installing..."
    & $Cmake --install $BuildDir
    if ($LASTEXITCODE -ne 0) { throw "[$Name] CMake install failed" }

    New-Item -ItemType File -Path $DoneMarker -Force | Out-Null
    Write-Host "  [$Name] Done."
}

function Clone-AndBuildDep {
    param(
        [string]$Name,
        [string]$Repo,
        [string]$Ref,
        [string[]]$ExtraArgs = @()
    )
    $SrcDir = Join-Path $RootDir "_deps\$Name"

    # Already built? skip clone entirely (avoids re-clone + file-lock issues)
    if (Test-Path (Join-Path $SrcDir "build\.build_done")) {
        Write-Host "  [$Name] Already built, skipping."
        return
    }

    if (-not (Test-Path (Join-Path $SrcDir ".git"))) {
        Write-Host "  [$Name] Cloning $Repo @ $Ref..."
        $DepsDir = Join-Path $RootDir "_deps"
        if (-not (Test-Path $DepsDir)) {
            New-Item -ItemType Directory -Path $DepsDir -Force | Out-Null
        }
        # Stale partial dir (no .git) left by an interrupted run? wipe it first.
        if (Test-Path $SrcDir) { Remove-Item -Recurse -Force $SrcDir -ErrorAction SilentlyContinue }
        # $Ref may be a tag/branch OR a full commit SHA. Try shallow tag/branch
        # clone first; if that fails (SHA), do a full clone then checkout.
        $isSha = $Ref -match '^[0-9a-fA-F]{40}$'
        $ok = $false
        if (-not $isSha) {
            Invoke-Git @('clone','--depth','1','--branch',$Ref,"https://github.com/$Repo.git",$SrcDir)
            if ($LASTEXITCODE -eq 0) { $ok = $true }
        }
        if (-not $ok) {
            if (Test-Path $SrcDir) { Remove-Item -Recurse -Force $SrcDir -ErrorAction SilentlyContinue }
            Invoke-Git @('clone',"https://github.com/$Repo.git",$SrcDir)
            if ($LASTEXITCODE -ne 0) { throw "[$Name] git clone failed" }
            Invoke-Git @('-C',$SrcDir,'checkout',$Ref)
        }
    }

    Build-Dep -Name $Name -SrcDir $SrcDir -ExtraArgs $ExtraArgs
}

function Download-AndBuildDep {
    param(
        [string]$Name,
        [string]$Url,
        [string[]]$ExtraArgs = @()
    )
    $SrcDir = Join-Path $RootDir "_deps\$Name"

    # Re-extract if missing OR a prior run left an empty/partial dir.
    if (-not (Test-Path (Join-Path $SrcDir "CMakeLists.txt"))) {
        Write-Host "  [$Name] Downloading $Url..."
        $DepsDir = Join-Path $RootDir "_deps"
        if (-not (Test-Path $DepsDir)) {
            New-Item -ItemType Directory -Path $DepsDir -Force | Out-Null
        }
        if (Test-Path $SrcDir) { Remove-Item -Recurse -Force $SrcDir -ErrorAction SilentlyContinue }
        $Archive = Join-Path $DepsDir "$Name.tar.gz"
        Invoke-WebRequest -Uri $Url -OutFile $Archive -UseBasicParsing
        New-Item -ItemType Directory -Path $SrcDir -Force | Out-Null
        # --force-local: GNU tar treats the 'C:\...' archive path as a remote
        # host (drive-letter colon) without it. Use Windows' bsdtar via cmd to
        # avoid Git-Bash tar path translation.
        & tar --force-local -xzf $Archive -C $SrcDir --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "[$Name] tar extract failed" }
        Remove-Item -Force $Archive -ErrorAction SilentlyContinue
    }

    Build-Dep -Name $Name -SrcDir $SrcDir -ExtraArgs $ExtraArgs
}

# 4a: rocm-cmake (build tools, needed by CK and MIGraphX)
Write-Host ""
Write-Host "  --- rocm-cmake ---"
Clone-AndBuildDep -Name "rocm-cmake" -Repo "ROCm/rocm-cmake" -Ref "b3959201846b9d9250a6f2a386ad43be04d1d051"

# 4b: ROCm recipes (provides sqlite3 cmake recipe)
Write-Host ""
Write-Host "  --- rocm-recipes ---"
$RocmRecipesDir = Join-Path $RootDir "_deps\rocm-recipes"
if (-not (Test-Path $RocmRecipesDir)) {
    Write-Host "  [rocm-recipes] Cloning..."
    Invoke-Git @('clone','--depth','1','https://github.com/ROCm/rocm-recipes.git',$RocmRecipesDir)
}
try {
    & pip install --quiet $RocmRecipesDir 2>$null
} catch {
    # rocm-recipes pip install may fail non-fatally
}

# 4c: abseil
Write-Host ""
Write-Host "  --- abseil ---"
# Pin ABSL_OPTION_USE_STD_STRING_VIEW to a DETERMINISTIC value before building.
# Default is 2 (auto-detect), which on this clang-cl produced a split: the compiled
# libs used std::string_view but the installed options.h header stayed at 0
# (absl's own class). protobuf then compiled against absl::string_view while the
# libs exported std::string_view -> migraphx_onnx.dll lld-link "undefined symbol
# absl::...string_view...". Force 1 (always std::string_view) so header+libs+protobuf
# all agree. Must clone first so the source file exists to patch.
$AbseilSrc = Join-Path $RootDir "_deps\abseil"
if (-not (Test-Path (Join-Path $AbseilSrc ".git"))) {
    if (Test-Path $AbseilSrc) { Remove-Item -Recurse -Force $AbseilSrc -ErrorAction SilentlyContinue }
    Invoke-Git @('clone','--depth','1','--branch','20250512.0','https://github.com/abseil/abseil-cpp.git',$AbseilSrc)
}
$AbseilOptions = Join-Path $AbseilSrc "absl\base\options.h"
if (Test-Path $AbseilOptions) {
    $opt = Get-Content $AbseilOptions -Raw
    if ($opt -notmatch '#define ABSL_OPTION_USE_STD_STRING_VIEW 1') {
        Write-Host "  Pinning ABSL_OPTION_USE_STD_STRING_VIEW=1..."
        $opt = $opt -replace '#define ABSL_OPTION_USE_STD_STRING_VIEW \d', '#define ABSL_OPTION_USE_STD_STRING_VIEW 1'
        [System.IO.File]::WriteAllText($AbseilOptions, $opt, (New-Object System.Text.UTF8Encoding $false))
    }
}
Clone-AndBuildDep -Name "abseil" -Repo "abseil/abseil-cpp" -Ref "20250512.0" `
    -ExtraArgs @("-DABSL_ENABLE_INSTALL=ON", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DABSL_MSVC_STATIC_RUNTIME=Off")

# 4d: protobuf (needs rc.exe in PATH)
Write-Host ""
Write-Host "  --- protobuf ---"
# Build protobuf. Key: use -Dprotobuf_ABSL_PROVIDER=package so it finds
# abseil from the cget prefix. Disable upb (not needed by MIGraphX).
# If protoc.exe linking fails with lld-link, the script falls back to
# building without protoc and downloading a pre-built protoc binary.
$ProtobufExtraArgs = @(
    "-DCMAKE_POSITION_INDEPENDENT_CODE=On",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    "-Dprotobuf_BUILD_TESTS=Off",
    "-Dprotobuf_BUILD_SHARED_LIBS=Off",
    "-Dprotobuf_MSVC_STATIC_RUNTIME=Off",
    "-Dprotobuf_BUILD_LIBPROTOC=On",
    "-Dprotobuf_BUILD_PROTOC_BINARIES=On",
    "-Dprotobuf_BUILD_PROTOBUF_BINARIES=On",
    "-Dprotobuf_BUILD_LIBUPB=Off",
    "-Dprotobuf_ABSL_PROVIDER=package"
)
$ProtobufSrc = Join-Path $RootDir "_deps\protobuf"
$ProtobufDone = Join-Path $ProtobufSrc "build\.build_done"
if (-not (Test-Path $ProtobufDone)) {
    try {
        Clone-AndBuildDep -Name "protobuf" -Repo "google/protobuf" -Ref "v30.0" `
            -ExtraArgs $ProtobufExtraArgs
    } catch {
        Write-Warning "protobuf build failed (likely protoc.exe linker error with lld-link + abseil)."
        Write-Host "  Retrying without protoc binaries + downloading pre-built protoc..."
        # Remove failed build
        $ProtobufBuild = Join-Path $ProtobufSrc "build"
        if (Test-Path $ProtobufBuild) { Remove-Item -Recurse -Force $ProtobufBuild }
        # Build libprotobuf only (no executables)
        Clone-AndBuildDep -Name "protobuf" -Repo "google/protobuf" -Ref "v30.0" `
            -ExtraArgs @(
                "-DCMAKE_POSITION_INDEPENDENT_CODE=On",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                "-Dprotobuf_BUILD_TESTS=Off",
                "-Dprotobuf_BUILD_SHARED_LIBS=Off",
                "-Dprotobuf_MSVC_STATIC_RUNTIME=Off",
                "-Dprotobuf_BUILD_LIBPROTOC=On",
                "-Dprotobuf_BUILD_PROTOC_BINARIES=Off",
                "-Dprotobuf_BUILD_PROTOBUF_BINARIES=On",
                "-Dprotobuf_BUILD_LIBUPB=Off",
                "-Dprotobuf_ABSL_PROVIDER=package"
            )
        # Download pre-built protoc for Windows
        $ProtocZip = Join-Path $RootDir "_deps\protoc-win64.zip"
        $ProtocUrl = "https://github.com/protocolbuffers/protobuf/releases/download/v30.0/protoc-30.0-win64.zip"
        Write-Host "  Downloading pre-built protoc from $ProtocUrl..."
        Invoke-WebRequest -Uri $ProtocUrl -OutFile $ProtocZip -UseBasicParsing
        Expand-Archive -Path $ProtocZip -DestinationPath (Join-Path $RootDir "_deps\protoc-bin") -Force
        $CgetBin = Join-Path $CgetPrefix "bin"
        if (-not (Test-Path $CgetBin)) { New-Item -ItemType Directory -Path $CgetBin -Force | Out-Null }
        Copy-Item -Path (Join-Path $RootDir "_deps\protoc-bin\bin\protoc.exe") -Destination (Join-Path $CgetBin "protoc.exe") -Force
        # Copy any protoc runtime DLLs (abseil etc.) shipped in the prebuilt zip
        Get-ChildItem -Path (Join-Path $RootDir "_deps\protoc-bin\bin") -Filter "*.dll" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName -Destination $CgetBin -Force }
        Remove-Item -Force $ProtocZip -ErrorAction SilentlyContinue
        # The libprotobuf-only install still declares a protobuf::protoc cmake
        # target but with no/blank IMPORTED_LOCATION -> MIGraphX's protobuf_generate
        # emits a malformed "protoc --cpp_out :path". Rewrite the target's location
        # in the installed protobuf cmake configs to point at the prebuilt protoc.
        $protocExe = (Join-Path $CgetBin "protoc.exe") -replace '\\', '/'
        $protobufCmakeDir = Join-Path $CgetPrefix "lib\cmake\protobuf"
        if (Test-Path $protobufCmakeDir) {
            Get-ChildItem -Path $protobufCmakeDir -Filter "*.cmake" | ForEach-Object {
                $c = Get-Content $_.FullName -Raw
                if ($c -match 'protobuf::protoc') {
                    $c = $c -replace '(set_target_properties\(protobuf::protoc PROPERTIES[^)]*IMPORTED_LOCATION(?:_[A-Z]+)?\s+)"[^"]*"', "`$1`"$protocExe`""
                    Set-Content -Path $_.FullName -Value $c -NoNewline
                }
            }
        }
        # Belt-and-suspenders: a small cmake file that forces the imported target
        $protocFix = Join-Path $protobufCmakeDir "protoc-prebuilt-fix.cmake"
        if (Test-Path $protobufCmakeDir) {
            $fixContent = "if(NOT TARGET protobuf::protoc)`n  add_executable(protobuf::protoc IMPORTED)`nendif()`nset_property(TARGET protobuf::protoc PROPERTY IMPORTED_LOCATION `"$protocExe`")`n"
            [System.IO.File]::WriteAllText($protocFix, $fixContent, (New-Object System.Text.UTF8Encoding $false))
            # The fix file is inert unless something includes it. protobuf-config.cmake
            # is what find_package(Protobuf CONFIG) loads, so append an include() at its
            # END (after protobuf-targets.cmake declares the empty protoc target).
            $protobufConfig = Join-Path $protobufCmakeDir "protobuf-config.cmake"
            if (Test-Path $protobufConfig) {
                $cfg = Get-Content $protobufConfig -Raw
                if ($cfg -notmatch 'protoc-prebuilt-fix') {
                    $cfg += "`n# Force prebuilt protoc IMPORTED_LOCATION (libprotobuf-only install left it blank)`ninclude(`"`${CMAKE_CURRENT_LIST_DIR}/protoc-prebuilt-fix.cmake`")`n"
                    [System.IO.File]::WriteAllText($protobufConfig, $cfg, (New-Object System.Text.UTF8Encoding $false))
                }
            }
        }
        Write-Host "  Pre-built protoc installed + cmake target patched at $CgetPrefix\bin\"
    }
} else {
    Write-Host "  [protobuf] Already built, skipping."
}

# 4e: nlohmann/json
Write-Host ""
Write-Host "  --- nlohmann/json ---"
Clone-AndBuildDep -Name "json" -Repo "nlohmann/json" -Ref "v3.8.0" `
    -ExtraArgs @("-DJSON_BuildTests=Off")

# 4f: pybind11
Write-Host ""
Write-Host "  --- pybind11 ---"
Clone-AndBuildDep -Name "pybind11" -Repo "pybind/pybind11" -Ref "3e9dfa2866941655c56877882565e7577de6fc7b" `
    -ExtraArgs @("-DPYBIND11_TEST=Off")

# 4g: msgpack-c (install to share/cmake so MIGraphX finds it)
Write-Host ""
Write-Host "  --- msgpack-c ---"
Clone-AndBuildDep -Name "msgpack-c" -Repo "msgpack/msgpack-c" -Ref "cpp-3.3.0" `
    -ExtraArgs @("-DMSGPACK_BUILD_TESTS=Off", "-DMSGPACK_BUILD_EXAMPLES=Off", "-DMSGPACK_CXX17=ON")

# 4h: sqlite3 (download amalgamation + compile directly, copy files to cget)
Write-Host ""
Write-Host "  --- sqlite3 ---"
$SqliteSrc = Join-Path $RootDir "_deps\sqlite3"
$SqliteLib = Join-Path $CgetPrefix "lib\sqlite3.lib"
$SqliteHdr = Join-Path $CgetPrefix "include\sqlite3.h"
# Only skip if BOTH the lib and the header are actually present in cget
if ((Test-Path $SqliteLib) -and (Test-Path $SqliteHdr)) {
    Write-Host "  [sqlite3] Already installed, skipping."
} else {
    if (-not (Test-Path (Join-Path $SqliteSrc "sqlite3.c"))) {
        Write-Host "  [sqlite3] Downloading amalgamation..."
        if (Test-Path $SqliteSrc) { Remove-Item -Recurse -Force $SqliteSrc }
        $SqliteZip = Join-Path $RootDir "_deps\sqlite3.zip"
        $SqliteUrl = "https://www.sqlite.org/2025/sqlite-amalgamation-3500400.zip"
        New-Item -ItemType Directory -Path (Join-Path $RootDir "_deps") -Force | Out-Null
        Invoke-WebRequest -Uri $SqliteUrl -OutFile $SqliteZip -UseBasicParsing
        Expand-Archive -Path $SqliteZip -DestinationPath (Join-Path $RootDir "_deps\sqlite3-tmp") -Force
        $SqliteInner = Get-ChildItem -Path (Join-Path $RootDir "_deps\sqlite3-tmp") -Directory | Select-Object -First 1
        Move-Item -Path $SqliteInner.FullName -Destination $SqliteSrc
        Remove-Item -Recurse -Force (Join-Path $RootDir "_deps\sqlite3-tmp") -ErrorAction SilentlyContinue
        Remove-Item -Force $SqliteZip -ErrorAction SilentlyContinue
    }
    Write-Host "  [sqlite3] Compiling sqlite3.c -> sqlite3.lib..."
    New-Item -ItemType Directory -Path (Join-Path $CgetPrefix "lib") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $CgetPrefix "include") -Force | Out-Null
    Push-Location $SqliteSrc
    try {
        # Compile object with clang-cl, then archive into a static lib
        & $ClangCl /nologo /c /O2 /MD /DSQLITE_ENABLE_COLUMN_METADATA "sqlite3.c" /Fosqlite3.obj
        if ($LASTEXITCODE -ne 0) { throw "sqlite3 compile failed" }
        $LlvmLib = Join-Path $RocmDevel "lib\llvm\bin\llvm-lib.exe"
        if (-not (Test-Path $LlvmLib)) { $LlvmLib = "llvm-lib.exe" }
        & $LlvmLib /nologo /out:sqlite3.lib sqlite3.obj
        if ($LASTEXITCODE -ne 0) { throw "sqlite3 archive failed" }
        Copy-Item -Path "sqlite3.lib" -Destination $SqliteLib -Force
        Copy-Item -Path "sqlite3.h" -Destination $SqliteHdr -Force
        if (Test-Path "sqlite3ext.h") {
            Copy-Item -Path "sqlite3ext.h" -Destination (Join-Path $CgetPrefix "include\sqlite3ext.h") -Force
        }
    } finally {
        Pop-Location
    }
    if (-not ((Test-Path $SqliteLib) -and (Test-Path $SqliteHdr))) {
        throw "sqlite3 install verification failed: missing lib or header in $CgetPrefix"
    }
    Write-Host "  [sqlite3] Installed sqlite3.lib + sqlite3.h to $CgetPrefix"
}

# 4i: eigen
Write-Host ""
Write-Host "  --- eigen ---"
Download-AndBuildDep -Name "eigen" `
    -Url "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz" `
    -ExtraArgs @("-DBUILD_TESTING=Off", "-DEIGEN_BUILD_DOC=Off", "-DEIGEN_BUILD_LAPACK=Off")

# 4j: Composable Kernel (codegen only)
Write-Host ""
Write-Host "  --- Composable Kernel (codegen) ---"
Build-Dep -Name "ck-codegen" -SrcDir (Join-Path $CkSrc "codegen") `
    -ExtraArgs @("-DCMAKE_POSITION_INDEPENDENT_CODE=On", "-DBUILD_TESTING=Off", "-DEMBED_USE=CArrays") `
    -BuildTarget "ck_host"

# 4k: rocMLIR (full LLVM/MLIR build - SLOW, 1-3 hours)
Write-Host ""
Write-Host "  --- rocMLIR (this takes 1-3 hours) ---"
Build-Dep -Name "rocMLIR" -SrcDir $RocmlirSrc `
    -ExtraArgs @("-DBUILD_FAT_LIBROCKCOMPILER=On", "-DLLVM_INCLUDE_TESTS=Off", "-DLLVM_DISABLE_ASSEMBLY_FILES=ON", "-DBUILD_SHARED_LIBS=OFF")

Write-Host ""
Write-Host ">>> All dependencies built."

# ---------------------------------------------------------------------------
# Step 5: Build MIGraphX
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Step 5: Build MIGraphX"

$MigraphxBuild = Join-Path $MigraphxSrc "build"
if (-not (Test-Path $MigraphxBuild)) {
    New-Item -ItemType Directory -Path $MigraphxBuild -Force | Out-Null
}

$ConfigDone = Join-Path $MigraphxBuild ".configure_done"
if (-not (Test-Path $ConfigDone)) {
    # A stale CMakeCache.txt from a failed/previous configure (e.g. different
    # GPU targets, or cached SQLite3_*-NOTFOUND) poisons this run. Start clean.
    $StaleCache = Join-Path $MigraphxBuild "CMakeCache.txt"
    if (Test-Path $StaleCache) {
        Write-Host "  Removing stale CMakeCache.txt for clean configure..."
        Remove-Item -Force $StaleCache -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force (Join-Path $MigraphxBuild "CMakeFiles") -ErrorAction SilentlyContinue
    }
    # protobuf was installed as libprotobuf-only with -Dprotobuf_ABSL_PROVIDER=package,
    # so libprotobuf.lib has unresolved abseil symbols (LogMessage, Cord, CatPieces...).
    # MIGraphX links the bare Protobuf_LIBRARY path (module mode) and drops protobuf's
    # transitive absl deps -> migraphx_onnx.dll lld-link "undefined symbol: absl::...".
    # Fix: make Protobuf_LIBRARY a list = libprotobuf + every absl/utf8 static lib.
    # lld-link only pulls referenced symbols, so over-listing is harmless.
    $LibDir = Join-Path $CgetPrefix "lib"
    $ProtobufLibs = @((Join-Path $LibDir "libprotobuf.lib"))
    Get-ChildItem -Path $LibDir -Filter "absl_*.lib" -ErrorAction SilentlyContinue |
        ForEach-Object { $ProtobufLibs += $_.FullName }
    foreach ($extra in @("utf8_range.lib", "utf8_validity.lib")) {
        $p = Join-Path $LibDir $extra
        if (Test-Path $p) { $ProtobufLibs += $p }
    }
    $ProtobufLibList = ($ProtobufLibs -join ';')
    Write-Host "  Protobuf_LIBRARY: libprotobuf + $($ProtobufLibs.Count - 1) absl/utf8 libs"

    Write-Host "  Configuring MIGraphX..."
    & $Cmake -G Ninja `
        "-DCMAKE_MAKE_PROGRAM=$Ninja" `
        "-DCMAKE_BUILD_TYPE=Release" `
        "-DCMAKE_C_COMPILER=$ClangCl" `
        "-DCMAKE_CXX_COMPILER=$ClangCl" `
        "-DCMAKE_HIP_COMPILER=$Clangxx" `
        "-DCMAKE_HIP_ARCHITECTURES=$GpuTargetsStr" `
        "-DCMAKE_PREFIX_PATH=$CgetPrefix;$RocmDevel;$RocmCore;$RocmLibs" `
        "-DCMAKE_INSTALL_PREFIX=$InstallDir" `
        "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW" `
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL" `
        "-DMIGRAPHX_ENABLE_GPU=ON" `
        "-DMIGRAPHX_ENABLE_CPU=OFF" `
        "-DMIGRAPHX_ENABLE_FPGA=OFF" `
        "-DMIGRAPHX_ENABLE_PYTHON=ON" `
        "-DMIGRAPHX_USE_COMPOSABLEKERNEL=ON" `
        "-DMIGRAPHX_USE_ROCBLAS=ON" `
        "-DMIGRAPHX_USE_HIPBLASLT=ON" `
        "-DMIGRAPHX_USE_MIOPEN=ON" `
        "-DMIGRAPHX_ENABLE_MLIR=ON" `
        "-DMIGRAPHX_WORKAROUND_HIP_MULTI_ARCH_BUG=ON" `
        "-DGPU_TARGETS=$GpuTargetsStr" `
        "-DBUILD_DEV=OFF" `
        "-DPython3_EXECUTABLE=$(Join-Path $VenvScripts 'python.exe')" `
        "-DPYTHON_EXECUTABLE=$(Join-Path $VenvScripts 'python.exe')" `
        "-DSQLite3_ROOT=$CgetPrefix" `
        "-DSQLite3_INCLUDE_DIR=$(Join-Path $CgetPrefix 'include')" `
        "-DSQLite3_LIBRARY=$(Join-Path $CgetPrefix 'lib\sqlite3.lib')" `
        "-Dmsgpackc-cxx_DIR=$(Join-Path $CgetPrefix 'lib\cmake\msgpack')" `
        "-DProtobuf_INCLUDE_DIR=$(Join-Path $CgetPrefix 'include')" `
        "-DProtobuf_LIBRARY=$ProtobufLibList" `
        "-DProtobuf_PROTOC_EXECUTABLE=$(Join-Path $CgetPrefix 'bin\protoc.exe')" `
        -B $MigraphxBuild `
        -S $MigraphxSrc
    if ($LASTEXITCODE -ne 0) { throw "MIGraphX CMake configure failed" }

    New-Item -ItemType File -Path $ConfigDone -Force | Out-Null
}

Write-Host "  Building MIGraphX (this may take 30-90 minutes)..."
& $Cmake --build $MigraphxBuild --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "MIGraphX build failed" }

Write-Host "  Installing MIGraphX..."
& $Cmake --install $MigraphxBuild
if ($LASTEXITCODE -ne 0) { throw "MIGraphX install failed" }

Write-Host ""
Write-Host ">>> MIGraphX build complete!"

# ---------------------------------------------------------------------------
# Step 6: Verify
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Step 6: Quick verification"

$MigraphxPython = Join-Path $MigraphxBuild "lib"
$MigraphxLib = Join-Path $InstallDir "lib"
$MigraphxBin = Join-Path $InstallDir "bin"

Write-Host "  Libraries:"
$dllLocations = @($MigraphxLib, $MigraphxBin, $MigraphxPython)
$foundDlls = $false
foreach ($loc in $dllLocations) {
    $dlls = Get-ChildItem -Path $loc -Filter "migraphx*.dll" -ErrorAction SilentlyContinue
    if ($dlls) {
        $dlls | ForEach-Object { Write-Host "    $($_.FullName)" }
        $foundDlls = $true
        break
    }
}
if (-not $foundDlls) {
    Write-Host "    (no migraphx DLLs found yet - check build output)"
}

Write-Host ""
Write-Host "  Python module:"
$pydFiles = Get-ChildItem -Path $MigraphxPython -Filter "*migraphx*.pyd" -ErrorAction SilentlyContinue
if ($pydFiles) {
    $pydFiles | ForEach-Object { Write-Host "    $($_.FullName)" }
} else {
    Write-Host "    (will check after install)"
}

Write-Host ""
Write-Host "============================================"
Write-Host "BUILD COMPLETE"
Write-Host "============================================"
Write-Host ""
Write-Host "GPU targets:  $GpuTargetsStr"
Write-Host "Install dir:  $InstallDir"
Write-Host "Build dir:    $MigraphxBuild"
Write-Host ""
Write-Host "To test Python bindings:"
Write-Host "  `$env:PATH = `"$MigraphxLib;`$env:PATH`""
Write-Host "  `$env:PYTHONPATH = `"$MigraphxPython;`$env:PYTHONPATH`""
Write-Host "  python -c `"import migraphx; print(migraphx)`""
Write-Host ""
Write-Host "Known patches applied:"
Write-Host "  1. CK codegen: /std:c++20 for clang-cl + placed before add_embed_library"
Write-Host "  2. CK utils.hpp: added #include <string>"
Write-Host "  3. CK codegen: EMBED_USE=CArrays (avoids RC long path issues)"
Write-Host "  4. rocMLIR: LLVM_DISABLE_ASSEMBLY_FILES=ON (avoids MASM/clang flag conflicts)"
Write-Host "  5. MIGraphX ck.hpp: removed #ifndef _WIN32 guard on env var declarations"
Write-Host "  6. git config core.longpaths=true (CK repo has long filenames)"
