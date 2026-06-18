# MIGraphX Build Script for Windows (ROCm 7.14 pip SDK)
# Prerequisites: pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx120X-all]"
#                pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ rocm-sdk-devel
#                Python 3.12, Visual Studio 2022 BuildTools, Git

param(
    [string]$GPU_TARGETS = "gfx1200;gfx1201"  # RDNA 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "F:\MIGraphxWin"
$VenvRoot = "$ProjectRoot\venv"
$VcpkgInstalled = "$ProjectRoot\vcpkg\installed\x64-windows"
$RocmCmakeRoot = "$ProjectRoot\rocm-cmake"
$SrcDir = "$ProjectRoot\src"
$BuildDir = "$ProjectRoot\build_gpu"
$Ninja = "$ProjectRoot\vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe"
$CmakeExe = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

# ROCm 7.14 from pip packages
$RocmSdkCore   = "$VenvRoot\Lib\site-packages\_rocm_sdk_core"
$RocmSdkLibBin  = "$VenvRoot\Lib\site-packages\_rocm_sdk_libraries\bin"
$RocmSdkDevel   = "$ProjectRoot\rocm-sdk\_rocm_sdk_devel"
$ClangDir       = "$RocmSdkCore\lib\llvm\bin"
$ClangDeps      = "$ProjectRoot\clang-deps"   # clang-built abseil + protobuf (for ONNX)

# ONNX/TF need protobuf+abseil built with clang++ (run build_deps.sh first).
# If clang-deps is missing, ONNX/TF are auto-disabled.
$EnableOnnx = Test-Path "$ClangDeps\lib\cmake\protobuf\protobuf-config.cmake"

Write-Host "=== MIGraphX Windows Build (ROCm 7.14 pip SDK) ===" -ForegroundColor Cyan
Write-Host "ROCm Core:  $RocmSdkCore"
Write-Host "ROCm Devel: $RocmSdkDevel"
Write-Host "GPU Targets: $GPU_TARGETS"
Write-Host ""

# Validate
if (-not (Test-Path $ClangDir\clang++.exe)) {
    Write-Host "ERROR: clang++ not found at $ClangDir\clang++.exe" -ForegroundColor Red
    Write-Host "Run: pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ rocm-sdk-devel"
    exit 1
}
if (-not (Test-Path $RocmSdkLibBin\MIOpen.dll)) {
    Write-Host "ERROR: MIOpen.dll not found. Run: pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ torch[device-gfx120X-all]" -ForegroundColor Red
    exit 1
}

$env:HIP_PATH = $RocmSdkCore
$env:ROCM_PATH = $RocmSdkCore

# Create build directory
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "--- CMake Configuration ---" -ForegroundColor Yellow

$cmakeArgs = @(
    "-S", $SrcDir, "-B", $BuildDir, "-G", "Ninja",
    "-DCMAKE_CXX_COMPILER=$ClangDir\clang++.exe",
    "-DCMAKE_C_COMPILER=$ClangDir\clang.exe",
    "-DCMAKE_RC_COMPILER=C:/Program Files/AMD/ROCm/7.1/bin/llvm-rc.exe",
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    # clang-deps FIRST so its clang abseil wins over vcpkg's MSVC abseil (same target names)
    "-DCMAKE_PREFIX_PATH=$ClangDeps;$RocmSdkDevel;$RocmSdkCore;$VcpkgInstalled;$RocmCmakeRoot",
    "-DGPU_TARGETS=$GPU_TARGETS",
    # ROCm device library path — pip SDK uses non-standard layout
    "-DCMAKE_CXX_FLAGS=--rocm-device-lib-path=$RocmSdkCore/lib/llvm/amdgcn/bitcode",
    # GPU libraries — all ON now, from pip ROCm 7.14
    "-DMIGRAPHX_USE_ROCBLAS=ON",
    "-DMIGRAPHX_USE_MIOPEN=ON",
    "-DMIGRAPHX_USE_HIPBLASLT=ON",
    "-DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF",  # CK not ported to Windows
    "-DMIGRAPHX_ENABLE_MLIR=OFF",            # rocMLIR not available on Windows
    # ONNX/TF parsers — enabled when clang-built protobuf is present (build_deps.sh)
    "-DMIGRAPHX_ENABLE_ONNX=$(if ($EnableOnnx) {'ON'} else {'OFF'})",
    "-DMIGRAPHX_ENABLE_TF=$(if ($EnableOnnx) {'ON'} else {'OFF'})",
    # Python bindings
    "-DMIGRAPHX_ENABLE_PYTHON=ON",
    "-DPYTHON_EXECUTABLE=$VenvRoot\Scripts\python.exe",
    # Windows-specific
    "-DMIGRAPHX_MSVC_STATIC_RUNTIME=OFF",
    "-DMIGRAPHX_WORKAROUND_HIP_MULTI_ARCH_BUG=ON",
    "-DBUILD_TESTING=OFF",
    "-Wno-dev"  # Suppress cmake policy warnings (hip-config symlink noise)
)

# When ONNX is enabled, point cmake explicitly at the clang-built protobuf/abseil
if ($EnableOnnx) {
    $cmakeArgs += @(
        "-Dprotobuf_DIR=$ClangDeps/lib/cmake/protobuf",
        "-Dabsl_DIR=$ClangDeps/lib/cmake/absl",
        "-Dutf8_range_DIR=$ClangDeps/lib/cmake/utf8_range"
    )
    Write-Host "ONNX/TF parsers: ENABLED (clang-deps found)" -ForegroundColor Green
} else {
    Write-Host "ONNX/TF parsers: DISABLED (run build_deps.sh to enable)" -ForegroundColor Yellow
}

& $CmakeExe @cmakeArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--- Building MIGraphX ---" -ForegroundColor Yellow
& $CmakeExe --build $BuildDir --config Release --parallel 8 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Build SUCCESS ===" -ForegroundColor Green

# Copy runtime DLLs to output directory
Write-Host ""
Write-Host "--- Copying runtime DLLs ---" -ForegroundColor Yellow

# UCRT API set DLLs (clang-cl creates api-ms-win-crt-*.dll imports)
$redistUcrt = "C:\Program Files (x86)\Windows Kits\10\Redist\10.0.26100.0\ucrt\DLLs\x64"
Get-ChildItem "$redistUcrt\api-ms-win-crt-*.dll" | Copy-Item -Destination "$BuildDir\bin" -Force

# VC++ runtime
Copy-Item "C:\Windows\System32\VCRUNTIME140.dll" "$BuildDir\bin" -Force
Copy-Item "C:\Windows\System32\VCRUNTIME140_1.dll" "$BuildDir\bin" -Force -ErrorAction SilentlyContinue
Copy-Item "C:\Windows\System32\MSVCP140.dll" "$BuildDir\bin" -Force
Copy-Item "C:\Windows\System32\MSVCP140_1.dll" "$BuildDir\bin" -Force -ErrorAction SilentlyContinue
Copy-Item "C:\Windows\System32\MSVCP140_ATOMIC_WAIT.dll" "$BuildDir\bin" -Force -ErrorAction SilentlyContinue

# SQLite3 from vcpkg
Copy-Item "$VcpkgInstalled\bin\sqlite3.dll" "$BuildDir\bin" -Force -ErrorAction SilentlyContinue

# ROCm SDK DLLs (needed by migraphx_gpu.dll at runtime)
Copy-Item "$RocmSdkCore\bin\amdhip64_7.dll" "$BuildDir\bin" -Force
Copy-Item "$RocmSdkCore\bin\hiprtc0714.dll" "$BuildDir\bin" -Force
Copy-Item "$RocmSdkCore\bin\amd_comgr.dll" "$BuildDir\bin" -Force
Copy-Item "$RocmSdkCore\bin\rocm_kpack.dll" "$BuildDir\bin" -Force
Copy-Item "$RocmSdkLibBin\MIOpen.dll" "$BuildDir\bin" -Force
Copy-Item "$RocmSdkLibBin\rocblas.dll" "$BuildDir\bin" -Force
Copy-Item "$RocmSdkLibBin\libhipblaslt.dll" "$BuildDir\bin" -Force

# Copy rocBLAS + hipBLASLt kernel databases
if (Test-Path "$RocmSdkLibBin\rocblas") {
    Copy-Item "$RocmSdkLibBin\rocblas" "$BuildDir\bin\rocblas" -Recurse -Force
}
if (Test-Path "$RocmSdkLibBin\hipblaslt") {
    Copy-Item "$RocmSdkLibBin\hipblaslt" "$BuildDir\bin\hipblaslt" -Recurse -Force
}

Write-Host ""
Write-Host "Output binaries: $BuildDir\bin"
Get-ChildItem $BuildDir\bin -Filter "*.dll" | ForEach-Object { Write-Host "  $($_.Name)" }
Get-ChildItem $BuildDir\bin -Filter "*.pyd" | ForEach-Object { Write-Host "  $($_.Name)" }

# Install built pyd + DLLs into venv for test
Write-Host ""
Write-Host "--- Installing into venv site-packages ---" -ForegroundColor Yellow
$SitePackages = "$VenvRoot\Lib\site-packages"
Copy-Item "$BuildDir\bin\migraphx*.pyd" $SitePackages -Force
Copy-Item "$BuildDir\bin\migraphx*.dll" $SitePackages -Force
Copy-Item "$BuildDir\bin\migraphx-hiprtc-driver.exe" $SitePackages -Force -ErrorAction SilentlyContinue
Copy-Item "$BuildDir\bin\sqlite3.dll" $SitePackages -Force -ErrorAction SilentlyContinue
# UCRT + VC++ runtime (needed by hiprtc-driver subprocess)
Get-ChildItem "$BuildDir\bin\api-ms-win-crt-*.dll" | Copy-Item -Destination $SitePackages -Force
@("VCRUNTIME140.dll","VCRUNTIME140_1.dll","MSVCP140.dll","MSVCP140_1.dll","MSVCP140_ATOMIC_WAIT.dll") | ForEach-Object {
    Copy-Item "$BuildDir\bin\$_" $SitePackages -Force -ErrorAction SilentlyContinue
}
# ROCm DLLs (needed by hiprtc-driver subprocess, can't inherit os.add_dll_directory)
@("amdhip64_7.dll","hiprtc0714.dll","hiprtc-builtins0714.dll","amd_comgr.dll","rocm_kpack.dll",
  "MIOpen.dll","rocblas.dll","libhipblaslt.dll") | ForEach-Object {
    Copy-Item "$BuildDir\bin\$_" $SitePackages -Force -ErrorAction SilentlyContinue
}
# Kernel databases (rocBLAS/hipBLASLt look for these next to DLL)
if (Test-Path "$BuildDir\bin\rocblas") {
    Copy-Item "$BuildDir\bin\rocblas" "$SitePackages\rocblas" -Recurse -Force
}
if (Test-Path "$BuildDir\bin\hipblaslt") {
    Copy-Item "$BuildDir\bin\hipblaslt" "$SitePackages\hipblaslt" -Recurse -Force
}

Write-Host ""
Write-Host "--- Running GPU smoke test (venv python) ---" -ForegroundColor Yellow
& "$VenvRoot\Scripts\python.exe" "$ProjectRoot\test_gpu.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test FAILED" -ForegroundColor Red
} else {
    Write-Host "Smoke test PASSED" -ForegroundColor Green
}

Write-Host ""
Write-Host "Python usage (venv):"
Write-Host "  $VenvRoot\Scripts\python.exe test_gpu.py"
