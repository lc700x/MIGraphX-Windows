# ============================================================================
# Package built MIGraphX into a Python wheel for Windows.
# Bundles MIGraphX DLLs + .pyd + ONNX backend. Excludes ROCm runtime DLLs
# (those come from rocm_sdk_core / rocm_sdk_libraries pip packages).
#
# Usage:
#   .\make-wheel.ps1 [-Arch gfx1103] [-Version 0.1.0]
# ============================================================================
[CmdletBinding()]
param(
    [string]$Arch = "gfx1103",
    [string]$Version = "0.1.0"
)
$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvScripts = Join-Path $RootDir ".env\Scripts"
$Py = Join-Path $VenvScripts "python.exe"
$BuildBin = Join-Path $RootDir "AMDMIGraphX\build\bin"
$BuildLib = Join-Path $RootDir "AMDMIGraphX\build\lib"
$WheelStage = Join-Path $RootDir "wheel-stage"
$PkgDir = Join-Path $WheelStage "migraphx"

Write-Host "Packaging MIGraphX wheel: arch=$Arch version=$Version"

# Clean stage
if (Test-Path $WheelStage) { Remove-Item -Recurse -Force $WheelStage }
New-Item -ItemType Directory -Path $PkgDir -Force | Out-Null

# Copy MIGraphX DLLs + pyd (exclude test-only kernel-check DLLs)
$dlls = Get-ChildItem -Path $BuildBin -Filter "migraphx*.dll" |
    Where-Object { $_.Name -notmatch 'kernel_file_check' }
foreach ($d in $dlls) { Copy-Item $d.FullName -Destination $PkgDir -Force }
Copy-Item (Join-Path $BuildBin "migraphx.cp312-win_amd64.pyd") -Destination $PkgDir -Force

# Copy ONNX backend package as a subpackage
$OnnxSrc = Join-Path $BuildLib "onnx_migraphx"
if (Test-Path $OnnxSrc) {
    Copy-Item -Recurse $OnnxSrc -Destination (Join-Path $PkgDir "onnx_migraphx") -Force
}

# __init__.py: register ROCm DLL dirs from installed pip packages, then load .pyd
$InitPy = @'
import os
import sys
import importlib.util
import importlib.machinery

_pkg_dir = os.path.dirname(os.path.abspath(__file__))

def _add_rocm_dll_dirs():
    # Bundled MIGraphX DLLs live next to this file
    dirs = [_pkg_dir]
    # ROCm runtime DLLs come from the rocm pip packages (site-packages/_rocm_sdk_*)
    try:
        import importlib.util as _u
        for mod in ("_rocm_sdk_core", "_rocm_sdk_libraries", "_rocm_sdk_devel"):
            spec = _u.find_spec(mod)
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    b = os.path.join(loc, "bin")
                    if os.path.isdir(b):
                        dirs.append(b)
    except Exception:
        pass
    # Fallback: scan site-packages for _rocm_sdk_*/bin
    for sp in site_packages_dirs():
        for name in ("_rocm_sdk_core", "_rocm_sdk_libraries", "_rocm_sdk_devel"):
            b = os.path.join(sp, name, "bin")
            if os.path.isdir(b) and b not in dirs:
                dirs.append(b)
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def site_packages_dirs():
    out = []
    for p in sys.path:
        if p and os.path.isdir(p) and p.lower().endswith("site-packages"):
            out.append(p)
    return out

_add_rocm_dll_dirs()

# Load the compiled extension (migraphx.cp3XX-win_amd64.pyd) sitting next to us.
# The .pyd exports PyInit_migraphx, so the spec name's LAST component must be
# "migraphx" (ExtensionFileLoader derives the init symbol from it). We use a
# distinct dotted prefix so it doesn't collide with this package in sys.modules.
def _load_ext():
    for fn in os.listdir(_pkg_dir):
        if fn.startswith("migraphx.") and fn.endswith(".pyd"):
            path = os.path.join(_pkg_dir, fn)
            loader = importlib.machinery.ExtensionFileLoader("_migraphx_native.migraphx", path)
            spec = importlib.util.spec_from_loader("_migraphx_native.migraphx", loader, origin=path)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            return mod
    raise ImportError("migraphx: compiled extension .pyd not found in " + _pkg_dir)

_core = _load_ext()
# Re-export everything from the compiled module
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)
del _core
'@
[System.IO.File]::WriteAllText((Join-Path $PkgDir "__init__.py"), $InitPy, (New-Object System.Text.UTF8Encoding $false))

# pyproject.toml
$Pyproject = @"
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "migraphx"
version = "$Version"
description = "AMD MIGraphX inference engine (Windows, $Arch) with GPU+CK+MLIR+ONNX"
requires-python = ">=3.12"
dependencies = [
    "rocm_sdk_core",
    "rocm_sdk_libraries",
]

[tool.setuptools]
packages = ["migraphx", "migraphx.onnx_migraphx"]

[tool.setuptools.package-data]
migraphx = ["*.dll", "*.pyd"]
"@
[System.IO.File]::WriteAllText((Join-Path $WheelStage "pyproject.toml"), $Pyproject, (New-Object System.Text.UTF8Encoding $false))

# Build the wheel. pip/build emit progress on stderr; don't treat that as fatal.
Push-Location $WheelStage
try {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Py -m pip install --upgrade build wheel setuptools 2>&1 | Out-Null
    & $Py -m build --wheel 2>&1 | ForEach-Object { Write-Host $_ }
    $ErrorActionPreference = $old
    if ($LASTEXITCODE -ne 0) { throw "wheel build failed" }
} finally {
    Pop-Location
}

# Retag the wheel as platform-specific (it contains a .pyd, so it is NOT py3-none-any)
$WheelOut = Get-ChildItem -Path (Join-Path $WheelStage "dist") -Filter "*.whl" | Select-Object -First 1
Push-Location (Join-Path $WheelStage "dist")
try {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Py -m wheel tags --python-tag cp312 --abi-tag cp312 --platform-tag win_amd64 --remove $WheelOut.Name 2>&1 | ForEach-Object { Write-Host $_ }
    $ErrorActionPreference = $old
} finally {
    Pop-Location
}
# Pick the retagged wheel (cp312-...-win_amd64)
$WheelOut = Get-ChildItem -Path (Join-Path $WheelStage "dist") -Filter "*win_amd64.whl" | Select-Object -First 1
if (-not $WheelOut) {
    $WheelOut = Get-ChildItem -Path (Join-Path $WheelStage "dist") -Filter "*.whl" | Select-Object -First 1
}

# Backup the wheel. Encode the arch into the version local-tag (+gfxNNNN) so the
# filename stays a VALID, pip-installable wheel name (arch in the wrong field
# makes pip reject it as "not supported on this platform").
$BackupDir = Join-Path $RootDir "wheels"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$ArchTag = ($Arch -replace '[^A-Za-z0-9]', '')
$BackupName = "migraphx-$Version+$ArchTag-cp312-cp312-win_amd64.whl"
Copy-Item $WheelOut.FullName -Destination (Join-Path $BackupDir $BackupName) -Force

Write-Host ""
Write-Host "============================================"
Write-Host "WHEEL BUILT"
Write-Host "  Source: $($WheelOut.FullName)"
Write-Host "  Backup: $(Join-Path $BackupDir $BackupName)"
Write-Host "============================================"
