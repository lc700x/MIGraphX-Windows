# Clone AMDMIGraphX at the pinned commit and apply Windows build patches.
# Run once before build_migraphx.ps1.

param(
    [string]$Commit = "0043a53c9"  # tested commit
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "F:\MIGraphxWin"
$SrcDir      = "$ProjectRoot\src"
$PatchFile   = "$ProjectRoot\patches\windows_build.patch"

# --- rocm-cmake (needed by AMDMIGraphX CMake) ---
if (-not (Test-Path "$ProjectRoot\rocm-cmake\CMakeLists.txt")) {
    Write-Host "--- Cloning rocm-cmake ---" -ForegroundColor Yellow
    git clone https://github.com/RadeonOpenCompute/rocm-cmake.git "$ProjectRoot\rocm-cmake" --depth 1
} else {
    Write-Host "rocm-cmake already present" -ForegroundColor Green
}

# --- AMDMIGraphX source ---
if (-not (Test-Path "$SrcDir\.git")) {
    Write-Host "--- Cloning AMDMIGraphX ---" -ForegroundColor Yellow
    git clone https://github.com/ROCm/AMDMIGraphX.git "$SrcDir" --depth 1
    Set-Location $SrcDir
    git fetch --depth 1 origin $Commit
    git checkout $Commit
    Set-Location $ProjectRoot
} else {
    Write-Host "src already present ($(git -C $SrcDir log --oneline -1))" -ForegroundColor Green
}

# --- Apply Windows patches ---
Write-Host "--- Applying Windows build patches ---" -ForegroundColor Yellow
Set-Location $SrcDir
$result = git apply --check "$PatchFile" 2>&1
if ($LASTEXITCODE -eq 0) {
    git apply "$PatchFile"
    Write-Host "Patches applied OK" -ForegroundColor Green
} else {
    Write-Host "Patch already applied or conflicts:" -ForegroundColor Yellow
    Write-Host $result
}
Set-Location $ProjectRoot

Write-Host ""
Write-Host "=== Source ready. Run .\build_migraphx.ps1 to build. ===" -ForegroundColor Cyan
