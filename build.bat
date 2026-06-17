@echo off
setlocal

set GPU_TARGETS=gfx1200;gfx1201
if not "%1"=="" set GPU_TARGETS=%1

echo === MIGraphX Build ===
echo GPU_TARGETS: %GPU_TARGETS%
echo.

powershell.exe -NoLogo -ExecutionPolicy Bypass -File "%~dp0build_migraphx.ps1" -GPU_TARGETS "%GPU_TARGETS%"
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo === Building Python wheel ===
"%~dp0venv\Scripts\python.exe" "%~dp0make_wheel.py"
if errorlevel 1 (
    echo.
    echo WHEEL BUILD FAILED
    pause
    exit /b 1
)

echo.
echo === Installing wheel into venv ===
"%~dp0venv\Scripts\pip.exe" install --force-reinstall "%~dp0dist\migraphx_rocm-2.16.0.dev0-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
    echo WHEEL INSTALL FAILED
    pause
    exit /b 1
)

echo.
echo === Smoke test ===
"%~dp0venv\Scripts\python.exe" "%~dp0test_gpu.py"
if errorlevel 1 (
    echo SMOKE TEST FAILED
    pause
    exit /b 1
)

echo.
echo === ALL DONE ===
pause
