
---

## 📁 File 10: `run.bat` (Windows Quick Run)

```batch
@echo off
echo 🚀 Nova Core Launcher
echo ====================
echo.
echo 1] Run single input
echo 2] Chat mode
echo 3] Benchmark
echo 4] Speed test
echo 5] Info
echo.
set /p choice="Select (1-5): "

if "%choice%"=="1" (
    set /p input="Enter text: "
    cargo run --release -- run --input "%input%"
)
if "%choice%"=="2" cargo run --release -- chat
if "%choice%"=="3" cargo run --release -- bench
if "%choice%"=="4" cargo run --release -- speed
if "%choice%"=="5" cargo run --release -- info
pause