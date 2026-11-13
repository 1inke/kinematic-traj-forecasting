@echo off
setlocal
cd /d "%~dp0"
set KIN_FORCE_CPU=1
set PYTHONUNBUFFERED=1
set PY=<PYTHON_ENV_ROOT>\envs\astra2\python.exe
if not exist "%PY%" set PY=python
echo Using %PY%
"%PY%" -u run_v1.py
if errorlevel 1 exit /b 1
"%PY%" -u run_improve.py
if errorlevel 1 exit /b 1
"%PY%" -u run_v2_ablation.py
if errorlevel 1 exit /b 1
"%PY%" -u write_report.py
echo DONE
