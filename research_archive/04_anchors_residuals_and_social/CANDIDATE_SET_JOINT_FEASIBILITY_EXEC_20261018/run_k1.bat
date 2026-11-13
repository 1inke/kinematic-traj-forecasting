@echo off
cd /d "%~dp0"
set JOINT_FEAS_ROOT=<WORKSPACE_ROOT>
set JOINT_FEAS_EXEC=%CD%
set JOINT_FEAS_K1=%CD%\k1_exec
if not exist .venv-k0\Scripts\python.exe (
  echo missing .venv-k0
  exit /b 1
)
.venv-k0\Scripts\python.exe -m pip install -q scikit-learn joblib
.venv-k0\Scripts\python.exe run_k1.py %*
