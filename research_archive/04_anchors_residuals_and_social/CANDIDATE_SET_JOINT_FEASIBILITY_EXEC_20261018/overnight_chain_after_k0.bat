@echo off
cd /d "<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
echo [%date% %time%] chain watcher start>> logs\overnight_chain.log
:waitk0
timeout /t 120 /nobreak >nul
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | findstr /I "run_k0" >nul
if %ERRORLEVEL%==0 goto waitk0
REM also check any python still on run_k0 via wmic-ish: look at log marker
if not exist results\seed44\FIT_cal\result_fields.json (
  echo [%date% %time%] K0 FIT_cal missing after exit>> logs\overnight_chain.log
  exit /b 3
)
if exist k1_exec\runtime_state.json (
  echo [%date% %time%] K1 already complete, skip>> logs\overnight_chain.log
  exit /b 0
)
if exist logs\k1_overnight.started (
  echo [%date% %time%] K1 already started marker, skip>> logs\overnight_chain.log
  exit /b 0
)
echo started> logs\k1_overnight.started
call run_k1_overnight.bat
