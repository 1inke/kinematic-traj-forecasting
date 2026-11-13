@echo off
cd /d "<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set NUMEXPR_NUM_THREADS=2
set VECLIB_MAXIMUM_THREADS=2
set JOINT_FEAS_ROOT=<WORKSPACE_ROOT>
set JOINT_FEAS_EXEC=<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018
set JOINT_FEAS_K1=<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\k1_exec
if not exist k1_exec mkdir k1_exec
echo [%date% %time%] K1 start>> logs\overnight_chain.log
".venv-k0\Scripts\python.exe" -u run_k1.py >> logs\k1_overnight.log 2>>&1
echo [%date% %time%] K1 exit=!ERRORLEVEL!>> logs\overnight_chain.log
