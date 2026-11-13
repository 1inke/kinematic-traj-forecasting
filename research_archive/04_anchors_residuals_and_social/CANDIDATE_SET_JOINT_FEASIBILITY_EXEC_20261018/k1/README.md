# K1 执行子目录说明

路径：`CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/k1_exec/`（由 `JOINT_FEAS_K1` 或默认）

唯一因素：I=individual-surrogate vs S=set-marginal。生成器冻结；禁止 K0 oracle teacher / outer GT 训练 / 配额扫参。

运行（用户机，在 EXEC 根目录）：

```bat
.\.venv-k0\Scripts\python.exe -m pip install scikit-learn joblib
set JOINT_FEAS_ROOT=<WORKSPACE_ROOT>
set JOINT_FEAS_EXEC=%CD%
set JOINT_FEAS_K1=%CD%\k1_exec
.\.venv-k0\Scripts\python.exe run_k1.py
```

Smoke：`python run_k1.py --smoke 8 --skip-boot`
