# 安全可达性与运动信息读出：实施与未启动报告

日期：2026-10-17（日本标准时间）。本报告不是模型效果报告。

## 结论

**已按文档实现并做实际预检，但未启动任何正式实验。** F0缺少已安装SOCP接口，且设计禁止本轮安装；I1按实际生产分块测得的完整流程规划76.588分钟，超过≤30分钟硬启动门；因此G1不触发。没有用预检的小模型/固定样本冒充6个正式25epoch模型，没有安全见证、信息读出效果或科学阴性结论。

人类给予充分执行、自修和合理安排授权；仍按“按照目录要求”保留科学因素、30分钟门、依赖与数据边界。没有减数据/epoch/候选数/审核，或把完整工作拆成隐性长任务。

## 1. 阶段状态

| 阶段 | 已完成 | 未执行及原因 | 科学结论 |
|---|---|---|---|
| F0 | 现有运行环境依赖审计、边界/状态报告 | 无可用已安装SOCP接口；不安装；48样本/640子问题均未开始 | SAFE_OPPORTUNITY_UNRESOLVED，不是安全域为空 |
| I1 | 完整实现、数值/GPU/保存/恢复/身份/源/资源检查，生产吞吐测量 | 六个正式25epoch模型、24科学预测job、人口效果与主CI未产生；总预算不达标 | 未检验信息迁移，不是历史无信息 |
| G1 | 条件门核对 | F0/I1均无所需科学证据，不触发 | NOT_TRIGGERED，不能计作完成 |

正式模型0、正式训练run0、正式科学job0、生产worker/Windows任务0、新自动跟进0。旧暂停任务、baseline、history cycle11与总体目标未动。

## 2. F0依赖阻碍

检查已有六个Python：astra2、astra、py311、用户Python3.11/3.10、应用bundled Python，均未发现cvxpy/clarabel/scs/ecos/mosek/gurobipy module spec。主环境另查cvxopt/picos也不存在。本结论限于已审计的运行接口，不声称机器所有软件均穷尽。

[02_F0_SAFE_REACHABILITY.md](02_F0_SAFE_REACHABILITY.md)明确本轮不下载或安装。没有以SciPy一般优化替代注册SOCP、没有改目标/阈值、没有求解任何窗口。solver_attempts=0与solver infeasible/超时完全不同；不生成GT教师、不外推安全不可达。I1按[README](README.md)独立继续准备，不因F0没有见证而假装信息问题已回答。

证据：[F0依赖审计](F0_execution_20261017/DEPENDENCY_AUDIT.json)、[F0报告](F0_execution_20261017/RESULT_ANALYSIS.md)。

## 3. I1实施与实际通过的检查

两臂S=末速度重复7次+零加速度6次，H=真实7速度+6加速度。共同P296/native24/history26=146输入，146→128→64→24 GELU、28632参数，同seed完整配对参数/buffers/统计与zero-last native identity。输入只有past8/P2/native；GT/groups/ID/logits不进forward。

共同FIT真H统计：float32特征、float64人口mean/std，std<1e-6置1。P2/history按window18909次，native按window×20=378180次；outer/inner/CAL不拟合统计。原24679 unique IDs、四人口及native/GT/group/track/scene/sequence与既有保存缓存逐项一致。原F16没有实例化或新forward，仅源码/模型/cache SHA保护。

DIRECT_READOUT_MSE为B×20×12平均二维平方范数，路径native.double+head.float32转double；没有winner、diversity、tanh、积分输出界、keep、ray/alpha/fallback、teacher、scorer或旧解冻。未来正式固定6fresh×25epoch/11100steps、outer final25 primary；所有模型/24job仍未执行。

实际预检包括：

- S/H各8次真实FIT256 GPU梯度，zero-last分阶段全部参数有限且连接、初始化native EXACT；
- 原始past8与既有observation首16列逐项对齐；共同FIT统计/分块累计通过，NumPy与GPU同输入history算术通过；
- 独立MSE标量、坐标梯度及固定非零double参数FD（eps1e-4、1e-5），误差约5.97e-13至1.71e-12；这是隔离double数学检查，不是生产float32量化证明；
- 两臂实际513FIT+12inner子集两epoch连续/中断，参数/buffers/optimizer/Torch CPU/CUDA/loader/NumPy/Python RNG/history EXACT；不是完整FIT两轮，更不是正式6run；
- 实际128窗口forward/save/read、samecandidate质量、原SciPy全K20 complete-link .10meanpath/.20endpoint、五pooled motion独立重算；
- 固定FIT前2048窗口的真实生产chunk管线与全部上述保存审核，两臂分别22.9871s/23.7846s；throwaway固定非零初始化，不保存benchmark模型，不按科学效果挑样本；
- 20k paired-seed/track bootstrap fixture以独立trackloop重建，通过；原family2与97.5%双侧口径不改。没有正式outer效果或正式主CI；
- 坏source/output/resume合同实际拒绝；三seed初始化SHA绑定和六实际生产manifest分支验证。原生zero、同slot门、20copies onecluster fixture通过。

代码与预检证据位于 [I1当前目录](I1_execution_pipeline_recovery_20261017/README.md)。父目录保留，不运行其旧入口。

## 4. 预算测量与为什么不能启动

| 项目 | 实测/推算 |
|---|---:|
| FIT256训练step，S（4warm+32measure） | .02750698s |
| FIT256训练step，H | .02291390s |
| 六正式训练共11100step nominal | 279.836s，约4.66min |
| 完整2048保存/读回/聚簇/五motion/独立审核，S | 22.9871s |
| 同管线，H | 23.7846s |
| 全197432重复输出window线性管线外推（尚未加1.5余量） | 2292.888s，38.215min |
| 管线增加1.5人口quantile/IO余量后 | 3439.332s，57.322min |
| 加训练、bootstrap/source余量并整体×1.2 | **76.588min** |
| 文档启动上限 | **30min** |

197432是(6科学模型+native/CV两参照)×24679 window的计算量，不是独立样本数。native/CV不是两个新训练模型。76.6分钟是完整生产估算，不是已经运行了76.6分钟，也不是已启动生产的ETA。

最初版本用128-window烟测的固定文件/GC开销放大，规划58.4727min，已标BUDGET_NOT_LAUNCHED。为避免只因烟测小批固定开销阻断，保留全部原证据，在新版本补测实际2048生产chunk。新版本仍超门；**即便拿掉1.5余量，管线本身38.2min已超30min**，不是只靠保守系数才失败。不选择较快seed/人口替代，不继续换预测公式直到低于门。

这是保存与完整审核吞吐阻碍，不是GPU训练自身需要76.6min，也不是内存不足。预算仍是估计、不保证真实耗时；但设计启动门要求估计通过，不能以可能更快为由忽略门。没有第三个预算变体或拆分多个长任务。

证据：[原预算阻断](I1_execution_20261017/PREFLIGHT.json)、[分块复核说明](I1_execution_pipeline_recovery_20261017/PIPELINE_BUDGET_REVIEW.md)、[新实际预检](I1_execution_pipeline_recovery_20261017/PREFLIGHT.json)。

## 5. 来源、修复与资源

首预检只在常速fixture失败：0.4间隔再float32保存已不是精确匀速，速度差3.5762786865234375e-7超2e-7测试。失败代码与trace完整保留。严格常速改用原先已有二进制0.5间隔fixture，小数观测量化差仅描述；真实数据、2e-7后端容差和全部科学门不变。

父完整代码/预检锁2677源，继承原T2全部2560摘要。分块复核版锁2717源，digest `4fcf681d4cb4e939e3e9e1e79b8c9eec5099908fe8bc974ed16962479c400e53`；预检初末verify通过。features/model/metrics/bootstrap/train/worker/prelaunch七模块AST EXACT，protocol与memory估计JSON EXACT；common只变源继承，preflight只增加同生产chunk计时与预算，resource guard只明确返回已执行的系统reserve阈值，Windows任务名仅隔离。source lock后没有改科学代码。

资源E4916164516bytes=4.578535GiB，1.3E=5.952095GiB；沿用current+Windows retainedpeak RSS/private检查，并按新设计保留系统physical/commit各6GiB，不是固定6GiB进程门。此次预检peakprivate2.33415GiB、peakRSS1.60067GiB，系统最小可用physical15.583GiB/commit19.738GiB（准确bytes见PREFLIGHT），均通过；不是生产峰值，因为没有生产。

预检elapsed201.689s，磁盘可用约32.73GiB，保守输出payload约1.845GiB，磁盘门通过。没有OS/pagefile变更、用户程序kill、安装、云资源或旧结果覆盖。旧T1历史资源偏差不被新检查洗掉。

## 6. 反方检查与停止

DA1已在锁源前保存：S通过P2/native已有历史，H只是同一观测的显式读出；I1回归可能塌缩，不能当多模态成功，F0负结果不代表域空。

DA2：最强反驳是测量有固定开销且预测保守，实际完整任务可能更快。已追加实际2048生产分块核对并保留首估算；原始线性管线仍38.2min，没有满足≤30门。没有获得任何关于H−S、H−CV或安全域的效果证据，因此不能写模型阴性。

DA3：全部正式run/job计0，正式signal为unknown而非false，G1为未触发而非已完成；未做生产source/audit/模型复放声明，未创建后台worker或空转automation。研究执行停在设计本身要求的依赖/预算门，不是随意人类暂停、也不假装总体目标完成。

检查采用 [Academic Research Suite](<USER_HOME>/.codex/skills/academic-research-suite/SKILL.md) 的证据—反证流程，同助手inline，非外部独立评审。

## 7. 后续边界

当前设计下停止。进一步正式执行需要先解决允许的求解器依赖和≤30min完整审核吞吐/预算合同；如以后修订，必须保留本轮版本和失败来源，在新版本明确修订范围与全部预检，不能直接改这份locked协议继续跑。

本次未引入新模型假设，不能自动加MoE/scorer、解冻旧上游、增加新数据、降低审核规格或调科学阈值。原F16已见、target非context-disjoint、3scene/3seed及适应性复用等局限仍保留；当前甚至没有正式效果估计，不作任何泛化/不可预测性/唯一根因推论。

