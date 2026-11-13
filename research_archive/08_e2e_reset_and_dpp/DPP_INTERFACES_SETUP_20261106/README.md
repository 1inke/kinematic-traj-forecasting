# DPP 场景地图与 DSF 接口配置（2026-11-06）

最新授权：用户要求“配好这两个接口”。新增本独立目录，不修改已经锁源的11-04 Stage A，不启动 Stage B/新生成器预训练，不删除任何东西。

## 已完成的工程部分

- `scene_interface.py`：真实记录ID注册、证据/文件SHA、独立米制配准残差检查、语义类别策略、像素语义图→各向同性米制栅格→SDF、unknown距离可靠域、沿段采样、缺图/图外/未知保留NA。**不能把透视像素EDT乘一个常数就叫米制距离。**
- `dsf_interface.py`：标准高斯潜变量合同，真实checkpoint的weights_only/strict加载，预训练生成器冻结但保留对潜变量的梯度，DSF context→20个latent→12步米制轨迹，B0原radius r/B1场景/B2运动学/B3乘积/B4均匀五种权重，DPP EC可微损失及场景/运动学Torch算术。无GT参数、h3 latent替换或自动生成器训练。
- `test_interfaces.py`：工程夹具，包括真实CUDA8样本、两固定eps FD、冻结权重/boolbuffer、非零latent梯度、NumPy对照、地图unknown、配准/坏prior/未训练权重拒绝。夹具随机生成器**不是科学预训练模型**；测试通过不等StageB生产前检。

## 新找到并下载的真实地图来源

Y-net作者官方README链接的数据包及作者Issue29确认ETH/UCY语义图/H：

- https://raw.githubusercontent.com/HarshayuGirase/Human-Path-Prediction/master/ynet/README.md
- https://github.com/HarshayuGirase/Human-Path-Prediction/issues/29
- https://raw.githubusercontent.com/HarshayuGirase/Human-Path-Prediction/master/ynet/utils/image_utils.py

`fetch_ynet_maps.py` 对995391402字节ZIP只做HTTP Range选择读取17个语义图/H小文件；CRC+各SHA记录`YNET_DOWNLOAD.json`，不下载/执行包内pkl、模型或代码，不修改旧Python环境。原始资产保留`official_ynet_assets/`。总ZIP SHA未下载验证，不冒称已核整包SHA或独立真实性签名。

实际覆盖当前7个recording；hotel依官方image2world先交换像素xy，再施H。students/uni、zara2/zara3存在相同图/H，**不能凭字节相同认定它们独立或每录像都标注完备**。uni_examples整个mask只有标签1；不能当作可靠的“全部可通行、无障碍”证书。

目前图像0/1的行人可通行规则、独立静态标注来源/使用权限、同录像真实米制配准/valid域还未核齐，故`map_registry.json`暂不科学准入。`REAL_ASSET_INSPECTION.json`只读ID/观测past做H投影分母检查，不用future/GT决定类别/配准，不把观测重投影当独立控制点证明。

地图证据齐后：按`annotation.template.json`填入真实证据，再运行：

```powershell
& .venv-governance/Scripts/python.exe -B qoder_work/DPP_INTERFACES_SETUP_20261106/scene_interface.py --annotation <annotation.json> --output <new_map_folder>
```

将新record.json登记到本目录新的registry版本，由MapRegistry消费。所有missing图仍保留原ID/分母，不删窗凑覆盖。

## 最新用户决定：先用现有生成器，只有不成立才另训CVAE

用户在本轮明确提出DSF应可推广到其他潜向量生成轨迹的方法，要求先尝试现有方法、不行才预训练CVAE。此最新指令覆盖了本页下面“必须先提供CVAE”的旧窄理解。原论文4.1也明确允许其他合适生成模型；**先前把Gaussian cVAE当DSF通用接口的必要条件过窄，现予纠正**。高斯前提只约束原radius质量的概率解释，不能照搬到确定性h3查询。

已新增`current_h3_adapter.py`和`CurrentDSFPipeline`：加载现有已训练fresh-B0 seed42 epoch8 EMA完整SHA绑定checkpoint；保持F16/h3/score全部权重与boolbuffer冻结，提取因果上下文64D，将DSF输出20×64的控制偏移加到已有mode query，在原mode_mlp前解码到4/8/12锚与世界坐标。默认zero-last sampler使原K20路径保持原样；这里只增加新的控制接口，不改旧模型源码/文件。

`current_h3_check/RESULT.json`实际GPU哈希固定FIT8窗通过：zero-control原路径**bitwise EXACT**；固定.01正弦控制扰动产生最大.00309658m路径变化；运动学+DPP对控制的绝对梯度和1.5090251且有限；DSF最后层梯度正，zero-last的第一步上游零梯度如实预期，旧模型全部权重/boolbuffer精确不变且grad none。无optimizer更新/训练/新科学结果。295474冻结原模型+173440新sampler参数，共468914，不把173440称整模/通过原305000目标。真实完整CPU/GPU成本仍需生产前测。

这说明**目前无需新建CVAE预训练**。确定性latent没有N(0,I)先验，不报告originalGaussian-radius臂；可支持uniform/kin/scene/joint权重，但新的匹配实验应明确为现有生成器上的DSF式适配而非原CVAE论文逐字复现。原slot专属anchor_templates和context scale均固定，相同latent不代表相同语义模式，控制可达域仍受tanh/锚结构限制；有梯度不保证覆盖收益或物理安全。正式训练前仍要本新方案完整paired/phased/resume/保存/真实地图与资源前检，不把8窗接口测试说成全部通过。

## 先前CVAE备用路线（保留来源，当前不启动）

论文要求预训练cVAE解码器和真正的Gaussian prior；目前F16+h3不是该接口。已查作者论文/主页未发现可直接用于本项目C0行人8→12步的DSF checkpoint。作者DLow公开权重为Human3.6M/HumanEva人体姿态，不可拿来替代；DIVA是nuScenes车辆及地图，也不能换任务冒充本轮。

- 原论文：https://arxiv.org/html/1907.04967 （Appendix B的synthetic为2past→3future、latent2；human motion为59D/30step、latent8，都不是本C0 8→12的现成权重）
- 作者DLow说明：https://raw.githubusercontent.com/Khrylx/DLow/master/README.md
- DIVA说明：https://raw.githubusercontent.com/lcalem/DIVA/main/README.md

如果现有生成器适配未来被实证为不适用，用户已条件性允许另训CVAE；届时需先确定独立FIT-only预训练合同、架构/归一化/预算与全部原前检，不能用人体姿态或车辆权重换任务。`generator_contract.template.json`和Gaussian接口保留为备用，不再成为当前非高斯适配的人为阻碍。

## ARS同助手反证（非外审）

1. DA1：接口能反传不等生成器已预训练；官方有语义图不等同当前米制/类别证据已齐。
2. DA2：正例包括已真实找到全部7recording对应的作者地图资产；反例uni全1、多个同图、缺真实CVAEcheckpoint都保留，不退回“完全没有地图”的旧结论，也不伪造准入。
3. DA3：工程测试不是真实B0–B4预检/训练/效果结果。原StageA保持独立，不替换候选或宣称覆盖/安全收益，不新增seed/调阈值求PASS。

当前状态：代码接口已实现并测试，真实资产部分配置；**尚不能声称两个生产接口全部就绪或启动B0–B4**。缺项及下一步见SETUP_STATUS.json。
