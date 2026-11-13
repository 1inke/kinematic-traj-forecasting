# 坐标系说明（过夜可视化）

## 当前帧：`world_display_H_aligned_v2`（2026-10-19）

网页轨迹图使用 **世界米制 + 按场景、由单应 H 验证过的显示变换**。  
**训练数据 / ADE / FDE 未改**；只改展示坐标，方便对照视频。

| 场景 | 显示坐标 | 视频含义（事实） |
|---|---|---|
| **students003** / **zara01** / **zara02** | `(x, y)` | 世界 **+x → 画面右**，世界 **+y → 画面上** |
| **hotel** / **eth** | `(y, −x)` | 世界 **+y → 画面右**，世界 **+x → 画面下**（故 −x → 画面上） |

轴标签会写当前样本的场景名与变换。

## 场景 id（与 `process_eth.data_to_idx` 一致）

| id | 场景 |
|---|---|
| 0 | eth |
| 1 | hotel |
| 2 | students003 |
| 3 | zara01 |
| 4 | zara02 |

来源：`eth_data_train.npy` 通道 0；本 overnight（eth leave-one-out 训练缓存）含 hotel/students/zara，无 eth 样本。

## 世界坐标是什么

ASTRA 处理后的轨迹最后两维是标签文件中的世界平面坐标（米）：

- 标签列：BIWI/UCY 文本 `parts[13]=x`, `parts[15]=y`（见 `ASTRA/data/eth.py` 读入 `[[13,15]]`）。
- 文件：
  - eth: `datasets/eth_ucy/labels/eth/biwi_eth.txt`
  - hotel: `datasets/eth_ucy/labels/hotel/biwi_hotel.txt`
  - students003: `datasets/eth_ucy/labels/univ/students003.txt`
  - zara01/02: `labels/zara01/crowds_zara01.txt`, `labels/zara02/crowds_zara02.txt`

## H：世界 ↔ 像素

本地 H 现置于 `ASTRA/datasets/eth_ucy/homography/H_{scene}.txt`（此前目录为空，已从公开源补齐）。

| 场景 | H 文件来源 | H 含义 |
|---|---|---|
| eth / hotel | GUIDE-CoT `data/eth5/{eth,hotel}/H.txt`（与 EWAP BIWI `H.txt` 同值） | **像素 → 世界** |
| students003 | GUIDE-CoT `llm_module/datasets/homography/students003_H.txt` | 像素 → 世界（近对角，`h22<0`） |
| zara01 / zara02 | 同上目录 `crowds_zara0{1,2}_H.txt` | 像素 → 世界 |

投影代码与 ASTRA 一致：`utils/visualizer.world_to_pixel`  
`pixel_h = world_h @ inv(H.T)`，再归一化；并对 **非 eth/hotel** 的 UCY 结果把 `(col,row)` 换成 `(row,col)`。

### 由 H 推出的轴向（单位米步进 → 像素增量）

在典型原点处，ASTRA 投影：

| 场景 | 世界 +1m x | 世界 +1m y |
|---|---|---|
| **hotel** | ≈ +56 row（下）, +9 col | ≈ −3 row, **+52 col（右）** |
| **eth** | ≈ +21 row（下）, +1 col | ≈ −2 row, **+23 col（右）** |
| **students003** | ≈ 0 row, **+45 col（右）** | ≈ **−40 row（上）**, 0 col |
| **zara01 / zara02** | 同 students 族：+x→右，+y→上 | 同左 |

因此「视频同向」展示映射的最优离散选择：

- hotel/eth → **`display = (world_y, −world_x)`**（相关 ~0.99）
- students/zara → **`display = (world_x, world_y)`**（相关 1.0）
- **仅翻 y `(x,−y)` 对 hotel 相关 ~0.05，不能纠正水平/竖直错位**（此前错误补丁）。

## 视频叠图验证

1. **hotel / 样本 740 / 行人 259**  
   - 世界：x≈3.45 几乎不变，y: 2.39 → −9.10（Δy≈−11.5 m）  
   - H 投影：Δrow≈0，Δcol≈−588 → **画面水平向左**  
   - 叠图：`hotel_Hproj_f1086.jpg` 等；红圈落在人上；蓝箭头 +x 朝下、橙箭头 +y 朝右。  
   - 显示变换后轨迹沿 **display x** 变化 → 与视频左右走一致。

2. **students003 / 行人 102**（对照）  
   - 世界 Δx>0、Δy<0 → 投影 Δcol>0、Δrow>0（右下）  
   - 叠图：`stu_Hproj_f160.jpg`；+x 右、+y 上；红圈落在白衣行人。  
   - 显示保持 `(x,y)`。

## ASTRA 相关代码线索

- `process_eth.py`：加载 `homography/H_{name}.txt`；注释掉的「Flip x and y」仅针对非 eth/hotel（世界轴对调），说明 eth/hotel 与 UCY 轴约定不同。  
- `visualizer.world_to_pixel`：明确 UCY H 输出为 `(col,row)`，eth/hotel 为 `(row,col)`。

## 数据从哪来（本页）

`build_overnight_visual.py` 把局部坐标候选逆变换回世界米；`patch_display_axes_v2.py` 再按上表做 **仅显示** 变换。  
`common.js`：`scenes` / `scene_ids` / `display_code`（0=恒等，1=`(y,−x)`）。

## 已废弃

1. 历史朝向局部框（不能对照视频）。  
2. 全场景一律「世界 = 视频同向 +y 上」。  
3. **仅对 hotel/eth 做 `(x,−y)`**（2026-10-19 早先错误尝试）：ped259 仍竖直，与视频左右走不符。
