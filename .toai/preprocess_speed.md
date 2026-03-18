# Preprocess 加速需求记录

## 1. 需求背景
当前项目在 preprocess 阶段默认使用 `RoMa` 做 pairwise correspondence，整体耗时很高，已经成为 prior-pose pipeline 的主要瓶颈之一。

新的目标是：

1. 尝试在当前 MBA 项目中接入“加速版 MASt3R”来替换或补充当前 preprocess 的 correspondence backend。
2. 保持后续 BA 主流程不变，优先在 preprocess 阶段完成加速。
3. 尽量复用已有的 `inference_pairwise -> intermediate -> hdf5` 数据协议，降低侵入性。

参考代码：

1. 当前 prior-pose 全流程入口：`experiments/custom/priorpose_pipeline.py`
2. 当前 preprocess 对应点入口：`MargBA/corres_estimator/corre_utils.py`
3. 当前项目内置 MASt3R backend：`MargBA/corres_estimator/mast3r.py`
4. 参考的加速版 MASt3R BA 项目入口：`Mast3rBA/mast3r_ba_pipeline_v1.py`

---

## 2. 当前项目 preprocess 的真实瓶颈

### 2.1 pair 数量本身是 O(N^2)
当前 custom 数据集会直接枚举所有 `i < j` 图像对，因此 pair 数随着帧数平方增长。

结论：

1. 即使 matcher 单对速度更快，只要 pair 策略不变，总时长仍然会被 `O(N^2)` 拖住。
2. 因此“换成加速版 MASt3R”只能解决 preprocess 的一部分瓶颈，不会一次性解决全部耗时问题。

### 2.2 当前 preprocess 的集成边界
当前 preprocess 的核心不是“必须使用 RoMa”，而是“必须产出当前 MBA 能消费的 correspondence/visibility/HDF5 格式”。

现有流程为：

1. `priorpose_pipeline.py` 调 `inference_pairwise(...)`
2. `inference_pairwise(...)` 调 `inference_pairwise_per_gpu(...)`
3. 每个 GPU 上实例化对应 matcher（`RoMa` 或 `MASt3R`）
4. matcher 输出：
   1. `corres_src_dst`
   2. `corres_dst_src`
   3. `certainty_src_dst`
   4. `certainty_dst_src`
   5. `visibility`
   6. `valid_src_dst`
   7. `valid_dst_src`
5. 最终写入：
   1. `intermediate/corres_i2j/*`
   2. `intermediate/visibility_i2j/*`
   3. 合并进 hdf5 的 `corres_i2j/visibility_i2j`

这意味着：

1. BA 本身并不关心 correspondence 是 RoMa 还是 MASt3R 产生的。
2. 只要我们能把“加速版 MASt3R 的 matches”转换成当前 `warp + certainty + visibility` 协议，就可以无缝接到现有 BA。

---

## 3. 当前项目内置 MASt3R 为什么偏慢
当前项目内置的 `MargBA/corres_estimator/mast3r.py` 不是简单的单次 descriptor NN matching，而是更重的一条链：

1. 先做 coarse matching
2. 再基于 coarse matches 选 crops
3. 再做 fine matching
4. 对 A->B 和 B->A 各跑一遍
5. 再把结果 rasterize 成当前项目需要的 dense-ish warp/certainty

这条链的优点是结果更完整、更接近 dense warp；
缺点是每个 pair 的推理和后处理都更重。

---

## 4. 参考的加速版 MASt3R 在做什么
`Mast3rBA/mast3r_ba_pipeline_v1.py` 中的加速版实现采用的是更轻的一条路径：

1. 先将图像 resize 到较小分辨率（默认最长边 512）
2. 单次前向得到 `pred1/pred2`
3. 直接在 descriptor 上做快速 NN matching
4. 在规则子采样网格上提 query 点，而不是追求全图 dense 匹配
5. 用 confidence 做筛选

这条路径的特点：

1. 每个 pair 的前向和匹配逻辑更简单
2. 很可能比当前项目内置 MASt3R 更快
3. 输出更偏 sparse / semi-dense matches，而不是当前 MASt3R 这种经 coarse-to-fine 整理后的结果

---

## 5. 可行性分析

### 5.1 结论
结论：可行，而且推荐尝试。

原因：

1. 当前系统的集成边界很清晰，真正需要兼容的是 preprocess 输出格式，不是 BA 主体。
2. 参考项目已经在本仓库内提供了可用的加速版 MASt3R 源码目录：`Mast3rBA/mast3r/`
3. 当前项目的 `MASt3R.format_match_confidences(...)` 已经提供了“把 sparse matches 写回 warp/certainty 图”的现成思路，可以直接复用或照着改。

### 5.2 最小侵入接法
推荐方案不是直接替换现有 `MASt3R`，而是新增一个 backend，例如：

1. `MASt3RFast`
2. 或 `MASt3RAccel`

接入方式：

1. 在 `MargBA/corres_estimator/` 下新增一个适配器文件
2. 保持 `inference_pairwise(...)` 和 HDF5 写盘逻辑不变
3. 只在 `init_corres_model(...)` 中新增一个可选后端
4. 在 `priorpose_pipeline.py` / `preprocess.py` 的 `--corres-model` 参数中暴露这个新选项

这样做的好处：

1. 不影响现有 `RoMa`
2. 不影响现有慢版 `MASt3R`
3. 方便 A/B test
4. 一旦质量不够，可以快速回退

---

## 6. 关键适配点

### 6.1 输出格式适配
加速版 MASt3R 当前直接输出的是：

1. `matches_im0`
2. `matches_im1`
3. `conf`

而当前 MBA preprocess 期望的是：

1. `warp`，shape 为 `[H, W, 4]`
2. `certainty`，shape 为 `[1, H, W]`
3. 双向输出
4. `visibility`
5. `valid mask`

因此需要新增一个适配层，把 sparse matches rasterize 回当前协议。

这件事本身是可做的，因为当前项目里已经有类似逻辑：

1. 把 query 像素位置写回 source 平面
2. 把对应目标坐标写到 `warp[..., 2:4]`
3. 把置信度写到 `certainty`

### 6.2 坐标恢复到原图
参考项目里的 `prepare_image(...)` 会先把图片 resize 到较小分辨率，因此匹配坐标默认处在 resize 后坐标系。

接入当前 MBA 时，必须补一层坐标恢复：

1. 从 resize 后分辨率恢复到原图分辨率
2. 再转换到当前项目保存 correspondence 时使用的 `grid_sample` 坐标系 `[-1, 1]`

### 6.3 双向匹配
当前 MBA 的 preprocess 输出要求保存 `src->dst` 和 `dst->src` 两个方向。

参考加速版代码虽然主要展示了一个方向的匹配提取，但同一次模型前向其实已经拿到了 `pred1/pred2`，理论上可以：

1. 从 `pred1 -> pred2` 提一次
2. 再从 `pred2 -> pred1` 提一次

这样可以避免像当前慢版 MASt3R 一样，再额外做一次完整反向推理。

### 6.4 visibility 定义
当前项目里的 visibility 实际是“有效 correspondence 占图像面积的比例”。

如果改成加速版 MASt3R，由于匹配会更稀疏，visibility 的数值分布大概率会和 RoMa / 当前慢版 MASt3R 明显不同。

这意味着：

1. `min_visibility` 阈值大概率需要重新调
2. `min_confidence` 也需要重新调
3. 不能直接沿用当前默认值并假设质量稳定

---

## 7. 风险与不确定性

### 7.1 模块命名冲突
当前仓库已经有 `third_party/mast3r`，参考项目又带了一份 `Mast3rBA/mast3r`，两边都暴露顶层包名 `mast3r` / `dust3r`。

风险：

1. 如果直接全局改 `sys.path`，很容易导入到错误版本
2. 两套实现混用时容易出现隐蔽的运行时问题

建议：

1. 新增 backend 时做局部 lazy import
2. 显式把 `Mast3rBA/mast3r` 放到最前
3. 不要在模块顶层长期污染全局 import 路径

### 7.2 质量可能下降
加速版的本质是“更轻、更 sparse 的 matching”，速度更快通常意味着约束更少。

可能带来的问题：

1. correspondence 更稀疏
2. 边缘区域覆盖率下降
3. visibility 偏低
4. BA 收敛性不如 RoMa 或慢版 MASt3R

所以需要把“是否更快”和“是否还能支撑 BA”分开验证。

### 7.3 总耗时未必线性下降
即使单 pair matcher 更快，以下开销仍然存在：

1. `O(N^2)` 图像对枚举
2. 中间 PNG/TXT 落盘
3. HDF5 合并
4. 可视化与 I/O

因此预期应该是：

1. 单 pair 速度会提升
2. 总 preprocess 速度会提升
3. 但总提升倍数不一定和 matcher 单对提升倍数一致

### 7.4 环境依赖差异
参考项目里 MASt3R 是从 `Mast3rBA/mast3r` 目录手动加入 `sys.path` 后再 import。

这说明：

1. 它依赖的是那份 vendored MASt3R / DUSt3R 实现
2. 不一定和当前 `third_party/mast3r` 完全兼容

这件事不是 blocker，但会影响实现方式。

---

## 8. 推荐实施方案

### Phase 1: 做一个并行存在的新 backend
目标：先不替换 RoMa，不替换当前慢版 MASt3R，只新增：

1. `MASt3RFast` backend
2. CLI 中新增 `--corres-model MASt3RFast`

验收：

1. 能跑通 preprocess
2. 能产出当前协议要求的 `corres_i2j/visibility_i2j`
3. 能被 `priorpose_pipeline.py` 正常消费

### Phase 2: 小场景 benchmark
目标：

1. 对比 `RoMa`
2. 对比当前慢版 `MASt3R`
3. 记录 preprocess 用时、有效 pair 数、平均 visibility、BA 是否收敛

重点不要只看速度，还要看：

1. HDF5 中 pair 数有没有明显掉太多
2. coarse/fine BA 是否还能正常下降

### Phase 3: 调参
优先要调的参数：

1. `subsample`
2. `conf_threshold`
3. `min_confidence`
4. `min_visibility`

目标是在速度和 BA 稳定性之间找一个平衡点。

### Phase 4: 若仍慢，再做 pair pruning
如果换成加速版 MASt3R 后 preprocess 仍明显偏慢，那么下一阶段最有效的抓手不是继续抠 matcher，而是直接减少 pair 数，例如：

1. 时序窗口
2. 基于先验 pose 的候选边筛选
3. 只保留几何上更可能重叠的 pairs

---

## 9. 当前判断

### 9.1 是否值得做
值得做。

### 9.2 是否能低风险接入
能，前提是：

1. 以“新增 backend”的方式做
2. 不直接破坏当前 `RoMa` / `MASt3R`
3. 保持 HDF5 协议和 BA 接口不变

### 9.3 是否保证一定比当前默认 RoMa 快
不能在未实测前保证。

更准确的表述应该是：

1. 它很可能比当前项目里的慢版 `MASt3R` 更快
2. 它也有机会比当前默认 `RoMa` 的 preprocess 更快
3. 但总加速幅度取决于 pair 数、I/O 和阈值配置，必须实测确认

---

## 10. 当前进度

### 已完成
1. 已梳理当前项目 preprocess 的真实调用链。
2. 已确认当前 BA 不依赖具体 matcher，只依赖 correspondence/visibility/HDF5 协议。
3. 已定位当前项目慢版 `MASt3R` 的主要耗时结构：coarse + crop + fine + 双向流程。
4. 已定位参考项目加速版 MASt3R 的入口和核心匹配函数。
5. 已完成可行性分析，结论为“可行，建议新增 backend 方式接入”。
6. 已新增 `MASt3RFast` backend 代码接入，位置为 `MargBA/corres_estimator/mast3r_fast.py`。
7. 已将 `init_corres_model(...)` 扩展为支持 `MASt3RFast`。
8. 已将 `corres_estimator/__init__.py` 改为惰性导入，降低 `mast3r` / `dust3r` 包名冲突风险。
9. 已在 `experiments/custom/priorpose_pipeline.py` 中开放 `--corres-model MASt3RFast`。
10. 已修复 `experiments/custom/preprocess.py` 里 `--corres-model` 参数存在但内部仍硬编码 `RoMa` 的问题。
11. 已同步放开其他实验脚本中的 `corres-model` 枚举，并修复多份 `preprocess.py` 内部硬编码 `RoMa` 的问题。

### 待验证
1. 用户侧运行 `MASt3RFast` preprocess，确认能成功导出 `corres_i2j/visibility_i2j` 和 hdf5。
2. 用户侧验证与 `RoMa` / 慢版 `MASt3R` 相比的 preprocess 用时变化。
3. 用户侧验证 `MASt3RFast` 产出的 correspondence 是否足以支撑 coarse/fine BA 正常收敛。

### 下一步建议
1. 用一个小场景先跑 `priorpose_pipeline.py --corres-model MASt3RFast`。
2. 如果 pair 数保留过少或 visibility 偏低，优先调：
   1. `min-confidence`
   2. `min-visibility`
   3. `mast3r_fast.py` 内部的 `subsample`
   4. `match_conf_thr`
3. 如果速度仍不够，再进入 pair pruning 阶段。
