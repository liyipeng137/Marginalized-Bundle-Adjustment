# Prior Pose + LiDAR Depth 接入 MBA 计划

## 1. 目标与已确认前提
1. 目标：在现有 MBA 项目中，使用一组全帧先验 pose（来自 COLMAP）和对齐的 LiDAR GT depth，对同一组图像执行 BA 优化。
2. 先验 pose 用法：用于写入 pr_pose_model_init.ckpt 初值，在 BA 中继续被优化。
3. 深度条件：每张 RGB 都有同名、同分辨率、单位为米，格式为uint16的PNG的 LiDAR depth，且尺度与 pose 一致。
4. 内参来源：使用 COLMAP 相机内参。
5. pose 来源：COLMAP 模型（可参考 `MargBA/utils/colmap_read_model.py` 或 `experiments/scannet/colmap.py` 的读取方式）。

## 2. 当前代码现状与关键结论
1. 项目支持 `pr_pose_model_init.ckpt` 作为 BA 初值，后续 coarse/fine BA 会继续优化 pose，不会锁死初始化值。
2. 当前 custom 流程默认用 twoview 初始化，不直接读外部先验 pose。
3. BA 采样器读取内参时使用 `intrinsic_gt`；如果 custom hdf5 里没有该组，流程会不完整。
4. BA 采样器对深度读取逻辑是按 `marker_query_map` 分支：`qry -> depth_pr`，`map -> depth_gt`。custom 默认全 `qry`，因此默认不会用 `depth_gt`。

## 3. 实施策略（最小改动版）
1. 保持现有 MBA loss 与优化器不变，不引入 pose-prior loss。
2. 新增“先验初始化入口”，替代 custom 中的 `twoview_sfm_initialization`。
3. 新增“全帧使用 GT depth”的开关，避免 custom 全 `qry` 时仍读取 `depth_pr`。
4. 在预处理后补全 hdf5 所需字段，使 BA 可直接读取 COLMAP 内参与 LiDAR 深度。

### 3.1 【用户补充】
1. 关于预处理的流程，需要把"稠密匹配"这一操作提出来，所以我们的新pipeline脚本需要是全流程的，完成预处理/prior pose加载/BA的全流程

## 4. 分阶段执行计划

### Phase A: 数据契约与映射（硬检查）
1. 明确三方文件映射关系：RGB 文件名、COLMAP image name、LiDAR depth 文件名必须一一对应。
2. 统一索引规则：按项目内 `000000, 000001, ...` 的顺序建立 name-to-index 映射，确保 pose/intrinsic/depth 写入同一索引。
3. 明确坐标约定：使用 COLMAP 的 world-to-camera (`w2c`) 作为初始化 pose。
4. 新增前置检查脚本输出：
   1. 匹配成功帧数、缺失帧列表、重复名列表。
   2. 若任一缺失，直接 fail-fast，不进入 BA。

### Phase B: 预处理与 HDF5 产物
1. 从 RGB 运行稠密匹配，生成 `corres_i2j` 与 `visibility_i2j`（保留当前 RoMa/MASt3R 路径）。
1. 写入 `intrinsic_gt`：从 COLMAP camera 参数生成每帧 3x3 内参，并写入 hdf5。
2. 写入/确认 `depth_gt`：将 LiDAR depth 以项目当前深度编码方式写入 hdf5。
3. 保留 `depth_pr`（可选，初版我们暂不需要depth_pr）：
   1. 若需要兼容现有代码路径可保留[初版我们暂不需要]。
   2. 但 BA 实跑将通过新开关优先读取 `depth_gt`。
4. 该阶段结束时，hdf5 至少包含：
   1. `rgb`
   2. `corres_i2j`
   3. `visibility_i2j`
   4. `intrinsic_gt`
   5. `depth_gt`

### Phase C: 先验 pose 初始化
1. 读取 COLMAP pose，按 hdf5 图像索引对齐到 `nfrm`。
2. 构建 `GlobalOptimizationPoseParameters` 并逐帧写入 `newpose`。
3. 保存为 `pr_pose_model_init.ckpt`，作为 BA coarse stage 输入。
4. `gt_pose_model.ckpt` 在本任务中非必须，可按评估需求决定是否写入占位。

### Phase D: BA 深度读取开关
1. 在采样器增加参数，例如 `depth_source in {pr, gt, mixed}`。[初版我们暂时只提供depth_gt]
2. 本任务默认设为 `gt`，使全帧都读取 `depth_gt`。
3. 兼容原有逻辑：若未指定则维持现有行为，避免影响其他实验脚本。

### Phase E: 新全流程入口（单脚本）
1. 新增 prior-pose 版 custom 入口脚本（覆盖预处理 + pose初始化 + BA 全流程）。
2. 关键参数：
   1. `--data-root`：RGB 根目录。
   2. `--depth-root`：LiDAR depth 根目录。
   3. `--prior-pose-colmap-model`：COLMAP 模型目录或文件路径。
   4. `--corres-model`：`RoMa|MASt3R`。
   5. `--use-gt-depth`：启用 GT depth 采样（默认 true）。
   6. `--calibrated`：建议开启，固定 COLMAP 内参初值策略。
   7. `--skip-preprocess`：若已有 hdf5 的 `corres_i2j/visibility_i2j` 则可跳过匹配。
3. 产出保持与现有一致：`pr_pose_model_coarse.ckpt`、`pr_pose_model_fine.ckpt`、可视化结果。

## 5. 验证与验收（实现侧给出可读日志）
1. 静态校验：
   1. hdf5 中 `rgb/depth_gt/intrinsic_gt/corres_i2j/visibility_i2j` 完整。
   2. pose 帧数与图像帧数一致，无缺帧。
2. 几何校验：
   1. 随机抽样若干对图像，检查初始化 pose 的重投影残差分布是否在合理范围。
   2. coarse 后 loss 下降，fine 后进一步下降或稳定。
3. 结果校验：
   1. 对比初始化与 BA 后 pose 的变化幅度。
   2. 导出点云/轨迹进行可视化 sanity check。

## 6. 风险与回退
1. 风险：文件名映射不一致导致 pose/depth 错帧（高优先级）。
2. 风险：COLMAP 相机模型参数到 3x3 内参转换不一致（不同 camera model）。
3. 风险：少量帧缺深度或深度无效值比例过高，影响采样稳定性。
4. 回退策略：
   1. 先在小子集场景跑通完整链路。
   2. 保留原 custom 流程不变，新功能走独立参数开关或独立脚本。

## 7. 执行边界
1. 本计划阶段不修改 MBA 核心损失定义，不新增 pose 正则项。
2. 若后续你希望“先验 pose 软约束”，再追加第二阶段方案（新增 prior loss 与权重调度）。

## 8. 立刻开工顺序（本轮实施）
1. 第一步：实现数据契约检查与 name/index 对齐工具（先做 fail-fast，避免后续错帧）。
2. 第二步：实现 COLMAP 内参/pose + LiDAR depth 写入 hdf5 的增量脚本。
3. 第三步：修改 BA 采样器，支持全帧 `depth_gt` 模式。
4. 第四步：新增 prior-pose 全流程入口脚本，串起匹配、初始化、BA。
5. 第五步：在一个小规模样例上跑通，产出日志与 ckpt 验收。

## 9. 当前进度（2026-03-16）
1. 已完成：Phase D（BA `depth_source` 已支持 `gt/pr/marker/mixed`，当前默认按配置走 `gt`）。
2. 已完成：Phase E（新增 `experiments/custom/priorpose_pipeline.py`，已串起预处理+初始化+BA）。
3. 已完成：输入源扩展，`priorpose_pipeline` 已支持 `--prior-pose-colmap-model` 和 `--input-transforms-json` 二选一。
4. 已完成：输出源扩展，BA 结束后自动导出 `optimized_transforms.json`（包含优化后外参 + `adjust_intrinsic` 后内参）。
5. 进行中：远程环境端到端验证（由用户执行），重点检查坐标系和文件映射一致性。

## 10. 预处理（默认 RoMA）在做什么、为了产出什么
1. 目标：为 BA 构建“跨帧2D稠密对应 + 可见性权重”数据，供后续随机采样和重投影残差优化使用。
2. 入口：`priorpose_pipeline.py` 在预处理阶段调用 `inference_pairwise(...)`，默认对应模型可选 `RoMa|MASt3R`。
3. RoMA 对每个图像对的核心计算：
   1. 双向匹配：计算 `src->dst` 与 `dst->src` 两个方向的 dense warp + certainty。
   2. 置信度筛选：低于 `min_confidence` 的匹配置零。
   3. 前后向一致性检查（cyclic check）：过滤几何不一致匹配。
   4. 可见性计算：根据双向有效像素占比得到 pair 级 `visibility`。
   5. pair 过滤：若 `visibility < min_visibility` 或有效匹配为空，则该 pair 不写入产物。
4. 中间产物（磁盘）：
   1. `intermediate/corres_i2j/<pair>/`：每个方向写三张 png（`x/y/conf`）。
   2. `intermediate/visibility_i2j/<pair>.txt`：pair 可见性。
   3. 若数据集提供 GT 相对位姿，也会写 `intermediate/pose_i2j_gt/`（custom 常规无此项）。
5. 合并后产物（HDF5）：
   1. `corres_i2j`：用于 BA 采样对应点。
   2. `visibility_i2j`：用于连接权重和子图构建。
   3. priorpose 流程另外保留 `rgb/depth_gt/depth_pr/intrinsic_gt/pose_w2c_gt`，一起组成 BA 输入。
6. 与 depth-model 的关系说明：
   1. 默认 custom 原流程里，`depth-model` 负责生成 `depth_pr`，会被 BA 使用。
   2. 当前 priorpose 流程里，我们直接写入 LiDAR depth 到 `depth_gt/depth_pr`，预处理阶段不再依赖 `inference_unary`，因此 `depth-model` 主要影响输出目录命名与实验隔离。
7. 计算复杂度与耗时来源（加速重点）：
   1. pair 数量近似 `O(N^2)`，这是最主要开销来源。
   2. 每个 pair 需要双向推理与一致性检查，GPU 时间与显存占用高。
   3. 大量 PNG/TXT 落盘和再合并到 HDF5，I/O 开销显著。
8. 后续预处理加速抓手（按优先级）：
   1. 减少 pair 数：只保留时序邻近窗口或基于先验位姿筛选候选边，避免全连接。
   2. 复用已有结果：支持 `--skip-preprocess`，并保证 scene 命名一致，避免重复推理。
   3. 降低匹配分辨率或提高阈值：调 `min_confidence/min_visibility`，减少无效 pair 写盘。
   4. 关闭/延后可视化：`vls` 仅做抽样调试，正式批跑可禁用。
   5. I/O 优化：减少中间文件落地次数，或改为直接写入 HDF5（需要额外改造）。

## 11. `priorpose_pipeline.py` 当前流程分阶段梳理
1. 参数与路径规范化阶段：
   1. 解析 CLI 参数，检查 GPU 可用性。
   2. 对 `data_root/depth_root` 做 `normpath`，统一 scene 命名，避免路径尾 `/` 导致的错名 hdf5。
2. 先验输入解析阶段：
   1. 二选一读取先验：`COLMAP model` 或 `input_transforms.json`。
   2. 将每帧 RGB、depth、K、w2c 对齐成 payload。
3. 数据契约检查阶段：
   1. 检查 RGB/先验/depth 一一对应。
   2. 若缺失或不一致，fail-fast 退出。
4. HDF5 准备阶段：
   1. 写入 `rgb/depth_gt/depth_pr/intrinsic_gt/pose_w2c_gt`。
   2. 写入 `mapper.txt`（索引到文件名映射）。
5. 稠密匹配预处理阶段（可跳过）：
   1. 调 `inference_pairwise` 生成 `corres_i2j/visibility_i2j`。
   2. `--skip-preprocess` 时复用已有 hdf5 对应组。
6. 初始化位姿模型阶段：
   1. 用 payload 中先验位姿构建并保存 `pr_pose_model_init.ckpt`。
7. BA 优化阶段：
   1. 读取 hdf5 与初始化 ckpt。
   2. 执行 coarse + fine 两阶段 BA，输出 `pr_pose_model_coarse.ckpt` 与 `pr_pose_model_fine.ckpt`。
8. 结果导出阶段：
   1. 加载 fine ckpt。
   2. 导出优化后的 `transforms.json`（含优化后外参与 `adjust_intrinsic` 后内参）。

## 12. coarse 和 fine 两阶段BA的目的
   1. coarse：先把“全局几何结构”拉到合理位置
      1. 用更鲁棒的 cdf_log_subgraph（对大残差不敏感，先稳住整体）
      2. 用按节点度数组织的 pair 分配，强调图连通和全局一致性
      3. 内参相关学习率更激进（lr_intrinsic_boost=50）
      4. 从 pr_pose_model_init.ckpt 开始优化
      参考：
         priorpose_pipeline.py (line 550)
         priorpose_pipeline.py (line 572)
         bundle_adjustment.py (line 47)
   2. fine：在 coarse 基础上做“精细收敛”
      1. 换成 cdf_log + cdf_euclidean，先稳再追像素级精度
      2. pair 分配更接近全量均匀覆盖
      3. 内参学习率降低（lr_intrinsic_boost=10）
      4. 从 pr_pose_model_coarse.ckpt 接着优化
      参考：
         priorpose_pipeline.py (line 579)
         priorpose_pipeline.py (line 601)
         bundle_adjustment.py (line 53)
   总结：coarse 解决“别跑飞、先对齐”，fine 解决“再抠细节、压低残差”。