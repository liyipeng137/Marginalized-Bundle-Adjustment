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
