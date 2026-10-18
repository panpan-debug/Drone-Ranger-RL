# Drone-Ranger-RL / DQN 分支项目交接文档

- **生成日期：** 2026-10-18
- **交接目标：** 将当前云端项目上下文完整迁移到 Kimi Work 桌面端本地项目，便于在不共享会话数据的环境中继续开发、训练和复盘。
- **项目仓库：** `RipZou/Drone-Ranger-RL`，当前关注其中的 DQN 分支及 `dqn_2d_standalone` 纯二维独立实验。
- **当前主线结论：** 无障碍纯导航已经收敛并验证为 100% 到达率；当前未解决问题集中在 **3 障碍物避障**，主要表现为障碍物前卡住、左右方向决策不稳定，以及少量近目标绕圈。

> **资料范围说明**
>
> 本文档依据当前会话可见材料整理：6 张项目截图、1 份终端实验记录、3 个已生成 Python 文件，以及这些文件中记录的实验历史。由于云端与本地项目数据不互通，本文无法直接读取你本地仓库中的 checkpoint、训练日志和未上传脚本；凡涉及本地文件存在性或训练状态的内容，均已标注“需本地确认”。

---

## 0. 一页速览

| 项目状态 | 结论 |
|---|---|
| 已解决 | `num_obstacles=0` 的纯导航任务；`nav100_final.pt` 在 200 个评估回合中 200/200 到达。 |
| 当前主攻 | `num_obstacles=3` 的课程式避障微调。 |
| 主要失败模式 | 障碍物前 stuck、不知道左转还是右转；另有少量 final-stage orbiting。 |
| 已定技术路线 | 先用 ground-truth 坐标的 2D standalone 环境快速迭代，再考虑回到 Isaac Lab；暂不恢复模拟传感器。 |
| 已定迁移方式 | 从无障碍 checkpoint 迁移到 3 障碍物网络：扩展 `fc1.weight`，新增障碍物输入列置零，保留原导航策略。 |
| 下一组核心实验 | 保持同一任务、同一起点、同样 3 个障碍物，只增加 `living_cost=0.02`，与既有 3 障碍物基线做严格对照。 |
| 当前最大阻塞 | 云端看不到本地 checkpoint、完整训练脚本和最新训练日志；需要先完成本地文件合并与状态确认。 |

---

## 1. 项目背景与目标

### 1.1 项目定位

`Drone-Ranger-RL` 的 DQN 分支用于训练一个基于 DQN 的无人机导航与避障策略。该分支是主 PPO 管线之外的实验路线；主管线另包含 CNN 深度图输入和 `SimpleObs` ground-truth-vector 变体。

仓库截图中的 README 将项目描述为：

> “A DQN-based obstacle-avoidance navigation policy for a simulated quadrotor in NVIDIA Isaac Lab, built as an experimental branch alongside this project's main PPO-based pipeline.”

项目动机来自 LT-DQN 论文提出的三层混合 UAV 导航框架：

1. 学习得到的 DQN 控制器负责常规开阔空间飞行；
2. 在接近障碍物的 warning zone 中由 LiDAR + A* 规划器接管；
3. 在紧急近距离情况下由简单的反应式规则层处理。

当前 DQN 分支对应其中的 **Part 1**：先训练一个能够独立完成完整导航任务的 standalone DQN policy，而不是一开始就接入完整三层混合架构。

### 1.2 当前为什么使用 2D standalone 环境

当前迭代已经明确把模拟传感器从训练中移除，直接使用目标与障碍物的 ground-truth 坐标。这样做的目的：

- 避开 Isaac Lab 和模拟相机/传感器带来的工程复杂度；
- 让实验可以在无 GPU 的普通本地机器上运行；
- 把问题先缩小为“状态表示、动作空间、奖励函数、DQN 训练稳定性”本身；
- 快速复现、可视化和定位失败模式；
- 在 2D 环境中完成课程式训练后，再评估是否以及如何迁回 Isaac Lab。

团队原始说明中明确写道：

> “I removed the simulated sensors, now we are simply using coordinates of the obstacle and target for the training.”

以及：

> “You don't need to run the Issac Lab version, simply 2D Stand alone. You don't need a GPU for the training.”

### 1.3 当前环境定义

当前已生成的 `dqn_2d_env.py` 是一个纯二维、无 Isaac Lab 依赖的运动学环境：

- 控制周期：`DT = 0.02 s`，对应 50 Hz；
- 单回合长度：`20.0 s`，即最多 1000 step；
- 起点：无人机初始位置在原点附近 `[-1, 1] m` 范围采样；
- 目标点：在 `[-5, 5] m` 方形区域内采样；
- 到达阈值：`0.75 m`；
- 出界阈值：相对目标距离超过 `40.0 m`；
- 动作空间：3 个离散动作，语义为 `forward`、`turn_left`、`turn_right`；
- 状态：机体坐标系下的目标相对位置，以及最近可见障碍物的位置和半径；
- 障碍物：圆形，使用解析圆碰撞检测，不经过传感器仿真。

状态维度由配置派生：

```text
state_dim = 2 + 3 * min(k_nearest, num_obstacles)
```

其中：

- 前 2 维是 `[dx, dy]`；
- 每个暴露给网络的障碍物贡献 `[ox, oy, radius]`；
- `num_obstacles=0` 时状态就是 `[dx, dy]`；
- `num_obstacles=3, k_nearest=3` 时状态维度为 11；
- 整个状态除以 `state_scale=10.0` 做归一化。

不可见障碍物槽位使用常量 sentinel：

```text
(sense_range, 0, 0) / state_scale
```

其中 `radius=0` 表示“该槽位没有障碍物”。

### 1.4 课程式训练路线

当前采用分阶段课程：

1. **Stage 0：纯导航。** 不放障碍物，先让模型稳定学会“朝目标移动并到达”。
2. **Stage 1：3 障碍物。** 从已收敛的纯导航 checkpoint 迁移，不从头训练。
3. **后续阶段：** 在 3 障碍物结果稳定、失败模式被拆清之后，再逐步增加障碍物数量或提高场景复杂度。

障碍物生成规则已经实现：

- 默认 `num_obstacles=0`，3 障碍物实验通过 CLI 改为 3；
- 默认最多向网络暴露最近的 `k_nearest=3` 个障碍物；
- 障碍物半径 `0.5 m`，无人机半径 `0.2 m`；
- 起点和终点周围保留额外 `0.5 m` clearance，避免 episode 初始就无解；
- `first_obstacle_on_path=1` 时，障碍物 0 会被强制放在起点到目标连线的 25%–75% 区间，并带有限横向偏移，以确保避障能力确实被训练；
- 障碍物按“表面距离”排序，只暴露最近者；
- 表面距离超过 `sense_range=10.0 m` 的障碍物不可见。

### 1.5 最终要达成的结果

短期目标：

1. 在 `num_obstacles=3` 的 2D standalone 环境中得到稳定可复现的高到达率；
2. 把失败分布从“stuck/orbiting 为主”压低到可接受水平；
3. 对“左右方向决策”和“近目标 orbiting”分别给出机制解释和修复证据；
4. 形成一组干净、可对比的 ablation 结果，尤其是 `living_cost=0.02` 对 stuck 的影响。

中长期目标：

1. 完成从 3 障碍物到更多障碍物的课程式扩展；
2. 将 2D standalone 中验证有效的策略设计、状态设计和奖励设计迁回 Isaac Lab；
3. 最终形成能在仿真四旋翼中执行导航与避障任务的完整 DQN policy；
4. 后续再决定是否接入 LiDAR + A* warning-zone 规划器和紧急反应层，完成 LT-DQN 式混合架构。

---

## 2. 关键结论与已经定下来的取舍

### 2.1 已验证结论

#### 结论 1：纯导航任务已经解决

对 `checkpoints/nav100_final.pt` 做过 200 回合评估，终端原始结果为：

```text
[DIAG2D] Outcome distribution (n=200):
  reached       :  200 (100.0%)
  out_of_bounds :    0 (  0.0%)
  timed_out     :    0 (  0.0%)

[DIAG2D] mean_reward=107.61  mean_episode_length=237.7
```

结论：`num_obstacles=0` 下，当前 checkpoint 已满足“纯导航基线 100% 到达率”的要求。

#### 结论 2：replay buffer 大小是关键训练变量

团队已验证：默认 `replay_buffer_size=50000` 时行为很差；增大到 `5000000` 后纯导航模型达到 100% 成功率。

结论：后续复现纯导航或做 3 障碍物微调时，不能用 50000 作为默认实验口径；除非专门做 buffer-size ablation，否则正式实验应继续使用：

```bash
--replay_buffer_size=5000000
```

注意：`dqn_config.py` 当前 dataclass 默认值仍是 `50000`，这是配置默认值，不是推荐正式训练口径；正式命令应显式传入 5000000。

#### 结论 3：旧的加法式奖励存在可被独立刷分的漏洞

旧设计包含相互独立的 `rew_distance`、`rew_yaw_aligned` 和 `rew_action_rate` 等项。通过 `trace_2d.py` 诊断出两类可被策略利用的漏洞：

1. 面对目标但不动，也能单独获得航向对齐奖励；
2. 已经接近目标但不继续取得进展，也能获得接近奖励。

因此当前奖励已重构为单一乘性 progress shaping：

```text
rew_progress = vx_signed * yaw_cos * K_PROGRESS
K_PROGRESS = 0.05
```

其关键性质是：转向动作的 `vx_signed=0`，整个 progress 项归零；模型不能“只转向不动”还拿到对齐收益。当前还保留：

```text
reached: +100
crashed: -50
living_cost: 默认 0，可通过配置开启
```

#### 结论 4：`yaw_cos**3` 版本已经明确废弃

代码注释记录了一个已完成的反向实验：将 progress 中的 `yaw_cos` 改为 `yaw_cos**3` 后，到达率从 **29.0% 降到 8.0%**，轨迹整体变差。

解释：立方会压制整个 heading-error 范围内的奖励，而不只是压制近目标情况；例如 `yaw_cos=0.5` 时奖励从 0.5 降至 0.125，影响早期“学会往哪边转”的学习过程。

取舍：当前已回退到线性 `yaw_cos`。不要再把 `yaw_cos**3` 当作当前方案继续跑；若要重新尝试，必须作为新的隔离实验并说明变量。

#### 结论 5：纯导航模型存在左右转向判断不对称，但没有阻止无障碍任务收敛

`q_sweep.py` 对无障碍状态做了不同距离和 heading 下的 Q 值扫描：

| 距离 | heading > +5° 时选左转 | heading < -5° 时选右转 | 结论 |
|---:|---:|---:|---|
| 2 m | 83.3% | 38.9% | 明显不对称 |
| 6 m | 100.0% | 22.2% | 明显不对称 |
| 10 m | 77.8% | 22.2% | 明显不对称 |

结论：模型学到了足以完成无障碍导航的策略，但 left/right turn 的决策结构并不对称。这可能会迁移到避障阶段，尤其是在障碍物正前方需要选边绕行时。

#### 结论 6：3 障碍物阶段的核心问题不是“完全不会导航”，而是避障决策失败

已有 3 障碍物测试结果记录了以下失败模式：

```text
44.2% success
50.9% stuck in front for obstacle, don't know turn left or right
3.6% orbiting in final stage without getting closer
```

这说明从无障碍 checkpoint 迁移后，模型仍保留了相当部分导航能力，但遇到障碍物时无法稳定选择绕行方向。后续应把“方向决策”和“近目标 orbiting”拆成两个独立问题处理。

#### 结论 7：另一个 3 障碍物基线记录为 `9% / 91% / 0%`，不能与上面的 44.2% 结果混写

后续操作说明中出现过另一个基线口径：

> “保持 3 障碍物是为了和你手上的基线（9%/91%/0%）严格可比。”

这三个数字很可能分别对应某次实验的 `success / stuck / orbiting`，但当前材料没有给出对应 checkpoint、episode 数和完整输出。它也可能是不同于 44.2% 那轮的新基线。

处理口径：两份结果都必须保留，但在报告和图表中应分开标注 checkpoint、run_name、episode 数和分类定义，不能合并成同一个结果。

### 2.2 已定方案

#### 方案 A：继续使用 ground-truth 2D standalone，而不是立即回 Isaac Lab

已定理由：

- 便于无 GPU 训练；
- 便于可视化轨迹和 Q 值；
- 便于快速做奖励、动作粒度、课程训练等隔离实验；
- 当前失败模式还没有在 2D 环境中解释清楚，直接回到 Isaac Lab 会混入更多变量。

#### 方案 B：3 障碍物训练不从零开始，而从 100% 纯导航 checkpoint 迁移

已实现 `migrate_ckpt_to_obstacles.py`，迁移逻辑：

- 旧 checkpoint：`state_dim=2`；
- 新网络：`state_dim=2+3*K`；
- 新 `fc1.weight` 的前两列保留旧模型参数；
- 新增障碍物输入列全部置零；
- 对无障碍 sentinel 状态，新网络初始策略与旧网络完全一致；
- 梯度仍能流入新增列，后续训练可在不破坏导航能力的前提下学习避障。

注意：

- optimizer state 不迁移，Adam 从零开始；
- resume 后 epsilon 使用 `resume_epsilon=0.3`；
- `--hidden_dim` 必须与 checkpoint 匹配，当前默认 128。

#### 方案 C：先做严格单变量对照，再扩大实验

当前定下的对照原则是：

- 同样是 3 个障碍物；
- 同样使用 `nav100_3obs_init.pt` 作为起点；
- 其他训练设置保持一致；
- 只新增 `living_cost=0.02`。

原因：这是判断 living cost 是否减少 stuck 的干净对照。不要同时改奖励形状、障碍物数量、turn rate 或网络结构，否则周五汇报时无法归因。

#### 方案 D：平台期且已退化的训练应立即停止，不要继续消耗时间

操作说明中已经明确：当前仍在跑的平台期/退化训练应 `Ctrl+C` 停止；`ep2000.pt` 快照已经保留在硬盘上，后续可以回溯。

结论：继续跑旧训练没有意义；优先完成代码合并、短期 smoke test 和新的 `living_cost=0.02` 对照实验。

#### 方案 E：团队分工已按两个失败模式拆开

截图中的既定分工：

- **方向决策问题：** Tomas、Naomi、Pan、Sicheng；
- **orbiting 问题：** 其余成员与 Xiaoman；
- 两组需要更频繁同步，避免各自只优化局部指标。

### 2.3 已废弃或暂不建议继续的路线

| 路线 / 设置 | 状态 | 原因 |
|---|---|---|
| 模拟传感器输入的 2D standalone 训练 | 已移除 | 当前训练直接使用障碍物和目标坐标，减少变量。 |
| 旧的加法式 `rew_distance + rew_yaw_aligned + rew_action_rate` | 已废弃 | 可被“对准但不动”“接近但不进展”等行为独立刷分。 |
| `yaw_cos**3` progress shaping | 已废弃 | 到达率从 29.0% 降至 8.0%，且整体轨迹变差。 |
| 用 `replay_buffer_size=50000` 跑正式纯导航复现 | 不建议 | 已验证行为很差；正式口径使用 5000000。 |
| 继续运行平台期且已退化的旧训练 | 已决定停止 | 再跑无新增信息；已有 `ep2000.pt` 可回滚。 |
| 同时修改多个变量来做 living-cost 验证 | 不建议 | 会破坏与现有 3 障碍物基线的严格可比性。 |

### 2.4 仍未确认、不能当作结论的内容

- `hidden_dim` 从 64 提高到 128 是否确实是容量瓶颈的关键修复：当前只记录了“正在测试”，没有足够结果支持最终结论。
- `test_turn_granularity.py` 是否已在本地完成，以及其结果是否证明“固定角步长导致近目标 orbiting”：当前只有代码注释中的假设，尚无实验输出。
- `9% / 91% / 0%` 基线的完整分类定义、checkpoint、episode 数和生成命令：需从本地日志补齐。
- `nav100_3obs_init.pt` 是否已经由迁移脚本在本地成功生成：需本地确认。
- `curr_3obs_lc` 是否已经开跑或跑完：当前只看到操作计划，没有看到训练结果。

---

## 3. 产出清单与状态

### 3.1 云端已生成、可直接迁移的代码文件

| 文件 | 状态 | 内容 | 迁移备注 | SHA-256 |
|---|---|---|---|---|
| `dqn_config.py` | 完成 | 统一 DQN 配置；包含网络、动作、训练、障碍物、`living_cost` 和 CLI override；`state_dim` 由障碍物配置派生。 | 合并到本地 `dqn_2d_standalone` 前，先与本地版本 `git diff`，避免覆盖未上传改动。 | `9ac5e90dcdaae6c067f4c929063aa8cee29a3ae74c695ced6d48bd678b120d46` |
| `dqn_2d_env.py` | 完成 | 纯 2D 运动学环境；支持圆形障碍物、最近 K 个障碍物状态、解析碰撞、奖励重构、`_pre_reset` 快照。 | 已按障碍物和 `living_cost` 口径更新；与本地 `train_dqn.py` / 诊断脚本兼容依赖其既有属性名。 | `1177b72dde1214d7c8751b6c69c97135a95365e540ef0f5a5f8e8ba2931450d5` |
| `migrate_ckpt_to_obstacles.py` | 完成 | 将 `state_dim=2` 的纯导航 checkpoint 扩展为障碍物输入网络；新增 `fc1` 输入列置零并做策略一致性 sanity check。 | 运行时需要本地存在可导入的 `dqn_action_space.py`，且其中 `QNetwork` 的参数名和 `hidden_dim` 与 checkpoint 匹配。 | `5d37f86fcb83f4e11866081544c14c00f20c9a0e4ebf23b5d1e944c340986e3a` |

以上 3 个 Python 文件已通过语法级校验：`py_compile OK`。

### 3.2 云端可见的实验记录与参考材料

| 材料 | 状态 | 内容 | 后续用途 |
|---|---|---|---|
| 终端粘贴记录 `user_pasted_clipboard_long_content_as_file_(.venv) panyah@MacBo.txt` | 完成 | 包含 200 回合纯导航评估、8 条轨迹绘制、2/6/10 m 下 Q-sweep 与左右对称性检查。 | 作为纯导航收敛和 Q 值不对称的原始证据。SHA-256：`923cca26f3886408a9ee738baeca3aff3bcec1d1c306bd00c2cbb13ade4d473c`。 |
| `image.png` | 完成 | GitHub 仓库与 README 截图，说明 DQN 分支定位、LT-DQN 动机和仓库目录。 | 用于恢复项目背景。 |
| `image(1).png` | 完成 | 团队消息：最终版 DQN 移除模拟传感器；100% 导航依赖 5000000 replay buffer；附 2D 轨迹图。 | 用于保留团队口径和关键结论。 |
| `image(2).png` | 完成 | 3 障碍物初步结果：44.2% success、50.9% stuck、3.6% orbiting；以及分工计划。 | 用于定义两大问题组。 |
| `image(3).png` | 完成 | 团队消息：本周所有人运行 2D standalone；不需要 Isaac Lab 和 GPU。 | 用于恢复项目协作背景。 |
| `image(4).png` | 完成 | 本地目录截图，标出 `NEU/Drone_Ranger` 项目文件夹。 | 用于确认本地迁移目标目录。 |
| `image(5).png` | 完成 | living-cost 实验操作顺序和正式训练命令。 | 用于恢复下一步实验设计。 |

### 3.3 本地仓库中被引用、但云端没有文件本体的产出

这些文件在当前材料中被命令、代码 import 或截图明确引用，但云端会话没有其完整内容；迁移时必须以本地文件为准。

| 文件 / 目录 | 当前证据 | 状态判断 | 阻塞点 / 备注 |
|---|---|---|---|
| `train_dqn.py` | 多条训练命令直接调用 | 本地应存在，且曾用于训练 | 云端未提供文件本体；合并环境代码后需本地运行 smoke test。 |
| `dqn_action_space.py` | `migrate_ckpt_to_obstacles.py` 中导入 `QNetwork` | 本地应存在 | 需确认 `QNetwork` 的第一层参数名确为 `fc1.weight`，且 `hidden_dim=128`。 |
| `dqn_diagnose_2d.py` | 200 回合评估命令 | 本地应存在，已使用 | 需确认它是否能输出 stuck/orbiting 分类，而不仅是 Gym 式终止原因。 |
| `plot_2d_trajectory.py` | 已生成 `nav_traj.png` | 本地应存在，已使用 | 避障图应继续保留起点、终点、障碍物和 crash 标记。 |
| `q_sweep.py` | 已输出 heading/Q 值扫描 | 本地应存在，已使用 | 后续要扩展为“障碍物正前方选边”分析。 |
| `trace_2d.py` | 环境代码注释称曾用它发现奖励漏洞 | 本地应存在或曾存在 | 云端未提供文件本体；建议纳入实验工具目录。 |
| `test_turn_granularity.py` | 环境代码注释中提到 | 状态不明 | 需本地确认是已完成、进行中，还是仅计划创建。 |
| `dqn/` | GitHub 截图中的目录 | 仓库已有 | 云端未展开内容；本地以 git 状态为准。 |
| `dqn_2d_standalone/` | GitHub 截图和终端路径均指向该目录 | 当前主工作目录 | 云端生成的 3 个文件应合并到这里。 |
| `isaaclab_env/` | GitHub 截图中的目录 | 仓库已有，当前不是主线 | 2D 问题未解决前不建议迁回。 |
| `README.md` | GitHub 截图显示已有 | 仓库已有 | 后续应补充当前 2D standalone 实验口径。 |

### 3.4 模型、数据和图像产出

| 文件 / run | 状态 | 已有证据 | 后续处理 |
|---|---|---|---|
| `checkpoints/nav100_final.pt` | 完成，已验证 | 200/200 reached；mean_reward=107.61；mean_episode_length=237.7。 | 作为纯导航标准起点，必须备份并记录 git commit / config。 |
| `nav_eval.csv` | 完成 | 评估脚本已写出。 | 保留为纯导航 100% 基线原始数据。 |
| `nav_traj.png` | 完成 | 8 个 episode 全部 reached，且 `obstacles=0`。 | 保留为纯导航轨迹图。 |
| `checkpoints/ep2000.pt` | 完成，保留快照 | 操作说明称旧训练平台期且退化，但 ep2000 快照已在硬盘上。 | 不要删除；用于回滚或对比。 |
| `checkpoints/nav100_3obs_init.pt` | 需本地确认 | 迁移脚本和后续命令均以它为 3 障碍物起点。 | 若不存在，先运行迁移脚本生成；若存在，记录生成命令和哈希。 |
| 3 障碍物初步模型 | 完成过测试，但 checkpoint 未上传 | 44.2% / 50.9% / 3.6% 的失败分布。 | 需找回对应 checkpoint、run_name 和评估日志。 |
| `9% / 91% / 0%` 基线模型 | 需本地确认 | 仅在操作说明中被引用。 | 必须补齐 checkpoint、分类定义、episode 数和完整命令。 |
| `curr_3obs_lc` | 待执行 / 状态未确认 | 已有正式训练命令。 | 迁移完成后优先确认是否已跑；若未跑，按命令启动。 |

### 3.5 当前代码中的关键配置口径

| 参数 | 当前值 | 备注 |
|---|---:|---|
| `num_actions` | 3 | forward / turn_left / turn_right。 |
| `hidden_dim` | 128 | 从 64 提高到 128；容量实验结论仍未完全确认。 |
| `forward_speed` | 1.5 | 动作空间物理参数。 |
| `turn_rate` | 0.8 | 动作空间物理参数。 |
| `max_xy_vel` | 2.5 | 环境 tanh 映射使用。 |
| `max_yaw_rate` | 1.57 | 环境 tanh 映射使用。 |
| `replay_buffer_size` | 50000 | 配置默认值；正式实验显式改为 5000000。 |
| `batch_size` | 64 | 当前默认。 |
| `learning_rate` | 1e-3 | 当前默认。 |
| `gamma` | 0.98 | 当前默认。 |
| `epsilon_start` | 1.0 | 新训练起始值。 |
| `epsilon_min` | 0.05 | 当前默认。 |
| `epsilon_decay` | 0.995 | 当前默认。 |
| `target_update_every` | 10 | 当前默认。 |
| `resume_epsilon` | 0.3 | 从 checkpoint 恢复时使用。 |
| `num_obstacles` | 0 | 配置默认；3 障碍物实验通过 CLI 改为 3。 |
| `k_nearest` | 3 | 暴露给网络的最近障碍物数量。 |
| `obstacle_radius` | 0.5 | 圆形障碍物半径。 |
| `drone_radius` | 0.2 | 碰撞判定使用。 |
| `obstacle_clearance` | 0.5 | 起点和终点额外安全间隔。 |
| `first_obstacle_on_path` | 1 | 强制障碍物 0 出现在起点—目标路径上。 |
| `sense_range` | 10.0 | 超过该表面距离的障碍物不可见。 |
| `state_scale` | 10.0 | 状态归一化尺度。 |
| `living_cost` | 0.0 | 默认关闭；下一组对照实验用 0.02。 |

---

## 4. 待办事项（按优先级）

### P0：必须先完成，否则后续实验不可信

| 优先级 | 待办 | 验收标准 | 阻塞点 |
|---|---|---|---|
| P0-1 | 完成本地代码合并和备份 | 云端 3 个 Python 文件合入本地 `dqn_2d_standalone`；合并前本地改动已 commit 或备份；`git diff` 已检查。 | 云端与本地数据不互通，必须手动复制文件；无法确认本地是否有未上传改动。 |
| P0-2 | 确认本地依赖文件与 checkpoint 匹配 | `dqn_action_space.py` 可导入；`QNetwork` 第一层参数名为 `fc1.weight`；`nav100_final.pt` 的 `hidden_dim` 与迁移脚本参数一致。 | 这些文件未上传到云端；只能在本机检查。 |
| P0-3 | 确认或重新生成 `checkpoints/nav100_3obs_init.pt` | 迁移脚本运行成功，并输出 `policy preserved on obstacle-free states: OK`。 | 如果本地已修改网络结构或 checkpoint 不是 `hidden_dim=128`，迁移会失败。 |
| P0-4 | 做 30 轮 smoke test | 修改 `living_cost` 后训练/评估流程能完整运行，无 shape mismatch、NaN、checkpoint 加载失败或评估脚本不兼容。 | 目前只有“30 轮冒烟测试”的要求，未看到固定 smoke-test 命令；可用现有训练或诊断脚本实现。 |
| P0-5 | 启动 `living_cost=0.02` 的正式 3 障碍物对照实验 | run_name、配置、随机种子、代码 commit、起始 checkpoint 全部记录。 | 需先停止平台期/退化的旧训练，并确保使用同一个 `nav100_3obs_init.pt` 起点。 |

正式训练命令按现有决定保持为：

```bash
python3 train_dqn.py --num_envs=64 --num_episodes=3000 --num_obstacles=3 \
  --replay_buffer_size=5000000 \
  --resume_from=checkpoints/nav100_3obs_init.pt \
  --run_name=curr_3obs_lc \
  --living_cost=0.02
```

### P1：决定 3 障碍物问题能否继续推进

| 优先级 | 待办 | 验收标准 | 阻塞点 |
|---|---|---|---|
| P1-1 | 统一定义并自动分类 `success / stuck / orbiting / crash / timeout / out_of_bounds` | 每个 episode 的分类可由 CSV 或诊断脚本复现；不再依赖人工看图。 | 目前终端脚本原生输出只有 `reached / out_of_bounds / timed_out`，collision、stuck、orbiting 口径未完整提供。 |
| P1-2 | 补齐 `9% / 91% / 0%` 基线的实验元数据 | 找到对应 checkpoint、run_name、episode 数、分类结果和完整命令；与 44.2% 那轮分开记录。 | 当前只有截图中的简称，无法确认三个数字的顺序和实验条件。 |
| P1-3 | 分析障碍物正前方的左右选边问题 | 在 stuck episode 中提取状态、Q 值、动作序列和障碍物相对位置；判断是 Q 值接近、状态盲区、奖励不足还是探索不足。 | 需要 stuck 样本的逐 step 轨迹；当前没有上传相关 CSV。 |
| P1-4 | 扩展 Q-sweep 到“有障碍物”状态 | 除无障碍 heading 扫描外，增加障碍物在正前方、左前方、右前方、不同距离下的 Q 值表。 | 现有 `q_sweep.py` 输出记录只覆盖 “no obstacle in state”。 |
| P1-5 | 完成近目标 orbiting 的隔离实验 | 用 `test_turn_granularity.py` 或等价脚本量化固定角步长、到达阈值和 heading error 的关系。 | 云端没有该脚本本体或结果；当前只有“固定角步长可能导致无法精确归零”的假设。 |
| P1-6 | 比较 `living_cost=0` 与 `living_cost=0.02` | 同一评估集上输出到达率、stuck 率、orbiting 率、平均步长和奖励；只改变 living cost。 | 需要先完成 P0 的代码合并、smoke test 和正式训练。 |

### P2：3 障碍物稳定后再做

| 优先级 | 待办 | 验收标准 | 阻塞点 |
|---|---|---|---|
| P2-1 | 课程式增加障碍物数量 | 3 障碍物结果稳定后，再增加数量或改变 placement；每个阶段都有独立 checkpoint 和评估。 | 3 障碍物基线尚未解决，直接增加难度会混淆问题。 |
| P2-2 | 重新评估网络容量 | 在相同任务和训练预算下比较 64 与 128 hidden units，记录是否影响到达率和 stuck 率。 | 目前 `hidden_dim=128` 已采用，但容量是否瓶颈没有结论。 |
| P2-3 | 对 turn rate / 动作粒度做 ablation | 只有在 turn-granularity 假设被隔离验证后，才修改动作空间；每次只改一个变量。 | 缺少 orbiting 的定量机制证据。 |
| P2-4 | 设计障碍物 spawn 难度课程 | 比较“障碍物强制在路径上”和“随机但保证起点/终点可行”的训练效果。 | 当前 `first_obstacle_on_path=1` 是强课程信号，但还没有系统 ablation。 |
| P2-5 | 更新 README 和实验日志 | README 能说明 2D standalone 的安装、训练、评估、绘图、Q-sweep 和当前最佳 checkpoint。 | 需要先统一实验口径，否则文档会过快过期。 |

### P3：长期迁移与整合

| 优先级 | 待办 | 验收标准 | 阻塞点 |
|---|---|---|---|
| P3-1 | 将 2D 中验证有效的状态、奖励和课程迁回 Isaac Lab | Isaac Lab 版本能复现核心趋势，而不仅是单次成功。 | 2D ground-truth 状态与 Isaac Lab 的传感器/动力学存在分布和动力学差异。 |
| P3-2 | 决定是否接入 LiDAR + A* warning-zone 层 | 明确 standalone DQN 与混合架构的边界和接管条件。 | 当前 Part 1 standalone DQN 尚未在障碍物场景中稳定。 |
| P3-3 | 整理周五汇报图表 | 至少包含纯导航 100% 基线、3 障碍物旧基线、living-cost 对照、典型轨迹和 Q 值/动作证据。 | 取决于 P0/P1 实验是否按时完成。 |

---

## 5. 偏好与要求

以下是根据当前材料观察到的协作、写作和实验偏好，后续在 Kimi Work 本地项目中应继续保持。

### 5.1 写作与沟通偏好

- **中文为主要沟通语言**，但代码、变量、文件名、命令、run name 保持英文。
- **结论先行，证据紧随其后。** 不接受只写“效果变好/变差”，必须给出成功率、失败分布、轨迹图或 Q 值表。
- **明确区分事实、推测和待验证假设。** 例如“固定角步长可能导致 orbiting”目前只能写成假设，不能写成已证实结论。
- **保留原始措辞和原始数字。** 团队消息中的 44.2%、50.9%、3.6%，以及另一个 9%/91%/0% 基线，必须按原始来源分开记录。
- **喜欢结构清晰的 Markdown。** 推荐使用标题、表格、编号列表和代码块；关键命令必须可原样复制。
- **喜欢直接指出问题。** 对平台期、退化、奖励漏洞、左右不对称等问题不要包装成模糊表述。

### 5.2 实验偏好

- **严格控制变量。** 同一个对照实验中，任务、起点 checkpoint、障碍物数量、训练预算保持一致，只改变被测试的变量。
- **先小后大。** 修改代码后先跑 30 轮 smoke test，再跑 3000 episode 正式训练。
- **及时止损。** 已平台期且退化的训练不要继续跑；保留 checkpoint 后转向更有信息量的实验。
- **每个实验保留可回溯信息。** 包括 checkpoint 路径、run_name、命令、参数、代码 commit、episode 数和评估输出。
- **失败模式要拆开。** “方向决策问题”和“orbiting 问题”不要混为一个“避障不好”。
- **优先解释机制。** 修改奖励或动作空间前，先用 trace、Q-sweep 或粒度测试定位机制。
- **评估口径要可复现。** `stuck` 和 `orbiting` 后续应尽量由脚本自动判定，而不是只凭截图人工归类。

### 5.3 代码偏好

- **代码注释保留实验历史。** 当前代码中会明确记录某项改动为什么被回退，例如 `yaw_cos**3` 的失败结果；这种风格应保留。
- **配置与任务逻辑分离。** DQN 算法和训练超参数放 `dqn_config.py`；环境、奖励、障碍物逻辑放环境文件。
- **CLI 优先于手改默认值。** A/B 实验通过命令行参数完成，避免为了单次实验直接改代码默认值。
- **兼容旧脚本。** 环境保留 `_last_reached_flag`、`_last_any_contact`、`_last_out_of_bounds` 等属性，是为了让已有 train/diagnose 脚本无需大改。
- **保存 reset 前状态。** `_pre_reset` 快照用于轨迹和绘图脚本读取 episode 的真实最终状态，不能随意删除。
- **不要引入 Isaac Lab 依赖。** `dqn_2d_standalone` 的价值就在于轻量、无 GPU、可快速迭代。

### 5.4 命名与汇报口径

- 推荐继续使用语义化 run name，例如：
  - `nav100_final.pt`：纯导航 100% checkpoint；
  - `nav100_3obs_init.pt`：从纯导航迁移到 3 障碍物的初始 checkpoint；
  - `curr_3obs_lc`：3 障碍物 + living cost 对照实验。
- 汇报时建议统一使用：
  - `success` 或 `reached`：到达；
  - `stuck`：障碍物前停止或长期无进展；
  - `orbiting`：近目标绕圈但不进一步接近；
  - `crash`：与障碍物碰撞；
  - `timed_out`：达到最大步数；
  - `out_of_bounds`：出界。
- 其中 `stuck` 与 `orbiting` 需要在本地诊断脚本中补正式阈值定义。

---

## 6. 重要原文与必须保留的数据

### 6.1 纯导航 100% 评估原始输出

```text
(.venv) panyah@MacBook-Air dqn_2d_standalone % python3 dqn_diagnose_2d.py --checkpoint=checkpoints/nav100_final.pt --num_envs=64 --num_episodes=200 --csv_out=nav_eval.csv
[DIAG2D] loaded checkpoint: checkpoints/nav100_final.pt

[DIAG2D] collected 200 episodes

[DIAG2D] Outcome distribution (n=200):
  reached       :  200 (100.0%)
  out_of_bounds :    0 (  0.0%)
  timed_out     :    0 (  0.0%)

[DIAG2D] mean_reward=107.61  mean_episode_length=237.7
[DIAG2D] CSV written to nav_eval.csv
```

### 6.2 无障碍轨迹绘制原始输出

```text
[PLOT] episode 0: outcome=reached, steps=534, start=(0.74,-0.99), target=(3.11,4.97), obstacles=0
[PLOT] episode 1: outcome=reached, steps=256, start=(-0.13,-0.55), target=(-4.59,-3.77), obstacles=0
[PLOT] episode 2: outcome=reached, steps=153, start=(-0.44,0.68), target=(0.37,3.15), obstacles=0
[PLOT] episode 3: outcome=reached, steps=151, start=(0.24,-0.04), target=(2.51,2.22), obstacles=0
[PLOT] episode 4: outcome=reached, steps=256, start=(-0.44,0.46), target=(-3.82,-3.38), obstacles=0
[PLOT] episode 5: outcome=reached, steps=59, start=(0.08,0.99), target=(2.41,1.84), obstacles=0
[PLOT] episode 6: outcome=reached, steps=327, start=(0.83,0.00), target=(3.67,-2.64), obstacles=0
[PLOT] episode 7: outcome=reached, steps=389, start=(0.90,0.71), target=(2.67,-2.73), obstacles=0
[PLOT] saved to nav_traj.png
```

### 6.3 团队关于移除传感器与 replay buffer 的原文

> “Hello guys, just in case that you cannot join today's sub meeting, I have push the final version of our DQN, I removed the simulated sensors, now we are simply using coordinates of the obstacle and target for the training.”

> “Current model can reach 100% navigation success rate, the default replay buffer size is 50000, but the buffer size is too small so the behavior is really bad. simply by increasing the buffer size to 5000000 can make it 100%.”

### 6.4 团队关于 3 障碍物失败分布和分工的原文

> “I have tested 100% navigation model with 3 obstacle for curriculum learning, and the outcome is: 44.2% success 50.9% stuck in front for obstacle don't know turn left or right, 3.6% of orbiting in final stage without getting closer.”

> “My plan is we separate into two group to work on the two problems, Tomas, Naomi, Pan, Sicheng will work on the direction decision making problem. The others will work on the orbiting problem with me.”

> “For specific steps on the experiments: 1. Do the training of pure Navigation locally! get the model with 100% success rate 2. use curriculum learning for the obstacle avoidance training (lets do it gradually, starting from 3 obstacle), and we will work on different problems in sub groups.”

### 6.5 2D standalone 运行要求的原文

> “For this week, everybody could try run the training in this repo. You don't need to run the Issac Lab version, simply 2D Stand alone. You don't need a GPU for the training, so hopefully everyone could run it.”

### 6.6 living-cost 对照实验的操作顺序原文

> “先把现在还在跑的训练 Ctrl+C 停掉（平台期+退化，再跑没意义；ep2000.pt 快照都在硬盘上，随时可以回头）；
>
> 改两行代码（config 加字段 + env 里减一项）；
>
> 30 轮冒烟测试确认没改坏；
>
> 正式开跑，起点用 nav100_3obs_init.pt，保持 3 障碍物不变。”

### 6.7 living-cost 对照实验正式命令

```bash
python3 train_dqn.py --num_envs=64 --num_episodes=3000 --num_obstacles=3 --replay_buffer_size=5000000 --resume_from=checkpoints/nav100_3obs_init.pt --run_name=curr_3obs_lc --living_cost=0.02
```

### 6.8 对照实验设计原则原文

> “保持 3 障碍物是为了和你手上的基线（9%/91%/0%）严格可比——同样的任务、同样的起点、只多一个 living_cost，这才是干净的对照实验，周五汇报时这个对比就是核心图表。”

### 6.9 checkpoint 迁移原理原文

```text
Why zero-padding works: dqn_2d_env.py pads unused/invisible obstacle slots
with a constant sentinel (sense_range, 0, 0) / state_scale. Setting the new
fc1 columns to zero means the obstacle inputs contribute exactly nothing at
initialization, so the migrated network's policy is IDENTICAL to the
navigation policy on obstacle-free states. Gradients still flow into the
new columns (dL/dW = delta * input), so obstacle avoidance is learned
on top of the intact navigation behavior instead of from scratch.
```

### 6.10 `yaw_cos**3` 回退原因原文摘要

```text
REVERTED from yaw_cos**3 (v4): that change reduced reached-rate
from 29.0% to 8.0% and made trajectories markedly worse overall,
not just at the near-target orbiting case it targeted.
```

---

## 7. 本地迁移检查清单

将本文档和云端生成的 3 个 Python 文件放入本地项目后，建议按以下顺序执行：

1. **确认本地工作区干净或已备份**

   ```bash
   git status
   ```

2. **备份本地同名文件**

   ```bash
   mkdir -p backups/before_cloud_handoff
   cp dqn_config.py dqn_2d_env.py backups/before_cloud_handoff/ 2>/dev/null || true
   cp migrate_ckpt_to_obstacles.py backups/before_cloud_handoff/ 2>/dev/null || true
   ```

3. **复制云端生成的 3 个文件到本地 `dqn_2d_standalone`**

   - `dqn_config.py`
   - `dqn_2d_env.py`
   - `migrate_ckpt_to_obstacles.py`

4. **检查差异**

   ```bash
   git diff -- dqn_config.py dqn_2d_env.py migrate_ckpt_to_obstacles.py
   ```

5. **做语法检查**

   ```bash
   python3 -m py_compile dqn_config.py dqn_2d_env.py migrate_ckpt_to_obstacles.py
   ```

6. **确认 checkpoint 和网络结构**

   ```bash
   ls -lh checkpoints/nav100_final.pt
   ls -lh checkpoints/nav100_3obs_init.pt 2>/dev/null || true
   ```

7. **如 3 障碍物初始 checkpoint 不存在，则重新生成**

   ```bash
   python3 migrate_ckpt_to_obstacles.py \
     --in=checkpoints/nav100_final.pt \
     --out=checkpoints/nav100_3obs_init.pt \
     --num_obstacles=3
   ```

   必须看到：

   ```text
   policy preserved on obstacle-free states: OK
   ```

8. **运行 30 轮 smoke test**

   使用本地现有训练或诊断入口完成 30 轮短测；重点检查：

   - state shape 是否为 11；
   - checkpoint 是否加载成功；
   - reward 是否出现 NaN；
   - terminated/truncated 标志是否正常；
   - 日志、CSV、checkpoint 是否写出。

9. **启动 living-cost 正式对照实验**

   ```bash
   python3 train_dqn.py --num_envs=64 --num_episodes=3000 --num_obstacles=3 \
     --replay_buffer_size=5000000 \
     --resume_from=checkpoints/nav100_3obs_init.pt \
     --run_name=curr_3obs_lc \
     --living_cost=0.02
   ```

10. **训练后立刻归档**

    至少保存：

    - 最终 checkpoint；
    - 最优 checkpoint；
    - 训练日志；
    - 评估 CSV；
    - 轨迹图；
    - 完整命令；
    - 当前 git commit；
    - 分类后的 outcome distribution。

---

## 8. 给下一个继续者的最短指令

如果你只想快速继续，不要先改模型结构。按下面顺序做：

1. 合并云端 3 个 Python 文件并跑 `py_compile`；
2. 确认 `nav100_final.pt` 与 `hidden_dim=128` 匹配；
3. 生成或确认 `nav100_3obs_init.pt`；
4. 跑 30 轮 smoke test；
5. 启动 `curr_3obs_lc`；
6. 同时补齐 `stuck/orbiting` 自动分类；
7. 用同一评估集比较旧 3 障碍物基线和 `living_cost=0.02`；
8. 分别给方向决策组和 orbiting 组提供逐 step 状态、动作、Q 值和轨迹证据。

**当前不要做的事：**

- 不要从零重训 3 障碍物；
- 不要同时改 living cost、turn rate、奖励形状和障碍物数量；
- 不要继续已经平台期且退化的旧 run；
- 不要把 44.2%/50.9%/3.6% 与 9%/91%/0% 混成同一个基线；
- 不要在缺少证据时宣称 fixed turn granularity 已被证明是 orbiting 的根因。
