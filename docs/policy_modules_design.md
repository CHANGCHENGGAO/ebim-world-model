# LLM + Diffusion Policy 可选模块设计文档

**日期**: 2026-08-22
**版本**: v1.0
**状态**: 设计完成

## 1. 设计目标

为 EBiM Task 3 自主控制器增加 LLM 高层规划和 Diffusion Policy 低层轨迹生成作为**可选模块**，硬编码逻辑作为**最终兜底**，确保系统安全性和可靠性。

## 2. 核心原则

1. **可选启用**: 默认完全关闭（`--policy hardcoded`），无任何额外开销
2. **逐层兜底**: 每个 AI 模块都有对应的硬编码 fallback
3. **超时保护**: 所有 AI 调用都有严格超时限制
4. **接口标准**: 策略模块与执行层通过标准接口解耦
5. **可观测**: 每次决策记录来源（LLM/Diffusion/Hardcoded）和耗时

## 3. 架构概览

```
Task3Controller
    │
    ├── Stage 1/2/3/4 (四阶段流程)
    │
    └── ActionExecutor (动作执行器)
            │
            └── PolicyManager (策略管理器)
                    │
                    ├── L1: LLM Planner + Diffusion Policy (hybrid)
                    ├── L2: LLM Planner + IK Executor (llm)
                    └── L3: Hardcoded + IK (hardcoded) ← 最终兜底
```

## 4. 模块设计

### 4.1 PolicyManager (`policy_manager.py`)

**职责**: 统一策略入口，管理兜底链路

**接口**:
```python
class PolicyManager:
    def __init__(self, policy_mode: str = "hardcoded")
    def plan_action(self, state: dict, goal: dict, fallback_fn: callable) -> dict
    def generate_trajectory(self, target_pos, current_q, fallback_fn) -> np.ndarray
```

**策略模式**:
| 模式 | 规划 | 执行 | 适用场景 |
|------|------|------|---------|
| `hardcoded` | 硬编码 | IK | 比赛默认，最稳定 |
| `llm` | LLM | IK | 测试规划能力 |
| `diffusion` | 硬编码 | Diffusion | 测试轨迹生成 |
| `hybrid` | LLM | Diffusion | 全 AI 模式 |

**兜底链路** (hybrid 模式):
1. 尝试 LLM 规划 + Diffusion 执行
2. LLM 失败 → 硬编码规划 + Diffusion 执行
3. Diffusion 失败 → LLM 规划 + IK 执行
4. 都失败 → 硬编码规划 + IK 执行（最终兜底）

### 4.2 LLM Planner (`llm_planner.py`)

**职责**: 基于当前状态和目标，输出结构化动作指令

**技术方案**:
- 推理框架: `llama-cpp-python` (GGUF 格式)
- 推荐模型: Qwen2.5-3B-Instruct-Q4_K_M (~2GB, 已验证)
- 推理方式: 本地 CPU/GPU 推理
- 超时: 5秒

**输入 Prompt 结构**:
```
System: 你是机器人操作规划器。根据当前状态和目标，输出下一步动作。
只输出JSON，不要其他文字。

当前状态:
- 物品位置: {item_positions}
- 左臂关节: {left_joints}
- 右臂关节: {right_joints}
- 底座位置: {base_pos}
- 夹爪状态: {gripper_state}

目标: {goal_description}
可用动作: grasp, place, move_arm, navigate, wait

输出JSON格式:
{"action": "...", "arm": "left/right", "target_object": "...", "params": {...}}
```

**输出格式** (严格 JSON):
```json
{
  "action": "grasp",
  "arm": "left",
  "target_object": "bowl2",
  "params": {
    "approach_height": 0.15,
    "grasp_z_offset": 0.02
  },
  "reasoning": "碗位于盘子左侧，左手抓取路径更优"
}
```

**失败判定**:
- 超时（>5s）
- JSON 解析失败
- 缺少必填字段（action/arm/target_object）
- action 不在白名单内

### 4.3 Diffusion Policy (`diffusion_policy.py`)

**职责**: 生成从当前关节到目标位姿的平滑轨迹

**技术方案**:
- 框架级实现（无预训练模型时的占位实现）
- 默认: IK 解 + 高斯噪声模拟 diffusion 去噪过程（10步）
- 接口设计兼容真实 Diffusion Policy（未来可直接替换）

**接口**:
```python
class DiffusionPolicy:
    def __init__(self, model_path=None, num_denoise_steps=10)
    def generate_trajectory(self, target_pos, target_rot, current_q, n_steps=50) -> np.ndarray
```

**去噪过程（模拟版）**:
1. 从纯高斯噪声轨迹开始
2. 每步向 IK 插值目标靠近一点
3. 加上条件噪声（模仿 diffusion 的条件注入）
4. 最终输出平滑轨迹

**失败判定**:
- 轨迹中任何点超出关节限位
- 轨迹不连续（相邻步差 > 阈值）
- 终点与目标偏差 > 1cm

## 5. 集成方式

### 5.1 命令行参数

```bash
python task3_autonomous.py --policy hardcoded   # 默认
python task3_autonomous.py --policy llm
python task3_autonomous.py --policy diffusion
python task3_autonomous.py --policy hybrid
```

### 5.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL_PATH` | `/root/models/qwen2.5-3b-instruct-q4_k_m.gguf` | LLM 模型路径 |
| `LLM_N_CTX` | `2048` | 上下文窗口 |
| `LLM_N_GPU_LAYERS` | `0` | GPU 层数 (0=CPU only) |
| `DIFFUSION_MODEL_PATH` | `None` | Diffusion 模型路径 |
| `POLICY_TIMEOUT` | `5.0` | 策略决策超时（秒） |

### 5.3 控制器集成

在 `Task3Controller` 中：
- 初始化时创建 `PolicyManager` 实例
- `grasp_only` / `place_only` / `move_arm_to_xyz` 等方法中调用 policy
- 所有阶段流程不变，只在动作执行层接入策略选择

## 6. 性能考量

| 模式 | 额外延迟 | 内存占用 | 稳定性 |
|------|---------|---------|--------|
| hardcoded | 0ms | 0MB | 100% |
| llm (3B, CPU) | ~2-5s | ~2GB | 95% (格式错误回退) |
| llm (3B, GPU) | ~0.2-0.3s | ~2GB | 100% (已验证) |
| diffusion (模拟) | ~100ms | ~50MB | 99% |
| hybrid | ~5-10s | ~4-6GB | 90% (两层兜底后 ~99%) |

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| LLM 输出格式错误 | 动作失败 → 回退硬编码 | 严格 JSON schema 校验 + 重试1次 |
| LLM 推理超时 | 动作延迟 → 回退硬编码 | 5秒超时 + 守护线程 |
| Diffusion 轨迹越界 | 机械臂可能碰撞 | 关节限位校验 + IK 兜底 |
| 模型文件不存在 | 模块初始化失败 | 自动降级到 hardcoded 模式 |
| 内存不足 | OOM → 进程崩溃 | 模型按需加载，失败自动降级 |

## 8. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `policy_manager.py` | 新增 | 策略管理器 + 兜底链路 |
| `llm_planner.py` | 新增 | LLM 本地推理规划器 |
| `diffusion_policy.py` | 新增 | Diffusion Policy 轨迹生成 |
| `task3_autonomous.py` | 修改 | 集成策略管理器 |
| `Dockerfile` | 修改 | 添加 llama-cpp-python 依赖 |
| `entrypoint.sh` | 修改 | 添加模型路径环境变量 |
