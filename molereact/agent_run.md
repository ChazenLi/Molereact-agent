# Agent Run (`agent_run.py`) Documentation

> **定位**：流程编排器 (Workflow Orchestrator) 与 CLI 入口。

该脚本是整个 Agent 的大脑，负责串联逆合成引擎、分析工具、库存系统和 LLM 决策层。

## 核心类与接口

### `CompleteWorkModuleRunner`

整个流程的控制者。

#### `__init__(self, use_llm=True, auto_mode=False)`
*   **use_llm**: 是否启用 LLM。如果为 False，将强制使用启发式规则。
*   **auto_mode**: 是否开启全自动模式（不请求用户手动确认）。

#### `run_work_module(...)`
核心工作流函数。

**方法签名**:
```python
def run_work_module(
    self, 
    target_smiles: str, 
    stage: int = 1, 
    topk: int = 10, 
    history_context: str = "", 
    path_id: str = "1", 
    cumulative_route: Dict = None
) -> StageResult
```

**输入参数**:
*   `target_smiles`: 当前需要逆合成的分子。
*   `stage`: 当前递归深度（Step 1, 2, ...）。
*   `topk`: 要求底层引擎返回多少个候选反应。
*   `history_context`: 用于构建 Prompt 的历史上下文。
*   `path_id`: 路径的唯一标识符（如 "1.2.1"），用于追踪谱系。

**返回值**:
*   `StageResult`: 包含本阶段所有候选、决策结果、未解决前体列表的数据类。

### 核心步骤详解 (Deep Dive)

执行顺序如下：

1.  **Engine Proposal (`engine.propose_precursors`)**
    *   调用单步模型，获取 top-k 候选（Templates + Models）。
2.  **Stock Check (Initial)**
    *   快速检查所有候选前体的库存状态。
3.  **Repair & Expand (`_standardize_and_repair_candidates`)**
    *   使用 LLM 修复无效的 SMILES，并标准化格式。
    *   *Deep Scan (Optional)*: 如果开启，在此处进行深度预演。
4.  **LLM Selection (`_llm_select_top_n`)**
    *   将经过清洗的候选列表发给 LLM。
    *   LLM 结合 `Global Strategy` 和 `History` 进行打分和排序。
5.  **Visualization**
    *   生成 `.png` 图片，展示 Top-N 推荐路线。
6.  **Queue Update**
    *   返回 `unsolved_leaves`，由外部循环将其加入待办队列。

## 输入输出契约

### CLI 调用
```bash
# 标准调用
python agent_run.py --smiles "c1ccccc1"

# 预期输出
# [INFO] ...
# [SUCCESS] Stage 1 ...
# 生成 output/agent_runs/node_1_route_1.png
```

### 异常处理
*   **ImportError**: 尝试自动降级或导入备用路径。
*   **API Error**: LLM 失败时捕获异常，回退到 `_heuristic_select`。
*   **Analysis Error**: 分子分析失败会在日志中标记，但不阻断流程（作为 "Analysis Failed" 处理）。

## 关键配置
*   `ZHIPUAI_API_KEY`: 环境变量或硬编码，用于 LLM 服务鉴权。
*   `SCENARIO_PROFILES`: 定义评分权重（学术界 vs 工业界偏好）。
*   `CURRENT_SCENARIO`: 默认使用了 "INDUSTRIAL"（注重成本和效率）。
