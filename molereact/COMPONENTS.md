# Components Documentation

> **定位**：支持核心流程的内部组件 (Managers & Handlers)。

本模块包含状态管理、LLM 交互与路径追踪的关键实现。

## 1. SessionLogger (`managers/session_logger.py`)

**功能**：负责会话的持久化存储、日志记录与状态恢复。

### 核心机制
*   **Dual-Format Storage**:
    *   `.md` (Human Readable): 用于人类阅读和 LLM 上下文回溯。
    *   `.json` (Machine State): 包含完整的 `StageResult` 和 `cumulative_route` 数据，用于精准 resume。
*   **State Recovery**:
    *   优先读取 JSON快照 (`restore_session_state`)。
    *   如果 JSON 损坏，尝试解析 Markdown 日志重建状态（Fallback 策略）。

### 关键方法
*   `log_decision(...)`: 记录每一步的选择结果。
*   `save_json_snapshot(...)`: 保存当前完整的反应树状态。
*   `get_latest_context()`: 查找最近一次未完成的会话以便自动继续。

---

## 2. LLMSelectionHandler (`handlers/llm_selection.py`)

**功能**：封装与 LLM 的交互逻辑，包括 Prompt 构建、调用与结果解析。

### 工作流
1.  **Prepare Candidates**: 将候选反应、库存状态、分析数据（Analysis Tool results）格式化为文本块。
2.  **Global Strategy Injection**: 如果存在全局策略（Agent-0 输出），将其注入到 Prompt 中作为高优先级指导。
3.  **LLM Call**: 调用 ZhipuAI/OpenAI API（支持 Retry 和 Exponential Backoff）。
4.  **Parse & Repair**:
    *   支持解析 Markdown 代码块、纯 JSON 或从杂乱文本中提取 JSON 对象。
    *   包含 `fix_common_json_issues` 方法，自动修复常见的 JSON 格式错误（如末尾逗号、单引号）。

### 异常处理
*   如果多次解析失败，会触发 Fallback 机制，返回基于置信度排序的原列表，确保护理化流程不中断。

---

## 3. RouteHistoryManager (`managers/route_history.py`)

**功能**：维护反应路径的谱系（Lineage）并检测环路。

### 核心概念
*   **Path ID (谱系 ID)**: 如 `1.2.3` 代表从节点 1 -> 节点 1.2 -> 节点 1.2.3 的路径。
*   **Reaction Vector (反应向量)**: 记录每一步变化的官能团与重原子数（Delta）。

### 循环检测逻辑 (`evaluate_reaction_vector_loop`)
不仅仅检测 A -> B -> A，还通过向量和同构检测识别复杂的“伪循环”（Functional Group Stagnation）：

1.  **Inverse Vector**: Step N 的反应向量与 Step N-1 互为相反数（即 A -> B -> A）。
2.  **Cumulative Zero**: 连续多步后累积变化量为零。
3.  **State Repeat**: 当前分子结构（SMILES或特征指纹）在历史路径中出现过。

如果检测到 `HIGH` 风险，Agent 会在评分时大幅惩罚该路径。
