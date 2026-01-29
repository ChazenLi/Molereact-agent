# MoleReact Agent (Multi-step Retrosynthesis)

> **一句话总结**：这是一个基于 LLM + 专家规则的逆合成路线规划 Agent，输入目标分子 SMILES，输出完整的逆合成路线及其可视化。

## TL;DR（30秒上手）

```bash
# 1) 确保依赖就绪 (通常在项目根目录)
pip install -r requirements.txt

# 2) 运行 Agent (自动/交互模式)
# 针对特定分子运行
python agent/agent_run.py --smiles "CCCCC1=NC(Cl)=C(CO)N1CC2=CC=C(C3=CC=CC=C3C4=NNN=N4)C=C2"

# 或者启动自动模式 (扫描待办列表)
python agent/agent_run.py --auto

# 3) 结果在哪里
# 可视化图片和日志位于 output/agent_runs/
ls output/agent_runs/
```

---

## 目录

* [背景与目标](#背景与目标)
* [功能概览](#功能概览)
* [快速开始](#快速开始)
* [输入输出与数据契约](#输入输出与数据契约)
* [项目结构](#项目结构)
* [核心流程 How it works](#核心流程-how-it-works)
* [调试与排错](#调试与排错)
* [文档导航](#文档导航)

---

## 背景与目标

### 问题定义 (Why)
*   **目标**：解决复杂有机分子的多步逆合成路径规划问题。不仅要找到前体，还要评估路径的可行性、购买性以及化学风险。
*   **核心挑战**：传统单步模型缺乏全局视野，容易陷入死循环或提出无法购买的起始原料。
*   **非目标**：本项目暂不涉及以第一性原理（量子化学计算）为基础的反应能垒精确计算，主要依赖规则和数据驱动的模型建议。

### 产物 (What)
*   **输入**：目标分子的 SMILES 字符串。
*   **输出**：
    *   `StageResult`：每一级推断的详细 JSON 数据（含 LLM 分析、打分）。
    *   可视化图像：`.png` 格式的反应路径树。
    *   日志：详细的 `session.log` 和结构化 JSON 快照。

---

## 功能概览

*   ✅ **多步规划**：支持自动递归规划直到找到市售原料。
*   ✅ **混合智能**：结合单步逆合成模型（规则/模型）与 LLM（GPT-4/GLM-4）的高级推理能力。
*   ✅ **库存检查**：内置实时库存检索引擎，优先选择可购买前体。
*   ✅ **Deep Scan**：支持 "Glass-Box" 模式，展示 Agent 的每一步思考过程。
*   ✅ **人机交互**：支持专家在运行时注入策略（Manual Expert Injection）。

---

## 快速开始

### 环境要求
*   Python 3.8+
*   RDKit
*   PyTorch (如果是本地模型)
*   API Key (如果是调用云端 LLM，如 ZhipuAI)

### 运行
入口文件：`agent/agent_run.py`

**基本用法**:
```bash
python agent/agent_run.py --smiles "你的分子SMILES"
```

**自动批处理模式**:
```bash
python agent/agent_run.py --auto
```

---

## 输入输出与数据契约

### 输入
主要通过命令行参数或代码硬编码的标准分子（`DEFAULT_TARGET`）。

### 输出 (Artifacts)
所有产物默认保存在 `output/agent_runs/` 目录下。

1.  **Session Log**: `session_YYYYMMDD_HHMMSS.log` - 包含人类可读的推理全过程。
2.  **Route Visualizations**: `node_{path_id}_route_{idx}.png` - 反应路径图。
3.  **Checkpoints**: `snapshot_round_{n}.json` - 任务状态快照，用于恢复中断的任务。

---

## 项目结构

```text
MoleReact/multistep/agent/
├── agent_run.py          # [主入口] 核心调度器，CLI 入口
├── core/                 # [核心模块] ReAct 引擎实现
├── tools/                # [工具箱] 
│   ├── analysis.py       # 分子结构分析
│   ├── inventory.py      # 库存检查
│   └── ...
├── handlers/             # [处理器]
│   └── llm_selection.py  # LLM 交互与策略选择
├── managers/             # [状态管理]
│   ├── route_history.py  # 路径追踪与循环检测
│   └── session_logger.py # 日志与持久化
└── README.md             # 本文件
```

**建议阅读顺序**:
1.  `agent_run.py`: 理解 `CompleteWorkModuleRunner` 类的 `run_work_module` 方法。
2.  `handlers/llm_selection.py`: 理解 LLM 如何做决策。
3.  `tools/`: 浏览可用的原子能力。

---

## 核心流程 How it works

数据流如下：

```text
CLI Input 
   -> Agent Orchestrator (agent_run.py)
       -> 1. Retro Engine (Propose Candidates)
       -> 2. Inventory Check (Filter Stock)
       -> 3. Standardization & Repair (Fix SMILES)
       -> 4. Analysis Tool (Feature Extraction)
       -> 5. LLM Selection (Decide Next Step)
           -> Global Strategy Check
           -> History & Cycle Check
       -> 6. Visualization (Draw Route)
   -> Update Queue (BFS/DFS) & Loop
```

---

## 调试与排错

*   **API 错误**: 如果 LLM 调用失败，系统会自动降级为启发式（Heuristic）选择策略，并在日志中通过 warning 提示。
*   **循环检测**: 如果发现 `Cycle Detected` 日志，说明路线陷入重复，请检查 `RouteHistoryManager` 或手动介入。
*   **Deep Scan**: 在 `config.yml` 中开启 `enable_deep_scan: true` 可以看到详细的 ReAct 思考步骤。

---

## 文档导航

详细的模块文档请参考：

*   [**主程序说明 (agent_run.py)**](./agent_run.md)
*   [**工具箱文档 (tools/)**](./tools/TOOLS_README.md)
*   [**组件详解 (Managers & Handlers)**](./COMPONENTS.md)
