# 01 快速开始指南

> **目标读者**: 新用户 | **预计阅读时间**: 10 分钟

## 目录

1. [环境安装](#环境安装)
2. [快速运行](#快速运行)
3. [交互式命令](#交互式命令)
4. [输出说明](#输出说明)
5. [常见问题](#常见问题)

---

## 环境安装

### 系统要求

| 要求 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10/11, Linux, macOS 11+ | - |
| Python | 3.8 | 3.10 |
| 内存 | 8 GB | 16 GB |
| 存储 | 20 GB | SSD 50 GB |

### 安装步骤

#### 1. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 或使用 conda
conda create -n molereact python=3.10
conda activate molereact
```

#### 2. 安装 RDKit

```bash
# 使用 conda (推荐)
conda install -c conda-forge rdkit
```

#### 3. 安装核心依赖

```bash
pip install torch>=1.12.0
pip install aizynthfinder>=4.0.0
pip install zai  # 或 zhipuai>=4.0.0
pip install pyyaml pandas networkx matplotlib
```

#### 4. 配置 API Key

```bash
# 设置 ZhipuAI API Key
export ZHIPUAI_API_KEY="your_api_key_here"  # Linux/macOS
set ZHIPUAI_API_KEY=your_api_key_here       # Windows
```

---

## 快速运行

### 自动模式（推荐新手）

```bash
python -m MoleReact.multistep.agent.agent_run --auto --smiles "c1ccccc1CCO"
```

系统将自动完成：
1. 生成全局策略
2. 迭代展开合成树
3. 自动选择最优路线
4. 生成可视化报告

### 交互模式

```bash
python -m MoleReact.multistep.agent.agent_run --smiles "c1ccccc1CCO"
```

在每个决策点暂停，允许您：
- 选择特定路线
- 切换分支
- 回溯重开
- 注入专家步骤

### LLM 深度分析

```bash
python MoleReact/multistep/llm_retro_analyzer.py --smiles "c1ccccc1CCO"
```

生成详细的化学分析报告，不进行完整规划。

---

## 交互式命令

### 基础命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `[回车]` | 选择推荐的第1条路线 | 直接按回车 |
| `[数字 N]` | 选择第N条路线 | `2` |
| `list` | 查看当前合成树状态 | `list` |
| `q / quit / stop` | 结束规划 | `q` |
| `verify` | 标记需实验验证 | `verify` |

### 高级命令

#### `switch Q[n]` - 切换分支

切换到队列中的另一个分支：
```
switch Q2
```

#### `reopen [PathID]` - 回溯重开

撤销之前的决策并重新规划：
```
reopen 1.1
```
> **注意**: 此操作会清除该节点的所有子分支（幽灵节点清理）

#### `expert [T] >> [P]` - 专家注入

强制执行指定的化学步骤：
```
expert C1=CC=CC=C1 >> C1=CC=C(Br)C=C1.Mg
```

---

## 输出说明

### 输出目录结构

```
output/agent_runs/<session_id>/
├── session_<timestamp>.md       # 人类可读日志
├── session_<timestamp>.json     # 状态快照
├── node_<id>_route_<n>.png     # 可视化图片
└── knowledge_base.db            # 知识库
```

### 日志文件格式

**Markdown 日志** 包含：
- 会话元信息
- 每阶段的 LLM 分析
- 候选路线概览
- 决策结果

**JSON 快照** 包含：
- 完整的 `cumulative_route` 树
- 所有阶段结果
- 可用于会话恢复

---

## 常见问题

### Q1: RDKit 安装失败

```bash
# 解决方案：使用 conda
conda install -c conda-forge rdkit
```

### Q2: API 超时

```python
# 增加超时时间
# 在 agent_run.py 中修改
response = llm_client.chat(..., timeout=120)
```

### Q3: 内存不足

```bash
# 减少候选数量
python -m MoleReact.multistep.agent.agent_run --auto --smiles "..." --topk 5
```

### Q4: 如何恢复中断的会话？

系统会自动尝试从 JSON 快照恢复最新会话。

---

## 下一步

- [系统架构](02_architecture.md) - 了解整体架构
- [CLI 命令参考](08_cli_reference.md) - 完整命令列表
- [配置说明](07_configuration.md) - 高级配置选项

---

**文档更新**: 2026-02-02
