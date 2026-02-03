# 05 开发者指南

> **版本**: V3.6 | **目标读者**: 贡献者、开发者

## 目录

1. [开发环境配置](#开发环境配置)
2. [代码规范](#代码规范)
3. [测试策略](#测试策略)
4. [添加新功能](#添加新功能)
5. [调试技巧](#调试技巧)

---

## 开发环境配置

### 系统要求

| 要求 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10/11, Linux, macOS 11+ | - |
| Python | 3.8 | 3.10 |
| 内存 | 8 GB | 16 GB |
| 存储 | 20 GB | SSD 50 GB |

### 依赖安装

#### requirements.txt

```txt
# 核心框架
torch>=1.12.0
rdkit>=2022.03.1
numpy>=1.21.0
pandas>=1.3.0

# 逆合成引擎
aizynthfinder>=4.0.0

# LLM 客户端
zai>=0.1.0  # 或 zhipuai>=4.0.0

# 配置
pyyaml>=6.0

# 可视化
matplotlib>=3.5.0
networkx>=2.6.0

# 开发工具
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

#### 安装步骤

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装 RDKit (使用 conda)
conda install -c conda-forge rdkit

# 安装其他依赖
pip install -r requirements.txt
```

---

## 代码规范

### Python 风格指南 (PEP 8)

```python
# 正确示例
def calculate_similarity(
    smiles_a: str,
    smiles_b: str,
    method: str = "tanimoto"
) -> float:
    """
    计算两个 SMILES 的相似度。

    Args:
        smiles_a: 第一个分子的 SMILES
        smiles_b: 第二个分子的 SMILES
        method: 相似度计算方法

    Returns:
        相似度分数 (0.0 到 1.0)
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        raise ValueError("Invalid SMILES")

    return similarity_score
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `RetroSynthesisAgent` |
| 函数/变量 | snake_case | `calculate_similarity` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RETRIES` |
| 私有方法 | 前缀单下划线 | `_private_method` |

---

## 测试策略

### 测试结构

```
multistep/
├── tests/
│   ├── __init__.py
│   ├── test_engine.py         # 引擎测试
│   ├── test_tools.py          # 工具测试
│   ├── test_handlers.py       # 处理器测试
│   └── integration/
│       └── test_full_workflow.py  # 集成测试
```

### 单元测试示例

```python
# tests/test_engine.py
import pytest
from multistep.single_step_engine import create_default_engine

class TestSingleStepEngine:
    def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = create_default_engine()
        assert engine is not None

    def test_propose_precursors(self):
        """测试前体生成"""
        engine = create_default_engine()
        result = engine.propose_precursors("CCO", topk_model=5)

        assert "model_candidates" in result
        assert len(result["model_candidates"]) <= 5
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_engine.py

# 生成覆盖率报告
pytest --cov=multistep --cov-report=html
```

---

## 添加新功能

### 添加新工具

**步骤 1**: 创建工具文件

```python
# multistep/agent/tools/new_tool.py
from .base import BaseTool
from typing import Dict, Any

class NewChemicalTool(BaseTool):
    """新化学工具的描述"""

    @property
    def name(self) -> str:
        return "NewChemicalTool"

    @property
    def description(self) -> str:
        return "执行新的化学分析功能"

    def execute(self, smiles: str, **kwargs) -> Dict[str, Any]:
        # 实现逻辑
        return {"status": "success", "data": result}
```

**步骤 2**: 导出工具

```python
# multistep/agent/tools/__init__.py
from .new_tool import NewChemicalTool

__all__ = ["NewChemicalTool", ...]
```

**步骤 3**: 注册到工具箱

```python
# multistep/agent/agent.py
def _register_default_tools(self):
    self.toolbox.register(NewChemicalTool())
```

### 集成新的 LLM Provider

```python
# multistep/agent/llm_adapters/openai_adapter.py

class OpenAIAdapter:
    def __init__(self, api_key: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def chat(self, messages: list, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
```

---

## 调试技巧

### 日志配置

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### VS Code 调试配置

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: MoleReact Agent",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/multistep/agent/agent_run.py",
            "args": ["--smiles", "CCO", "--auto"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```

---

## 贡献流程

1. Fork 项目
2. 创建功能分支
3. 编写代码和测试
4. 确保测试通过
5. 运行代码格式化
6. 提交并创建 Pull Request

---

**相关文档**:
- [API 参考](04_api_reference.md) - 接口文档
- [系统架构](02_architecture.md) - 架构设计

**文档更新**: 2026-02-02
