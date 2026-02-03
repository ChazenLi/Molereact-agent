# 04 API 参考手册

> **版本**: V3.6 | **目标读者**: 开发者

## 目录

1. [Agent 核心](#agent-核心)
2. [Agent Tools](#agent-tools)
3. [Core Engines](#core-engines)
4. [数据结构](#数据结构)
5. [入口脚本](#入口脚本)

---

## Agent 核心

### RetroSynthesisAgent

**位置**: `agent/agent.py`

```python
class RetroSynthesisAgent:
    """主代理类，负责编排逆合成工作流"""

    def __init__(self, config=None, engine=None, session=None, llm_client=None):
        """初始化代理并注册工具"""

    def run_work_module(self, target_smiles, stage=1, **kwargs) -> WorkModuleResult:
        """执行单阶段标准工作流 (生成 -> 检查 -> 选择 -> 可视化)"""

    def plan_interactive(self, target_smiles, **kwargs) -> Dict:
        """执行多阶段交互式规划"""
```

### CompleteWorkModuleRunner

**位置**: `agent/agent_run.py`

```python
class CompleteWorkModuleRunner:
    """CLI 接口主编排器"""

    def __init__(self, config_path: str = None):
        """初始化运行器"""

    def initialize(self) -> None:
        """初始化引擎、工具、管理器"""

    def run_work_module(self, target_smiles: str) -> StageResult:
        """运行单阶段工作模块"""

    def plan_interactive(self, target_smiles: str) -> Dict:
        """运行交互式规划"""

    def _cmd_reopen(self, path_id: str) -> None:
        """回溯并重开节点"""

    def _cmd_switch(self, queue_index: int) -> None:
        """切换队列分支"""

    def _cmd_expert(self, target: str, precursors: str) -> None:
        """注入专家步骤"""
```

---

## Agent Tools

### BaseTool

**位置**: `agent/tools/base.py`

```python
class BaseTool(ABC):
    """所有工具的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一标识符"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述 (供 LLM 使用)"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑"""
        pass
```

### ToolRegistry

**位置**: `agent/tools/base.py`

```python
class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        """初始化注册表"""

    def register(self, tool: BaseTool) -> None:
        """注册工具"""

    def get_tool(self, name: str) -> BaseTool:
        """获取工具"""

    def list_tools(self) -> List[str]:
        """列出所有工具"""

    def get_tool_descriptions(self) -> str:
        """获取工具描述 (供 LLM 使用)"""
```

### Chemistry Tools

**位置**: `agent/tools/chemistry.py`

```python
class RetroSingleStepTool(BaseTool):
    """单步逆合成工具"""

    def execute(self, target_smiles: str, topk_model: int = 10,
                topk_template: int = 10) -> Dict[str, Any]:
        """生成候选前体"""

class ReactionClassifyTool(BaseTool):
    """反应分类工具"""

    def execute(self, product: str, precursors: Tuple[str, ...]) -> Dict:
        """分类反应类型 (BISECT/MODIFY/MIXED)"""
```

### Analysis Tools

**位置**: `agent/tools/analysis.py`

```python
class MoleculeAnalysisTool(BaseTool):
    """分子分析工具"""

    def execute(self, smiles: str) -> Dict[str, Any]:
        """计算分子性质: MW, LogP, TPSA, HBD, HBA, RotatableBonds"""
```

### Inventory Tools

**位置**: `agent/tools/inventory.py`

```python
class StockCheckTool(BaseTool):
    """库存检查工具"""

    def execute(self, smiles_list: List[str]) -> Dict[str, bool]:
        """检查分子是否可购买"""
```

### Planning Tools

**位置**: `agent/tools/planning.py`

```python
class RouteSelectionTool(BaseTool):
    """路线选择工具"""

    def execute(self, candidates: List[Dict], context: Dict) -> Dict:
        """使用 LLM 选择最佳路线"""
```

---

## Core Engines

### SingleStepRetroEngine

**位置**: `single_step_engine.py`

```python
class SingleStepRetroEngine:
    """单步逆合成引擎"""

    def __init__(self, retro_model, forward_model, aizynth_session):
        """初始化引擎"""

    def propose_precursors(
        self,
        target_smiles: str,
        topk_model: int = 10,
        topk_template: int = 10
    ) -> Dict[str, Any]:
        """
        提议候选前体

        Returns:
            {
                "model_candidates": List[RetroStep],
                "template_candidates": List[RetroStep],
                "union": List[RetroStep],
                "stats": Dict
            }
        """
```

### AizynthSession

**位置**: `aizynthsession.py`

```python
class AizynthSession:
    """AiZynthFinder 会话封装"""

    def __init__(self, config_path: str):
        """初始化会话"""

    def is_in_stock(self, smiles: str) -> bool:
        """检查分子是否在库存中"""

    def expand_once(self, smiles: str, topk: int = 50) -> List[ExpansionResult]:
        """单步模板展开"""

    def search(self, smiles: str, **kwargs) -> SearchResult:
        """完整深度搜索"""
```

---

## 数据结构

### RetroStep

```python
@dataclass
class RetroStep:
    """单步逆合成结果"""
    target: str                    # 目标分子 SMILES
    precursors: Tuple[str, ...]    # 前体集合
    source: str                     # "model" | "template" | "both"
    confidence: float               # 综合置信度 (0-1)
    retro_score: float              # 逆合成分数
    fwd_score: float                # 前向验证分数
    reaction_type: str = "Unknown"  # 反应类型
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### CandidateRoute

```python
@dataclass
class CandidateRoute:
    """候选路线"""
    rank: int
    precursors: Tuple[str, ...]
    source: str
    reaction_type: str
    stock_status: Dict[str, bool]
    risk_analysis: Dict[str, Any]
    confidence_reasoning: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### StageResult

```python
@dataclass
class StageResult:
    """单阶段结果"""
    stage: int
    target_smiles: str
    model_candidates: List[Dict]
    template_candidates: List[Dict]
    stock_results: Dict
    llm_selected_top_n: List[Dict]
    unsolved_leaves: List[str]
    is_complete: bool
    path_id: str
    image_paths: List[str]
```

### WorkModuleResult

```python
@dataclass
class WorkModuleResult:
    """工作模块结果"""
    success: bool
    target_smiles: str
    stages_completed: int
    final_tree: Dict[str, Any]
    statistics: Dict[str, Any]
```

---

## 入口脚本

### agent_run.py

```bash
# 运行默认测试
python -m MoleReact.multistep.agent.agent_run

# 指定分子
python -m MoleReact.multistep.agent.agent_run --smiles "c1ccccc1CCO"

# 自动模式
python -m MoleReact.multistep.agent.agent_run --auto --smiles "..."

# 指定配置
python -m MoleReact.multistep.agent.agent_run --config custom_config.yml
```

### llm_retro_analyzer.py

```bash
# LLM 深度分析
python MoleReact/multistep/llm_retro_analyzer.py --smiles "CCO"

# 指定候选数量
python MoleReact/multistep/llm_retro_analyzer.py --smiles "CCO" --topk-model 5
```

---

**相关文档**:
- [系统架构](02_architecture.md) - 组件设计
- [开发者指南](05_developer_guide.md) - 开发环境配置

**文档更新**: 2026-02-02
