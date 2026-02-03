# 07 配置说明

> **版本**: V3.6 | **目标读者**: 运维、开发者

## 目录

1. [配置文件](#配置文件)
2. [环境变量](#环境变量)
3. [Agent 配置](#agent-配置)
4. [LLM 配置](#llm-配置)
5. [场景配置](#场景配置)

---

## 配置文件

### config.yml

**位置**: `multistep/config.yml`

```yaml
# ========== 扩展模型配置 ==========
expansion:
  uspto:
    - uspto_model.onnx
    - uspto_templates.csv.gz
  ringbreaker:
    - uspto_ringbreaker_model.onnx
    - uspto_ringbreaker_templates.csv.gz

# ========== 过滤模型 ==========
filter:
  uspto: uspto_filter_model.onnx

# ========== 库存数据库 ==========
stock:
  zinc: zinc_stock.hdf5
  enamine: enamine_stock.hdf5  # 可选

# ========== Agent 配置 ==========
agent:
  # Deep Scan 开关
  enable_deep_scan: true

  # 最大阶段数
  max_stages: 10

  # 交互超时 (秒)
  interaction_timeout: 300

  # 场景配置
  scenario_profile: "INDUSTRIAL"

# ========== LLM 配置 ==========
llm:
  provider: "zhipuai"  # zhipuai | openai | anthropic
  model: "glm-4.7"
  temperature: 0.1
  max_tokens: 4096
  timeout: 120
  retry_attempts: 3
```

---

## 环境变量

### 必需变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `ZHIPUAI_API_KEY` | ZhipuAI API 密钥 | `42591f6c...` |

### 可选变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MOLEREACT_DB_PATH` | 知识库路径 | `output/agent_runs/knowledge_base.db` |
| `MOLEREACT_LOG_LEVEL` | 日志级别 | `INFO` |
| `MOLEREACT_OUTPUT_DIR` | 输出目录 | `output/agent_runs/` |

### 设置方式

```bash
# Linux/macOS
export ZHIPUAI_API_KEY="your_api_key_here"
export MOLEREACT_LOG_LEVEL="DEBUG"

# Windows
set ZHIPUAI_API_KEY=your_api_key_here
set MOLEREACT_LOG_LEVEL=DEBUG

# 或使用 .env 文件
echo "ZHIPUAI_API_KEY=your_api_key_here" > .env
```

---

## Agent 配置

### 运行模式

```python
# agent/config.py

@dataclass
class AgentConfig:
    """Agent 配置类"""

    # 运行模式
    auto_mode: bool = False
    enable_deep_scan: bool = True

    # 搜索限制
    max_stages: int = 10
    max_depth: int = 6
    queue_size: int = 50

    # 候选生成
    topk_model: int = 10
    topk_template: int = 10
    selection_count: int = 8

    # 交互设置
    interaction_timeout: int = 300

    # 场景配置
    scenario_profile: str = "INDUSTRIAL"
```

### 场景配置

```python
SCENARIO_PROFILES = {
    "ACADEMIC": {
        "description": "学术研究场景",
        "complexity": 0.4,
        "reactivity": 0.3,
        "selectivity": 0.3,
        "efficiency": 0.0,
        "pg_cost": 0.0,
    },
    "INDUSTRIAL": {
        "description": "工业生产场景",
        "complexity": 0.15,
        "reactivity": 0.15,
        "selectivity": 0.2,
        "efficiency": 0.25,
        "pg_cost": 0.25,
    },
    "PRODUCTION": {
        "description": "规模化生产场景",
        "complexity": 0.1,
        "reactivity": 0.1,
        "selectivity": 0.3,
        "efficiency": 0.3,
        "pg_cost": 0.2,
    }
}
```

---

## LLM 配置

### ZhipuAI (默认)

```python
# agent/llm_adapters/zhipuai_adapter.py

ZHIPUAI_CONFIG = {
    "provider": "zhipuai",
    "api_key": os.environ.get("ZHIPUAI_API_KEY"),
    "model": "glm-4.7",
    "temperature": 0.1,
    "max_tokens": 4096,
    "timeout": 120,
    "retry_attempts": 3,
    "retry_delay": 1.0,  # 指数退避基数
}
```

### OpenAI

```python
# agent/llm_adapters/openai_adapter.py

OPENAI_CONFIG = {
    "provider": "openai",
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 4096,
    "timeout": 120,
}
```

---

## 模型路径配置

### 默认模型路径

```python
# 相对于 MoleReact/ 目录
DEFAULT_MODEL_PATHS = {
    "retro": "retro_transformer/runs/20260117_004131/best_exact.pt",
    "forward": "retro_transformer/runs/20260116_122126/best_exact_forward.pt",
    "vocab": "retro_transformer/runs/20260117_004131/vocab.json",
}
```

### 自定义模型路径

```bash
# 通过环境变量
export MOLEREACT_RETRO_MODEL="/path/to/retro.pt"
export MOLEREACT_FORWARD_MODEL="/path/to/forward.pt"

# 或在代码中指定
engine = SingleStepRetroEngine(
    retro_model_path="/custom/path/retro.pt",
    forward_model_path="/custom/path/forward.pt"
)
```

---

## 数据库配置

### SQLite 知识库

```python
# agent/knowledge_base.py

DATABASE_CONFIG = {
    "type": "sqlite",
    "path": "output/agent_runs/knowledge_base.db",
    "tables": {
        "routes": "routes",
        "route_steps": "route_steps"
    },
    "indexes": [
        "CREATE INDEX idx_target_smiles ON routes(target_smiles);",
        "CREATE INDEX idx_session_id ON routes(session_id);",
        "CREATE INDEX idx_fingerprint ON routes(target_fingerprint);"
    ]
}
```

### Redis 缓存 (可选)

```python
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "cache_ttl": 3600,  # 1 小时
    "key_prefix": "molereact:"
}
```

---

## 验证配置

### 检查脚本

```bash
# 验证配置
python -c "
from multistep.agent.config import load_config
config = load_config()
print('Config loaded successfully')
print(f'Agent mode: {config.agent.scenario_profile}')
print(f'LLM provider: {config.llm.provider}')
"
```

---

**相关文档**:
- [快速开始](01_quick_start.md) - 安装与配置
- [开发者指南](05_developer_guide.md) - 开发环境

**文档更新**: 2026-02-02
