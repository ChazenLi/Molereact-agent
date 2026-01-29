# Tools Package Documentation

> **定位**：提供原子能力的工具库 (Atomic Capabilities)。

本目录包含 Agent 所需的各类化学分析、库存检查和标准化工具。所有工具均继承自 `base.BaseTool`，遵循统一的执行接口。

## 统一接口 (BaseTool)

```python
def execute(self, *args, **kwargs) -> Dict[str, Any]:
    ...
```

*   **Result Format**: 返回字典，通常包含 `status` ("success"/"error"), `message`, 和具体数据字段。

---

## 核心工具

### 1. MoleculeAnalysisTool (`analysis.py`)

**功能**: 计算分子基础理化性质和结构警报。

*   **输入**: `smiles` (str)
*   **输出**:
    ```json
    {
        "status": "success",
        "formatted_report": "MW: 320.5, LogP: 2.1 ...",
        "properties": {"MW": ..., "LogP": ..., "TPSA": ...}
    }
    ```
*   **规则**:
    *   MW > 500 或 LogP > 5 会被标记为 "Violations"。

### 2. StockCheckTool (`inventory.py`)

**功能**: 检查分子是否市售（In-Stock）。

*   **输入**: `smiles_list` (List[str])
*   **输出**:
    ```json
    {
        "results": [
            {"smiles": "...", "in_stock": true, "database": "ZINC"}
        ],
        "stock_rate": 0.5,
        "in_stock_count": 5
    }
    ```
*   **依赖**: 需要 `multistep/config.yml` 正确配置数据库路径，否则会 fallback 到 demo 模式或引发异常。

### 3. SupplyChainTool (`inventory.py`)

**功能**: 模拟查询供应商、价格和货期。

*   **输入**: `smiles_list` (List[str]), `preferred_region` (default "china")
*   **输出**:
    ```json
    {
        "materials": [
            {
                "smiles": "...",
                "vendors": [{"name": "Sigma", "price": ..., "lead_time": ...}],
                "best_price": 1200
            }
        ],
        "critical_path_items": ["... (Long lead time)"]
    }
    ```

### 4. SafetyCheckTool (`analysis.py`)

**功能**: 基于规则库检查高危试剂或官能团。

*   **输入**: `route_json` (包含 steps/reagents)
*   **输出**: `{"risk_level": "HIGH/LOW", "hazards": [...]}`

---

## 扩展指南

要添加新工具：

1.  在 `agent/tools/` 下新建脚本（如 `web_search.py`）。
2.  继承 `BaseTool` 并实现 `execute` 方法。
3.  在 `name` 属性中定义工具唯一标识。
4.  (可选) 将其注册到 `ToolRegistry` (如果在 ReAct 循环中使用)。
