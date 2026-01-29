# -*- coding: utf-8 -*-
"""
Module: multistep.agent.agent
Called By: llm_retro_analyzer.py, tests
Role: Core Agent Logic Encapsulation

Functionality:
    Defines the `RetroSynthesisAgent` class which serves as the "Brain" of the system.
    It encapsulates:
    - Tool Management (via ToolRegistry)
    - State Management (Work Module execution)
    - Configuration (AgentConfig)

Key Classes:
    - RetroSynthesisAgent: High-level API for running retrosynthesis tasks.
    
Relations:
    - Uses `multistep.agent.tools` for actual execution.
    - Uses `multistep.agent.config` for settings.
"""
"""
Retrosynthesis Agent
=====================

主 Agent 类，协调各 Skill 完成逆合成分析工作流。

工作流:
1. Work Module (5 步封闭模块)
2. 人机交互
3. 迭代循环直到完成
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from .config import AgentConfig, InteractionMode, AgentMode
from .agent_types import StageResult, CandidateRoute
from .tools import (
    ToolRegistry, RetroSingleStepTool, StockCheckTool, 
    ReactionClassifyTool, RouteSelectionTool, VisualizationTool,
    MoleculeAnalysisTool, AtomMappingTool, ScaffoldAnalysisTool
)

logger = logging.getLogger(__name__)


@dataclass
class WorkModuleResult:
    """单个工作模块的结果"""
    stage_number: int
    target_smiles: str
    top_n_routes: List[Dict]
    unsolved_leaves: List[str]
    stage_image_path: Optional[str]
    is_complete: bool
    audit_info: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "stage_number": self.stage_number,
            "target_smiles": self.target_smiles,
            "top_n_routes": self.top_n_routes,
            "unsolved_leaves": self.unsolved_leaves,
            "stage_image_path": self.stage_image_path,
            "is_complete": self.is_complete,
            "audit_info": self.audit_info,
        }


class RetroSynthesisAgent:
    """
    逆合成分析 Agent
    
    整合各 Skill 完成迭代式逆合成规划，支持人机交互。
    """
    
    def __init__(
        self,
        config: AgentConfig = None,
        engine=None,
        session=None,
        llm_client=None,
    ):
        """
        初始化 Agent
        """
        self.config = config or AgentConfig.for_research()
        self._engine = engine
        self._session = session
        self.llm_client = llm_client
        
        # Initialize Tool Registry
        self.toolbox = ToolRegistry()
        self._register_default_tools()
        
        # State
        self.cumulative_route = {"stages": [], "metadata": {}}
        self.history: List[WorkModuleResult] = []
    
    def _register_default_tools(self):
        """Register default tools."""
        self.toolbox.register(RetroSingleStepTool(engine=self._engine))
        self.toolbox.register(StockCheckTool(session=self._session))
        self.toolbox.register(ReactionClassifyTool())
        # Planning tool needs client update if client changes
        self.toolbox.register(RouteSelectionTool(llm_client=self.llm_client))
        self.toolbox.register(VisualizationTool())
        self.toolbox.register(MoleculeAnalysisTool())
        self.toolbox.register(AtomMappingTool())
        self.toolbox.register(ScaffoldAnalysisTool())

    @property
    def engine(self):
        """Lazy load engine via tool if needed, or direct."""
        retro_tool = self.toolbox.get_tool("RetroSingleStep")
        if retro_tool._engine is None:
             from multistep.single_step_engine import create_default_engine
             retro_tool._engine = create_default_engine(
                config_path=self.config.config_path,
                retro_path=self.config.retro_model_path,
                fwd_path=self.config.forward_model_path,
                vocab_path=self.config.vocab_path,
            )
        return retro_tool._engine
    
    @property
    def session(self):
        """Lazy load session via tool."""
        stock_tool = self.toolbox.get_tool("StockCheck")
        # trigger lazy load inside tool if accessed, but here exposing property
        # Tool handles lazy load execution, but if we need object:
        if stock_tool._session is None:
             from multistep.aizynthsession import AizynthSession
             import os
             config_path = self.config.config_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
             stock_tool._session = AizynthSession(config_path)
        return stock_tool._session
    
    def run_work_module(
        self,
        target_smiles: str,
        stage: int = 1,
        top_n: int = 3,
        topk_model: int = 10,
        topk_template: int = 10,
    ) -> WorkModuleResult:
        """
        运行单个工作模块 (5 步封闭分析)
        """
        logger.info(f"=== Work Module Stage {stage} ===")
        logger.info(f"Target: {target_smiles[:50]}...")
        
        audit_info = {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "config_mode": self.config.mode.value,
        }
        
        # Step 1: 单步逆合成分析
        logger.info("Step 1: RetroSingleStep")
        retro_tool = self.toolbox.get_tool("RetroSingleStep")
        retro_result = retro_tool.execute(
            target_smiles,
            topk_model=topk_model,
            topk_template=topk_template
        )
        audit_info["retro_stats"] = retro_result["stats"]
        
        # Merge candidates
        all_candidates = (
            retro_result["model_candidates"] + 
            retro_result["template_candidates"]
        )
        
        # Fix: Convert dicts back to objects if needed, OR adjust downstream logic to handle dicts.
        # The tools return dicts. The original code dealt with objects sometimes but converted to dicts early.
        # Let's assume dicts are fine.
        
        if not all_candidates:
            logger.warning("No candidates found!")
            return WorkModuleResult(
                stage_number=stage,
                target_smiles=target_smiles,
                top_n_routes=[],
                unsolved_leaves=[],
                stage_image_path=None,
                is_complete=False,
                audit_info=audit_info,
            )
        
        # Step 2: 风险分析
        logger.info("Step 2: Risk analysis")
        risk_analysis = self._analyze_risks(all_candidates)
        audit_info["risk_summary"] = risk_analysis.get("summary", "")
        
        # Step 3: 可购买性检查
        logger.info("Step 3: StockCheck")
        all_precursors = []
        for cand in all_candidates:
            # Handle both dict and object if mixed (Tool returns dicts candidates)
            precursors = cand.get("precursors", [])
            all_precursors.extend(precursors)
        all_precursors = list(set(all_precursors))
        
        stock_tool = self.toolbox.get_tool("StockCheck")
        stock_result = stock_tool.execute(all_precursors)
        audit_info["stock_rate"] = stock_result["stock_rate"]
        
        # Step 4: LLM 筛选
        logger.info("Step 4: RouteSelection")
        plan_tool = self.toolbox.get_tool("RouteSelection")
        # Ensure client is set
        if self.llm_client and plan_tool._client is None:
            plan_tool.set_client(self.llm_client)
            
        selection_result = plan_tool.execute(
            all_candidates,
            stock_result,
            risk_analysis,
            top_n=top_n
        )
        
        top_n_routes = selection_result["selected_routes"]
        audit_info["selection_summary"] = selection_result.get("summary", "")
        
        # Step 5: 阶段可视化
        logger.info("Step 5: StageVisualization")
        if top_n_routes:
            best_precursors = top_n_routes[0].get("precursors", [])
        else:
            best_precursors = []
        
        viz_tool = self.toolbox.get_tool("StageVisualization")
        viz_result = viz_tool.execute(
            target_smiles,
            best_precursors,
            stage,
            output_dir=None # Default
        )
        
        # Update cumulative route
        # Tool returns simple dict, we might need to update own state
        # Original stage_visualize updated cumulative_route dict.
        # Here we manually update state
        if "stages" not in self.cumulative_route: self.cumulative_route["stages"] = []
        self.cumulative_route["stages"].append({
            "stage": stage,
            "target": target_smiles,
            "precursors": best_precursors
        })
        
        # Determine unsolved leaves
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_result["results"]}
        unsolved_leaves = [p for p in best_precursors if not stock_map.get(p, False)]
        
        # Check if complete
        is_complete = len(unsolved_leaves) == 0
        
        result = WorkModuleResult(
            stage_number=stage,
            target_smiles=target_smiles,
            top_n_routes=top_n_routes,
            unsolved_leaves=unsolved_leaves,
            stage_image_path=viz_result.get("image_path"),
            is_complete=is_complete,
            audit_info=audit_info,
        )
        
        self.history.append(result)
        logger.info(f"Stage {stage} complete. Solved: {is_complete}, Unsolved: {len(unsolved_leaves)}")
        
        return result
    
    def _analyze_risks(self, candidates: List[Dict]) -> Dict[str, Any]:
        """
        分析候选路线的风险
        
        TODO: 扩展更详细的风险分析
        """
        high_risk_count = 0
        for cand in candidates:
            risk = cand.get("metadata", {}).get("risk", {})
            if risk.get("risk_level") == "High":
                high_risk_count += 1
        
        return {
            "total_candidates": len(candidates),
            "high_risk_count": high_risk_count,
            "summary": f"{high_risk_count}/{len(candidates)} 候选存在高风险",
        }
    
    def format_stage_output(self, result: WorkModuleResult) -> str:
        """
        格式化阶段输出 (用于人机交互)
        
        Returns:
            Markdown 格式的输出
        """
        output = f"""
## 📊 阶段 {result.stage_number} 逆合成分析报告

### 🎯 当前目标
- **分子**: `{result.target_smiles}`

---

### 🔬 推荐路线 (Top-{len(result.top_n_routes)})

"""
        for i, route in enumerate(result.top_n_routes, 1):
            precursors = route.get("precursors", [])
            reason = route.get("reason", "")
            priority = route.get("priority", "MEDIUM")
            
            star = "⭐" if priority == "HIGH" else ""
            output += f"""#### 路线 {i} {star}
- **前体**: {' + '.join(f'`{p}`' for p in precursors)}
- **推荐理由**: {reason}

"""
        
        output += f"""---

### 🔍 审计信息
- 时间: {result.audit_info.get('timestamp', '')}
- 模式: {result.audit_info.get('config_mode', '')}
- 可购买率: {result.audit_info.get('stock_rate', 0):.1%}

---

### ⏭️ 状态
"""
        if result.is_complete:
            output += "✅ **路线完成** - 所有前体可购买\n"
        else:
            output += f"⏳ **待继续** - 未解决分子: {', '.join(f'`{s[:20]}...`' for s in result.unsolved_leaves)}\n"
        
        output += """
---

### 👨‍🔬 交互选项
- 📝 输入改进建议
- ▶️ 输入 `继续` 进入下一阶段
- ⏹️ 输入 `终止` 结束规划
"""
        return output
    
    def plan_interactive(
        self,
        target_smiles: str,
        max_stages: int = 10,
        interaction_callback: Callable[[str], str] = None,
    ) -> Dict[str, Any]:
        """
        交互式完整规划
        
        Args:
            target_smiles: 目标分子
            max_stages: 最大阶段数
            interaction_callback: 交互回调函数 (输入输出均为字符串)
            
        Returns:
            完整规划结果
        """
        current_target = target_smiles
        stage = 1
        
        while stage <= max_stages:
            # Run work module
            result = self.run_work_module(current_target, stage=stage)
            
            # Format output
            output = self.format_stage_output(result)
            
            # Check if complete
            if result.is_complete:
                return self._finalize_plan("complete", result)
            
            # Interaction
            if self._should_interact(stage):
                if interaction_callback:
                    user_input = interaction_callback(output)
                else:
                    print(output)
                    user_input = input("\n请输入 (继续/终止/建议): ").strip()
                
                if user_input.lower() in ["终止", "stop", "quit"]:
                    return self._finalize_plan("terminated", result)
                
                if user_input.lower() not in ["继续", "continue", ""]:
                    # Process user feedback (TODO: implement)
                    logger.info(f"User feedback: {user_input}")
            
            # Prepare next stage
            if result.unsolved_leaves:
                current_target = result.unsolved_leaves[0]
            else:
                break
            
            stage += 1
        
        return self._finalize_plan("max_stages_reached", result)
    
    def _should_interact(self, stage: int) -> bool:
        """判断是否需要在当前阶段交互"""
        mode = self.config.interaction.default_mode
        
        if mode == InteractionMode.AUTO:
            return False
        elif mode == InteractionMode.INTERACTIVE:
            return True
        else:  # MIXED
            return stage <= self.config.interaction.interaction_stages
    
    def _finalize_plan(self, status: str, last_result: WorkModuleResult) -> Dict[str, Any]:
        """生成最终规划结果"""
        return {
            "status": status,
            "total_stages": len(self.history),
            "is_solved": status == "complete",
            "cumulative_route": self.cumulative_route,
            "history": [r.to_dict() for r in self.history],
            "last_result": last_result.to_dict(),
        }
