# -*- coding: utf-8 -*-
"""
Module: multistep.agent.agent_run
Called By: User (CLI Main Entry)
Role: Workflow Orchestrator / User Interaction Handler

Functionality:
    Orchestrates the complete 5-step retrosynthesis workflow:
    1. Generation (RetroSingleStep)
    2. Analysis (MoleculeAnalysis)
    3. Inventory Check (StockCheck)
    4. Planning (LLM/Heuristic Selection)
    5. Visualization (StageVisualization)

    Manages the "Agentic Loop" including:
    - Task Queue Management (Global Unsolved Queue)
    - User Interaction (CLI: Selection, Switch, Verify)
    - Session Logging and Report Generation
    
Key Classes:
    - CompleteWorkModuleRunner: Main controller class.

Features:
    - Integration of new `tools` package (RDKit, Inventory).
    - ReAct Loop Interface for dynamic analysis.
    - Robust Error Handling and Resume capability.

Usage:
    python agent/agent_run.py --auto
    python agent/agent_run.py --smiles "TargetSMILES"
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Setup path (MUST BE BEFORE IMPORTS)
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTISTEP_DIR = os.path.dirname(_SCRIPT_DIR)
_MOLEREACT_ROOT = os.path.dirname(_MULTISTEP_DIR)

if _MOLEREACT_ROOT not in sys.path:
    sys.path.insert(0, _MOLEREACT_ROOT)
if _MULTISTEP_DIR not in sys.path:
    # Also add parent dir for fallback
    sys.path.insert(0, _MULTISTEP_DIR)

try:
    from multistep.agent.config import AgentConfig, InteractionMode, AgentMode
    from multistep.agent.agent_react import ReActRetroAgent
    from multistep.agent.tools.visualization import VisualizationTool
except ImportError:
    from config import AgentConfig, InteractionMode, AgentMode
    from agent_react import ReActRetroAgent
    from tools.visualization import VisualizationTool
from dataclasses import dataclass, asdict


try:
    from multistep.agent.session_logger import SessionLogger
    from multistep.agent.prompts import get_system_role_prompt, get_selection_v2_prompt, get_smiles_repair_prompt
    from multistep.agent.smiles_standardizer import Standardizer
    from multistep.agent.tools.analysis import MoleculeAnalysisTool
    from multistep.agent.core.react import ReActSession
except ImportError:
    sys.path.append(os.path.join(_MULTISTEP_DIR, "agent"))
    from session_logger import SessionLogger
    from prompts import get_system_role_prompt, get_selection_v2_prompt, get_smiles_repair_prompt
    from smiles_standardizer import Standardizer
    from tools.analysis import MoleculeAnalysisTool
    from tools.advanced_analysis import toolbox as advanced_toolbox
    from core.react import ReActSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 默认测试分子
DEFAULT_TARGET = "FC1=CC=C(C=C1)[C@H](CN2C=NC3=C2C=CN3C4=CC=C(C=C4)N5CCN(C5)C6=CC(=NC=N6)C7=CN(C=N7)C)N"

ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY", "fe03944e939a4cd08084203ab88ccf8d.wF2T0LQjxkwR0lJv")
CHATGLM_MODEL = "glm-4.7"  # 使用最新模型，支持深度思考

# V2.2: 场景评分权重配置
SCENARIO_PROFILES = {
    "ACADEMIC": {
        "complexity": 0.4, "reactivity": 0.3, "selectivity": 0.3, "efficiency": 0.0, "pg_cost": 0.0
    },
    "INDUSTRIAL": {
        "complexity": 0.15, "reactivity": 0.15, "selectivity": 0.2, "efficiency": 0.25, "pg_cost": 0.25
    }
}
# 默认为工业模式
CURRENT_SCENARIO = "INDUSTRIAL"


@dataclass
class StageResult:
    """阶段结果"""
    stage: int
    target_smiles: str
    model_candidates: List[Dict]
    template_candidates: List[Dict]
    stock_results: Dict
    llm_selected_top_n: List[Dict]
    unsolved_leaves: List[str]
    is_complete: bool
    llm_analysis: str
    timestamp: str
    image_paths: List[str] = None  # 可视化图片路径列表

    def to_dict(self):
        return asdict(self)


class CompleteWorkModuleRunner:
    """完整工作模块运行器"""
    
    def __init__(self, use_llm: bool = True, auto_mode: bool = False):
        self.use_llm = use_llm
        self.auto_mode = auto_mode
        self.engine = None
        self.standardizer = Standardizer()
        self.analyzer = MoleculeAnalysisTool()
        self.llm_client = None
        self.history: List[StageResult] = []
        self.output_dir = os.path.join(_MULTISTEP_DIR, "output", "agent_runs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_logger = SessionLogger(self.output_dir)
        print(f"📄 Session Log: {self.session_logger.log_path}")
        self.stock_cache = {} # Cache for SMILES -> stock_status
        # self.visited_smiles = set() # 旧版: 全局去重 (弃用)
        # self.lineage_map = {} # 新版: 基于路径的谱系跟踪
        
    def initialize(self):
        """初始化引擎和 LLM 客户端"""
        print("\n" + "=" * 70)
        print(" 初始化 Agent 工作模块")
        print("=" * 70)
        
        # 加载逆合成引擎
        print("\n📦 加载逆合成引擎...")
        from multistep.single_step_engine import create_default_engine
        self.engine = create_default_engine()
        print("✅ 引擎加载完成")
        
        # 初始化 LLM 客户端 (使用新版 zai SDK)
        if self.use_llm:
            print("\n 初始化 LLM 客户端 (ZhipuAI)...")
            try:
                try:
                    from zai import ZhipuAiClient
                    self.llm_client = ZhipuAiClient(api_key=ZHIPUAI_API_KEY)
                    print("✅ LLM 客户端初始化成功 (glm-4.7 深度思考模式)")
                except ImportError:
                    print("⚠️ zai 未安装，尝试旧版 zhipuai...")
                    try:
                        from zhipuai import ZhipuAI
                        self.llm_client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
                        print("✅ LLM 客户端初始化成功 (zhipuai 兼容模式)")
                    except ImportError:
                        print("⚠️ zhipuai 也未安装，将使用启发式筛选")
                        self.llm_client = None
            except Exception as e:
                print(f"⚠️ LLM 初始化失败: {e}，将使用启发式筛选")
                self.llm_client = None
        
        # Initialize ReAct Agent (Subclass)
        # This allows us to use the encapsulated logic of the new agent
        self.agent = ReActRetroAgent(
            config=AgentConfig.for_research(), # Default config
            engine=self.engine,
            session=None, # Will be lazy loaded
            llm_client=self.llm_client
        )
        print("✅ ReAct 代理 (Sandbox) 已挂载")
    
    def run_work_module(self, target_smiles: str, stage: int = 1, topk: int = 10, history_context: str = "", path_id: str = "1") -> StageResult:
        """
        V3.0: Delegate fully to the Autonomous Meta-Controller.
        Replaces the old 5-step linear pipeline.
        """
        if hasattr(self.agent, "run_autonomous_loop"):
            print(f"\n🚀 Initiating Autonomous Agent V3.0 Protocol (ID: {path_id})...")
            final_report = self.agent.run_autonomous_loop(target_smiles)
            print(f"\n🏁 Autonomous Loop Finished.\nReport: {final_report}")
            
            # Check if really complete
            is_done = False
            if hasattr(self.agent, "tree") and self.agent.tree:
                is_done = not self.agent.tree.get_open_nodes()
                
            return StageResult(
                stage=stage,
                target_smiles=target_smiles,
                model_candidates=[], template_candidates=[], stock_results={}, 
                llm_selected_top_n=[], unsolved_leaves=[], is_complete=is_done,
                llm_analysis=final_report, timestamp=datetime.now().isoformat()
            )
        else:
            # Fallback if somehow agent is not V3
            return super().run_work_module(target_smiles, stage, topk, history_context, path_id=path_id)
    
    def _llm_select_top_n(
        self,
        target: str,
        candidates: List,
        stock_results: Dict,
        stage: int = 1,
        top_n: int = 7,  # Default increased to 7
        history_context: str = "",
        cumulative_route: Dict = None 
    ) -> Tuple[List[Dict], str]:
        """使用 LLM 筛选 Top-N 并提出新颖路线 (Holistic V2.0)"""
        print("  调用 ChatGLM 深度分析...")
        
        # 构建 prompt (完整 SMILES)
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_results["results"]}
        
        # 构建 Prompt (使用 prompts.py 模块)
        # 预先格式化候选路线文本
        candidate_blocks = []
        for i, cand in enumerate(candidates[:15], 1):
            source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
            precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
            confidence = cand.confidence if hasattr(cand, 'confidence') else cand.get('confidence', 0)
            
            precursor_lines = []
            analysis_report = [] # V2.2: Autonomous Analysis Report
            
            for p in precursors:
                status = "✅可购买" if stock_map.get(p, False) else "❌需合成"
                precursor_lines.append(f"  - `{p}` ({status})")
                
                # Autonomous Analysis
                try:
                    if self.analyzer:
                        props = self.analyzer.execute(p)
                        # Format compact analysis line
                        # Violations: specific check
                        v_count = props.get('LipinskiViolations', 0)
                        v_str = f"⚠️Violations={v_count}" if v_count > 0 else "✅Lipinski OK"
                        
                        report_line = (
                            f"     * Analysis for `{p}`: "
                            f"MW={props.get('MolecularWeight', 0):.1f}, "
                            f"LogP={props.get('LogP', 0):.2f}, "
                            f"TPSA={props.get('TPSA', 0):.1f}, "
                            f"{v_str}"
                        )
                        analysis_report.append(report_line)
                except Exception as e:
                    analysis_report.append(f"     * Analysis Error for `{p}`: {str(e)}")

            # --- [NEW] Advanced Analysis Toolbox Integration ---
            try:
                # 1. Run Advanced Analysis (AtomEconomy, ESOL, Bertz, etc.)
                adv_results = advanced_toolbox.analyze_candidate(target, precursors)
                
                # 2. Append to Analysis Report
                if adv_results:
                    analysis_report.append("\n     [Advanced Metrics]")
                    analysis_report.append(f"     {adv_results.get('formatted_report', '')}")
            except Exception as e:
                print(f"Warning: Advanced analysis failed: {e}")
                # analysis_report.append(f"     [Advanced Analysis Failed]: {str(e)}")
            # ---------------------------------------------------

            analysis_block = "\n".join(analysis_report) if analysis_report else "     (No analysis data available)"
            
            # Explicitly mark scores as deprecated/reference only per user request
            block = (
                f"### 路线 {i} [{source.upper()}]\n"
                f"   [Comparison]:\n"
                f"       Target   : {target}\n"
                f"       Precursors: {', '.join(precursors)}\n"
                f"   [Internal Scores - IGNORE]: Confidence={confidence:.4f} (Provided for ref only)\n"
                f"   [Precursors List]:\n" + "\n".join(precursor_lines) + "\n"
                f"   [Component Analysis Report - USE THIS]:\n{analysis_block}\n"
            )
            candidate_blocks.append(block)
        
        candidates_text = "\n\n".join(candidate_blocks)
        
        prompt = get_selection_v2_prompt(
            target=target,
            stage=stage,
            candidates_text=candidates_text,
            stock_rate=stock_results['stock_rate'],
            history_context=history_context,
            top_n=top_n
        )
        
        print(f"  📝 Prompt Length: {len(prompt)} chars")
        
        try:
            print(f"  🧠 Switching to ReAct Selection Mode provided by ReActRetroAgent...")
            
            # Use the ReAct Agent helper directly
            # Note: We pass the constructed candidates_text
            llm_text = self.agent.evaluate_candidates_with_react(
                target=target,
                candidates_text=candidates_text,
                context=history_context,
                criteria="Select Top-N valid routes. Ignore scores if analysis contradicts."
            )
            
            # ReAct returns the final answer string directly
            print(f"\n  📝 [ReAct 结论]:\n{llm_text[:200]}...")
            
            # Compatibility: Fake full_response list for downstream logic if needed, 
            # but we define llm_text directly so it's fine.
            full_response = [llm_text] 
            
            # 解析推荐 (从 LLM 响应中提取)
            # 解析 Task 1 的结果
            # Parse JSON Response (V3.4 Agentic Protocol)
            print("\n  🔍 Parsing JSON Response...", flush=True)
            
            import re
            import json
            
            results_data = {}
            llm_text = "".join(full_response)
            
            # Robust JSON Parsing Loop (V3.4 Fix)
            max_retries = 2
            last_error = ""
            current_attempt = 0
            
            while current_attempt <= max_retries:
                if current_attempt > 0:
                     print(f"  🔄 Retry Attempt {current_attempt}/{max_retries} due to JSON error...")
                     # In a real retry loop involving LLM, we would re-call the API here.
                     # However, since agent_full.py calls ReAct (external class) which handles the prompt internally,
                     # we can't easily re-inject the prompt here without refactoring ReAct.evaluate_candidates_with_react.
                     # For now, we will just try to repair the string locally or fail.
                     # But wait! We can't re-call LLM here easily because 'llm_text' is already returned.
                     # The retry loop logic in agent_run.py re-called the API.
                     # Here, we should rely on ReAct to be robust, OR accept that we only parse what we got.
                     # Given the structure, maybe we just improve local cleaning first.
                     pass 

                # Parsing Logic
                json_str = llm_text
                if "```json" in llm_text:
                    match = re.search(r"```json\s*(.*?)\s*```", llm_text, re.DOTALL)
                    if match: json_str = match.group(1)
                elif "```" in llm_text:
                    match = re.search(r"```\s*(.*?)\s*```", llm_text, re.DOTALL)
                    if match: json_str = match.group(1)
                
                try:
                    json_str = json_str.strip()
                    if not json_str.endswith("}"):
                        last_brace = json_str.rfind("}")
                        if last_brace != -1: json_str = json_str[:last_brace+1]
                    
                    results_data = json.loads(json_str)
                    break # Success
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON Parse Error: {e}")
                    # If we had a mechanism to feedback to ReAct we would used it.
                    # Since we don't, we break to avoid infinite local loop (logic difference from agent_run)
                    break
                
                current_attempt += 1
            
            final_selection_list = []
            seen_indices = set()
            
            # --- Logic V2: Process JSON 'routes' and 'shortlist' ---
            if results_data and "routes" in results_data and "shortlist" in results_data:
                # 1. Index all analyzed routes by ID
                analyzed_routes_map = {str(r.get("route_id", "0")): r for r in results_data["routes"]}
                
                # 2. Process Shortlist
                top_ids = results_data["shortlist"].get("top_ids", [])
                
                print(f"  🤖 LLM Shortlisted IDs: {top_ids}")
                
                for tid in top_ids:
                    # Clean ID (e.g., "1" or 1)
                    tid_str = str(tid).strip()
                    route_info = analyzed_routes_map.get(tid_str)
                    
                    if not route_info: 
                        print(f"  ⚠️ Warning: Shortlisted ID {tid} not found in routes detail.")
                        continue
                        
                    # Map back to original candidate index (1-based -> 0-based)
                    try:
                        # Assuming route_id corresponds to the "Route X" header which was i+1
                        # So ID "1" -> Index 0
                        orig_idx = int(tid_str) - 1
                    except:
                        continue
                        
                    if 0 <= orig_idx < len(candidates) and orig_idx not in seen_indices:
                        cand = candidates[orig_idx]
                        precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
                        source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
                        
                        # Extract rich reasoning from JSON
                        reason_parts = []
                        if route_info.get("rxn_type_from_FG"):
                            reason_parts.append(f"Type: {route_info['rxn_type_from_FG']}")
                        if route_info.get("selectivity_check", {}).get("risk"):
                            risk = route_info["selectivity_check"]["risk"]
                            reason_parts.append(f"Selectivity Risk: {risk}")
                            
                        # Check status
                        status = route_info.get("status", "PASS")
                        if status == "PASS_COND":
                            reason_parts.append(f"[CONDITIONALLY PASSED]: {route_info.get('revision_hint', 'Needs Revision')}")
                        elif status == "FAIL":
                            reason_parts.append(f"[FAILED]: {', '.join(route_info.get('fail_codes', []))}")
                        
                        full_reason = "; ".join(reason_parts)
                        
                        # Fix: Extract actual scores, preserving status
                        extracted_scores = route_info.get("scores", {})
                        if not isinstance(extracted_scores, dict): extracted_scores = {}
                        extracted_scores["status"] = status

                        final_selection_list.append({
                            "rank": len(final_selection_list) + 1,
                            "precursors": list(precursors),
                            "source": source,
                            "reason": full_reason,
                            "scores": extracted_scores, # Contain C/R/S + status
                            "original_index": orig_idx,
                            "analysis_data": route_info # Keep all audit data
                        })
                        seen_indices.add(orig_idx)

                # 3. Process Patched Routes (Auto-Spawn)
                # Look for 'patched_precursors' in ALL routes in JSON (not just shortlisted)
                for r in results_data["routes"]:
                    patch = r.get("patched_precursors")
                    # Check if patch exists and is valid list of strings
                    if patch and isinstance(patch, list) and len(patch) > 0 and isinstance(patch[0], str):
                          # In autonomous mode, we might just log this, BUT agent_full.py can also benefit
                         # from having these in the list.
                         # The autonomous loop uses these candidates in _generate_meta_prompt -> candidates text.
                         
                         # [NEW] Sanity Check for Hallucination
                         from rdkit import Chem
                         valid_patch = True
                         for smi in patch:
                            if not Chem.MolFromSmiles(smi):
                                print(f"  ⚠️ Warning: LLM suggested invalid SMILES '{smi}'. Discarding patch.")
                                valid_patch = False
                                break
                        
                         if not valid_patch: continue

                         feasibility = r.get("patch_feasibility", "Feasibility check passed")
                         
                         final_selection_list.append({
                            "rank": len(final_selection_list) + 1,
                            "precursors": patch,
                            "source": "LLM_Patch", # Special source ID
                            "reason": f"Auto-Patch from Route {r.get('route_id')}: {feasibility}",
                            "scores": {"status": "PATCHED"},
                            "original_index": -1, # Virtual
                            "reaction_type": r.get("rxn_type_from_FG", "Patched_Rxn")
                         })
            else:
                 print("  ⚠️ No valid 'routes' or 'shortlist' found in JSON. Falling back to simple parsing or system default.")

            # B. 填充剩余系统路线 (系统默认排序)
            for i, cand in enumerate(candidates):
                if i not in seen_indices and len(final_selection_list) < top_n:
                     precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
                     source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
                     conf = getattr(cand, 'confidence', 0.0)
                     final_selection_list.append({
                        "rank": len(final_selection_list) + 1,
                        "precursors": list(precursors),
                        "source": source,
                        "reason": f"System Candidate (Confidence: {conf:.4f}) - Not specifically prioritized by LLM JSON.",
                        "scores": {},
                        "original_index": i
                     })
                     seen_indices.add(i)

            # Note: Removed "LLM Novel" parsing for now as V3.4 prompt focuses on verifying Model/Template candidates.
            # If standardizer/patcher suggests new smiles, they would appear in "patched_route" logic requires Agent 2.
            
            # --- End Logic V2 ---
            # 最后再截断或者保留前 7+N
            # User wants 7 output. Let's return top 7 + Novel (if any)
            # 或者 strictly 7 system + novel.
            # "将每一次 llm 推荐的路线数量还是输出为七条"
            # Let's allow a bit more flexibility but default to showing top 7.
            
            # Logic to Combine and Slice (User Request: System(7) + Novel(All) = 7+N)
            
            # 1. Normalize source casing
            for r in final_selection_list:
                if r['source'].lower() == 'llm_novel':
                    r['source'] = 'llm_novel'
                elif r['source'] == 'LLM_Novel': # safe check
                    r['source'] = 'llm_novel'
            
            # 2. Separate candidates
            system_cands = [x for x in final_selection_list if x['source'] != 'llm_novel']
            novel_cands = [x for x in final_selection_list if x['source'] == 'llm_novel']
            
            # 3. Slice System routes to top_n (e.g., 7)
            # 确保系统路线最多只显示 user 设定的配额，避免过多
            system_selected = system_cands[:top_n]
            
            # 4. Append ALL Novel routes at the end
            # "预留席位: 无论系统推荐了多少条... 合并显示总路线"
            # "队尾追加... 绝不会被随意丢弃"
            selected = system_selected + novel_cands
            
            # 5. Re-rank (1..N)
            for i, r in enumerate(selected, 1):
                r['rank'] = i
            
            return selected, llm_text
            
        except Exception as e:
            print(f"  [FAIL] LLM Selection Error: {e}")
            return self._heuristic_select(candidates, stock_results, top_n), f"LLM error: {e}"
    
    def _standardize_and_repair_candidates(self, target: str, candidates: List) -> List[Dict]:
        """
        对候选路线进行标准化和智能修复
        1. 检查 RDKit 能否解析
        2. 如果解析失败，调用 LLM 尝试结合目标分子上下文进行逻辑修复
        3. 修复后再次验证，若仍失败则舍弃
        4. canonicalize 所有成功的 SMILES
        """
        print(f"  正在处理 {len(candidates)} 条候选路线的标准化...")
        
        cleaned_candidates = []
        
        for i, cand in enumerate(candidates, 1):
            # 获取前体 (注意处理对象或是字典)
            if hasattr(cand, 'precursors'):
                precursors = list(cand.precursors)
                source = getattr(cand, 'source', 'unknown')
            else:
                precursors = list(cand.get("precursors", []))
                source = cand.get("source", "unknown")
            
            valid_precursors = []
            is_broken = False
            
            for smi in precursors:
                # 1. 直接尝试规范化
                canon = self.standardizer.canonicalize(smi)
                if canon:
                    valid_precursors.append(canon)
                else:
                    # 2. 如果失败，尝试 LLM 修复
                    print(f"    [FAIL] Parse Error in Route {i} ({source}). Invalid SMILES: `{smi}`")
                    repaired = self._repair_broken_smiles(target, smi)
                    
                    if repaired and repaired != "INVALID":
                        # 3. 再次验证修复后的结果
                        canon_repaired = self.standardizer.canonicalize(repaired)
                        if canon_repaired:
                            print(f"    [SUCCESS] Repaired -> `{canon_repaired}`")
                            valid_precursors.append(canon_repaired)
                        else:
                            print(f"    [FAIL] Invalid Repair: `{repaired}` still cannot be parsed.")
                            is_broken = True
                            break
                    else:
                        print(f"    [FAIL] LLM could not repair the route.")
                        is_broken = True
                        break
            
            if not is_broken and valid_precursors:
                # 转换回字典以保持后续处理的一致性
                if hasattr(cand, 'to_dict'):
                    cand_dict = cand.to_dict()
                else:
                    cand_dict = cand.copy()
                
                cand_dict["precursors"] = valid_precursors
                cleaned_candidates.append(cand_dict)

        # 打印最终标准化的路线 (回复用户要求：输出标准化后的所有路线)
        print(f"\n   [DONE] Standardization & Repair Completed. {len(cleaned_candidates)} valid routes found:")
        for i, c in enumerate(cleaned_candidates, 1):
            src = c.get('source', 'unknown')
            ps = c.get('precursors', [])
            print(f"    [{i:2d}] {src.upper():<10} | {' + '.join(ps)}")
            
        return cleaned_candidates

    def _repair_broken_smiles(self, target: str, broken_smiles: str) -> Optional[str]:
        """利用 LLM 结合上下文强制修补 SMILES"""
        if not self.llm_client:
            return None
            
        print(f"    🧠 LLM 正在分析目标 `{target[:40]}...` 并重构前体结构...")
        prompt = get_smiles_repair_prompt(target, broken_smiles)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=CHATGLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是在有机合成和化学信息学领域极其严谨的专家，只输出 SMILES。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.1 # 极致确定性
            )
            
            repaired_text = response.choices[0].message.content.strip()
            # 过滤多余文字 (有些模型喜欢啰嗦)
            import re
            smi_match = re.search(r'([a-zA-Z0-9@\+\-\[\]\(\)\/\\=#%]{3,})', repaired_text)
            if smi_match:
                return smi_match.group(1)
            return repaired_text
        except Exception as e:
            logger.warning(f"SMILES 修复过程中发生错误: {e}")
            return None

    def _parse_llm_system_selection(self, llm_text: str) -> List[Dict]:
        """解析 Task 1: 解析 LLM 推荐的系统路线索引和理由"""
        import re
        results = []
        try:
            # 查找 "推荐路线:" 区块
            match = re.search(r'推荐路线[:：]([\s\S]*?)(?:### 任务 2|### Task 2|$)', llm_text)
            if not match:
                return []
            
            block = match.group(1)
            # 匹配行: "1. 路线 X:" 或 "1. 路线X"
            # 提取 X (原始索引)
            items = re.finditer(r'(\d+)\.\s*路线\s*(\d+)', block)
            
            for item in items:
                rank = int(item.group(1))
                route_idx = int(item.group(2)) # 这是 Prompt 中的 "路线 i"
                
                # 尝试提取理由和分数 (简单提取)
                # 寻找该项之后的内容，直到下一项
                start = item.end()
                # 找下一个 "d. " 改为找行首数字
                next_item = re.search(r'\n\d+\.\s*路线', block[start:])
                end = start + next_item.start() if next_item else len(block)
                
                content = block[start:end].strip()
                
                # 提取理由 (非贪婪匹配，直到打分行或下一项开始)
                # 修改正则以支持中英文“理由”标签，并更精准地截断
                reason_match = re.search(r'理由[:：]\s*([\s\S]+?)(?=\n\s*[C|c][:：]|\n\d+\.|$)', content)
                reason = reason_match.group(1).strip() if reason_match else ""
                
                # 清洗理由中的干扰项
                reason = re.sub(r'\[必须包含.*?\]', '', reason) # 移除提示占位符
                reason = reason.strip()
                
                # 提取分数 (新版: C/R/S/E/P)
                scores = {}
                # 尝试匹配 C:x R:x S:x E:x P:x
                score_match = re.search(r'[C|c][:：]\s*(\d+).*?[R|r][:：]\s*(\d+).*?[S|s][:：]\s*(\d+)(?:.*?[E|e][:：]\s*(\d+))?(?:.*?[P|p][:：]\s*(\d+))?', content)
                if score_match:
                    try:
                        scores['complexity'] = int(score_match.group(1))
                        scores['reactivity'] = int(score_match.group(2))
                        scores['selectivity'] = int(score_match.group(3))
                        if score_match.group(4): scores['efficiency'] = int(score_match.group(4))
                        if score_match.group(5): scores['pg_cost'] = int(score_match.group(5))
                        
                        # V2.2: 加权综合评分逻辑
                        weights = SCENARIO_PROFILES.get(CURRENT_SCENARIO, SCENARIO_PROFILES["ACADEMIC"])
                        weighted_sum = sum(
                            scores.get(k, 0) * weights.get(k, 0) 
                            for k in ["complexity", "reactivity", "selectivity", "efficiency", "pg_cost"]
                        )
                        # 如果是 Academic 且 E/P 为 0，重算权重分母
                        weight_sum_val = sum(weights.get(k, 0) for k in ["complexity", "reactivity", "selectivity", "efficiency", "pg_cost"] if k in scores or weights.get(k,0) > 0)
                        
                        scores['strategic'] = int(weighted_sum / (weight_sum_val if weight_sum_val > 0 else 1))
                        scores['feasibility'] = scores['strategic']
                    except (ValueError, TypeError):
                        pass
                else:
                    # 旧版兼容
                    strat_match = re.search(r'战略.*(\d+)', content)
                    feas_match = re.search(r'可行性.*(\d+)', content)
                    if strat_match: scores['strategic'] = int(strat_match.group(1))
                    if feas_match: scores['feasibility'] = int(feas_match.group(1))
                
                results.append({
                    "index": route_idx,
                    "reason": reason,
                    "scores": scores
                })
                
        except Exception as e:
            print(f"  ⚠️ 解析选路失败: {e}")
            
        return results

    def _parse_llm_novel_routes(self, llm_text: str) -> List[Dict]:
        """从 LLM 响应中解析新颖路线提案"""
        import re
        
        novel_routes = []
        
        try:
            # 尝试匹配 "LLM 新颖路线" 或类似模式
            # 查找 "反应类型:" 和 "前体 SMILES:" 模式
            
            # 模式1: 结构化格式 (增强版: 兼容 Markdown bold, list identifiers)
            # 允许前缀如 "1. " 或 "* " 或 "**"
            reaction_pattern = r'(?:[\*\-]?\s*\d+\.?\s*)?[\*]*反应类型[\*]*[：:]\s*(.+?)(?:\n|$)'
            # 允许 "前体 SMILES" 或 "前体" (注意 \s* 处理空格)
            precursor_pattern_token = r'[\*]*前体\s*(?:SMILES)?[\*]*[：:]'
            reason_pattern = r'[\*]*理由[\*]*[：:]\s*(.+?)(?:\n\n|\n(?=\d\.)|\n[CSR][:：]|$)'
            
            # 查找所有匹配
            text_lower = llm_text
            
            # 分段查找 (保留 split 逻辑不变，主要增强内部提取)
            sections = re.split(r'\n(?=\d+\.)', text_lower)
            
            for section in sections:
                if '反应类型' in section or 'LLM' in section.upper():
                    route = {}
                    
                    # 提取反应类型
                    reaction_match = re.search(reaction_pattern, section)
                    if reaction_match:
                        # 移除可能的 Markdown bold markers
                        rtype = reaction_match.group(1).strip()
                        route['reaction_type'] = rtype.replace('**', '').replace('__', '')
                    
                # 提取前体 (Generalized for 1 or more components)
                # Pattern: "前体 ... : [smiles] + [smiles]"
                # 策略: 捕获冒号后的整行，然后分割
                precursor_line_match = re.search(fr'{precursor_pattern_token}\s*([^\n]+)', section, re.IGNORECASE)
                if precursor_line_match:
                    raw_line = precursor_line_match.group(1).strip()
                    # 移除可能的反引号和 bold
                    clean_line = raw_line.replace('`', '').replace('**', '')
                    # 分割 (支持 + 或 ,)
                    parts = re.split(r'\s*[+,]\s*', clean_line)
                    parts = [p.strip() for p in parts if p.strip()]
                    
                    if parts:
                        route['precursors'] = " + ".join(parts)
                        route['precursors_list'] = parts
                    
                    # 提取理由 (增加长度限制到 200)
                    reason_match = re.search(reason_pattern, section, re.DOTALL)
                    if reason_match:
                        route['reason'] = reason_match.group(1).strip()[:200]
                    
                    # 提取打分 (C:x R:y S:z E:a P:b)
                    scores_match = re.search(r'C[:：]\s*(\d+)\s*R[:：]\s*(\d+)\s*S[:：]\s*(\d+)(?:\s*E[:：]\s*(\d+))?(?:\s*P[:：]\s*(\d+))?', section)
                    if scores_match:
                        route['scores'] = {
                            'complexity': int(scores_match.group(1)),
                            'reactivity': int(scores_match.group(2)),
                            'selectivity': int(scores_match.group(3))
                        }
                        if scores_match.group(4): route['scores']['efficiency'] = int(scores_match.group(4))
                        if scores_match.group(5): route['scores']['pg_cost'] = int(scores_match.group(5))
                    else:
                        route['scores'] = {}
                    
                    if route.get('precursors_list'):
                        novel_routes.append(route)
            
        except Exception as e:
            logger.debug(f"解析 LLM 新颖路线失败: {e}")
        
        return novel_routes
    
    def _heuristic_select(self, candidates: List, stock_results: Dict, top_n: int) -> List[Dict]:
        """启发式选择"""
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_results["results"]}
        
        scored = []
        for cand in candidates:
            score = 0
            precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
            source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
            confidence = cand.confidence if hasattr(cand, 'confidence') else cand.get('confidence', 0)
            
            # 来源加分
            if source == "template":
                score += 10
            elif source == "both":
                score += 15
            
            # 可购买加分
            stock_count = sum(1 for p in precursors if stock_map.get(p, False))
            score += stock_count * 20
            
            # 置信度
            score += confidence * 10
            
            scored.append((score, cand))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for rank, (score, cand) in enumerate(scored[:top_n], 1):
            precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
            source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
            
            results.append({
                "rank": rank,
                "precursors": list(precursors),
                "source": source,
                "reason": f"启发式评分: {score:.1f}",
            })
        
        return results
    
    def run_full_planning(self, target_smiles: str, max_stages: int = None) -> Dict:
        """
        运行完整的多阶段规划
        
        循环迭代工作模块直到：
        - 所有分子可购买 (成功)
        - 用户终止
        - (可选) 达到最大阶段数
        
        Args:
            target_smiles: 目标分子
            max_stages: 最大阶段数，默认为 None (无限制，由合成人员判断)
        """
        print("\n" + "*" * 70)
        print("****** 启动全流程逆合成规划 (MoleReact V2.2) ******")
        print("*" * 70)
        print(f"  目标分子: {target_smiles[:60]}...")
        print(f"  阶段限制: {max_stages if max_stages else '无限制 (人工控制)'}")
        print(f"  运行模式: {'自动' if self.auto_mode else '人机交互'}")
        
        current_target = target_smiles
        stage = 1
        history_context = ""
        
        # 尝试恢复会话
        # 尝试恢复会话
        if not self.auto_mode:
            latest_session = self.session_logger.get_latest_context()
            if latest_session["exists"]:
                print(f"\n[警告] 发现之前的会话记录 ({latest_session['session_id']})")
                resume_input = input("是否恢复会话? (输入路径或 [y/n]) [y]: ").strip()
                
                # ... 
                load_path = latest_session.get("path")
                if resume_input.lower() in ["", "y", "yes"]:
                    pass # 使用发现的最新的路径
                elif resume_input.lower() not in ["n", "no"]:
                    if os.path.exists(resume_input):
                        load_path = resume_input
                    else:
                        print(f"[失败] 未找到文件: {resume_input}. 跳过恢复。")
                
                if resume_input.lower() not in ["n", "no"]:
                    print("  [等待] 正在重建会话上下文中...")
                    
                    # 1. 恢复 Cumulative Route (核心状态)
                    restored_route = self.session_logger.restore_session_state(load_path)
                    
                    if restored_route and restored_route.get("stages"):
                        print(f"  [成功] 已恢复 {len(restored_route['stages'])} 个历史阶段数据。")
                        cumulative_route = restored_route
                        
                        # ...
                        
                        if global_unsolved_queue:
                            current_target = global_unsolved_queue[0][0] if isinstance(global_unsolved_queue[0], tuple) else global_unsolved_queue[0]
                            print(f"  ➡️ 恢复目标: {current_target} (待解队列: {len(global_unsolved_queue)})")
                        else:
                            print(f"  ⚠️ 上一阶段无未解决分子，可能已完成？重置目标为原始目标。")
                            current_target = target_smiles
                            
                        # 2. 恢复 LLM Context String
                        history_context = self.session_logger.load_history_context(load_path)
                        print(f"  📜 已恢复历史文本上下文 ({len(history_context)} 字符)")
                    else:
                        print("  ⚠️ 状态重建为空，将仅加载文本上下文。")
                        history_context = self.session_logger.load_history_context(load_path)
        
        print(f"  📝 Session Log loaded? {'Yes' if history_context else 'No'}")
        
        # 累积路线数据 (用于最终报告)
        # 如果未从 session 恢复，则初始化
        if 'cumulative_route' not in locals() or cumulative_route is None:
            cumulative_route = {
                "target": target_smiles,
                "stages": [],
                "status": "running",
                "global_unsolved_queue": [(target_smiles, [], "1")] # 存储 (SMILES, Lineage, PathID)
            }
            # 确保 current_target 初始正确
            current_target = target_smiles
            
        # 确保 global_unsolved_queue 存在 (兼容旧日志恢复)
        if "global_unsolved_queue" not in cumulative_route:
            cumulative_route["global_unsolved_queue"] = [(current_target, [], "1")] if current_target else []

        # 针对历史数据恢复的兼容处理 (如果存的是 SMILES 或 (SMILES, []) 则转为 (SMILES, [], PathID))
        fixed_queue = []
        for i, item in enumerate(cumulative_route["global_unsolved_queue"]):
            if isinstance(item, str):
                fixed_queue.append((item, [], f"{i+1}"))
            elif len(item) == 2:
                fixed_queue.append((item[0], item[1], f"1.{i+1}"))
            else:
                fixed_queue.append(item)
        cumulative_route["global_unsolved_queue"] = fixed_queue

        # 同步本地变量引用
        global_unsolved_queue = cumulative_route["global_unsolved_queue"]
        
        try:
            while True:
                # 检查最大阶段限制 (如果设置)
                if max_stages and stage > max_stages:
                    print(f"\n⚠️ 已达到最大阶段数 ({max_stages})")
                    break
                
                # 运行工作模块 (传入历史上下文 和 cumulative_route)
                
                # 从队列中取出当前目标
                if not global_unsolved_queue:
                    print("\n✅ 所有分子均已解决或可购买！")
                    break
                
                current_node = global_unsolved_queue.pop(0)
                current_target, current_lineage, current_path_id = current_node
                
                # Path-Aware Loop Detection
                if current_target in current_lineage:
                    print(f"\n[严重警告: 发现路线死循环] 分子 `{current_target}` 在其所属谱系路径中重复出现！")
                    print(f"  路径: {' -> '.join(current_lineage + [current_target])}")
                    if not self.auto_mode:
                        c_choice = input("是否强制重新处理此节点? (y/n) [n]: ").strip().lower()
                        if c_choice != 'y':
                            continue
                
                # 更新 Lineage
                new_lineage = current_lineage + [current_target]
                
                # 构建结构化的全景背景 Context (V2.2 强化)
                # 1. 全局进度 (已解出的分子对)
                global_progress = "### 1. 全局合成进度 (Global Progress):\n"
                if cumulative_route["stages"]:
                    for s in cumulative_route["stages"]:
                        t_smi = s.get("target", "Unknown")
                        p_smis = s.get("precursors", [])
                        pid = s.get("path_id", "Unknown")
                        global_progress += f"- [Node {pid}] {t_smi} => {' + '.join(p_smis)}\n"
                else:
                    global_progress += "- (初始目标，正在开启第一步)\n"
                
                # 2. 当前路径追踪 (Path-Aware Lineage)
                path_lineage = f"\n### 2. 当前分子谱系路径 (Current Path: {current_path_id}):\n"
                if current_lineage:
                    path_lineage += " -> ".join(current_lineage) + f" -> **{current_target}**"
                else:
                    path_lineage += f"**{current_target}** (根节点目标)"
                
                full_context = global_progress + path_lineage
                
                result = self.run_work_module(current_target, stage=stage, topk=10, history_context=full_context, path_id=current_path_id)
                
                # Check if complete
                if result.is_complete:
                    print("\n" + "*" * 70)
                    print("****** 规划任务圆满完成 ******")
                    print("*" * 70)
                    cumulative_route["status"] = "completed"
                    return self._generate_final_report("complete")
                
                # Interactive Mode
                if not self.auto_mode:
                    print("\n" + "*" * 60)
                    print(f"****** 人机交互决策 (节点: {current_path_id}) ******")
                # ==========================================================================================
                # 🔄 交互块 (Interaction Block)
                # ==========================================================================================
                interaction_active = True
                selected_route_idx = -1
                
                while interaction_active:
                    if self.auto_mode:
                        # 自动模式下跳过交互，默认选择第 0 条 (Top-1)
                        selected_route_idx = 0
                        interaction_active = False # Exit loop
                    else:
                        print("*" * 60)
                        # 显示待拆解分子队列
                        if global_unsolved_queue:
                            print(f"  📋 待合成分子队列 (待解分支):")
                            for idx, (mol, lineage, pid) in enumerate(global_unsolved_queue, 1):
                                depth = len(lineage)
                                print(f"    [Q{idx}] {mol[:40]}... (ID: {pid}, 深度: {depth})")
                        
                        print("-" * 60)
                        print("  操作指令:")
                        print("    [回车]         - 对当前目标使用路线 1")
                        print("    数字           - 选择当前目标的特定路线 (如: 2)")
                        print("    switch [Qn]    - 切换到另一个待解分子 (如: switch Q1)")
                        print("    list           - 查看当前完整的合成树方案与进度")
                        print("    reopen [ID]    - 重新打开并调整某个已处理的节点 (如: reopen 1.1)")
                        print("    q/stop/退出    - 终止规划")
                        print("    verify/验证    - 标记当前阶段待实验验证")
                        print("-" * 60)
                    
                        user_input = input(">>> (请选择或输入指令): ").strip()
                        
                        # Command 1: 终止
                        if user_input.lower() in ["终止", "stop", "quit", "q", "退出"]:
                            print("\n 用户终止规划")
                            return self._generate_final_report("terminated_by_user")
                        
                        # Command 2: 方案查看 (List) - 保持在交互块
                        if user_input.lower() in ["list", "方案", "查看"]:
                            print("\n" + "=" * 60)
                            print("📜 当前合成方案汇总 (Current Tree Summary):")
                            for s_idx, s in enumerate(cumulative_route["stages"], 1):
                                print(f"  [{s['path_id']}] {s['target'][:40]} => {' + '.join(s['precursors'])}")
                            if global_unsolved_queue:
                                print(f"  ⏳ 待合成分子: {len(global_unsolved_queue)} 个")
                            print("=" * 60)
                            continue # Stay in interaction loop
                        
                        # Command 3: 验证标记 (Verify) - 保持在交互块，支持进一步交互
                        if any(x in user_input for x in ["待验证", "验证", "verify"]):
                            print("  [提示] 已将当前阶段标记为待实验验证。")
                            self.session_logger.log_event(
                                title="实验验证标记 (Verification Required)",
                                content=f"用户将对节点 `{current_path_id}` (目标: `{current_target}`) 的决策标记为需要进一步实验室验证。",
                                level="WARNING"
                            )
                            
                            # V2.2: True Tool Use (ReAct Hook)
                            # User requested explicit verification, allowing LLM to dynamically call tools.
                            if self.use_llm and self.llm_client:
                                try:
                                    print("\n  🔍 [ReAct] Initializing Dynamic Analysis Session...")
                                    from multistep.agent.tools.base import ToolRegistry
                                    # Create specific registry for this session
                                    temp_registry = ToolRegistry()
                                    # Register available tools (Anal analysis is most relevant here)
                                    temp_registry.register(self.analyzer) # MoleculeAnalysisTool
                                    
                                    # Instantiate ReAct Session
                                    react = ReActSession(self.llm_client, temp_registry)
                                    
                                    # Define Goal
                                    goal = f"Verify the chemical stability and potential risks for molecule: {current_target}. Use the MoleculeAnalysisTool to get properties."
                                    
                                    # Run
                                    print("  🤖 Agent is thinking and acting...")
                                    react_result = react.run(goal)
                                    
                                    print(f"  📝 [ReAct Conclusion]: {react_result}")
                                    
                                    # Inject into context for next turn
                                    # We append this to a temporary note or history
                                    self.session_logger.log_event("ReAct Analysis", react_result, "INFO")
                                    print("  [Info] ReAct analysis result logged.")
                                    
                                except Exception as e:
                                    print(f"  [Error] ReAct execution failed: {e}")
                            else:
                                print("  [Info] LLM not available for dynamic analysis.")

                            print("  [提示] 您可以继续输入指令 (如选择路线或切换分支)。")
                            continue

                        # Command 4: 分支切换 (Switch) - 退出交互块，重新开始大循环
                        if user_input.lower().startswith("switch"):
                            target_match = re.search(r'[Qq](\d+)', user_input)
                            if target_match:
                                q_idx = int(target_match.group(1)) - 1
                                if 0 <= q_idx < len(global_unsolved_queue):
                                    # Logic to switch queue
                                    global_unsolved_queue.insert(0, current_node)
                                    selected_node = global_unsolved_queue.pop(q_idx + 1)
                                    global_unsolved_queue.insert(0, selected_node)
                                    
                                    self.session_logger.log_event(
                                        title="分支切换 (Branch Switch)",
                                        content=f"用户通过 `switch Q{q_idx+1}` 切换了分支。\n- 原目标: `{current_node[0]}`\n- 新目标: `{selected_node[0]}` (ID: {selected_node[2]})",
                                        level="INFO"
                                    )
                                    print(f"  🔄 分支已切换！下一个目标将是: {selected_node[0][:40]}")
                                    selected_route_idx = -999 # Signal to skip current route processing
                                    interaction_active = False # Break interaction loop
                                    break 
                                else:
                                    print(f"  [警告] 无效的任务编号: Q{q_idx+1}")
                                    continue
                            else:
                                print(f"  [用法参考] switch Q1 / switch 1")
                                continue

                        # Command 5: 节点重启 (Reopen) - 退出交互块，重新开始大循环
                        if user_input.lower().startswith("reopen"):
                            pid_match = re.search(r'([\d\.]+)', user_input[6:])
                            if pid_match:
                                target_pid = pid_match.group(1)
                                found_idx = -1
                                target_mol = None
                                target_lineage = []
                                for i, s in enumerate(cumulative_route["stages"]):
                                    if s["path_id"] == target_pid:
                                        found_idx = i
                                        target_mol = s["target"]
                                        target_lineage = s.get("lineage", [])
                                        break
                                
                                if found_idx != -1:
                                    remaining_stages = []
                                    for i, s in enumerate(cumulative_route["stages"]):
                                        if s["path_id"] == target_pid or s["path_id"].startswith(target_pid + "."):
                                            continue
                                        remaining_stages.append(s)
                                    cumulative_route["stages"] = remaining_stages
                                    
                                    self.session_logger.log_reopen(path_id=target_pid, target_smiles=target_mol, reason="用户主动请求重新评估")
                                    global_unsolved_queue.insert(0, (target_mol, target_lineage, target_pid))
                                    print(f"  ♻️ 节点 {target_pid} 已重新开启。")
                                    selected_route_idx = -999 # Signal skip
                                    interaction_active = False
                                    break
                                else:
                                    print(f"  [ERROR] Path ID {target_pid} not found.")
                                    continue
                            else:
                                print(f"  [USAGE] reopen 1.1")
                                continue

                        # Command 6: 路线选择 (Default) - 退出交互块，继续流程
                        import re
                        digit_match = re.search(r'(\d+)', user_input)
                        if digit_match:
                             route_num = int(digit_match.group(1))
                             if 1 <= route_num <= len(result.llm_selected_top_n):
                                 selected_route_idx = route_num - 1
                                 print(f"  [确定] 已切换至路线 {route_num}")
                                 interaction_active = False # Break interaction loop
                                 # Fall through to process selection
                             else:
                                 print(f"  [警告] 无效的路线编号 {route_num}")
                                 continue
                        elif user_input == "" or user_input.lower() in ["继续", "continue"]:
                            selected_route_idx = 0 # Default to 1
                            print(f"  [确定] 默认使用路线 1")
                            interaction_active = False
                        else:
                             # treat as note or invalid
                             print(f"  [提示] 无法识别指令 '{user_input}'. 输入数字选择路线，或 'list' 查看详情。")
                             continue

                # ==========================================================================================
                # 🔄 处理交互结果 (Process Result)
                # ==========================================================================================
                
                # Check for Skip signals (Switch/Reopen triggered)
                if selected_route_idx == -999:
                     continue # Skip to next outer loop iteration (Queue has been modified)

                # Process Route Selection
                if 0 <= selected_route_idx < len(result.llm_selected_top_n):
                    chosen_route = result.llm_selected_top_n[selected_route_idx]
                    route_desc = f"Stage {stage} 选择了路线 {selected_route_idx+1} ({chosen_route.get('source')})"
                    
                    # Update History Context
                    history_context = f"- 上一阶段 ({stage}) 决策: {route_desc}\n"
                    # Add user note if any (Simplified refactor: previously checked user_input again)
                    # self.session_logger.log_decision(...) 
                    
                    # Reconstruction stock_map
                    current_stock_map = {r["smiles"]: r["in_stock"] for r in result.stock_results.get("results", [])}
                    if "stock_check" in chosen_route:
                        for smi, info in chosen_route["stock_check"].items():
                            current_stock_map[smi] = info.get("in_stock", False)

                    # LLM Correction Logic (Keep existing)
                    precursors = chosen_route.get("precursors", [])
                    reason_text = chosen_route.get("reason", "")
                    correction_match = re.search(r'已修正\s*SMILES[:：]\s*(\[?[^\]\n]+\]?)', reason_text)
                    if correction_match:
                        correction_str = correction_match.group(1).strip()
                        clean_corr = correction_str.replace('`', '').replace('[', '').replace(']', '')
                        corr_parts = [p.strip() for p in re.split(r'\s*[+,]\s*', clean_corr) if p.strip()]
                        valid_corr_parts = []
                        for p in corr_parts:
                            canon_p = self.standardizer.canonicalize(p)
                            if canon_p: valid_corr_parts.append(canon_p)
                        if len(valid_corr_parts) == len(corr_parts):
                            precursors = valid_corr_parts
                            chosen_route["precursors"] = precursors

                    result.unsolved_leaves = [p for p in precursors if not current_stock_map.get(p, False)]
                    
                    # Log Decision
                    self.session_logger.log_decision(stage, selected_route_idx, chosen_route, "", global_unsolved_queue=global_unsolved_queue)

                    # Update Cumulative Route
                    cumulative_route["stages"].append({
                        "stage": stage,
                        "path_id": current_path_id,
                        "target": current_target,
                        "lineage": current_lineage,
                        "action": f"选择了路线 {selected_route_idx + 1}",
                        "precursors": precursors,
                        "unsolved_leaves": result.unsolved_leaves,
                        "reaction_type": chosen_route.get("reaction_type", ""),
                        "reason": chosen_route.get("reason", "")
                    })
            
                # Update Global Queue (Depth-First)
                if result.unsolved_leaves:
                    new_nodes = []
                    for child_idx, m in enumerate(result.unsolved_leaves, 1):
                        if m not in new_lineage:
                            child_pid = f"{current_path_id}.{child_idx}"
                            new_nodes.append((m, new_lineage, child_pid))
                    
                    cumulative_route["global_unsolved_queue"] = new_nodes + global_unsolved_queue
                    global_unsolved_queue = cumulative_route["global_unsolved_queue"]
                    print(f"\n[下一步] 新增分支: {len(new_nodes)}, 队列总计: {len(global_unsolved_queue)}")
                else:
                    if global_unsolved_queue:
                        next_node = global_unsolved_queue[0]
                        self.session_logger.log_event("分枝已解决", f"节点 {current_path_id} 完成。切换至: {next_node[0]}", "SUCCESS")
                        print(f"\n✨ [分枝解决] 节点 {current_path_id} 已成功拆解。")
                    else:
                        print("\n[顺利完成] 逆合成树解析完毕。")
                        break
                
                stage += 1
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断 (KeyboardInterrupt)")
            cumulative_route["status"] = "interrupted"
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
            cumulative_route["status"] = "error"
        finally:
            # 无论如何退出 (完成、终止、报错)，都生成最终可视化和报告
            print("\n🔄 正在生成最终会话报告与路径图...")
            self._finalize_session(cumulative_route)
        
        return self._generate_final_report("completed")

    def _convert_to_aizynth_dict(self, cumulative_route: Dict) -> Dict:
        """Convert cumulative route stages to AiZynthFinder tree dict"""
        target_to_stage = {}
        
        # Build lookup map: Target -> Stage Data
        for s in cumulative_route["stages"]:
            t = s["target"]
            target_to_stage[t] = s
        
        def build_node(mol_smiles: str, depth: int = 0, built_visited: set = None) -> Dict:
            if built_visited is None:
                built_visited = set()
                
            # 防御性编程：防止循环引用导致的无限递归
            if mol_smiles in built_visited or depth > 20:
                return {
                    "type": "mol",
                    "smiles": mol_smiles,
                    "is_chemical": True,
                    "in_stock": False,
                    "metadata": {"warning": "Recursive path detected or depth limit exceeded"}
                }
            
            built_visited.add(mol_smiles)
            
            node = {
                "type": "mol",
                "smiles": mol_smiles,
                "is_chemical": True, 
                "in_stock": self.stock_cache.get(mol_smiles, False) # 使用缓存的库存状态
            }
            
            # If this mol was a target in some stage, it means we expanded it
            if mol_smiles in target_to_stage:
                stage_data = target_to_stage[mol_smiles]
                precursors = stage_data.get("precursors", [])
                
                # Construct reaction child
                rxn_smiles = ".".join(precursors) + ">>" + mol_smiles
                
                reaction_node = {
                    "type": "reaction",
                    "smiles": rxn_smiles,
                    "metadata": {
                        "path_id": stage_data.get("path_id", "Unknown"),
                        "reaction_type": stage_data.get("reaction_type", "Unknown"),
                        "reason": stage_data.get("action", "")
                    },
                    "children": []
                }
                
                for p in precursors:
                    reaction_node["children"].append(build_node(p, depth + 1, built_visited.copy()))
                
                node["children"] = [reaction_node]
            
            return node

        if not cumulative_route.get("target"):
            return {}
            
        return build_node(cumulative_route["target"])

    def _finalize_session(self, cumulative_route: Dict):
        """Finalizing session: Generate summary visualization and log."""
        print("\n" + "*" * 70)
        print("****** 正在结项并生成汇总数据 ******")
        print("*" * 70)
        try:
            image_path = None
            
            # 1. 尝试使用 AiZynthFinder 风格高级可视化 (Reference 1.py)
            try:
                from aizynthfinder.reactiontree import ReactionTree
                tree_dict = self._convert_to_aizynth_dict(cumulative_route)
                
                if tree_dict:
                    img_name = f"tree_full_{datetime.now().strftime('%H%M%S')}.png"
                    img_path_aizynth = os.path.join(self.output_dir, img_name)
                    
                    # 生成图片
                    ReactionTree.from_dict(tree_dict).to_image().save(img_path_aizynth)
                    print(f"  🖼️ [AiZynth] 全景逆合成路线图已生成: {img_path_aizynth}")
                    image_path = img_path_aizynth
            except ImportError:
                print("  ⚠️ 未找到 aizynthfinder 库，尝试使用普通可视化。")
            except Exception as e:
                print(f"  ⚠️ AiZynth 可视化生成失败: {e}")
            
            # 2. 如果高级可视化失败或未启用，使用 stage_visualize (Fallback)
            if not image_path:
                from multistep.agent.tools import VisualizationTool
                # 提取最后一步的未解决分子作为 leaves
                last_stage = cumulative_route["stages"][-1] if cumulative_route["stages"] else {}
                leaves = last_stage.get("unsolved_leaves", [])
                
                # VisualizationTool.execute signature is slightly different
                # execute(self, target_smiles: str, selected_precursors: List[str], stage_number: int, output_dir: str = None)
                # Note: VisualizationTool.execute currently only visualizes ONE stage (precursors).
                # The original stage_visualize could handle cumulative_route to some extent or fallback?
                # Actually, original stage_visualize logic was simple: draw target and precursors.
                # It updated cumulative_route inline.
                
                viz_tool = VisualizationTool()
                viz_result = viz_tool.execute(
                    cumulative_route["target"], 
                    leaves, 
                    stage_number=0, 
                    output_dir=None
                )
                image_path = viz_result.get("image_path")
                image_path = viz_result.get("stage_image_path")
                print(f"  [OK] Full retrosynthesis map generated: {image_path}")
            
            # 3. 写入 Session Log
            self.session_logger.log_session_summary(cumulative_route, image_path)
            print(f"  [DONE] Session summary written: {self.session_logger.log_path}")
            
        except Exception as e:
            print(f"[FAIL] Final report generation failed: {e}")
    
    def _generate_final_report(self, status: str) -> Dict:
        """生成最终报告"""
        report = {
            "status": status,
            "total_stages": len(self.history),
            "history": [h.to_dict() for h in self.history],
            "timestamp": datetime.now().isoformat(),
        }
        
        # 保存报告
        report_path = os.path.join(
            self.output_dir,
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成 Markdown 报告
        self._generate_markdown_report(report_path.replace(".json", ".md"), status)
        
        print(f"\n[DONE] Final Report Saved: {report_path}")
        print(f"****** FINAL REPORT: {status.upper()} ******")
        print("*" * 60)
        print(f"Final Status: {status}")
        print(f"Total Stages Processed: {len(self.history)}")
        
        return report
    
    def _generate_markdown_report(self, md_path: str, status: str):
        """生成 Markdown 格式报告"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# MoleReact 逆合成规划报告\n\n")
            f.write(f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **状态**: {status}\n")
            f.write(f"- **总阶段数**: {len(self.history)}\n")
            
            # 如果有图片，使用相对路径
            md_dir = os.path.dirname(md_path)
            
            for stage_res in self.history:
                f.write(f"\n## Stage {stage_res.stage}: {stage_res.target_smiles[:40]}...\n\n")
                f.write(f"- **目标**: `{stage_res.target_smiles}`\n")
                f.write(f"- **完成状态**: {'✅ 完成' if stage_res.is_complete else '⏳ 待继续'}\n")
                if not stage_res.is_complete:
                    f.write(f"- **未解决分子**: {len(stage_res.unsolved_leaves)}\n")
                
                # 推荐路线表格
                f.write(f"\n### 🏆 推荐路线\n")
                f.write("| 排名 | 来源 | 前体 | 理由 |\n")
                f.write("|------|------|------|------|\n")
                
                for cand in stage_res.llm_selected_top_n:
                    rank = cand.get('rank', '-')
                    source = cand.get('source', 'unknown')
                    precursors = "<br>".join([f"`{p}`" for p in cand.get('precursors', [])])
                    reason = cand.get('reason', '').replace('\n', ' ')
                    f.write(f"| {rank} | {source} | {precursors} | {reason} |\n")
                
                # 可视化图片展示
                if stage_res.image_paths:
                    f.write(f"\n### 📊 路线可视化\n")
                    f.write("| 路线 | 可视化 |\n")
                    f.write("|------|--------|\n")
                    
                    for i, img_abs_path in enumerate(stage_res.image_paths):
                        try:
                            # 尝试计算相对路径
                            rel_path = os.path.relpath(img_abs_path, md_dir).replace("\\", "/")
                            # 找到对应的路线信息
                            if i < len(stage_res.llm_selected_top_n):
                                route_info = stage_res.llm_selected_top_n[i]
                                desc = f"**Route {route_info.get('rank')}**<br>Source: {route_info.get('source')}"
                            else:
                                desc = f"Route {i+1}"
                                
                            f.write(f"| {desc} | ![{desc}]({rel_path}) |\n")
                        except Exception as e:
                            f.write(f"| Route {i+1} | (图片路径错误: {e}) |\n")

                # LLM 分析详情 (折叠)
                if stage_res.llm_analysis:
                    f.write(f"\n<details>\n<summary>🧠 LLM 详细分析 (点击展开)</summary>\n\n")
                    f.write(stage_res.llm_analysis)
                    f.write(f"\n\n</details>\n")
                
                f.write(f"\n---\n")
            
        print(f"📄 Markdown 报告已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="完整 Agent 工作模块")
    parser.add_argument("--smiles", default=DEFAULT_TARGET, help="目标分子 SMILES")
    parser.add_argument("--stages", type=int, default=None, help="最大阶段数 (默认无限制)")
    parser.add_argument("--auto", action="store_true", help="自动模式 (不交互)")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM 分析")
    parser.add_argument("--single", action="store_true", help="只运行单个工作模块")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         MoleReact Agent - 完整工作模块                            ║
║     Complete Work Module with LLM Analysis                       ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    try:
        runner = CompleteWorkModuleRunner(
            use_llm=not args.no_llm,
            auto_mode=args.auto,
        )
        
        runner.initialize()
        
        if args.single:
            # 只运行单个工作模块
            result = runner.run_work_module(args.smiles, stage=1, topk=10)
            print(f"\n完成状态: {'✅ 可购买' if result.is_complete else '⏳ 待继续'}")
        else:
            # 运行完整规划
            report = runner.run_full_planning(args.smiles, max_stages=args.stages)
            print(f"\n最终状态: {report['status']}")
            print(f"总阶段数: {report['total_stages']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
