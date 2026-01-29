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
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Setup path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTISTEP_DIR = os.path.dirname(_SCRIPT_DIR)
_MOLEREACT_ROOT = os.path.dirname(_MULTISTEP_DIR)

if _MOLEREACT_ROOT not in sys.path:
    sys.path.insert(0, _MOLEREACT_ROOT)
if _MULTISTEP_DIR not in sys.path:
    sys.path.insert(0, _MULTISTEP_DIR)

try:
    from multistep.agent.session_logger import SessionLogger
    from multistep.agent.prompts import get_system_role_prompt, get_selection_v2_prompt, get_smiles_repair_prompt, get_global_strategy_prompt
    from multistep.agent.tools.fg_cycle_detector import CycleDetector
    from multistep.agent.managers.route_history import RouteHistoryManager
    from multistep.agent.handlers.llm_selection import LLMSelectionHandler
    from multistep.agent.managers.expert_memory import ExpertMemoryManager
    from multistep.agent.smiles_standardizer import Standardizer
    from multistep.agent.tools.advanced_analysis import AdvancedAnalysisToolbox # V3.5 Analysis
    from multistep.agent.tools.analysis import MoleculeAnalysisTool
    from multistep.agent.tools.advanced_analysis import toolbox as advanced_toolbox
    from multistep.agent.core.react import ReActSession
    from multistep.agent.tools.base import ToolRegistry, BaseTool
    from multistep.agent.tools.inventory import StockCheckTool
    from multistep.agent.knowledge_base import get_knowledge_base
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.path.append(os.path.join(_MULTISTEP_DIR, "agent"))
    from session_logger import SessionLogger
    from prompts import get_system_role_prompt, get_selection_v2_prompt, get_smiles_repair_prompt, get_global_strategy_prompt
    from tools.fg_cycle_detector import CycleDetector
    from managers.route_history import RouteHistoryManager
    from handlers.llm_selection import LLMSelectionHandler
    from smiles_standardizer import Standardizer
    from tools.analysis import MoleculeAnalysisTool
    from tools.advanced_analysis import toolbox as advanced_toolbox
    from core.react import ReActSession
    from knowledge_base import get_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Default test molecule
DEFAULT_TARGET = "CCCCC1=NC(Cl)=C(CO)N1CC2=CC=C(C3=CC=CC=C3C4=NNN=N4)C=C2"

ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY", "bf6ff4aee52342f0a2315ff6f37b77a8.LnB0oQdUrSiQSReg")
CHATGLM_MODEL = "glm-4.7"  # Flagship model with better reasoning

# V2.2: 场景评分权重配置
SCENARIO_PROFILES = {
    "ACADEMIC": {
        "complexity": 0.4, "reactivity": 0.3, "selectivity": 0.3, "efficiency": 0.0, "pg_cost": 0.0
    },
    "INDUSTRIAL": {
        "complexity": 0.15, "reactivity": 0.15, "selectivity": 0.2, "efficiency": 0.25, "pg_cost": 0.25
    }
}
# Default to INDUSTRIAL mode
CURRENT_SCENARIO = "INDUSTRIAL"


@dataclass
class StageResult:
    """Stage result"""
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
    path_id: str = "1"
    metadata: Dict[str, Any] = None
    image_paths: List[str] = None  # List of paths for visualized images

    def to_dict(self):
        return asdict(self)


class CompleteWorkModuleRunner:
    """Complete work module runner"""
    
    def __init__(self, use_llm: bool = True, auto_mode: bool = False):
        self.use_llm = use_llm
        self.auto_mode = auto_mode
        self.engine = None
        self.standardizer = Standardizer()
        self.analyzer = MoleculeAnalysisTool()
        self.cycle_detector = CycleDetector() # Maintained for legacy access if any
        self.route_manager = RouteHistoryManager()
        self.expert_memory = ExpertMemoryManager()
        self.llm_handler = None # Initialized in initialize()
        self.llm_client = None
        self.history: List[StageResult] = []
        self.output_dir = os.path.join(_MULTISTEP_DIR, "output", "agent_runs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_logger = SessionLogger(self.output_dir)
        logger.info(f"Session Log: {self.session_logger.log_path}")
        self.stock_cache = {} # Cache for SMILES -> stock_status
        self.global_strategy = None  # V3.5: LLM global planning result
        self.global_strategy_cached = False  # Track if strategy is from cache
        self.knowledge_base = get_knowledge_base()  # V3.5: Long-term memory
        # self.visited_smiles = set() # Old version: global deduplication (Deprecated)
        # self.lineage_map = {} # New version: path-based lineage tracking
        
        # Load Config for Deep Scan
        self.enable_deep_scan = False
        try:
             config_path = os.path.join(_MULTISTEP_DIR, "config.yml")
             if os.path.exists(config_path):
                 with open(config_path, 'r', encoding='utf-8') as f:
                     import yaml
                     cfg = yaml.safe_load(f)
                     if cfg and 'agent' in cfg and 'enable_deep_scan' in cfg['agent']:
                         self.enable_deep_scan = cfg['agent']['enable_deep_scan']
                         logger.info(f"Deep Scan (ReAct) Mode: {'ENABLED' if self.enable_deep_scan else 'DISABLED'}")
        except Exception as e:
             logger.warning(f"Config load failed: {e}")

    def initialize(self):
        """Initialize engine and LLM client"""
        logger.info("=" * 70)
        logger.info(" Initializing Agent Work Module")
        logger.info("=" * 70)
        
        # Load retrosynthesis engine
        logger.info("\nLoading retrosynthesis engine...")
        from multistep.single_step_engine import create_default_engine
        self.engine = create_default_engine()
        logger.info("Engine loaded successfully")
        
        # Initialize LLM client (ZhipuAI)
        if self.use_llm:
            logger.info("\nInitializing LLM client (ZhipuAI)...")
            try:
                from zai import ZhipuAiClient
                self.llm_client = ZhipuAiClient(api_key=ZHIPUAI_API_KEY)
                logger.info("LLM client initialized successfully (glm-4.7 reasoning mode)")
                
                # Initialize Handler
                self.llm_handler = LLMSelectionHandler(
                    llm_client=self.llm_client,
                    analyzer_tool=self.analyzer,
                    route_manager=self.route_manager
                )
                
            except ImportError:
                logger.warning("zai not installed, trying legacy zhipuai...")
                try:
                    from zhipuai import ZhipuAI
                    self.llm_client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
                    logger.info("LLM client initialized successfully (zhipuai legacy mode)")
                    
                    # Initialize Handler
                    self.llm_handler = LLMSelectionHandler(
                        llm_client=self.llm_client,
                        analyzer_tool=self.analyzer,
                        route_manager=self.route_manager
                    )
                    
                except ImportError:
                    logger.warning("zhipuai not installed, falling back to heuristic selection")
                    self.llm_client = None
                    self.llm_handler = None
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}, falling back to heuristic selection")
                self.llm_handler = None
    
    def manual_add_expert_step(self, command_str: str, session_context: Dict = None):
        """
        Public API: Input a manual step into the Knowledge Base (DB) AND Active Session.
        Format: "TARGET >> PRECURSOR1.PRECURSOR2"
        
        Args:
            command_str: The reaction string.
            session_context: Optional dict containing {
                'cumulative_route': ..., 
                'unsolved_queue': ...,
                'current_path_id': str,
                'current_lineage': List[str]
            }
        """
        try:
            if ">>" not in command_str:
                logger.error("Error: Format must be 'target >> precursor1.precursor2'")
                return
            
            parts = command_str.split(">>")
            t = parts[0].strip()
            p_str = parts[1].strip()
            p_list = [x.strip() for x in p_str.split(".")]
            
            logger.info(f"Processing expert strategy input: {t} -> {p_list}")
            
            # 1. Store in KnowledgeBase (Long Term Memory)
            # Analyze to get metadata (vector, etc.)
            analysis_meta = {}
            reaction_type = "Manual_Expert_Entry"
            try:
                # V3.5: Perform real chemical analysis on expert input
                # This ensures bond formation, reaction types, and vectors are calculated and stored
                logger.info("Performing advanced chemical analysis...")
                temp_toolbox = AdvancedAnalysisToolbox()
                report = temp_toolbox.analyze_candidate(t, p_list)
                
                # Extract key fields
                if "reaction_type" in report:
                    reaction_type = report["reaction_type"]
                
                analysis_meta = report
                logger.info(f"Analysis complete: Detected reaction type '{reaction_type}'")
                
            except Exception as e:
                logger.warning(f"Analysis tool call failed: {e}, storing text info only")

            single_stage = {
                "stage": 1,
                "target": t,
                "precursors": p_list,
                "reaction_type": reaction_type,
                "reason": "Manually added by human expert.",
                "analysis_metadata": analysis_meta,
                "source": "Human_Expert"
            }
            
            fake_route_data = {
                "target": t,
                "stages": [single_stage],
                "status": "manual_success",
                "global_strategy": {"note": "Manual Entry"}
            }
            
            if hasattr(self, 'knowledge_base') and self.knowledge_base:
                rid = self.knowledge_base.store_route(
                    target_smiles=t,
                    route_data=fake_route_data,
                    llm_score=10.0,
                    quality_notes="Manual Expert Injection",
                    session_id="manual_cli"
                )
                if rid > 0:
                    logger.info(f"Successfully stored in Knowledge Base (Route ID: {rid})")
            
            # 2. Inject into Active Session (Short Term Memory / Execution State)
            if session_context:
                cumulative_route = session_context.get('cumulative_route')
                unsolved_queue = session_context.get('unsolved_queue')
                pid = session_context.get('current_path_id', '?')
                lineage = session_context.get('current_lineage', [])
                
                if cumulative_route and unsolved_queue is not None:
                    # Update Stages
                    if "stages" not in cumulative_route: cumulative_route["stages"] = []
                    
                    # Create a "Sold" stage entry for the session history
                    new_stage_entry = {
                        "path_id": pid,
                        "target": t,
                        "precursors": p_list,
                        "action": "Manual Expert Solve",
                        "reaction_type": "Manual",
                        "metadata": {"source": "Human_Expert", "score": 10.0}
                    }
                    cumulative_route["stages"].append(new_stage_entry)
                    
                    # Update Queue (Remove target if strictly enforcing, but here just ADD precursors)
                    # We assume the caller handles popping the current target from queue if needed.
                    # Or we just push the NEW children to the FRONT.
                    
                    logger.info(f"Injecting {len(p_list)} precursors into synthesis queue...")
                    
                    # Add new precursors
                    for i, p in enumerate(p_list, 1):
                        child_id = f"{pid}.{i}"
                        new_lineage = lineage + [t] # Target is now parent
                        # Check inventory
                        in_stock = self.stock_cache.get(p, False)
                        if not in_stock:
                            unsolved_queue.insert(0, (p, new_lineage, child_id))
                            logger.info(f"Added to queue: {p[:15]}... (ID: {child_id})")
                        else:
                            logger.info(f"Resolved (in stock): {p[:15]}...")
                    
                    logger.info("Session updated. Agent will process new precursors in the next round.")

        except Exception as e:
            logger.error(f"Expert strategy addition failed: {e}")
            import traceback
            traceback.print_exc()

    def _call_global_strategy(self, target_smiles: str, history_context: str = "", force_regenerate: bool = False) -> Dict:
        """
        V3.5: Global Strategic Planning - Call LLM to generate strategy at workflow start
        
        Args:
            target_smiles: Target molecule SMILES
            history_context: Historical context (when resuming session)
            force_regenerate: Force regenerate (even if cached)
            
        Returns:
            Dict: Global planning JSON, including global_direction, loop_rules, hard_gates
        """
        # If cached and not forced to regenerate, use cache
        if self.global_strategy and not force_regenerate:
            logger.info("Using cached global strategy (use --regenerate to force regeneration)")
            self.global_strategy_cached = True
            return self.global_strategy
        
        if not self.llm_client or not self.use_llm:
            logger.warning("LLM not available, skipping global strategy")
            return {}
        
        logger.info("*" * 60)
        logger.info(" STEP 0: Global Strategy Planning (Agent-0) ")
        logger.info("*" * 60)
        logger.info(f"Target Molecule: {target_smiles[:50]}...")
        
        # Get component analysis report for target molecule
        component_report = ""
        try:
            if self.analyzer:
                analysis = self.analyzer.execute(target_smiles)
                if analysis.get("status") == "success":
                    component_report = analysis.get("formatted_report", "")
        except Exception as e:
            logger.warning(f"Target analysis failed: {e}")
        
        # V3.5 AGENT-0 V2: ReAct Upgrade
        # Instead of static prompt, we launch a dynamic investigation
        try:
             logger.info("Launching ReAct Global Strategist (Agent-0 V2)...")
             
             # Reuse registry from deep scan
             registry = self._build_react_registry()
             
             session = ReActSession(self.llm_client, registry, max_steps=6) # Allow more steps for strategy
             
             goal = f"""
             Analyze the target molecule '{target_smiles}'. 
             Your goal is to formulate a Global Synthesis Strategy.
             
             REQUIRED STEPS:
             1. Use tools to analyze structure (Cycle, Chiral Centers, Functional Groups).
             2. Identify KEY RISKS (e.g. sensitive groups, high strain).
             3. Formulate a Disconnection Strategy (Convergent vs Linear, where to cut).
             
             FINAL ANSWER FORMAT:
             You must output a strictly valid JSON string as your Final Answer.
             JSON Schema:
             {{
                "global_direction": {{
                    "preferred_convergences": ["..."],
                    "late_stage_sensitive_FG": ["..."],
                    "pg_principles": ["..."]
                }},
                "loop_rules": {{ "aba_rule": "..." }},
                "hard_gates": {{ "forbidden_reactions": ["..."] }}
             }}
             """
             
             # Context
             ctx = f"History: {history_context}\nPre-calculated Report: {component_report}"
             
             # Run ReAct
             final_answer_text = session.run(goal, context=ctx)
             
             # Parse result (Cleaning potential Markdown wrapper)
             json_str = final_answer_text.strip()
             if "```json" in json_str:
                 match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
                 if match: json_str = match.group(1)
             elif "```" in json_str:
                 match = re.search(r"```\s*(.*?)\s*```", json_str, re.DOTALL)
                 if match: json_str = match.group(1)
             elif "Final Answer:" in json_str:
                 json_str = json_str.split("Final Answer:")[-1].strip()

             strategy = json.loads(json_str)

             # 存储到实例
             self.global_strategy = strategy
             self.global_strategy_cached = False
             
             # 记录到 Session Logger (Enhanced)
             self.session_logger.log_event(
                 "🎯 Global Strategy (ReAct)",
                 f"**ReAct Investigation Complete**\n\n**Strategy JSON**:\n```json\n{json.dumps(strategy, indent=2, ensure_ascii=False)[:500]}...\n```",
                 level="SUCCESS"
             )
             
             logger.info("Global strategy planning complete (Agent-0 V2)")
             
             # 显示关键规划点
             if strategy.get("global_direction"):
                 direction = strategy["global_direction"]
                 logger.info(f"Preferred convergences: {direction.get('preferred_convergences', ['N/A'])[:2]}")
                 logger.info(f"Late-stage sensitive FG: {direction.get('late_stage_sensitive_FG', ['N/A'])[:2]}")
                 
             return strategy

        except Exception as e:
            logger.warning(f"Agent-0 V2 failed ({e}), falling back to legacy one-shot...")
            # Fallback code or empty return (Original logic had prompts)
            return {}
    
    def run_work_module(self, target_smiles: str, stage: int = 1, topk: int = 10, history_context: str = "", path_id: str = "1", cumulative_route: Dict = None) -> StageResult:
        """
        Run complete 5-step work module
        
        Steps:
        1. retro_single_step - Get candidates
        2. Risk analysis
        3. molecule_stock_check - Purchasability
        4. llm_select_top_n - LLM Selection (with history)
        5. stage_visualize - Visualization
        """
        logger.info("*" * 70)
        logger.info(f"WORKING MODULE: NODE {path_id} (STAGE {stage})")
        logger.info("*" * 70)
        logger.info(f"Target Molecule: {target_smiles[:60]}{'...' if len(target_smiles) > 60 else ''}")
        
        # Record stage start
        self.session_logger.log_stage_start(stage, target_smiles)
        
        if history_context:
            logger.info("HISTORY CONTEXT")
            logger.info(f"  {history_context}")
        
        timestamp = datetime.now().isoformat()
        
        # ========== Step 1: Single-step Retrosynthesis Analysis ==========
        logger.info("*" * 60)
        logger.info(" STEP 1: Single-step Retrosynthesis Analysis ")
        logger.info("*" * 60)
        
        result = self.engine.propose_precursors(
            target_smiles,
            topk_model=topk,
            topk_template=topk,
        )
        
        model_cands = result.get("model_candidates", [])
        template_cands = result.get("template_candidates", [])
        
        logger.info(f"Model candidates: {len(model_cands)}")
        logger.info(f"Template candidates: {len(template_cands)}")
        
        # Convert to dictionary format
        model_dicts = [c.to_dict() for c in model_cands]
        template_dicts = [c.to_dict() for c in template_cands]
        
        all_candidates = model_cands + template_cands
        
        # ========== Step 2: Stock Availability Check (Pre-check) ==========
        logger.info("*" * 60)
        logger.info(" STEP 2: Stock Availability Check ")
        logger.info("*" * 60)
        
        all_precursors = set()
        for cand in all_candidates:
            all_precursors.update(cand.precursors)
        
        stock_results = {"results": [], "stock_rate": 0}
        stock_map = {}  # smiles -> bool
        in_stock_count = 0
        
        print(f"  Checking {len(all_precursors)} unique precursors...")
        
        for smi in all_precursors:
            if smi in self.stock_cache:
                is_stock = self.stock_cache[smi]
            else:
                try:
                    is_stock = self.engine.template_engine.is_in_stock(smi)
                    self.stock_cache[smi] = is_stock
                except:
                    is_stock = False
            
            stock_map[smi] = is_stock
            stock_results["results"].append({"smiles": smi, "in_stock": is_stock})
            if is_stock: in_stock_count += 1
            
        stock_results["stock_rate"] = in_stock_count / len(all_precursors) if all_precursors else 0
        logger.info(f"Stock Coverage: {stock_results['stock_rate']:.1%}")

        # ========== Step 3: Candidate Repair & Expansion (Restored) ==========
        logger.info("*" * 60)
        logger.info(" STEP 3: Candidate Repair & Expansion ")
        logger.info("*" * 60)

        # 1. Standardize and Repair (LLM-based)
        # Ensure we pass the objects or dicts correctly. 
        # _standardize_and_repair_candidates expects a list of objects or dicts, returns a list of Dicts.
        final_candidates_for_llm = self._standardize_and_repair_candidates(target_smiles, all_candidates)
        
        # 2. Re-verify Stock for VALID Candidates Only
        # Sync stock_map for new/repaired precursors
        all_p_updated = set()
        for c in final_candidates_for_llm:
            # precursors is already a list of strings in the returned dicts
            all_p_updated.update(c.get('precursors', []))
        
        # Update cache for any new smiles
        for s in all_p_updated:
            if s not in self.stock_cache:
                try:
                    self.stock_cache[s] = self.engine.template_engine.is_in_stock(s)
                except:
                    self.stock_cache[s] = False
        
        updated_stock_results = {"results": [], "stock_rate": 0}
        updated_in_stock = 0
        for s in all_p_updated:
            is_stk = self.stock_cache.get(s, False)
            updated_stock_results["results"].append({"smiles": s, "in_stock": is_stk})
            if is_stk: updated_in_stock += 1
            
        updated_stock_results["stock_rate"] = updated_in_stock / len(all_p_updated) if all_p_updated else 0
        stock_results = updated_stock_results
        
        # V3.5: DEEP SCAN (Glass-Box Observability) - PRE-SELECTION
        # Now we audit ALL candidates (or top 20 limit) before asking LLM to select
        if self.enable_deep_scan and final_candidates_for_llm:
             logger.info("*" * 60)
             logger.info(" STEP 3.5: Full-Scale Deep Scan (Audit) ")
             logger.info("*" * 60)
             # Limit to 20 to avoid excessive latency if list is huge, but user asked for "All"
             # Let's trust the user or cap at 15 for sanity.
             final_candidates_for_llm = self._deep_scan_candidates(final_candidates_for_llm, target_smiles, limit=None)

        # ========== Step 4: Holistic LLM Selection Top-N ==========
        logger.info("*" * 60)
        logger.info(" STEP 4: Holistic LLM Selection Top-N ")
        logger.info("*" * 60)
        
        llm_analysis = ""
        selected_top_n = []
        
        if self.llm_client and self.use_llm:
            selected_top_n, llm_analysis = self._llm_select_top_n(
                target_smiles, final_candidates_for_llm, stock_results, stage=stage, top_n=7,
                history_context=history_context, cumulative_route=cumulative_route, path_id=path_id,
                global_strategy=self.global_strategy
            )
            # Record LLM analysis
            self.session_logger.log_llm_analysis(llm_analysis)
            self.session_logger.log_candidates_summary(selected_top_n)
        else:
            logger.info("Using heuristic selection...")
            selected_top_n = self._heuristic_select(all_candidates, stock_results, top_n=7)
            llm_analysis = "Heuristic selection (No LLM)"
            self.session_logger.log_candidates_summary(selected_top_n)
        
        # V3.5: DEEP SCAN - Moved to Pre-Selection (Above)
        # if self.enable_deep_scan and selected_top_n:
        #      selected_top_n = self._deep_scan_candidates(selected_top_n, target_smiles)

        # V3.5: Show global analysis summary
        if cumulative_route and cumulative_route.get("current_stage_analysis"):
             ga = cumulative_route["current_stage_analysis"]
             logger.info("[Global Analysis Summary]")
             logger.info(f"    • Synthesis direction: {ga.get('synthesis_direction', 'N/A')}")
             logger.info(f"    • Key challenges: {ga.get('key_challenges', 'N/A')}")
             logger.info(f"    • Recommended strategy: {ga.get('recommended_strategy', 'N/A')}")

        logger.info(f"Recommended Top-{len(selected_top_n)}:")
        for i, sel in enumerate(selected_top_n, 1):
            source = sel.get('source', 'N/A')
            # Highlight Expert/KB Support
            if source == 'Human_Expert' or source == 'Knowledge_Base' or sel.get('metadata', {}).get('score', 0) >= 9.0:
                 source_display = f"✨ {source} (High Confidence)"
            else:
                 source_display = source

            print(f"    [{i}] {source_display} | 前体: {sel.get('precursors', [])[:2]}")
            if sel.get('reason'):
                logger.info(f"        Rationale: {sel['reason']}") # Show fully
        
        # ========== Step 5: Route Visualization ==========
        logger.info("-" * 60)
        logger.info(f"Step 5: Recommended Route Visualization (ID: {path_id})")
        logger.info("-" * 60)
        
        # Identify unsolved molecules (from each route)
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_results["results"]}
        unsolved_leaves = []
        route_images = []
        
        # Generate separate visualization for each recommended route
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            logger.info(f"Generating visualizations for {len(selected_top_n)} routes...")
            
            for route_idx, route in enumerate(selected_top_n, 1):
                precursors = route.get("precursors", [])
                source = route.get("source", "N/A")
                
                # Collect unsolved molecules
                for p in precursors:
                    if not stock_map.get(p, False) and p not in unsolved_leaves:
                        unsolved_leaves.append(p)
                
                # Build visualization
                mols = []
                legends = []
                
                # Target
                target_mol = Chem.MolFromSmiles(target_smiles)
                if target_mol:
                    mols.append(target_mol)
                    legends.append(f"Target ({path_id})")
                
                # Precursors
                for p in precursors:
                    mol = Chem.MolFromSmiles(p)
                    if mol:
                        status = "✓" if stock_map.get(p, False) else "✗"
                        mols.append(mol)
                        legends.append(f"{status} P{len(mols)-1}")
                
                if len(mols) > 1:
                    # 使用 path_id 和 route_idx 确保文件名唯一且与数据树一致
                    # Fix: Ensure .png extension is not removed by replace
                    safe_pid = str(path_id).replace(".", "_")
                    filename = f"node_{safe_pid}_route_{route_idx}.png"
                    img_path = os.path.join(self.output_dir, filename)
                    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300), legends=legends)
                    img.save(img_path)
                    route_images.append(img_path)
                    logger.info(f"[DATA] Node {path_id} Route {route_idx} [{source}]: {img_path}")
            
            logger.info("Successfully generated visualizations.")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            
        is_complete = len(unsolved_leaves) == 0
        
        # ========== STAGE EVALUATION SUMMARY ==========
        logger.info("*" * 60)
        logger.info(" STAGE EVALUATION SUMMARY ")
        logger.info("*" * 60)
        
        # Categorize and display routes
        system_routes = [r for r in selected_top_n if r.get('source') != 'llm_novel']
        llm_routes = [r for r in selected_top_n if r.get('source') == 'llm_novel']
        
        logger.info("*" * 80)
        logger.info(f" CONSOLIDATED RECOMMENDATIONS (Top-{len(selected_top_n)}) ")
        logger.info("*" * 80)
        
        for i, r in enumerate(selected_top_n, 1):
            source = r.get('source', 'Unknown')
            # Use text-based markers
            source_mark = "[NOVEL]" if source == 'llm_novel' else f"[SYSTEM:{source}]"
            
            # Stock info
            precursors = r.get('precursors', [])
            stock_status = sum(1 for p in precursors if stock_map.get(p, False))
            stock_str = f"[STOCK] {stock_status}/{len(precursors)}"
            
            # Print Header
            print(f"  {i}. {source_mark} {stock_str}")
            
            # Print Precursors
            for p in precursors:
                status = "[S]" if stock_map.get(p, False) else "[M]"
                print(f"     {status} {p}")
                
            # Print Scores if available
            scores = r.get('scores', {})
            if scores:
                c = scores.get('complexity', '-')
                r_score = scores.get('reactivity', '-')
                s = scores.get('selectivity', '-')
                logger.info(f"     Scores: C:{c} | R:{r_score} | S:{s}")
            
            # Print Reason (Full)
            if r.get('reason'):
                 print(f"     [INFO] Reason: {r['reason']}") # No truncation
            
            print(f"{'*' * 80}")
            
        logger.info(f"Total candidates: {len(selected_top_n)} (Novel: {len(llm_routes)})")
        logger.info(f"Overall Stock Rate: {stock_results['stock_rate']:.1%}")
        
        if is_complete:
            logger.info(f"All precursors in stock, Stage {stage} complete!")
        else:
            logger.info(f"Need disassembly: {len(unsolved_leaves)} molecules")
            for j, leaf in enumerate(unsolved_leaves[:5], 1):
                logger.info(f"     [{j}] `{leaf}`")
        
        stage_result = StageResult(
            stage=stage,
            target_smiles=target_smiles,
            model_candidates=model_dicts,
            template_candidates=template_dicts,
            stock_results=stock_results,
            llm_selected_top_n=selected_top_n,
            unsolved_leaves=unsolved_leaves,
            is_complete=is_complete,
            llm_analysis=llm_analysis,
            timestamp=timestamp,
            path_id=path_id,
            image_paths=route_images
        )
        
        self.history.append(stage_result)
        return stage_result
    
    def _llm_select_top_n(
        self,
        target: str,
        candidates: List,
        stock_results: Dict,
        stage: int = 1,
        top_n: int = 7,  # Default increased to 7
        history_context: str = "",
        cumulative_route: Dict = None,
        path_id: str = "1",
        global_strategy: Dict = None
    ) -> Tuple[List[Dict], str]:
        """Delegates to LLMSelectionHandler."""
        if not self.llm_handler:
             logger.warning("No LLM handler initialized.")
             return candidates[:top_n], "No LLM"
             
        return self.llm_handler.select_top_n(
            target, candidates, stock_results, stage, top_n, history_context, cumulative_route, path_id, global_strategy
        )

    # --- DELEGATED METHODS (Wrapped for compatibility) ---

    def _get_path_prefixes(self, path_id: str) -> List[str]:
        return self.route_manager._get_path_prefixes(path_id)

    def _collect_path_vector_history(self, cumulative_route: Dict, path_id: str) -> List[Dict]:
        return self.route_manager.collect_path_vector_history(cumulative_route, path_id)

    # --- V3.5 GLASS-BOX OBSERVABILITY ---
    
    def _build_react_registry(self) -> ToolRegistry:
         registry = ToolRegistry()
         
         # Wrapper for Advanced Analysis
         class AnalysisTool(BaseTool):
             @property
             def name(self):
                 return "ChemicalAnalysis"
             
             @property
             def description(self):
                 return "Analyzes a molecule (MW, LogP, Alerts). Input: SMILES"
                 
             def execute(self, args):
                 # Hardening: Handle if ReAct splits args into list
                 target = args
                 if isinstance(args, (list, tuple)):
                     target = args[0] if args else ""
                 return advanced_toolbox.analyze_candidate(target, []) # Simplified
         
         # Stock Tool
         stock_tool = StockCheckTool(self.engine.template_engine) # Need engine access
         
         registry.register(AnalysisTool())
         registry.register(stock_tool)
         return registry

    def _deep_scan_candidates(self, candidates: List[Dict], target: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Run ReAct Session on candidates to provide 'Glass-Box' thinking trace.
        If limit is None, scans ALL candidates (Phase 2 Full Audit).
        REVERTED to Per-Candidate Autonomous Logic (V2.2 Behavior + Retry)
        """
        if not self.enable_deep_scan or not self.llm_client:
            return candidates

        scan_count = len(candidates) if limit is None else min(limit, len(candidates))
        logger.info("="*60)
        logger.info(f" DEEP SCAN: Initializing ReAct Audit for {scan_count} Candidates... ")
        logger.info("="*60)
        
        registry = self._build_react_registry()
        
        subset = candidates[:limit] if limit is not None else candidates
        
        for i, cand in enumerate(subset, 1):
             logger.info(f"Scanning Candidate # {i}...")
             precursors = cand.get('precursors', [])
             p_str = " + ".join(precursors)
             source = cand.get('source', 'unknown')
             
             # Create Session (One per candidate)
             # V2.2 autonomous logic: Let LLM decide what tools to use for THIS specific route
             session = ReActSession(self.llm_client, registry, max_steps=5)
             
             goal = f"Verify if the reaction '{target} >> {p_str}' is chemically sound. Check for toxicity or stock availability risks."
             
             try:
                 # RUN (Logs will act as the 'Show Thinking' UI)
                 context_str = f"Candidate Source: {source}"
                 result = session.run(goal, context=context_str)
                 
                 # Append result to explanation
                 audit_note = result[:200] + "..." if len(result) > 200 else result
                 if 'reason' in cand:
                     cand['reason'] += f" [DeepScan: {audit_note}]"
                 else:
                     cand['reason'] = f"[DeepScan] {audit_note}"
                     
             except Exception as e:
                 logger.error(f"DeepScan failed for candidate {i}: {e}")
                 cand['reason'] = cand.get('reason', '') + " [DeepScan Failed]"

        logger.info("Deep Scan Complete.")
        return candidates

    def _sum_vector(self, vectors: List[Dict[str, int]]) -> Dict[str, int]:
        return self.route_manager._sum_vector(vectors)

    def _is_inverse_vector(self, left: Dict[str, int], right: Dict[str, int]) -> bool:
        return self.route_manager._is_inverse_vector(left, right)

    def _evaluate_reaction_vector_loop(
        self,
        cumulative_route: Dict,
        path_id: str,
        candidate_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.route_manager.evaluate_reaction_vector_loop(cumulative_route, path_id, candidate_meta)

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
        """利用 LLM 结合上下文强制修补 SMILES (Enhanced with Retry)"""
        if not self.llm_client:
            return None
            
        print(f"    🧠 LLM 正在分析目标 `{target[:40]}...` 并重构前体结构...")
        prompt = get_smiles_repair_prompt(target, broken_smiles)
        
        import time
        import random
        max_retries = 3
        base_delay = 3.0
        
        for attempt in range(max_retries + 1):
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
                # 过滤多余文字
                import re
                smi_match = re.search(r'([a-zA-Z0-9@\+\-\[\]\(\)\/\\=#%]{3,})', repaired_text)
                if smi_match:
                    return smi_match.group(1)
                return repaired_text
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"SMILES repair request failed ({e}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"SMILES repair failed after retries: {e}")
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
        # [FEATURE] Input SMILES Standardization
        logger.info(f"Standardizing input SMILES: {target_smiles[:40]}...")
        std_smiles = self.standardizer.canonicalize(target_smiles)
        if std_smiles:
            if std_smiles != target_smiles:
                logger.info(f"Standardized: {target_smiles[:20]}... -> {std_smiles[:20]}...")
            target_smiles = std_smiles
        else:
            logger.warning("Could not standardize input SMILES. Proceeding with original.")
            
        logger.info("*" * 70)
        logger.info(" Starting Full Retrosynthesis Planning (MoleReact V2.2) ")
        logger.info("*" * 70)
        logger.info(f"Target Molecule: {target_smiles[:60]}...")
        logger.info(f"Stage Limit: {max_stages if max_stages else 'Unlimited (Manual)'}")
        logger.info(f"Run Mode: {'Automatic' if self.auto_mode else 'Interactive'}")
        
        current_target = target_smiles
        stage = 1
        history_context = ""
        
        # 尝试恢复会话
        # Attempt session recovery
        if not self.auto_mode:
            latest_session = self.session_logger.get_latest_context()
            if latest_session["exists"]:
                logger.warning(f"Previous session record found ({latest_session['session_id']})")
                resume_input = input("Resume session? (enter path or [y/n]) [y]: ").strip()
                
                # ... 
                load_path = latest_session.get("path")
                if resume_input.lower() in ["", "y", "yes"]:
                    pass # 使用发现的最新的路径
                elif resume_input.lower() not in ["n", "no"]:
                    if os.path.exists(resume_input):
                        load_path = resume_input
                    else:
                        logger.error(f"File not found: {resume_input}. Skipping recovery.")
                
                if resume_input.lower() not in ["n", "no"]:
                    logger.info("Reconstructing session context...")
                    
                    # 1. Recover Cumulative Route (Core State)
                    restored_route = self.session_logger.restore_session_state(load_path)
                    
                    if restored_route and restored_route.get("stages"):
                        logger.info(f"Successfully restored {len(restored_route['stages'])} historical stages.")
                        cumulative_route = restored_route
                        
                        # ...
                        
                        if global_unsolved_queue:
                            current_target = global_unsolved_queue[0][0] if isinstance(global_unsolved_queue[0], tuple) else global_unsolved_queue[0]
                            logger.info(f"Restored target: {current_target} (Unsolved queue: {len(global_unsolved_queue)})")
                        else:
                            logger.warning("No unsolved molecules in previous stage. Resetting to original target.")
                            current_target = target_smiles
                            
                        # 2. Recover LLM Context String
                        history_context = self.session_logger.load_history_context(load_path)
                        logger.info(f"Restored historical text context ({len(history_context)} chars)")
                    else:
                        logger.warning("State reconstruction empty, loading text context only.")
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
        
        # ========== V3.5: 全局战略规划 (Step 0) ==========
        # 在主循环开始前调用 LLM 生成全局战略
        force_regen = getattr(self, '_force_regenerate_strategy', False)
        strategy = self._call_global_strategy(target_smiles, history_context, force_regenerate=force_regen)
        
        # Build strategy summary for subsequent context
        strategy_summary = ""
        if strategy and strategy.get("global_direction"):
            direction = strategy["global_direction"]
            strategy_summary = f"""### 0. Global Strategy:
- **Preferred bonds**: {', '.join(direction.get('preferred_convergences', ['N/A'])[:3])}
- **Late-stage sensitive FG**: {', '.join(direction.get('late_stage_sensitive_FG', ['N/A'])[:3])}
- **PG Principles**: {len(direction.get('pg_principles', []))} rules
- **Anti-loop Rule**: {strategy.get('loop_rules', {}).get('aba_rule', 'N/A')}
"""
            # 存储到 cumulative_route 供恢复使用
            cumulative_route["global_strategy"] = strategy
        
        # ========== V3.5: 知识库检索 ==========
        # 检索结构相似的历史成功路线作为参考
        kb_context = ""
        try:
            similar_routes = self.knowledge_base.retrieve_similar_routes(target_smiles, top_k=3)
            if similar_routes:
                logger.info(f"Found {len(similar_routes)} similar historical routes")
                for i, r in enumerate(similar_routes, 1):
                    logger.info(f"    [{i}] Similarity: {r['similarity']:.2f} | Score: {r['llm_score']:.1f} | {r['total_steps']} steps")
                kb_context = self.knowledge_base.format_for_context(similar_routes)
                
                # 记录到 Session Log
                self.session_logger.log_event(
                    "📚 Knowledge Base",
                    f"Retrieved {len(similar_routes)} similar route references",
                    level="INFO"
                )
            else:
                logger.info("No similar historical routes found in Knowledge Base")
        except Exception as e:
            logger.warning(f"Knowledge Base retrieval failed: {e}")
        
        # 将知识库上下文加入战略摘要
        if kb_context:
            strategy_summary = strategy_summary + "\n" + kb_context
        
        try:
            while True:
                # Check stage limit
                if max_stages and stage > max_stages:
                    logger.warning(f"Maximum stage limit reached ({max_stages})")
                    break
                
                # Run work module
                
                # Pop current target from queue
                if not global_unsolved_queue:
                    logger.info("All molecules resolved or available!")
                    
                    # V3.5: 存储成功路线到知识库
                    try:
                        if cumulative_route.get("stages"):
                            route_id = self.knowledge_base.store_route(
                                target_smiles=target_smiles,
                                route_data=cumulative_route,
                                llm_score=7.5,  # 默认成功分数
                                quality_notes="Auto-stored on successful completion",
                                session_id=self.session_logger.session_id
                            )
                            if route_id > 0:
                                logger.info(f"Route stored in Knowledge Base (ID: {route_id})")
                                stats = self.knowledge_base.get_statistics()
                                logger.info(f"Knowledge Base Statistics: {stats['total_routes']} routes, average score: {stats['avg_score']}")
                    except Exception as e:
                        logger.warning(f"Knowledge Base storage failed: {e}")
                    
                    break
                
                current_node = global_unsolved_queue.pop(0)
                current_target, current_lineage, current_path_id = current_node
                
                # Path-Aware Loop Detection
                if current_target in current_lineage:
                    logger.critical(f"CRITICAL: Infinite loop detected! Molecule `{current_target}` reappears in its lineage!")
                    logger.critical(f"  Path: {' -> '.join(current_lineage + [current_target])}")
                    if not self.auto_mode:
                        c_choice = input("Force re-process this node? (y/n) [n]: ").strip().lower()
                        if c_choice != 'y':
                            continue
                
                # 更新 Lineage
                new_lineage = current_lineage + [current_target]
                
                # 构建结构化的全景背景 Context (V2.2 强化: 加入高价值化学快照)
                # 1. 全局进度 (已解出的分子对 + 战略指标)
                global_progress = "### 1. 全局合成历史 (Global Synthesis History):\n"
                active_pgs = []
                
                if cumulative_route["stages"]:
                    for s in cumulative_route["stages"]:
                        pid = s.get("path_id", "Unknown")
                        tgt = s.get("target", "Unknown")
                        p_smis = s.get("precursors", [])
                        meta = s.get("analysis_metadata", {})
                        
                        # 重点提取：策略、骨架相似度、保护基提示、反应标志
                        strat = meta.get("reaction_strategy", "N/A")
                        scaf_sim = meta.get("scaffold_sim", 0.0)
                        pg_warn = meta.get("pg_warning", "")
                        tsign = meta.get("transform_signature", "Unknown")
                        ssign = meta.get("state_signature", "Unknown")
                        
                        if pg_warn and "mask" in pg_warn.lower() or "protect" in pg_warn.lower():
                            active_pgs.append(f"{tgt[:10]}...({pg_warn})")
                        
                        strat_str = f" [Strat: {strat}, Scaf-Sim: {scaf_sim:.2f}, Sign: {tsign}]"
                        global_progress += f"- [Node {pid}]{strat_str} {tgt[:20]}... => {' + '.join([p[:15]+'...' for p in p_smis])}\n"
                else:
                    global_progress += "- (初始目标，规划首步)\n"
                
                # 2. 累积保护基与路径追踪
                pg_inventory = f"\n### 2. 活跃保护基清单 (Active PG Inventory):\n"
                pg_inventory += ", ".join(active_pgs) if active_pgs else "None"
                
                path_lineage = f"\n\n### 3. Current Path ({current_path_id}):\n"
                if current_lineage:
                    path_lineage += " -> ".join([p[:20]+"..." for p in current_lineage]) + f" -> **{current_target}**"
                else:
                    path_lineage += f"**{current_target}** (Root Target)"
                
                # V3.5: 将全局战略摘要加入上下文
                full_context = strategy_summary + global_progress + pg_inventory + path_lineage
                
                result = self.run_work_module(
                    current_target,
                    stage=stage,
                    topk=10,
                    history_context=full_context,
                    path_id=current_path_id,
                    cumulative_route=cumulative_route
                )
                
                # Check if complete
                if result.is_complete:
                    logger.info("*" * 70)
                    logger.info(" Planning Task Completed Successfully ")
                    logger.info("*" * 70)
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
                        # Show pending molecule queue
                        if global_unsolved_queue:
                            logger.info("Unsolved Molecule Queue (Pending Branches):")
                            for idx, (mol, lineage, pid) in enumerate(global_unsolved_queue, 1):
                                depth = len(lineage)
                                logger.info(f"    [Q{idx}] {mol[:40]}... (ID: {pid}, Depth: {depth})")
                        
                        print("-" * 60)
                        print("    [Enter]        - Use Route 1 for current target")
                        print("    [Number]       - Select specific route (e.g. 2)")
                        print("    switch [Qn]    - Switch to another pending molecule (e.g. switch Q1)")
                        print("    list           - View current synthesis tree and progress")
                        print("    reopen [ID]    - Reopen and adjust a processed node (e.g. reopen 1.1)")
                        print("    expert [CMD]   - Inject expert strategy (e.g. expert A >> B.C)")
                        print("    q/stop/exit    - Terminate planning")
                        print("    verify         - Mark current stage for experimental verification")
                        print("-" * 60)
                    
                        user_input = input(">>> (Select route or enter command): ").strip()
                        
                        # Command 1: Terminate
                        if user_input.lower() in ["终止", "stop", "quit", "q", "退出", "exit"]:
                            logger.info("User terminated planning")
                            return self._generate_final_report("terminated_by_user")
                        
                        # Command 2: View Scheme (List)
                        if user_input.lower() in ["list", "方案", "查看"]:
                            logger.info("=" * 60)
                            print("\n" + "=" * 60)
                            logger.info("Current Tree Summary:")
                            if not cumulative_route["stages"]:
                                logger.info("  (No confirmed stages yet - processing root nodes)")
                            else:
                                for s_idx, s in enumerate(cumulative_route["stages"], 1):
                                    # Ensure display uses correct fields
                                    pid = s.get('path_id', '?')
                                    tgt = s.get('target', 'Unknown')
                                    prec = s.get('precursors', [])
                                    act = s.get('action', '')
                                    logger.info(f"  [Node {pid}] {tgt[:30]}... => {' + '.join(prec)} ({act})")
                                    
                            if global_unsolved_queue:
                                logger.info("Unsolved Molecule Queue (Pending Branches):")
                                for i, item in enumerate(global_unsolved_queue, 1):
                                    # Handle flexible unpacking (structure might vary slightly)
                                    if len(item) == 3:
                                        m, lin, pid = item
                                    else:
                                        m = item[0]
                                        lin = []
                                        pid = "?"
                                    
                                    # Calculate depth from path_id (e.g. "1.1" -> 2)
                                    depth = pid.count('.') + 1 if isinstance(pid, str) else 1
                                    
                                    logger.info(f"    [Q{i}] {m[:30]}... (ID: {pid}, Depth: {depth})")
                            logger.info("=" * 60)
                            continue # Stay in interaction loop
                        
                        # Command 3: Verification (Verify)
                        if any(x in user_input for x in ["待验证", "验证", "verify"]):
                            logger.info("Current stage marked for experimental verification.")
                            self.session_logger.log_event(
                                title="Verification Required",
                                content=f"User marked decision for node `{current_path_id}` (target: `{current_target}`) as needing lab verification.",
                                level="WARNING"
                            )
                            
                            # V2.2: True Tool Use (ReAct Hook)
                            # User requested explicit verification, allowing LLM to dynamically call tools.
                            if self.use_llm and self.llm_client:
                                try:
                                    logger.info("Initializing ReAct Dynamic Analysis Session...")
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
                                    logger.info("Agent is thinking and acting...")
                                    react_result = react.run(goal)
                                    
                                    logger.info(f"ReAct Conclusion: {react_result}")
                                    
                                    # Inject into context for next turn
                                    # We append this to a temporary note or history
                                    self.session_logger.log_event("ReAct Analysis", react_result, "INFO")
                                    logger.info("ReAct analysis result logged.")
                                    
                                except Exception as e:
                                    logger.error(f"ReAct execution failed: {e}")
                            else:
                                logger.info("LLM not available for dynamic analysis.")

                            logger.info("Please continue with commands (e.g. choose route or switch branch).")
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
                                        title="Branch Switch",
                                        content=f"User switched branch via `switch Q{q_idx+1}`.\n- Old target: `{current_node[0]}`\n- New target: `{selected_node[0]}` (ID: {selected_node[2]})",
                                        level="INFO"
                                    )
                                    logger.info(f"Branch switched! Next target: {selected_node[0][:40]}")
                                    selected_route_idx = -999 # Signal to skip current route processing
                                    interaction_active = False # Break interaction loop
                                    break 
                                else:
                                    logger.warning(f"Invalid task number: Q{q_idx+1}")
                                    continue
                            else:
                                logger.info("Usage: switch Q1 / switch 1")
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
                                    
                                    # Remove queued nodes that belong to the reopened subtree.
                                    pruned_queue, removed = self._prune_queue_by_path_prefix(
                                        global_unsolved_queue,
                                        target_pid
                                    )
                                    global_unsolved_queue = pruned_queue
                                    cumulative_route["global_unsolved_queue"] = global_unsolved_queue
                                    
                                    self.session_logger.log_reopen(path_id=target_pid, target_smiles=target_mol, reason="User requested re-evaluation")
                                    if removed:
                                        logger.info(f"Cleared {removed} queued nodes under subtree {target_pid}.")
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

                        # Command 6: 专家策略注入 (Expert)
                        if user_input.lower().startswith("expert") or user_input.lower().startswith("manual"):
                            # Format: expert TARGET >> PREC.PREC
                            cmd_body = user_input[6:].strip() # remove "expert"
                            if ">>" in cmd_body:
                                # Prepare Session Context
                                context = {
                                    'cumulative_route': cumulative_route,
                                    'unsolved_queue': global_unsolved_queue,
                                    'current_path_id': current_path_id,
                                    'current_lineage': current_lineage
                                }
                                self.manual_add_expert_step(cmd_body, session_context=context)
                                print("  [提示] 专家策略已应用。系统已自动更新队列，即将开始处理新前体。")
                                
                                # Since we solved this node manually, we want to STOP the current loop 
                                # and go back to the MAIN loop which picks from global_unsolved_queue.
                                interaction_active = False 
                                selected_route_idx = -999 # Skip standard processing
                                break # Break Inner Loop
                            else:
                                print("  [用法参考] expert 目标SMILES >> 前体1.前体2")
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
                    route_desc = f"Stage {stage} selected route {selected_route_idx+1} ({chosen_route.get('source')})"
                    
                    # Update History Context
                    history_context = f"- Previous stage ({stage}) decision: {route_desc}\n"
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
                    correction_match = re.search(r'(?:已修正|Corrected)\s*SMILES[:：]\s*(\[?[^\]\n]+\]?)', reason_text)
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
                    self.session_logger.log_decision(stage, selected_route_idx, chosen_route, "", global_unsolved_queue=global_unsolved_queue, path_id=current_path_id)

                    # Update Cumulative Route
                    cumulative_route["stages"].append({
                        "stage": stage,
                        "path_id": current_path_id,
                        "target": current_target,
                        "lineage": current_lineage,
                        "action": f"Selected Route {selected_route_idx + 1}",
                        "precursors": precursors,
                        "unsolved_leaves": result.unsolved_leaves,
                        "reaction_type": chosen_route.get("reaction_type", ""),
                        "reason": chosen_route.get("reason", ""),
                        "analysis_metadata": chosen_route.get("analysis_metadata", {})
                    })
                    
                    # V3.6: Auto-save JSON snapshot after each decision for robust recovery
                    self.session_logger.save_json_snapshot(
                        cumulative_route, 
                        extra_data={"global_strategy": self.global_strategy}
                    )

            
                # Update Global Queue (Depth-First)
                if result.unsolved_leaves:
                    new_nodes = []
                    for child_idx, m in enumerate(result.unsolved_leaves, 1):
                        if m not in new_lineage:
                            child_pid = f"{current_path_id}.{child_idx}"
                            new_nodes.append((m, new_lineage, child_pid))
                    
                    cumulative_route["global_unsolved_queue"] = new_nodes + global_unsolved_queue
                    global_unsolved_queue = cumulative_route["global_unsolved_queue"]
                    logger.info(f"Added {len(new_nodes)} new branches, total queue size: {len(global_unsolved_queue)}")
                else:
                    if global_unsolved_queue:
                        next_node = global_unsolved_queue[0]
                        self.session_logger.log_event("Branch Resolved", f"Node {current_path_id} completed. Switching to: {next_node[0]}", "SUCCESS")
                        logger.info(f"Branch Resolved: Node {current_path_id} successfully disassembled.")
                    else:
                        logger.info("Retrosynthesis tree parsing completed.")
                        break
                
                stage += 1
        
        except KeyboardInterrupt:
            logger.warning("User interrupted (KeyboardInterrupt)")
            cumulative_route["status"] = "interrupted"
        except Exception as e:
            logger.error(f"Execution error: {e}")
            import traceback
            traceback.print_exc()
            cumulative_route["status"] = "error"
        finally:
            # Generate final reports either way
            logger.info("Generating final session report and path map...")
            self._finalize_session(cumulative_route)
        
        return self._generate_final_report("completed")

    def _convert_to_aizynth_dict(self, cumulative_route: Dict) -> Dict:
        """Convert cumulative route stages to AiZynthFinder tree dict"""
        path_to_stage = {}
        
        # Build lookup map: PathID -> Stage Data
        for s in cumulative_route["stages"]:
            pid = s.get("path_id")
            if pid:
                path_to_stage[pid] = s
        
        def build_node(node_id: str, mol_smiles: str, depth: int = 0) -> Dict:
            # Defensive programming: depth limit
            if depth > 20:
                return {
                    "type": "mol",
                    "smiles": mol_smiles,
                    "is_chemical": True,
                    "in_stock": False,
                    "metadata": {"warning": "Depth limit exceeded"}
                }
            
            node = {
                "type": "mol",
                "smiles": mol_smiles,
                "is_chemical": True, 
                "in_stock": self.stock_cache.get(mol_smiles, False)
            }
            
            # Use path_id to find the corresponding stage expansion
            if node_id in path_to_stage:
                stage_data = path_to_stage[node_id]
                precursors = stage_data.get("precursors", [])
                
                # Check if this stage actually matches the target (sanity check)
                # Note: stage_data['target'] should match mol_smiles ideally, 
                # but we trust path_id as the structural truth.
                
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
                
                for i, p in enumerate(precursors, 1):
                    # Deterministic ID generation for children: current_id.child_index
                    child_id = f"{node_id}.{i}"
                    reaction_node["children"].append(build_node(child_id, p, depth + 1))
                
                node["children"] = [reaction_node]
            
            return node

        if not cumulative_route.get("target"):
            return {}
            
        # Start from root (ID "1")
        return build_node("1", cumulative_route["target"])

    def _prune_queue_by_path_prefix(self, queue: List[Tuple], path_prefix: str) -> Tuple[List[Tuple], int]:
        """Remove queued nodes that belong to a subtree rooted at path_prefix."""
        if not path_prefix:
            return queue, 0
        pruned = []
        removed = 0
        for item in queue:
            if isinstance(item, tuple) and len(item) >= 3:
                pid = item[2]
                if isinstance(pid, str) and (pid == path_prefix or pid.startswith(path_prefix + ".")):
                    removed += 1
                    continue
            pruned.append(item)
        return pruned, removed

    def _finalize_session(self, cumulative_route: Dict):
        """Finalizing session: Generate summary visualization and log."""
        logger.info("=" * 70)
        logger.info(" Finalizing session and generating summary data ")
        logger.info("=" * 70)
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
                    logger.info(f"[AiZynth] Full retrosynthesis map generated: {img_path_aizynth}")
                    image_path = img_path_aizynth
            except ImportError:
                logger.warning("aizynthfinder library not found, trying basic visualization.")
            except Exception as e:
                logger.warning(f"AiZynth visualization failed: {e}")
            
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
                    target_smiles=cumulative_route["target"], 
                    selected_precursors=leaves, 
                    node_id="FINAL_SUMMARY", 
                    output_dir=None
                )
                image_path = viz_result.get("image_path")
                image_path = viz_result.get("stage_image_path")
                logger.info(f"Full retrosynthesis map generated: {image_path}")
            
            # 3. Write Session Log
            self.session_logger.log_session_summary(cumulative_route, image_path)
            logger.info(f"Session summary written: {self.session_logger.log_path}")
            
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")
    
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
        
        # Build Markdown report
        self._generate_markdown_report(report_path.replace(".json", ".md"), status)
        
        logger.info(f"Final Report Saved: {report_path}")
        logger.info("*" * 60)
        logger.info(f" FINAL REPORT: {status.upper()} ")
        logger.info("*" * 60)
        logger.info(f"Final Status: {status}")
        logger.info(f"Total Stages Processed: {len(self.history)}")
        
        return report
    
    def _generate_markdown_report(self, md_path: str, status: str):
        """Generate Markdown format report"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# MoleReact Retrosynthesis Planning Report\n\n")
            f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Status**: {status}\n")
            f.write(f"- **Total Stages**: {len(self.history)}\n")
            
            # Use relative paths for images
            md_dir = os.path.dirname(md_path)
            
            for stage_res in self.history:
                f.write(f"\n## Stage {stage_res.stage}: {stage_res.target_smiles[:40]}...\n\n")
                f.write(f"- **Target**: `{stage_res.target_smiles}`\n")
                f.write(f"- **Completion Status**: {'✅ Completed' if stage_res.is_complete else '⏳ Pending'}\n")
                if not stage_res.is_complete:
                    f.write(f"- **Unsolved Molecules**: {len(stage_res.unsolved_leaves)}\n")
                
                # Recommended Route Table
                f.write(f"\n### 🏆 Recommended Routes\n")
                f.write("| Rank | Source | Precursors | Rationale |\n")
                f.write("|------|------|------|------|\n")
                
                for cand in stage_res.llm_selected_top_n:
                    rank = cand.get('rank', '-')
                    source = cand.get('source', 'unknown')
                    precursors = "<br>".join([f"`{p}`" for p in cand.get('precursors', [])])
                    reason = cand.get('reason', '').replace('\n', ' ')
                    f.write(f"| {rank} | {source} | {precursors} | {reason} |\n")
                
                # Visualization
                if stage_res.image_paths:
                    f.write(f"\n### 📊 Route Visualization\n")
                    f.write("| Route | Visualization |\n")
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

                # LLM Analysis (Folded)
                if stage_res.llm_analysis:
                    f.write(f"\n<details>\n<summary>LLM Detailed Analysis (Click to expand)</summary>\n\n")
                    f.write(stage_res.llm_analysis)
                    f.write(f"\n\n</details>\n")
                
                f.write(f"\n---\n")
            
        logger.info(f"Markdown report saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Complete Agent Work Module")
    parser.add_argument("--smiles", default=DEFAULT_TARGET, help="Target molecule SMILES")
    parser.add_argument("--stages", type=int, default=None, help="Maximum stages (default: no limit)")
    parser.add_argument("--auto", action="store_true", help="Automatic mode (no interaction)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--single", action="store_true", help="Run single work module only")
    
    args = parser.parse_args()
    
    logger.info("""
╔══════════════════════════════════════════════════════════════════╗
║     🧪 MoleReact Agent - Complete Work Module                     ║
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
            # Run single work module only
            result = runner.run_work_module(args.smiles, stage=1, topk=10)
            logger.info(f"Completion Status: {'✅ Available' if result.is_complete else '⏳ Pending'}")
        else:
            # Run full planning
            report = runner.run_full_planning(args.smiles, max_stages=args.stages)
            logger.info(f"Final Status: {report['status']}")
            logger.info(f"Total Stages Processed: {len(runner.history)}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("User Interrupted")
        return 1
    except Exception as e:
        logger.error(f"Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
