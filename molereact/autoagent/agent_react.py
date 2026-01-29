# -*- coding: utf-8 -*-
"""
Module: multistep.agent.agent_react
Called By: agent_full.py
Role: ReAct-Enhanced Agent Subclass

Functionality:
    Inherits from RetroSynthesisAgent but overrides work modules to use ReAct logic.
    Provides a safe sandbox for developing full-flow ReAct capabilities without breaking the main agent.

Key Classes:
    - ReActRetroAgent: The ReAct-enabled agent.
"""
import logging
from typing import Dict, Any, Optional

try:
    from multistep.agent.agent import RetroSynthesisAgent
    from multistep.agent.core.react import ReActSession
    from multistep.agent.tools.base import ToolRegistry
    from multistep.agent.tools.analysis import MoleculeAnalysisTool
    from multistep.agent.core.tree import SynthesisTree, NodeStatus
except ImportError:
    # Fallback for local execution/testing
    import sys
    import os
    # Add root to path if needed (though usually caller handles this)
    # Assuming standard structure
    from agent import RetroSynthesisAgent
    from core.react import ReActSession
    from tools.base import ToolRegistry
    from tools.analysis import MoleculeAnalysisTool
    from core.tree import SynthesisTree, NodeStatus

logger = logging.getLogger(__name__)

class ReActRetroAgent(RetroSynthesisAgent):
    """
    Subclass of RetroSynthesisAgent.
    Integrates the ReAct loop directly into the main workflow phases.
    V3.0: Supports Fully Autonomous "Meta-Controller" Mode.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We can register extra ReAct-specific tools here if needed
        logger.info("ReActRetroAgent initialized (Sandbox Mode)")
        self.tree = None # Global state for autonomous mode
        self.log_file = "autonomous_walkthrough.md"

    def _log_md(self, content: str):
        """Helper to append content to the autonomous walkthrough log."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(content + "\n\n")
        except Exception as e:
            logger.error(f"Failed to log to MD: {e}")

    def run_work_module(self, target_smiles: str, stage: int = 1, context: str = "", **kwargs) -> Dict[str, Any]:
        """
        Original semi-autonomous module.
        """
        # For now, we behave exactly like the parent (compatibility check)
        # TODO: Replace with ReAct orchestration logic
        logger.info(f"ReActAgent: Starting Module for {target_smiles} (Currently passthrough)")
        
        return super().run_work_module(target_smiles, stage, context, **kwargs)

    def run_autonomous_loop(self, target_smiles: str, max_steps: int = 20) -> str:
        """
        V3.1 Entry Point: Fully Autonomous Controller Loop.
        
        The LLM manages a 'SynthesisTree' and decides strategies (Expand, Backtrack, Assess).
        Includes Global Analysis and Pre-Expansion Chemistry Checks.
        """
        print(f"\n🚀 Starting Autonomous Agent V3.1 for Target: {target_smiles}")
        
        # 1. Initialize Tree
        self.tree = SynthesisTree(target_smiles)
        
        # Ensure Logging is set up (Simple file append for V3.1)
        self.log_file = "autonomous_walkthrough.md"
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# Autonomous Session: {target_smiles}\n\n")

        # 0. Initial Global Blueprint
        blueprint = self._generate_global_blueprint(target_smiles)
        self.blueprint = blueprint
        self._log_md(f"### Global Synthesis Blueprint\n{blueprint}")

        step_count = 0
        action_history = [] # For loop detection
        
        while step_count < max_steps:
            step_count += 1
            header = f"\n--- 🤖 Meta-Controller Step {step_count}/{max_steps} ---"
            print(header)
            self._log_md(f"## Step {step_count}")
            
            # 2. Check Solved Status
            open_nodes = self.tree.get_open_nodes()
            if not open_nodes:
                msg = "🎉 All branches are solved (Stock/Available)!"
                print(msg)
                self._log_md(f"**Result**: {msg}")
                return f"SUCCESS: Plan Complete. {self.tree.serialize_state()}"
            
            # 3. Serialize State (FOCUSED VIEW to prevent context explosion)
            state_text = self.tree.serialize_state(focused=True)
            self._log_md(f"### Current State\n```text\n{state_text}\n```")
            
            # 4. Meta-Reasoning Prompt (Enhanced with History and Blueprint)
            history_text = "\n".join(action_history[-5:]) # Last 5 actions
            prompt = self._build_meta_prompt(state_text, history_text, blueprint)
            
            # 5. Call LLM
            response_text = self._call_llm_meta(prompt)
            print(f"🧠 Thought:\n{response_text}")
            self._log_md(f"### Meta-Analysis & Decision\n{response_text}")
            
            # 6. Parse and Execute Strategy
            action_done, action_desc = self._execute_meta_action_safe(response_text)
            if action_done:
                action_history.append(action_desc)
            else:
                print("⚠️ No valid meta-action parsed. Continuing...")
                self._log_md("**Action Failed**: No valid command parsed.")
        
        return "STOPPED: Max steps reached."

    def _generate_global_blueprint(self, target_smiles: str) -> str:
        """Generate a rough synthesis strategy at the start."""
        print("🌍 Generating Global Synthesis Blueprint...")
        prompt = f"""
You are a world-class synthetic chemist. Provide a high-level retrosynthetic blueprint for the molecule: {target_smiles}.

Requirements:
1. Identify key structural challenges (rings, stereocenters, sensitive groups).
2. Propose a rough multi-step strategy (e.g., "Convergent synthesis using a Wittig coupling at the late stage").
3. Suggest 2-3 major building blocks.
4. Highlight potential pitfalls (e.g., functional group compatibility, protection needs).

Keep it concise but strategic.
"""
        return self._call_llm_meta(prompt)
    
    def _execute_meta_action_safe(self, response_text: str) -> (bool, str):
        """Wrapper for _execute_meta_action to return description and handle errors."""
        # Simple wrapper to extract the action string for history
        # We need to parse strict action for history
        import re
        
        # We re-use the parsing logic slightly or modify _execute_meta_action to return (bool, str)
        # For now, let's just delegate and return True/False + crude extraction
        # But wait, original _execute_meta_action returns bool only.
        # I should REFACTOR _execute_meta_action to return (success, description)
        # But to minimal diff, I will rely on side-effects? No, history needs the string.
        # Let's modify _execute_meta_action signature in next step.
        # For this step, I will assumes _execute_meta_action returns a bool, and I'll extract string here for history.
        
        success = self._execute_meta_action(response_text)
        
        # Extract action string for history
        action_match = re.search(r"(EXPAND|SWITCH|MARK_DEAD)[:\s]+([a-zA-Z0-9_\-]+)", response_text, re.IGNORECASE)
        desc = action_match.group(0) if action_match else "UNKNOWN_ACTION"
        
        return success, desc

    def _build_meta_prompt(self, state_text: str, history_text: str = "", blueprint: str = "") -> str:
        return f"""
You are the Chief Scientific Officer managing a complex retrosynthesis project.

Original Target Blueprint:
{blueprint}

Current Synthesis Tree:
{state_text}

Recent Actions:
{history_text or "None"}

Required Analysis Process (Look-ahead & Look-back):
1. **GLOBAL ALIGNMENT**: Does the current tree align with the global blueprint? Have we veered into inefficient loops?
2. **PROTECTION CHECK**: Are there functional group conflicts (e.g., A -> B -> A loops)? If a loop is detected (marked in tree), AVOID expanding that branch and instead MARK_DEAD or SWITCH.
3. **STRATEGIC LOOK-AHEAD**: If we expand Node X, will we get closer to simple building blocks or just more complex intermediates? 
4. **DECISION**: Choose the next High-Level Action to optimize the overall route.

Available Strategy Actions (Output strict format):
1. **EXPAND: [NodeID]**
   - Triggers: Atomic Chemistry Analysis -> Retrosynthesis -> ReAct Route Selection.
2. **SWITCH: [NodeID]**
   - Change focus to a different branch to explore alternative pathways.
3. **MARK_DEAD: [NodeID]**
   - Abandon this branch (e.g., if it's a loop or chemically impossible).

Output Format:
[GLOBAL ANALYSIS]
...
[STRATEGIC THOUGHT]
...
[ACTION]
Action: EXPAND: NodeID | SWITCH: NodeID | MARK_DEAD: NodeID
"""

    def _call_llm_meta(self, prompt: str) -> str:
        """Streaming wrapper around client to show progress and reasoning."""
        if not self.llm_client:
            return "Error: No LLM Client"
            
        print("\n  💬 LLM Analysis in progress (Streaming)...")
        print("  " + "-" * 40)
        
        try:
            # Check if model supports thinking mode (usually glm-4.7 or similar)
            # We use stream=True to provide feedback for the '異常久' (unusually long) wait
            response = self.llm_client.chat.completions.create(
                model="glm-4.7",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                thinking={
                    "type": "enabled",  # Enable depth thinking if supported
                },
                stream=True
            )
            
            full_response = []
            reasoning_content = []
            is_reasoning = False
            
            print("  💭 [Thinking Process]", flush=True)
            
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    reasoning = chunk.choices[0].delta.reasoning_content
                    if not is_reasoning:
                        is_reasoning = True
                    print(reasoning, end="", flush=True)
                    reasoning_content.append(reasoning)
                
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if is_reasoning:
                        is_reasoning = False
                        print("\n\n  📝 [Analysis Conclusion]", flush=True)
                    print(content, end="", flush=True)
                    full_response.append(content)
            
            print("\n  " + "-" * 40 + "\n")
            return "".join(full_response)
            
        except Exception as e:
            print(f"  ❌ LLM Call Failed: {e}")
            # Fallback to non-streaming if thinking/streaming parameters cause issues 
            # (though with ZhipuAI SDK they shouldn't if configured correctly)
            return f"ERROR: {str(e)}"

    def _execute_meta_action(self, response_text: str) -> bool:
        """Parse Action and Execute on Tree."""
        import re
        
        # Robust Regex Patterns
        # Allows: "Action: EXPAND: NodeID", "Action: EXPAND NodeID", "EXPAND: NodeID", "EXPAND NodeID"
        # We look for the keyword EXPAND/SWITCH/MARK_DEAD followed by optional colon/whitespace and then the ID.
        
        # 1. EXPAND
        # Matches "EXPAND[: ]+ID" anywhere in the text (assuming LLM output focus)
        match_exp = re.search(r"EXPAND[:\s]+([a-zA-Z0-9_\-]+)", response_text, re.IGNORECASE)
        if match_exp:
            node_id = match_exp.group(1).strip()
            self._log_md(f"**Action Executed**: EXPAND {node_id}")
            return self._action_expand_node(node_id)
            
        # 2. SWITCH
        match_sw = re.search(r"SWITCH[:\s]+([a-zA-Z0-9_\-]+)", response_text, re.IGNORECASE)
        if match_sw:
            node_id = match_sw.group(1).strip()
            if node_id in self.tree.nodes:
                self.tree.cursor_id = node_id
                print(f"👉 Switched focus to {node_id}")
                self._log_md(f"**Action Executed**: SWITCH {node_id}")
                return True
            else:
                print(f"❌ Node {node_id} not found.")
                # Fallback: don't crash loop, just log
                self._log_md(f"**Action Failed**: Switch target {node_id} not found.")
                return False
        
        # 3. MARK_DEAD
        match_dead = re.search(r"MARK_DEAD[:\s]+([a-zA-Z0-9_\-]+)", response_text, re.IGNORECASE)
        if match_dead:
             node_id = match_dead.group(1).strip()
             if node_id in self.tree.nodes:
                 self.tree.set_status(node_id, NodeStatus.DEAD)
                 print(f"💀 Marked {node_id} as DEAD.")
                 self._log_md(f"**Action Executed**: MARK_DEAD {node_id}")
                 return True

        return False

    def _action_expand_node(self, node_id: str) -> bool:
        """
        ATOMIC OPERATION V3.4: 
        Analysis -> Retro -> Stock -> V3.4 Selector (JSON) -> Auto-Spawn Patch -> Update Tree
        """
        node = self.tree.get_node(node_id)
        if not node: return False
        
        print(f"🧪 [Atomic Expand V3.4] Processing {node.smiles}...")
        self._log_md(f"#### Expanding Node {node_id} ({node.smiles})")
        
        # --- Step 0: Pre-Expansion Chemistry Analysis ---
        print("  0️⃣  Pre-Flight Chemistry Analysis...")
        analysis_summary = []
        try:
            analysis_tool = self.toolbox.get_tool("MoleculeAnalysis")
            analysis_res = analysis_tool.execute(node.smiles)
            analysis_summary.append(f"Basic Properties: {analysis_res.get('formatted_report', 'N/A')}")
            
            scaffold_tool = self.toolbox.get_tool("ScaffoldAnalysis")
            scaffold_res = scaffold_tool.execute(node.smiles)
            if scaffold_res.get("status") == "success":
                analysis_summary.append(f"Murcko Scaffold: {scaffold_res.get('scaffold', 'None')} (Rings: {scaffold_res.get('num_rings', 0)})")
            
            combined_analysis = "\n".join(analysis_summary)
            # print(f"     [Analysis Summary]: {combined_analysis[:100]}...") 
            self._log_md(f"**Chemical Analysis**:\n{combined_analysis}")
        except Exception as e:
            print(f"     [Warning] Analysis failed: {e}")

        if not self.engine:
            print("❌ Engine not loaded.")
            return False
            
        try:
            # --- Step 1: Retrosynthesis (RetroSingleStep) ---
            print("  1️⃣  Running Retrosynthesis...")
            res = self.engine.propose_precursors(node.smiles, topk_model=10, topk_template=10)
            candidates = res.get("model_candidates", []) + res.get("template_candidates", [])
            
            if not candidates:
                print("  ❌ No routes found. Marking DEAD.")
                self.tree.set_status(node_id, NodeStatus.DEAD)
                return True 

            # --- Step 2: Stock Check & Advanced Analysis ---
            print("  2️⃣  Checking Inventory & Formatting Candidates...")
            # Collect all unique precursors
            all_precursors = set()
            for c in candidates:
                precursors = c.precursors if hasattr(c, 'precursors') else c.get('precursors', [])
                all_precursors.update(precursors)
            
            stock_tool = self.toolbox.get_tool("StockCheck")
            stock_res = stock_tool.execute(list(all_precursors))
            stock_map = {r["smiles"]: r["in_stock"] for r in stock_res["results"]}

            # Helper for advanced analysis (Using AdvancedAnalysisToolbox if available)
            # We must import it locally or assume it's in self.toolbox?
            # agent_full.py imports 'advanced_toolbox'. We can reuse if injected.
            # Assuming 'advanced_toolbox' is available globally or we re-init.
            # For massive efficiency, let's assume we can import it.
            try:
                from multistep.agent.tools.advanced_analysis import toolbox as advanced_toolbox
            except ImportError:
                advanced_toolbox = None

            candidate_blocks = []
            for i, cand in enumerate(candidates[:15], 1): 
                precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
                source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
                conf = cand.confidence if hasattr(cand, 'confidence') else cand.get('confidence', 0)
                
                p_lines = []
                for p in precursors:
                    status = "✅(Stock)" if stock_map.get(p) else "❌(Buy/Make)"
                    p_lines.append(f"      - {p} {status}")
                
                # Run Advanced Analysis
                analysis_block = "(No advanced analysis available)"
                if advanced_toolbox:
                     adv = advanced_toolbox.analyze_candidate(node.smiles, precursors)
                     analysis_block = adv.get('formatted_report', '')

                block = (
                    f"### Route {i} [{source.upper()}]\n"
                    f"   [Comparison]: Target: {node.smiles} | Precursors: {', '.join(precursors)}\n"
                    f"   [Internal Scores - IGNORE]: Conf={conf:.4f}\n"
                    f"   [Precursors List]:\n" + "\n".join(p_lines) + "\n"
                    f"   [Component Analysis Report - USE THIS]:\n{analysis_block}\n"
                )
                candidate_blocks.append(block)
            
            candidates_text = "\n\n".join(candidate_blocks)

            # --- Step 3: Call V3.4 Selector Agent (JSON Mode) ---
            print("  3️⃣  Calling V3.4 Selector Agent (JSON)...")
            from multistep.agent.prompts import get_selection_v2_prompt
            
            prompt = get_selection_v2_prompt(
                target=node.smiles,
                stage=len(self.tree.nodes), # roughly
                candidates_text=candidates_text,
                stock_rate=stock_results['stock_rate'] if 'stock_results' in locals() else 0.0,
                history_context="Autonomous Mode",
                top_n=3 # Focus on top 3 for expansion
            )
            
            # Using _call_llm_meta (streaming enabled)
            llm_response = self._call_llm_meta(prompt)
            
            # --- Step 4: Parse JSON & Auto-Spawn ---
            import json, re
            json_str = llm_response
            if "```json" in llm_response:
                match = re.search(r"```json\s*(.*?)\s*```", llm_response, re.DOTALL)
                if match: json_str = match.group(1)
            elif "```" in llm_response:
                match = re.search(r"```\s*(.*?)\s*```", llm_response, re.DOTALL)
                if match: json_str = match.group(1)
                
            results_data = {}
            try:
                results_data = json.loads(json_str.strip())
            except:
                print("  ❌ JSON Parse Failed. Using fallback top 1.")
            
            selected_indices = []
            patched_routes = []
            
            if results_data and "routes" in results_data:
                # 4.1 Collect valid selections
                if "shortlist" in results_data:
                     for tid in results_data["shortlist"].get("top_ids", []):
                         try: selected_indices.append(int(tid)-1)
                         except: pass
                
                # 4.2 Auto-Spawn Patched Routes
                for r in results_data["routes"]:
                    # Check for patch
                    # 1. Has 'patched_precursors'
                    # 2. 'patch_feasibility' is present (simple valid check)
                    patch = r.get("patched_precursors")
                    if patch and isinstance(patch, list) and len(patch) > 0:
                        print(f"  🛠️  Found LLM Patch for Route {r.get('route_id')}: {patch}")
                        patched_routes.append({
                            "precursors": patch,
                            "source": "LLM_Patch",
                            "reaction_type": r.get("rxn_type_from_FG", "Patched_Rxn"),
                            "reason": f"Patched from Route {r.get('route_id')}: {r.get('patch_feasibility', 'Feasibility check')}"
                        })

            # --- Step 5: Update Tree (Union of Selected + Patched) ---
            
            # Add Traditional Selected
            for idx in selected_indices:
                if 0 <= idx < len(candidates):
                    cand = candidates[idx]
                    precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
                    print(f"  ✅ Adding System Route {idx+1}")
                    self._add_child_nodes(node_id, precursors, cand)

            # Add Patched Routes (Auto-Spawn)
            for patch in patched_routes:
                print(f"  ✨ Spawning Patched Branch: {patch['precursors']}")
                self._log_md(f"**Auto-Spawn**: Patched Route spawned: `{patch['precursors']}`\n*Reason*: {patch['reason']}")
                # Reuse _add_child_nodes logic but with dict wrapper
                self._add_child_nodes(node_id, patch['precursors'], patch)
                
            if not selected_indices and not patched_routes:
                print("  ⚠️ No valid routes selected by LLM. Adding Top 1 System fallback.")
                if candidates:
                    self._add_child_nodes(node_id, candidates[0].precursors, candidates[0])

            self.tree.set_status(node_id, NodeStatus.VISITED)
            return True

        except Exception as e:
            print(f"❌ Atomic Expand Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _add_child_nodes(self, parent_id, precursors, route_info):
        """Helper to add child nodes from a route object/dict"""
        # Determine reaction type and source safely
        rtype = "unknown"
        stype = "unknown"
        
        if isinstance(route_info, dict):
            rtype = route_info.get("reaction_type", "unknown")
            stype = route_info.get("source", "unknown")
        else: # Object
            rtype = getattr(route_info, "reaction_type", "unknown")
            stype = getattr(route_info, "source", "unknown")
            
        stock_tool = self.toolbox.get_tool("StockCheck")
        # Quick stock check for new patches or reuse cache
        # Ideally we batch this but for now per-route is fine
        stock_res = stock_tool.execute(list(precursors))
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_res["results"]}

        for p in precursors:
            child_id = self.tree.add_child(
                parent_id, p, 
                reaction_rule=rtype,
                source_type=stype
            )
            if stock_map.get(p):
                self.tree.set_status(child_id, NodeStatus.SOLVED)

    def run_react_reasoning(self, goal: str, context: str = "") -> str:
        """
        Dedicated entry point for a pure ReAct session within the agent's context.
        """
        if not self.toolbox:
            return "Error: Tools not initialized."
            
        session = ReActSession(self.llm_client, self.toolbox)
        return session.run(goal, context)

    def evaluate_candidates_with_react(self, target: str, candidates_text: str, context: str, criteria: str) -> str:
        """
        Use ReAct to evaluate candidates dynamically.
        
        Args:
            target: Target SMILES
            candidates_text: Pre-formatted candidate list
            context: History context
            criteria: Selection criteria
            
        Returns:
            LLM textual response (Final Answer)
        """
        goal = f"""
Analyze the provided candidate routes for target {target}.
Candidates are listed below:
{candidates_text}

Your Task:
1. Use 'MoleculeAnalysisTool' to check properties of key precursors if you need more data than provided.
2. Select the best valid routes (up to requested N).
3. Provide mechanistic reasoning for each.
4. Output the final selection in the standard "Recommended Routes" format.

Criteria: {criteria}
"""
        return self.run_react_reasoning(goal, context)
