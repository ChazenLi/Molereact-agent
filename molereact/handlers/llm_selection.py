# -*- coding: utf-8 -*-
"""
Module: multistep.agent.handlers.llm_selection
Description: Handles LLM interaction for candidate selection (prompting, parsing).
Refactored from agent_run.py.
"""

from typing import Dict, Any, List, Tuple, Optional
import json
import logging
from multistep.agent.prompts import get_selection_v2_prompt
from multistep.agent.tools.advanced_analysis import toolbox as advanced_toolbox
from multistep.agent.managers.route_history import RouteHistoryManager

logger = logging.getLogger(__name__)

class LLMSelectionHandler:
    """
    Handles Top-N selection using LLM.
    Responsibilties:
    1. Analyzing candidates (Autonomous Analysis)
    2. Constructing Prompts (Integration with RouteHistory)
    3. Calling LLM
    4. Parsing and Repairing JSON Response
    """
    
    def __init__(self, llm_client, analyzer_tool=None, route_manager: RouteHistoryManager = None):
        self.llm_client = llm_client
        self.analyzer = analyzer_tool
        self.route_manager = route_manager or RouteHistoryManager()
        
    def select_top_n(
        self,
        target: str,
        candidates: List,
        stock_results: Dict,
        stage: int = 1,
        top_n: int = 7, 
        history_context: str = "",
        cumulative_route: Dict = None,
        path_id: str = "1",
        global_strategy: Dict = None
    ) -> Tuple[List[Dict], str]:
        """
        Orchestrates the selection process.
        """
        logger.info("Calling LLM for deep analysis...")
        
        # 1. Prepare Candidate Blocks (with analysis)
        candidates_text = self._prepare_candidate_text(
            target, candidates, stock_results, cumulative_route, path_id
        )
        
        # 2a. Inject Global Strategy if available (Optimization Fusion)
        if global_strategy:
             strat_ctx = f"\n\n### 🌍 [GLOBAL STRATEGY GUIDANCE]:\n"
             if global_strategy.get("global_direction"):
                 direction = global_strategy["global_direction"]
                 strat_ctx += f"- Preferred Convergences: {direction.get('preferred_convergences', [])}\n"
                 strat_ctx += f"- Sensitive FG to Avoid Early: {direction.get('late_stage_sensitive_FG', [])}\n"
                 strat_ctx += f"- PG Rules: {direction.get('pg_principles', [])}\n"
             if global_strategy.get("hard_gates"):
                 strat_ctx += f"- HARD GATES: {global_strategy.get('hard_gates')}\n"
             
             # Append to existing history context so it appears in the prompt's context block
             history_context += strat_ctx
        
        # 2b. Build Prompt
        effective_top_n = min(len(candidates), top_n)
        prompt = get_selection_v2_prompt(
            target=target,
            stage=stage,
            candidates_text=candidates_text,
            stock_rate=stock_results.get('stock_rate', 0.0),
            history_context=history_context,
            top_n=effective_top_n
        )
        
        logger.info(f"Prompt Length: {len(prompt)} chars. Sending request to LLM...")

        # 3. Call LLM (with Retry and Exponential Backoff)
        max_retries = 3  # V3.6: Increased from 2 to 3
        last_error = ""
        
        for attempt in range(max_retries + 1):
            try:
                # Prepare API call arguments
                current_prompt = prompt
                if last_error:
                    current_prompt += f"\n\n[SYSTEM ERROR]: Previous JSON was invalid: {last_error}. Please output STRICT JSON."

                # Determine correct API method based on client type
                # ZhipuAI (official) uses .chat.completions.create
                # ZhipuAiClient (custom/legacy) might use .chat() ??
                # Current user env seems to use ZhipuAI or ZhipuAiClient that has .chat.completions.create structure per original file
                
                # Check for "CHATGLM_MODEL" availability, if not imported, default it
                model_name = "glm-4.7" # Default fallback
                
                # Access global config if possible, else hardcode common
                # In agent_run.py it was CHATGLM_MODEL global. Here we might need to rely on passed args or safe attributes.
                # Just use "glm-4" or whatever passed.
                
                is_official_client = hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions")
                
                full_response_text = ""
                
                logger.info(f"Sending request to LLM (Attempt {attempt+1}/{max_retries+1})...")

                if is_official_client:
                    # Generic ZhipuAI / OpenAI compatible
                    api_kwargs = {
                        "model": model_name, 
                        "messages": [
                             # self.route_manager doesn't provide system prompt. 
                             # We should import get_system_role_prompt
                            {"role": "system", "content": "You are a specialized chemist assistant."}, 
                            {"role": "user", "content": current_prompt}
                        ],
                        "stream": True,
                        "temperature": 0.8,
                    }
                    if "plus" in model_name or "9b" in model_name:
                         api_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

                    response = self.llm_client.chat.completions.create(**api_kwargs)
                    
                    full_response = []
                    is_reasoning = False
                    
                    for chunk in response:
                        # Reasoning
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            reasoning = chunk.choices[0].delta.reasoning_content
                            if not is_reasoning:
                                is_reasoning = True
                                logger.info("\n[Thought Process] ", end="")
                            print(reasoning, end="", flush=True)
                        
                        # Content
                        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            if is_reasoning:
                                is_reasoning = False
                                logger.info("\n\n[Analysis Conclusion]")
                            print(content, end="", flush=True)
                            full_response.append(content)
                    print("\n  " + "-" * 60)
                    full_response_text = "".join(full_response)
                
                else:
                    # Legacy or Custom Client fallback
                    # Assuming .chat(prompt) works if it's the specific wrapper from 'zai'
                    # But user error says 'Chat object is not callable', meaning self.llm_client.chat IS an object (likely ZhipuAI.chat resource)
                    # Implementation detail: 'zai' client usually wraps check.
                    # If we aren't sure, we try the explicit create call.
                    # If previous code worked with client.chat.completions.create, we stick to that.
                    pass 

                # 4. Parse Response (using collected text)
                response = full_response_text
                parsed_json = self._parse_llm_json(response)
                
                # Validation
                if "routes" not in parsed_json or "shortlist" not in parsed_json:
                     raise ValueError("Missing 'routes' or 'shortlist' in JSON")
                
                # Extract Top N
                top_ids = parsed_json["shortlist"].get("top_ids", [])
                if not top_ids:
                    # Fallback to defaults? Or parse all routes with PASS?
                    # agent_run.py didn't have explicit fallback here beyond error catching
                    logger.warning("No top_ids found in shortlist.")
                
                # Collect selected candidates
                selected = []
                # Create map of route_id -> candidate
                # We assume candidates list is 1-indexed implicitly by order?
                # agent_run.py loop used enumerate(candidates, 1) to build text.
                # So route_id "1" maps to candidates[0].
                
                # Also we need to merge LLM reasoning back into candidates
                for r_eval in parsed_json.get("routes", []):
                    try:
                        rid_str = str(r_eval.get("route_id", "0"))
                        # Handle "Route 1", "1", "#1"
                        import re
                        digits = re.findall(r"\d+", rid_str)
                        if not digits: continue
                        rid = int(digits[0])
                        
                        if 1 <= rid <= len(candidates):
                            cand = candidates[rid-1]
                            # Attach LLM evaluation
                            cand_meta = cand.get('metadata', {}) if isinstance(cand, dict) else getattr(cand, 'metadata', {})
                            cand_meta['llm_evaluation'] = r_eval
                            if isinstance(cand, dict):
                                cand['metadata'] = cand_meta
                                cand['llm_reasoning'] = r_eval.get("revision_hint", "")
                                # Map recommendation_reason to top-level key for easy access
                                cand['reason'] = r_eval.get("recommendation_reason", "")
                            else:
                                setattr(cand, 'metadata', cand_meta)
                                setattr(cand, 'llm_reasoning', r_eval.get("revision_hint", ""))
                                setattr(cand, 'reason', r_eval.get("recommendation_reason", ""))
                            
                            if rid in top_ids:
                                selected.append(cand)
                    except:
                        continue
                
                # Fallback if selected is empty but top_ids wasn't
                # (Maybe route objects weren't found?)
                if not selected and len(candidates) > 0:
                     logger.warning("Fallback: LLM selection failed to map, picking top confidence.")
                     selected = sorted(candidates, key=lambda x: x.get('confidence', 0) if isinstance(x, dict) else x.confidence, reverse=True)[:top_n]

                # Store global analysis if available
                if "shortlist" in parsed_json and "global_analysis" in parsed_json["shortlist"]:
                    if cumulative_route is not None:
                        cumulative_route["current_stage_analysis"] = parsed_json["shortlist"]["global_analysis"]

                return selected, response

            except Exception as e:
                last_error = str(e)
                print(f"    [Attempt {attempt+1}] LLM Parsing Error: {e}")
                if attempt < max_retries:
                    # V3.6: Exponential backoff
                    import time
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    logger.info("Retrying...")
        
        logger.error(f"All attempts failed. Error: {last_error}")
        # Final Fallback
        return candidates[:top_n], f"LLM FAILED: {last_error}"

    def _prepare_candidate_text(self, target, candidates, stock_results, cumulative_route, path_id) -> str:
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_results.get("results", [])}
        candidate_blocks = []
        
        for i, cand in enumerate(candidates[:15], 1): # Limit to 15 context
            source = cand.source if hasattr(cand, 'source') else cand.get('source', 'unknown')
            precursors = cand.precursors if hasattr(cand, 'precursors') else cand.get('precursors', [])
            confidence = cand.confidence if hasattr(cand, 'confidence') else cand.get('confidence', 0)
            
            precursor_lines = []
            analysis_report = [] 
            
            for p in precursors:
                status = "✅Available" if stock_map.get(p, False) else "❌Needs Synthesis"
                precursor_lines.append(f"  - `{p}` ({status})")
                
                # Autonomous Analysis
                try:
                    if self.analyzer:
                        ans_res = self.analyzer.execute(p)
                        if ans_res.get("status") == "success":
                            props = ans_res.get("properties", {})
                            mw = props.get("MW", 0)
                            logp = props.get("LogP", 0)
                            tpsa = props.get("TPSA", 0)
                            report_line = f"     * Analysis for `{p}`: MW={mw:.1f}, LogP={logp:.2f}, TPSA={tpsa:.1f}. {ans_res.get('formatted_report', '').split('. ')[-1]}"
                            analysis_report.append(report_line)
                        else:
                             analysis_report.append(f"     * Analysis Error for `{p}`: {ans_res.get('message', 'Unknown Error')}")
                except Exception as e:
                    analysis_report.append(f"     * Analysis Error for `{p}`: {str(e)}")

            # --- Advanced Analysis Toolbox Integration ---
            try:
                adv_results = advanced_toolbox.analyze_candidate(target, precursors)
                
                if adv_results:
                    analysis_report.append("\n     [Advanced Metrics]")
                    analysis_report.append(f"     {adv_results.get('formatted_report', '')}")

                    # Use RouteHistoryManager for loop eval
                    loop_eval = {}
                    if cumulative_route:
                        loop_eval = self.route_manager.evaluate_reaction_vector_loop(cumulative_route, path_id, adv_results)
                        adv_results["reaction_vector_loop"] = loop_eval
                        adv_results["fg_effectiveness"] = loop_eval.get("fg_effectiveness", "MED")
                        analysis_report.append(
                            "     [Loop Vector Check] "
                            f"Risk={loop_eval.get('loop_risk', 'N/A')}; "
                            f"FG_Effectiveness={loop_eval.get('fg_effectiveness', 'N/A')}; "
                            f"Reason={loop_eval.get('loop_reason', 'N/A')}"
                        )
                    
                    # Store as structured metadata
                    if isinstance(cand, dict):
                        cand['analysis_metadata'] = adv_results
                    else:
                        setattr(cand, 'analysis_metadata', adv_results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Advanced analysis failed: {e}")
            
            # Include Deep Scan (ReAct) Reasoning if available
            deep_reason = cand.reason if hasattr(cand, 'reason') else cand.get('reason', '')
            if "[DeepScan]" in deep_reason or "Audit Result" in deep_reason:
                analysis_report.append(f"\n     [🔍 DEEP SCAN AUDIT]:\n     {deep_reason}")
            
            analysis_block = "\n".join(analysis_report) if analysis_report else "     (No analysis data available)"
            
            block = (
                f"### Route {i} [{source.upper()}]\n"
                f"   [Comparison]:\n"
                f"       Target   : {target}\n"
                f"       Precursors: {', '.join(precursors)}\n"
                f"   [Internal Scores - IGNORE]: Confidence={confidence:.4f} (Provided for ref only)\n"
                f"   [Precursors List]:\n" + "\n".join(precursor_lines) + "\n"
                f"   [Component Analysis Report - USE THIS]:\n{analysis_block}\n"
            )
            candidate_blocks.append(block)
            
        return "\n\n".join(candidate_blocks)

    def _parse_llm_json(self, text: str) -> Dict:
        """
        Parses LLM output, extracting JSON block with multiple fallback strategies.
        
        V3.6: Enhanced robustness with multi-level fallback:
        1. Standard Markdown code block extraction
        2. Brace-based extraction
        3. JSON repair for common issues (trailing commas, single quotes)
        """
        import re
        
        def try_parse(s: str) -> Optional[Dict]:
            """Attempt JSON parse with error handling."""
            try:
                return json.loads(s)
            except:
                return None
        
        def fix_common_json_issues(s: str) -> str:
            """Fix common JSON formatting issues from LLM output."""
            # Remove trailing commas before } or ]
            s = re.sub(r',\s*([}\]])', r'\1', s)
            # Replace single quotes with double quotes (careful with apostrophes)
            # Only replace quotes around keys/values, not contractions
            s = re.sub(r"(?<=[{,:\[\s])'([^']*)'(?=[}\],:\s])", r'"\1"', s)
            # Remove comments (// style)
            s = re.sub(r'//.*?(?=\n|$)', '', s)
            # Fix unquoted keys (simple cases)
            s = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', s)
            return s
        
        cleaned = text.strip()
        
        # ========== Strategy 1: Markdown code block ==========
        if "```json" in cleaned:
            try:
                json_block = cleaned.split("```json")[1].split("```")[0].strip()
                result = try_parse(json_block)
                if result:
                    return result
                # Try with fixes
                result = try_parse(fix_common_json_issues(json_block))
                if result:
                    logger.debug("JSON parsed after fixing common issues (strategy 1)")
                    return result
            except:
                pass
        
        # ========== Strategy 2: Generic code block ==========
        if "```" in cleaned:
            try:
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    json_block = parts[1].strip()
                    # Remove language identifier if present
                    if json_block.startswith(('json', 'JSON')):
                        json_block = json_block[4:].strip()
                    result = try_parse(json_block)
                    if result:
                        return result
                    result = try_parse(fix_common_json_issues(json_block))
                    if result:
                        return result
            except:
                pass
        
        # ========== Strategy 3: Brace extraction ==========
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_block = cleaned[start:end+1]
            result = try_parse(json_block)
            if result:
                return result
            # Try with fixes
            result = try_parse(fix_common_json_issues(json_block))
            if result:
                logger.debug("JSON parsed after fixing common issues (strategy 3)")
                return result
        
        # ========== Strategy 4: Regex extraction for nested JSON ==========
        # Find the largest balanced JSON object
        try:
            json_pattern = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', cleaned, re.DOTALL)
            if json_pattern:
                json_block = json_pattern.group(0)
                result = try_parse(json_block)
                if result:
                    return result
                result = try_parse(fix_common_json_issues(json_block))
                if result:
                    return result
        except:
            pass
        
        # All strategies failed
        raise ValueError(f"Failed to parse JSON from LLM output. Text preview: {cleaned[:200]}...")

