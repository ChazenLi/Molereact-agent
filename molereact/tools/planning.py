# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.planning
Called By: agent.py
Role: Strategic Route Selection & Decision Making

Functionality:
    Encapsulates the logic for selecting the best retrosynthesis route from candidates.
    Integrates LLM-based reasoning (Holistic V2) and Heuristic selection fallback.

Key Classes:
    - RouteSelectionTool: The primary decision-maker.

Relations:
    - Uses `prompts.py` for LLM interaction.
    - Operates on `CandidateRoute` objects.
"""
"""
Planning Tools
==============
"""
import logging
import json
from typing import Dict, Any, List
from .base import BaseTool

logger = logging.getLogger(__name__)

class RouteSelectionTool(BaseTool):
    """Encapsulates LLM-based Route Selection."""
    def __init__(self, llm_client=None):
        self._client = llm_client

    @property
    def name(self) -> str:
        return "RouteSelection"
    
    @property
    def description(self) -> str:
        return "Selects the best reaction routes using LLM or heuristics."
    
    def set_client(self, client):
        self._client = client

    def execute(self, candidates: List[Dict], stock_results: Dict, risk_analysis: Dict=None, selection_criteria: Dict=None, top_n: int=3, **kwargs) -> Dict[str, Any]:
        criteria = selection_criteria or {
            "prefer_stock": True,
            "avoid_protection": False,
            "prefer_common_reactions": True,
        }
        
        # If client available, try LLM
        if self._client:
            try:
                # We need to reconstruct the prompt logic here or import it
                # For simplicity, we import the helper from prompts.py if needed, OR reimplement
                # Let's rely on heuristic fallback for this refactor if prompt construction is too complex to copy-paste
                # UNLESS we assume we can import `multistep.agent.prompts`
                from multistep.agent.prompts import get_simple_selection_prompt
                
                candidates_summary = ""
                for i, cand in enumerate(candidates[:15], 1):
                    precursors = cand.get('precursors', [])
                    source = cand.get('source', 'unknown')
                    candidates_summary += f"\n{i}. [{source}] {' + '.join(precursors)}"
                
                prompt = get_simple_selection_prompt(
                    candidates_summary=candidates_summary,
                    stock_results_json=json.dumps(stock_results, ensure_ascii=False, indent=2),
                    criteria=criteria,
                    top_n=top_n
                )
                
                response = self._client.chat(prompt)
                return self._parse_llm_response(response, candidates)
            except Exception as e:
                logger.warning(f"LLM selection failed: {e}, using heuristic fallback")
        
        return self._heuristic_select(candidates, stock_results, criteria, top_n)

    def _heuristic_select(self, candidates, stock_results, criteria, top_n):
        scored = []
        stock_map = {r["smiles"]: r["in_stock"] for r in stock_results.get("results", [])}
        
        for i, cand in enumerate(candidates):
            score = 0
            # Source bonus
            if cand.get("source") == "template": score += 10
            elif cand.get("source") == "both": score += 15
            
            # Stock bonus
            precursors = cand.get("precursors", [])
            stock_count = sum(1 for p in precursors if stock_map.get(p, False))
            score += stock_count * 20
            
            # Confidence
            score += cand.get("confidence", 0) * 10
            scored.append((score, i, cand))
            
        scored.sort(reverse=True, key=lambda x: x[0])
        
        selected_routes = []
        for rank, (score, orig_idx, cand) in enumerate(scored[:top_n], 1):
            selected_routes.append({
                "rank": rank,
                "precursors": cand.get("precursors", []),
                "reason": f"Heuristic Score: {score:.1f}",
                "priority": "HIGH" if rank == 1 else "MEDIUM"
            })
            
        return {
            "selected_routes": selected_routes,
            "summary": f"Heuristically selected {len(selected_routes)} routes.",
            "next_step_suggestion": "Review manual"
        }

    def _parse_llm_response(self, response, candidates):
        # Simplified parser
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                selected_ranks = parsed.get("selected_ranks", [1, 2, 3])
                selected_routes = []
                for rank in selected_ranks:
                    if 1 <= rank <= len(candidates):
                        cand = candidates[rank-1]
                        selected_routes.append({
                            "rank": rank,
                            "precursors": cand.get("precursors", []),
                            "reason": parsed.get("reasons", [""])[len(selected_routes)] if len(parsed.get("reasons", [])) > len(selected_routes) else "",
                            "priority": "HIGH"
                        })
                return {
                    "selected_routes": selected_routes,
                    "summary": parsed.get("summary", ""),
                    "next_step_suggestion": parsed.get("next_step_suggestion", "")
                }
        except Exception:
            pass
        return self._heuristic_select(candidates, {}, {}, 3)
