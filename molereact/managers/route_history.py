# -*- coding: utf-8 -*-
"""
Module: multistep.agent.managers.route_history
Description: Manages synthesis path history and detects cycles.
Refactored from agent_run.py.
"""

from typing import Dict, Any, List, Optional
from multistep.agent.tools.fg_cycle_detector import CycleDetector

class RouteHistoryManager:
    """
    Manages the history of reaction vectors and checks for loops.
    """
    def __init__(self):
        self.cycle_detector = CycleDetector()

    def _get_path_prefixes(self, path_id: str) -> List[str]:
        """
        Parses path_id "1.2.3" -> ["1", "1.2", "1.2.3"] to trace lineage.
        """
        parts = path_id.split(".")
        prefixes = []
        for i in range(1, len(parts) + 1):
            prefixes.append(".".join(parts[:i]))
        return prefixes

    def collect_path_vector_history(self, cumulative_route: Dict, path_id: str) -> List[Dict]:
        """Collects reaction vectors and state signatures for the full lineage of path_id."""
        history = []
        if not cumulative_route or "stages" not in cumulative_route:
            return history

        prefixes = self._get_path_prefixes(path_id)
        
        stages = cumulative_route["stages"]
        
        # Normalize stages to list of (stage_index, stage_data)
        normalized_stages = []
        if isinstance(stages, list):
            for i, s_data in enumerate(stages):
                 normalized_stages.append((str(i+1), s_data))
        elif isinstance(stages, dict):
             # Sort logic for dict keys "1", "2"
             keys = sorted(stages.keys(), key=lambda x: int(x))
             for k in keys:
                 normalized_stages.append((k, stages[k]))
        
        for s_key, stage_data in normalized_stages:
            # stage_data is either the StageResult dict directly OR a container of selections
            # V3.5 format: StageResult dict.
            
            selected_routes = stage_data.get('llm_selected_top_n', [])
            if not selected_routes and isinstance(stage_data, list):
                # Legacy compatibility: stage_data might be the list of routes directly
                selected_routes = stage_data
            
            for p in prefixes:
                found = next((r for r in selected_routes if str(r.get('path_id', r.get('route_id'))) == p), None)
                
                if found and 'metadata' in found:
                    meta = found['metadata']
                    if 'analysis_metadata' in found:
                         meta = found['analysis_metadata']
                    
                    rv = meta.get("reaction_vector", {})
                    history.append({
                        "path_id": p,
                        "state_signature": meta.get("state_signature", ""),
                        "fg_equivalence_state": meta.get("fg_equivalence_state", ""),
                        "exact_state_signature": meta.get("exact_state_signature", ""),
                        "reaction_vector": rv
                    })
        return history

    def _sum_vector(self, vectors: List[Dict[str, int]]) -> Dict[str, int]:
        """Sums a list of vector dicts."""
        total = {}
        for v in vectors:
            for k, val in v.items():
                total[k] = total.get(k, 0) + val
        # Filter 0
        return {k: v for k, v in total.items() if v != 0}

    def _is_inverse_vector(self, left: Dict[str, int], right: Dict[str, int]) -> bool:
        """Checks if right is the exact inverse of left."""
        if not left and not right: return False
        if not left or not right: return False
        
        all_keys = set(left.keys()) | set(right.keys())
        for k in all_keys:
            if left.get(k, 0) + right.get(k, 0) != 0:
                return False
        return True

    def evaluate_reaction_vector_loop(
        self,
        cumulative_route: Dict,
        path_id: str,
        candidate_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyzes the proposed step (candidate_meta) against the history (cumulative_route)
        to detect cycles or stagnation.
        """
        if not candidate_meta:
            return {"loop_risk": "LOW", "loop_reason": "no_metadata", "fg_effectiveness": "MED"}

        # 1. Collect History
        history = self.collect_path_vector_history(cumulative_route, path_id)
        
        # 2. Get Current Signatures
        current_eq_sig = candidate_meta.get("fg_equivalence_state", "")
        current_exact_sig = candidate_meta.get("exact_state_signature", "")
        
        rv = candidate_meta.get("reaction_vector") or {}
        
        if not history:
             return {"loop_risk": "LOW", "loop_reason": "no_history", "fg_effectiveness": "HIGH"}

        # 3. Detect Cycle (Using Component 1 CycleDetector)
        res = self.cycle_detector.detect_complex_cycle(history, current_eq_sig, current_exact_sig)
        
        # 4. Integrate legacy inverse vector check if not already HIGH
        # Use existing utility methods
        if res["loop_risk"] != "HIGH":
             last = history[-1]["reaction_vector"]
             inverse_last = self._is_inverse_vector(last.get("fg_delta", {}), rv.get("fg_delta", {}))
             if inverse_last:
                 res["loop_risk"] = "MED"
                 res["loop_reason"] += " + inverse_of_previous_vector"
                 res["fg_effectiveness"] = "LOW"
                 res["inverse_last"] = True

        # 5. Legacy Cumulative Zero Check (Optional but good for robustness)
        # Calculates if net vector sum is zero
        fg_history = [h["reaction_vector"].get("fg_delta", {}) for h in history]
        ha_history = [h["reaction_vector"].get("heavy_atom_delta", {}) for h in history]
        
        fg_sum = self._sum_vector(fg_history + [rv.get("fg_delta", {})])
        ha_sum = self._sum_vector(ha_history + [rv.get("heavy_atom_delta", {})])
        cumulative_zero = not fg_sum and not ha_sum
        
        state_sig = candidate_meta.get("state_signature", "")
        state_repeats = bool(state_sig) and any(state_sig == h.get("state_signature") for h in history)
        
        if cumulative_zero and state_repeats and res["loop_risk"] == "LOW":
            # Catch legacy simple loop
             res["loop_risk"] = "HIGH"
             res["loop_reason"] = "cumulative_vector_zero_with_state_repeat"
             res["fg_effectiveness"] = "LOW"
             
        res["cumulative_zero"] = cumulative_zero
        res["state_repeats"] = state_repeats
        
        return res
