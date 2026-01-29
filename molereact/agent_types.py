# -*- coding: utf-8 -*-
"""
Agent Data Types
================
Shared data structures and type definitions.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class CandidateRoute:
    """候选路线"""
    rank: int
    precursors: Tuple[str, ...]
    source: str                     # "model" / "template" / "both"
    reaction_type: str
    stock_status: Dict[str, bool]   # {smiles: in_stock}
    risk_analysis: Dict[str, Any]
    confidence_reasoning: str       # 化学合理性推理 (非数值)
    confidence: float = 0.0         # Added to support access in agent_run.py
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class StageResult:
    """阶段分析结果"""
    stage_number: int
    target_smiles: str
    candidates: List[CandidateRoute]
    selected_top_n: List[CandidateRoute]
    unsolved_leaves: List[str]
    stage_image_path: Optional[str]
    timestamp: str
    audit_info: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        # Custom serialization logic if needed
        return {
            "stage_number": self.stage_number,
            "target_smiles": self.target_smiles,
            "candidates": [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.candidates],
            "selected_top_n": [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.selected_top_n],
            "unsolved_leaves": self.unsolved_leaves,
            "stage_image_path": self.stage_image_path,
            "timestamp": self.timestamp,
            "audit_info": self.audit_info
        }
