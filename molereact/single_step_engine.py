# -*- coding: utf-8 -*-
"""
Module: multistep.single_step_engine
Called By: agent/tools/chemistry.py, agent_run.py (legacy)
Role: Core Single-Step Retrosynthesis Engine

Functionality:
    Wraps the underlying single-step model (e.g., local model or API).
    Provides methods to propose precursors given a target SMILES.
    Aggregates results from template-based and model-based sources.

Key Classes:
    - SingleStepEngine: The main engine class.
    
Usage:
    engine = create_default_engine()
    result = engine.propose_precursors("CCO")
"""
"""
Single-Step Retrosynthesis Engine (单步逆合成分析引擎)
=====================================================

功能概述 (Overview):
    本模块提供一个统一的"单步逆合成推理引擎"，用于为给定的目标分子生成可能的前体(Precursor)集合。
    它并行运行两条独立的推理分支，并将结果合并、去重、排序后输出：
    
    1. **Deep Learning Branch (深度学习分支)**:
       - 使用训练好的 Retro Transformer 模型预测可能的断键方式。
       - 使用 Forward Transformer 模型进行"往返验证"(Round-Trip Validation)，评估预测的可靠性。
       - 综合打分：Confidence = sqrt(P_retro * Score_forward)。
       
    2. **Template Branch (模板分支)**:
       - 使用 AiZynthFinder (USPTO/RingBreaker 模板库) 进行基于已知反应规则的逆合成展开。
       - 使用模板匹配概率进行打分。

    最终输出一个标准化的 `RetroStep` 对象列表，可直接供 LLM 或化学家进行决策分析。

核心类 (Core Classes):
    - `SingleStepRetroEngine`: 主引擎类，整合模型和模板两条分支。
    - `RetroStep`: 标准化的单步逆合成结果数据结构。

工厂函数 (Factory):
    - `create_default_engine()`: 使用预训练模型创建引擎的便捷函数。

默认模型路径 (Default Model Paths):
    - Retro Model: MoleReact/retro_transformer/runs/20260117_004131/best_exact.pt
    - Forward Model: MoleReact/retro_transformer/runs/20260116_122126/best_exact_forward.pt
    - Vocab: MoleReact/retro_transformer/runs/20260117_004131/vocab.json

使用示例 (Usage Example):
    ```python
    from multistep.single_step_engine import create_default_engine
    
    engine = create_default_engine()
    candidates = engine.propose_precursors("c1ccccc1C(=O)O")
    
    for c in candidates[:5]:
        print(f"Score: {c.confidence:.3f}, Precursors: {c.precursors}")
    ```


"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from .aizynthsession import AizynthSession
from .hybrid_retro_planner import RetroModel, ForwardModel
from .scoring_utils import compute_tanimoto_similarity, analyze_selectivity_risk

logger = logging.getLogger(__name__)

# ============================================================================
# Default Model Paths (relative to MoleReact root)
# ============================================================================
_MOLEREACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RETRO_MODEL_PATH = os.path.join(
    _MOLEREACT_ROOT, "retro_transformer", "runs", "20260117_004131", "best_exact.pt"
)
DEFAULT_FORWARD_MODEL_PATH = os.path.join(
    _MOLEREACT_ROOT, "retro_transformer", "runs", "20260116_122126", "best_exact_forward.pt"
)
DEFAULT_VOCAB_PATH = os.path.join(
    _MOLEREACT_ROOT, "retro_transformer", "runs", "20260117_004131", "vocab.json"
)
DEFAULT_CONFIG_PATH = os.path.join(
    _MOLEREACT_ROOT, "multistep", "config.yml"
)


def create_default_engine(
    config_path: str = None,
    retro_path: str = None,
    fwd_path: str = None,
    vocab_path: str = None,
    use_parallel: bool = True
) -> "SingleStepRetroEngine":
    """
    Factory function to create a SingleStepRetroEngine with default models.
    
    Args:
        config_path: Path to AiZynthFinder config. Defaults to multistep/config.yml.
        retro_path: Path to retro model .pt file.
        fwd_path: Path to forward model .pt file.
        vocab_path: Path to vocab.json.
        use_parallel: Whether to run branches in parallel.
        
    Returns:
        Configured SingleStepRetroEngine instance.
    """
    from .model_adapters import TransformerRetroModel, TransformerForwardModel
    
    config_path = config_path or DEFAULT_CONFIG_PATH
    retro_path = retro_path or DEFAULT_RETRO_MODEL_PATH
    fwd_path = fwd_path or DEFAULT_FORWARD_MODEL_PATH
    vocab_path = vocab_path or DEFAULT_VOCAB_PATH
    
    logger.info(f"Loading AiZynthSession from {config_path}")
    session = AizynthSession(config_path)
    
    logger.info(f"Loading Retro Model from {retro_path}")
    retro_model = TransformerRetroModel(retro_path, vocab_path)
    
    logger.info(f"Loading Forward Model from {fwd_path}")
    fwd_model = TransformerForwardModel(fwd_path, vocab_path)
    
    return SingleStepRetroEngine(retro_model, fwd_model, session, use_parallel=use_parallel)



@dataclass
class RetroStep:
    """Standardized single retrosynthetic step candidate"""
    target: str
    precursors: Tuple[str, ...]
    source: str         # "model", "template", "both"
    confidence: float   # 0.0 to 1.0
    retro_score: float  # Raw P(retro) or template score
    fwd_score: float    # Round-trip consistency score or forward P
    reaction_type: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "target": self.target,
            "precursors": list(self.precursors),
            "source": self.source,
            "confidence": self.confidence,
            "retro_score": self.retro_score,
            "fwd_score": self.fwd_score,
            "risk": self.metadata.get("risk", {}),
            "reaction_type": self.reaction_type
        }

class SingleStepRetroEngine:
    """
    Unified Single-Step Retrosynthesis Engine.
    Combines Deep Learning Models and Template-Based Rules.
    """
    
    def __init__(
        self,
        retro_model: RetroModel,
        forward_model: ForwardModel,
        template_engine: AizynthSession,
        use_parallel: bool = True
    ):
        self.retro_model = retro_model
        self.forward_model = forward_model
        self.template_engine = template_engine
        self.use_parallel = use_parallel

    def propose_precursors(
        self, 
        target_smiles: str, 
        topk_model: int = 10,
        topk_template: int = 10
    ) -> Dict[str, Any]:
        """
        Generate precursor candidates for a target molecule.
        
        Args:
            target_smiles: Target molecule SMILES.
            topk_model: Number of top candidates from Model branch.
            topk_template: Number of top candidates from Template branch.
            
        Returns:
            Dict with structure:
            {
                "target": str,
                "model_candidates": List[RetroStep],  # Top K from model, sorted by model score
                "template_candidates": List[RetroStep],  # Top K from template, sorted by template score
                "union": List[RetroStep],  # Deduplicated union (for reference)
                "stats": {"model_count": int, "template_count": int, "overlap_count": int}
            }
            
        NOTE: Model and Template scores are NOT comparable! 
        Each list is ranked within its own source only.
        """
        # Execute branches
        if self.use_parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_model = executor.submit(self._run_model_branch, target_smiles, topk_model * 2)
                future_template = executor.submit(self._run_template_branch, target_smiles, topk_template * 2)
                
                cands_model = future_model.result()
                cands_template = future_template.result()
        else:
            cands_model = self._run_model_branch(target_smiles, topk_model * 2)
            cands_template = self._run_template_branch(target_smiles, topk_template * 2)
        
        # Sort WITHIN each source
        cands_model = sorted(cands_model, key=lambda x: x.confidence, reverse=True)[:topk_model]
        cands_template = sorted(cands_template, key=lambda x: x.confidence, reverse=True)[:topk_template]
        
        # Add rank_in_source to metadata
        for i, c in enumerate(cands_model):
            c.metadata['rank_in_source'] = i + 1
        for i, c in enumerate(cands_template):
            c.metadata['rank_in_source'] = i + 1
            
        # Calculate union for reference (detect overlaps)
        model_keys = {tuple(sorted(c.precursors)) for c in cands_model}
        template_keys = {tuple(sorted(c.precursors)) for c in cands_template}
        overlap_keys = model_keys & template_keys
        
        # Build union list (mark overlaps as "both")
        union = []
        seen = set()
        for c in cands_model + cands_template:
            key = tuple(sorted(c.precursors))
            if key not in seen:
                seen.add(key)
                if key in overlap_keys:
                    c.source = "both"
                union.append(c)
        
        return {
            "target": target_smiles,
            "model_candidates": cands_model,
            "template_candidates": cands_template,
            "union": union,
            "stats": {
                "model_count": len(cands_model),
                "template_count": len(cands_template),
                "overlap_count": len(overlap_keys)
            }
        }

    def _run_model_branch(self, target: str, topk: int) -> List[RetroStep]:
        """Deep Learning Branch: Predict -> Forward Check -> Score"""
        candidates = []
        try:
            # 1. Retro Prediction
            preds = self.retro_model.predict(target, topk=topk)
            
            for pred in preds:
                # 2. Forward Verification (Round-Trip)
                fwd_score = 0.0
                try:
                    # Logic: If fwd model predicts 'target' from 'precursors' with high prob
                    # Or we can generate product and check similarity
                    # Here we use the abstract 'score' method which gives P(target | precursors)
                    fwd_score = self.forward_model.score(list(pred.precursors), target)
                    
                    # Optional: We could also GENERATE the product and check Tanimoto
                    # But ForwardModel interface currently only provides score()
                except Exception:
                    fwd_score = 0.0
                
                # Check for log-probs (negative values)
                r_score = pred.retro_score
                if r_score < 0: r_score = 0.001 # Treat log-prob as small prob or handle appropriately
                
                f_score = fwd_score
                if f_score < 0: f_score = 0.001

                # 3. Combined Confidence
                # Geometric mean often balances precision better than arithmetic
                confidence = (r_score * f_score) ** 0.5
                if isinstance(confidence, complex): confidence = 0.0
                
                # 4. Risk Analysis
                risk = analyze_selectivity_risk(pred.precursors)
                
                candidates.append(RetroStep(
                    target=target,
                    precursors=pred.precursors,
                    source="model",
                    confidence=confidence,
                    retro_score=pred.retro_score,
                    fwd_score=fwd_score,
                    metadata={"risk": risk}
                ))
        except Exception as e:
            logger.warning(f"Model branch failed: {e}")
            
        return candidates

    def _run_template_branch(self, target: str, topk: int) -> List[RetroStep]:
        """Template Branch: AiZynthFinder Expansion"""
        candidates = []
        try:
            # 1. Expand
            results = self.template_engine.expand_once(target, topk=topk)
            
            for res in results:
                # AiZynth doesn't provide Fwd score directly, assume default or run fwd model
                fwd_score = 0.5 # Default neutral
                try:
                    fwd_score = self.forward_model.score(list(res.reactants), target)
                except: pass
                
                # Template score (if available in metadata) or default
                retro_score = float(res.metadata.get("policy_probability", 0.5))
                if retro_score > 1.0: retro_score = 0.99 # Cap
                
                # Confidence
                confidence = (retro_score * fwd_score) ** 0.5
                
                risk = analyze_selectivity_risk(res.reactants, res.template_smarts)
                
                # Ensure reactants are strings
                precursors_str = tuple(str(x) if not isinstance(x, str) else x for x in res.reactants)

                candidates.append(RetroStep(
                    target=target,
                    precursors=precursors_str,
                    source="template",
                    confidence=confidence,
                    retro_score=retro_score,
                    fwd_score=fwd_score,
                    reaction_type=res.template_smarts, # Identify by template
                    metadata={"risk": risk, "template": res.template_smarts}
                ))
        except Exception as e:
            logger.warning(f"Template branch failed: {e}")
            
        return candidates

    def _merge_candidates(self, list_a: List[RetroStep], list_b: List[RetroStep]) -> List[RetroStep]:
        """Refined merge with deduplication"""
        
        # Map by sorted precursors signature
        merged_map = {}
        
        for cand in list_a + list_b:
            # Create a unique key for the chemical transformation
            key = tuple(sorted(cand.precursors))
            
            if key in merged_map:
                existing = merged_map[key]
                
                # Update source flag
                if existing.source != cand.source:
                    existing.source = "both"
                    
                # Boost confidence if found by both methods
                if cand.source != existing.source: 
                    # Bonus for consensus
                    existing.confidence = min(0.99, existing.confidence * 1.2)
                    
                # Keep highest metadata/scores
                existing.retro_score = max(existing.retro_score, cand.retro_score)
                existing.fwd_score = max(existing.fwd_score, cand.fwd_score)
            else:
                merged_map[key] = cand
                
        return list(merged_map.values())
