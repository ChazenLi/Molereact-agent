# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.chemistry
Called By: agent.py
Role: Chemical Reaction & Structure Handling

Functionality:
    Provides basic chemical intelligence tools:
    - RetroSingleStep: Generates precursors for a target structure (Main Engine Interface).
    - SmilesStandardize: Validates and canonicalizes SMILES strings.
    - ReactionClassify: (Mock) Identifies reaction types.

Key Classes:
    - RetroSingleStepTool: Interface to the SingleStepEngine.
    - SmilesStandardizeTool: Interface to conversion utilities.

Relations:
    - Wraps methods from `multistep.single_step_engine` and `smiles_standardizer`.
"""
"""
Chemistry Domain Tools
======================
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from .base import BaseTool

logger = logging.getLogger(__name__)

class RetroSingleStepTool(BaseTool):
    """Encapsulates SingleStepRetroEngine."""
    def __init__(self, engine=None):
        self._engine = engine

    @property
    def name(self) -> str:
        return "RetroSingleStep"
    
    @property
    def description(self) -> str:
        return "Proposes precursors for a target molecule using retrosynthesis models."
    
    def execute(self, target_smiles: str, topk_model: int = 10, topk_template: int = 10, **kwargs) -> Dict[str, Any]:
        if self._engine is None:
            # Lazy init
            from multistep.single_step_engine import create_default_engine
            self._engine = create_default_engine()
            
        result = self._engine.propose_precursors(
            target_smiles,
            topk_model=topk_model,
            topk_template=topk_template
        )
        
        # Format output
        return {
            "target": result["target"],
            "model_candidates": [c.to_dict() for c in result["model_candidates"]],
            "template_candidates": [c.to_dict() for c in result["template_candidates"]],
            "stats": result["stats"],
            "timestamp": datetime.now().isoformat()
        }

class ReactionClassifyTool(BaseTool):
    """Encapsulates ReactionClassifier."""
    @property
    def name(self) -> str:
        return "ReactionClassify"
    
    @property
    def description(self) -> str:
        return "Classifies a reaction type (BISECT/MODIFY/MIXED) and calculates scaffold similarity."

    def execute(self, product_smiles: str, precursor_smiles: List[str], **kwargs) -> Dict[str, Any]:
        try:
            from multistep.reaction_classifier import ReactionClassifier
            classifier = ReactionClassifier()
            result = classifier.classify(product_smiles, tuple(precursor_smiles))
            
            explanation = self._generate_explanation(result)
            return {
                "reaction_type": result.reaction_type.value,
                "confidence": result.confidence,
                "scaffold_similarity": result.scaffold_similarity,
                "heavy_atom_delta": result.heavy_atom_delta,
                "explanation": explanation
            }
        except ImportError:
            return {"status": "error", "message": "ReactionClassifier module not found"}

    def _generate_explanation(self, result):
        if result.reaction_type.value == "BISECT":
            return f"骨架断裂反应: 产物分解为多个片段，相似度 {result.scaffold_similarity:.2f}"
        elif result.reaction_type.value == "MODIFY":
            return f"官能团修饰: 骨架保持，相似度 {result.scaffold_similarity:.2f}"
        else:
            return f"混合类型反应，相似度 {result.scaffold_similarity:.2f}"

class SmilesStandardizeTool(BaseTool):
    """Encapsulates Standardizer."""
    @property
    def name(self) -> str:
        return "SmilesStandardize"
    
    @property
    def description(self) -> str:
        return "Standardizes and canonicalizes SMILES strings."
    
    def execute(self, smiles: str, mode: str = "canon", **kwargs) -> Dict[str, Any]:
        try:
            from multistep.agent.smiles_standardizer import Standardizer
            # Use module level or passed instance? New instance is safer for now.
            standardizer = Standardizer()
            
            if mode == "full":
                res = standardizer.standardize_full(smiles)
            else:
                res = standardizer.canonicalize(smiles)
                
            return {
                "original": smiles,
                "standardized": res,
                "success": res is not None,
                "mode": mode
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

class AtomMappingTool(BaseTool):
    """
    Atom Mapping Tool using RXNMapper.
    Maps atoms between reactants and products to identify bond changes.
    """
    @property
    def name(self) -> str:
        return "AtomMapping"
    
    @property
    def description(self) -> str:
        return "Maps atoms in a chemical reaction (SMILES) to identify which atoms in reactants correspond to which in products. Useful for bond-change analysis."
    
    def execute(self, reaction_smiles: str, **kwargs) -> Dict[str, Any]:
        """
        Args:
            reaction_smiles: 'reactant1.reactant2>>product' or similar.
        """
        try:
            from rxnmapper import RXNMapper
            rxn_mapper = RXNMapper()
            results = rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])
            result = results[0]
            
            return {
                "status": "success",
                "mapped_rxn": result.get("mapped_rxn"),
                "confidence": result.get("confidence"),
                "message": "Atom mapping completed successfully."
            }
        except ImportError:
            return {
                "status": "error", 
                "message": "rxnmapper is not installed in the environment. Please install with 'pip install rxnmapper'."
            }
        except Exception as e:
            return {"status": "error", "message": f"Mapping failed: {str(e)}"}

class ScaffoldAnalysisTool(BaseTool):
    """
    Analyzes the molecular scaffold (Murcko Scaffold).
    """
    @property
    def name(self) -> str:
        return "ScaffoldAnalysis"
    
    @property
    def description(self) -> str:
        return "Extracts and analyzes the Murcko Scaffold of a molecule. Useful for understanding the core structural framework."
    
    def execute(self, smiles: str, **kwargs) -> Dict[str, Any]:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"status": "error", "message": "Invalid SMILES"}
            
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if not scaffold_mol:
                return {"status": "success", "scaffold": "", "message": "No scaffold found (likely a small acyclic molecule)."}
                
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
            
            # Additional analysis: ring count in scaffold
            from rdkit.Chem import rdMolDescriptors
            ring_count = rdMolDescriptors.CalcNumRings(scaffold_mol)
            
            return {
                "status": "success",
                "scaffold": scaffold_smiles,
                "num_rings": ring_count,
                "heavy_atom_count": scaffold_mol.GetNumHeavyAtoms()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
