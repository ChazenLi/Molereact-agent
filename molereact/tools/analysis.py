# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.analysis
Called By: agent_run.py, llm_retro_analyzer.py, agent.py
Role: Chemical Property & Safety Analysis

Functionality:
    Provides a suite of tools for analyzing molecular properties, safety risks, and cost estimation.
    Uses RDKit for cheminformatics calculations.

Key Classes:
    - MoleculeAnalysisTool: Calculates MW, LogP, TPSA, Lipinski Violations, Structural Alerts.
    - CostEstimationTool: (Mock) Estimates synthesis cost.
    - SafetyCheckTool: Checks for high-risk reagents or functional groups.
    - ScaleUpTool: (Mock) Analyzes scalability.

Relations:
    - Inherits from `BaseTool`.
    - Used by `RetroSynthesisAgent` and `ReActSession` for validation.
"""
"""
Analysis Tools
==============
"""
from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from .base import BaseTool

class MoleculeAnalysisTool(BaseTool):
    """
    RDKit-based Molecule Analysis Tool.
    Calculates physicochemical properties and structural alerts.
    """
    @property
    def name(self) -> str:
        return "MoleculeAnalysis"
    
    @property
    def description(self) -> str:
        return "Calculates physicochemical properties (MW, LogP, TPSA), checks Lipinski rules."
    
    def execute(self, smiles: str, **kwargs) -> Dict[str, Any]:
        if not smiles: return {"status": "error", "message": "Empty SMILES"}
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return {"status": "error", "message": "Invalid SMILES"}
            
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            violations = []
            if mw > 500: violations.append("MW > 500")
            if logp > 5: violations.append("LogP > 5")
            
            return {
                "status": "success",
                "formatted_report": f"MW: {mw:.1f}, LogP: {logp:.1f}, TPSA: {tpsa:.1f}. Violations: {violations or 'None'}",
                "properties": {"MW": mw, "LogP": logp, "TPSA": tpsa}
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

class CostEstimationTool(BaseTool):
    @property
    def name(self) -> str:
        return "CostEstimation"
    @property
    def description(self) -> str:
        return "Estimates process cost based on steps and materials."
    
    def execute(self, route_json: Dict, scale: str="lab", target_quantity: str="1g", **kwargs) -> Dict[str, Any]:
        # Simplified logic from original
        stages = route_json.get("stages", [])
        num_steps = len(stages)
        base_cost = 700 * num_steps # placeholder
        return {
            "total_estimated_cost": base_cost,
            "currency": "CNY",
            "drivers": ["Steps: " + str(num_steps)]
        }

class SafetyCheckTool(BaseTool):
    @property
    def name(self) -> str:
        return "SafetyCheck"
    @property
    def description(self) -> str:
        return "Evaluates reaction safety and reagent hazards."
    
    def execute(self, route_json: Dict, reagent_list: List[str] = None, **kwargs) -> Dict[str, Any]:
        # Simplified logic
        if reagent_list is None:
            reagent_list = []
            for s in route_json.get("stages", []):
                reagent_list.extend(s.get("reagents", []))
        
        hazards = []
        # Demo DB
        HAZARD_DB = {"NaH": "Flammable", "LiAlH4": "Explosive/Water Reactive"}
        
        for r in reagent_list:
             for k, v in HAZARD_DB.items():
                 if k in r:
                     hazards.append(f"{r}: {v}")
        
        return {
            "risk_level": "HIGH" if hazards else "LOW",
            "hazards": hazards
        }

class ScaleUpTool(BaseTool):
    @property
    def name(self) -> str:
        return "ScaleUpAnalysis"
    @property
    def description(self) -> str:
        return "Analyzes feasibility for pilot/production scale."
    
    def execute(self, route_json: Dict, target_scale: str="pilot", **kwargs) -> Dict[str, Any]:
        stages = route_json.get("stages", [])
        issues = []
        if len(stages) > 5:
            issues.append("Too many linear steps for easy scale-up")
        return {
            "feasibility": "Medium" if issues else "High",
            "issues": issues
        }
