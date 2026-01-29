# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.visualization
Called By: agent_run.py
Role: Visual Report Generation

Functionality:
    Generates images of molecules and reaction pathways using RDKit.
    Produces grid images for candidate routes to aid human review.

Key Classes:
    - VisualizationTool: Main visualization utility.

Relations:
    - Uses RDKit `Draw` and `AllChem`.
"""
"""
Visualization Tools
===================
"""
from typing import Dict, Any, List
import os
from rdkit import Chem
from rdkit.Chem import Draw
from .base import BaseTool

class VisualizationTool(BaseTool):
    @property
    def name(self) -> str:
        return "StageVisualization"
    @property
    def description(self) -> str:
        return "Generates images for reaction stages."
    
    def execute(self, target_smiles: str, selected_precursors: List[str], node_id: str, output_dir: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generates a visualization for a specific node in the synthesis tree.
        
        Args:
            target_smiles: The SMILES of the target molecule.
            selected_precursors: List of SMILES for the precursors.
            node_id: The unique ID from the SynthesisTree (e.g., '1.1').
            output_dir: Parent directory for images.
        """
        if output_dir is None:
            output_dir = "output/agent_stages"
        os.makedirs(output_dir, exist_ok=True)
        
        mols = []
        labels = [f"Node: {node_id}\n{target_smiles[:15]}..."]
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol: mols.append(target_mol)
        
        for i, s in enumerate(selected_precursors):
            m = Chem.MolFromSmiles(s)
            if m:
                mols.append(m)
                labels.append(f"P{i+1}")
        
        # Use node_id for naming to ensure consistency with the Data Tree
        filename = f"node_{node_id}.png".replace(".", "_") # Replace dots to avoid FS issues if needed, but '.' is usually fine.
        img_path = os.path.join(output_dir, filename)
        
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300), legends=labels)
            img.save(img_path)
            
        return {"image_path": img_path, "node_id": node_id}
