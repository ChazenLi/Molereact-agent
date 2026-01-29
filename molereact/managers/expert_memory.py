# -*- coding: utf-8 -*-
"""
Module: multistep.agent.managers.expert_memory
Description: Manages Long-Term Expert Memory (human-in-the-loop learning).
"""

import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from multistep.agent.tools.advanced_analysis import toolbox as advanced_toolbox

class ExpertMemoryManager:
    """
    Manages a persistent store of high-quality expert reactions.
    """
    def __init__(self, storage_path: str = None):
        if storage_path:
             self.storage_path = storage_path
        else:
             # Default to multistep/data/expert_memory.jsonl
             base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             self.storage_path = os.path.join(base_dir, "data", "expert_memory.jsonl")
        
        self._ensure_storage()
        self.memory_cache = []
        self.load_memory()

    def _ensure_storage(self):
        if not os.path.exists(os.path.dirname(self.storage_path)):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
    def load_memory(self):
        """Loads entries into RAM for simpler similarity search (acceptable for <100k entries)."""
        self.memory_cache = []
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.memory_cache.append(json.loads(line))
                        except:
                            continue
        print(f"Memory loaded: {len(self.memory_cache)} expert entries.")

    def add_expert_reaction(
        self, 
        target_smiles: str, 
        precursor_smiles: List[str], 
        reaction_type: str = "Manual Entry",
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes and saves a manual reaction step.
        """
        # 1. Standardize (Basic)
        t_mol = Chem.MolFromSmiles(target_smiles)
        target_std = Chem.MolToSmiles(t_mol, isomericSmiles=True) if t_mol else target_smiles
        
        p_std_list = []
        for p in precursor_smiles:
            m = Chem.MolFromSmiles(p)
            if m: p_std_list.append(Chem.MolToSmiles(m, isomericSmiles=True))
            else: p_std_list.append(p)

        # 2. Advanced Analysis (Compute Vector, Mapping, FGs)
        # This makes the "unstructured" input "structured"
        analysis_data = {}
        try:
            analysis_data = advanced_toolbox.analyze_candidate(target_std, p_std_list)
        except Exception as e:
            print(f"Warning: Expert analysis failed: {e}")
            analysis_data = {"error": str(e)}

        # 3. Construct Entry
        entry = {
            "id": str(uuid.uuid4()),
            "target_smiles": target_std,
            "precursors_smiles": p_std_list,
            "reaction_type": reaction_type,
            "metadata": analysis_data,
            "tags": tags or ["manual"],
            "timestamp": time.time(),
            # Pre-compute fingerprint for searching
            "fingerprint": self._compute_fp_hex(target_std) 
        }

        # 4. Save
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        self.memory_cache.append(entry)
        return entry

    def retrieve_similar_reactions(
        self, 
        target_smiles: str, 
        top_k: int = 3, 
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Finds reactions where the stored target is similar to the query target.
        """
        query_fp = self._get_fp(target_smiles)
        if not query_fp: return []
        
        results = []
        for entry in self.memory_cache:
            entry_fp = self._get_fp_from_hex(entry.get("fingerprint"))
            if entry_fp:
                sim = DataStructs.TanimotoSimilarity(query_fp, entry_fp)
                if sim >= threshold:
                    results.append({"entry": entry, "similarity": sim})
        
        # Sort by similarity desc
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return [r["entry"] for r in results[:top_k]]

    def _compute_fp_hex(self, smiles: str) -> str:
        fp = self._get_fp(smiles)
        return DataStructs.BitVectToText(fp) if fp else ""

    def _get_fp_from_hex(self, hex_str: str):
        try:
            if not hex_str: return None
            # Need reliable hex/binary serialization. 
            # RDKit Text is hex string of bits? 
            # Actually simpler: just recompute from SMILES if dataset is small, 
            # or store as explicit bitstring.
            # For robustness in this MVP, let's just recompute if 'target_smiles' is present.
            # (Caching raw objects in memory_cache is better)
            return None 
        except:
            return None

    def _get_fp(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except:
            pass
        return None

    # Overriding retrieve to use on-the-fly component for accuracy in MVP
    def retrieve_similar_reactions_mvp(
        self, 
        target_smiles: str, 
        top_k: int = 5, 
        threshold: float = 0.4 # Lower threshold for substructure-ish
    ) -> List[Dict[str, Any]]:
        query_fp = self._get_fp(target_smiles)
        if not query_fp: return []
        
        scored = []
        for entry in self.memory_cache:
            # Recompute from SMILES (safer than implementing hex deserialization right now)
            tgt = entry.get("target_smiles")
            if tgt:
                fp = self._get_fp(tgt)
                if fp:
                    sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                    if sim >= threshold:
                        scored.append((sim, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]
