# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.fg_cycle_detector
Description: Core Functional Group Equivalence and Cycle Detection Logic.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
import collections
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# 1. FG Equivalence Classes
# Groups chemically related functional groups that often participate in reversible cycles.
FG_EQUIVALENCE_CLASSES = {
    "carbonyl_acid_derivative": [
        "Acid", "AcidChloride", "AcidBromide", "Anhydride", 
        "Ester", "Amide", "Thioester", "Carboxylate", "Hydrazide", "HydroxamicAcid"
    ],
    "nitrogen_amine_derivative": [
        "Amine_Primary", "Amine_Secondary", "Amine_Tertiary", 
        "Amide", "Imide", "Imine", "Carbamate", "Urea", "Enamine", "Amidine",
        "Azide", "Isocyanate", "Isothiocyanate", "Sulfonamide"
    ],
    "carbonyl_ox_state": [
        "Alcohol", "Aldehyde", "Ketone", "Acid", "Carboxylate", "Ester", "Hemiacetal", "Acetal"
    ],
    "halide_leaving": [
        "Halide_Cl", "Halide_Br", "Halide_I", "Sulfonate", "SulfonylChloride", "Diazo"
    ],
    "sulfur_ox_state": [
        "Thiol", "Thioether", "Sulfoxide", "Sulfone", "SulfonicAcid", "Sulfonamide"
    ],
    "boron_coupling": [
        "Boronate", "BoronicAcid", "Halide_Br", "Halide_I", "Triflate"
    ],
    "protection_ether": [
        "Alcohol", "Ether", "SilylEther", "Acetal", "Ketal", "BenzylEther", "Ester", "Carbonate"
    ],
    "azole_ring": [
         "Imidazole", "Pyrazole", "Triazole", "Tetrazole", "Oxazole", "Isoxazole", "Thiazole", "Isothiazole"
    ]
}

# Reverse mapping for fast lookup
FG_TO_CLASS = collections.defaultdict(list)
for cls_name, fgs in FG_EQUIVALENCE_CLASSES.items():
    for fg in fgs:
        FG_TO_CLASS[fg].append(cls_name)

class FGStateTracker:
    """
    Maintains a scaffold + FG state signature for each node to detect equivalence cycles.
    """
    
    @staticmethod
    def get_scaffold_smiles(smiles: str) -> str:
        """Extracts Murcko Scaffold."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return "Invalid"
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if not scaf or scaf.GetNumAtoms() == 0:
                return "Acyclic"
            return Chem.MolToSmiles(scaf)
        except:
            return "Error"

    @staticmethod
    def get_active_equivalence_classes(present_fgs: List[str]) -> Set[str]:
        """Maps a list of present FGs to their equivalence classes."""
        active_classes = set()
        for fg in present_fgs:
            if fg in FG_TO_CLASS:
                active_classes.update(FG_TO_CLASS[fg])
        return active_classes

    @classmethod
    def generate_equivalence_state_signature(cls, smiles: str, present_fgs: List[str]) -> str:
        """
        Generates a signature representing the molecule's equivalence class state:
        Signature = Scaffold + Sorted(EquivalenceClasses)
        """
        scaf = cls.get_scaffold_smiles(smiles)
        classes = sorted(list(cls.get_active_equivalence_classes(present_fgs)))
        if not classes:
            # Fallback to raw FGs if no class mapped
            classes = sorted(present_fgs)
            
        return f"{scaf}|{','.join(classes)}"

    @classmethod
    def generate_exact_state_signature(cls, smiles: str, present_fgs: List[str]) -> str:
        """
        Generates a signature representing the molecule's exact FG state:
        Signature = Scaffold + Sorted(SpecificFGs)
        """
        scaf = cls.get_scaffold_smiles(smiles)
        return f"{scaf}|{','.join(sorted(present_fgs))}"

def calculate_fg_effectiveness(fg_delta: Dict[str, int], current_classes: Set[str], previous_classes: Set[str]) -> str:
    """
    Quantifies whether an FGI provides net progress.
    """
    class_diff_added = current_classes - previous_classes
    class_diff_removed = previous_classes - current_classes
    
    if class_diff_added or class_diff_removed:
        return "HIGH"
        
    return "LOW" 

class CycleDetector:
    """
    Detects complex cycles using FG equivalence history.
    """
    
    def detect_complex_cycle(
        self, 
        history: List[Dict[str, Any]], 
        current_eq_sig: str, 
        current_exact_sig: str
    ) -> Dict[str, Any]:
        """
        Args:
            history: List of prior steps with signatures
            current_eq_sig: Equivalence signature
            current_exact_sig: Exact signature
            
        Returns:
             Dict with keys: is_cycle, cycle_length, reason, risk_level
        """
        is_cycle = False
        loop_risk = "LOW"
        reason = "no_cycle"
        cycle_length = 0
        
        # 1. Check for EXACT signature repeats (Strict Cycle)
        match_index = -1
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("exact_state_signature") == current_exact_sig:
                match_index = i
                break
                
        if match_index != -1:
            cycle_length = len(history) - match_index
            if cycle_length == 1:
                # Immediate reversal or identity (A->A)
                # If vectors are inverse, it's a loop. If identity, it's stagnation.
                # We assume caller checks inverse vector separately, but here we flag "Return to state".
                loop_risk = "HIGH"
                reason = "Direct return to exact prior state (Stagnation/Loop)"
                is_cycle = True
            else:
                loop_risk = "HIGH"
                reason = f"Complex cycle (len={cycle_length+1}) detected: Return to exact prior FG state"
                is_cycle = True
        
        else:
            # 2. Check for EQUIVALENCE signature repeats (Class Loop / Stagnation)
            eq_match_index = -1
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("fg_equivalence_state") == current_eq_sig:
                    eq_match_index = i
                    break
            
            if eq_match_index != -1:
                cycle_length = len(history) - eq_match_index
                if cycle_length == 1:
                    # Stagnation (Staying in same class)
                    # Use lower risk, let effectiveness score decide
                    loop_risk = "MED" # Warning
                    reason = "Stagnation: Remaining in same FG equivalence class"
                    is_cycle = False # Not a "Cycle" per se, but risk
                else:
                    # Loop in equivalence class (A->A...->A)
                    loop_risk = "MED"
                    reason = f"Equivalence Loop (len={cycle_length+1}): Returning to same FG class state"
                    is_cycle = True

        fg_effectiveness = "LOW" if loop_risk in ("MED", "HIGH") else "HIGH"

        return {
            "loop_risk": loop_risk,
            "is_cycle": is_cycle,
            "cycle_length": cycle_length,
            "loop_reason": reason,
            "fg_effectiveness": fg_effectiveness
        }
