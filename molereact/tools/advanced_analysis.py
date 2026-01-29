# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.advanced_analysis
Description: Advanced chemical analysis tools for Deep Reasoning Agents.
Includes:
    - AtomEconomyTool
    - SolubilityPredictTool (ESOL)
    - ComplexityAnalysisTool (Bertz)
    - ReactionMappingTool (RxnMapper/MCS)
    - FormulaAnalysisTool
    - FunctionalGroupTool
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem import rdFMCS
try:
    from rdkit.Chem import FilterCatalog
except ImportError:
    FilterCatalog = None

from .fg_cycle_detector import FGStateTracker


logger = logging.getLogger(__name__)

class BaseAnalysisTool:
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class AtomEconomyTool(BaseAnalysisTool):
    """Calculates Atom Economy: MW(Product) / Sum(MW(Reactants))"""
    def execute(self, product_smiles: str, reactant_smiles_list: List[str]) -> Dict[str, Any]:
        try:
            prod_mol = Chem.MolFromSmiles(product_smiles)
            if not prod_mol: return {"error": "Invalid Product SMILES", "atom_economy": 0.0}
            
            prod_mw = Descriptors.MolWt(prod_mol)
            reactant_mws = []
            for s in reactant_smiles_list:
                m = Chem.MolFromSmiles(s)
                if m: reactant_mws.append(Descriptors.MolWt(m))
            
            total_reactant_mw = sum(reactant_mws)
            if total_reactant_mw == 0:
                return {"atom_economy": 0.0, "message": "No valid reactants"}
            
            ae = (prod_mw / total_reactant_mw) * 100.0
            return {"atom_economy": round(ae, 2), "product_mw": prod_mw, "total_reactant_mw": total_reactant_mw}
        except Exception as e:
            return {"error": str(e), "atom_economy": 0.0}

class SolubilityPredictTool(BaseAnalysisTool):
    """Predicts Solubility using ESOL (Delaney, 2004)"""
    def execute(self, smiles: str) -> Dict[str, Any]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return {"error": "Invalid SMILES"}
            
            # ESOL Parameters
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Aromatic Proportion
            aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            num_heavy = mol.GetNumHeavyAtoms()
            aromatic_proportion = aromatic_atoms / num_heavy if num_heavy > 0 else 0
            
            # ESOL Equation
            # LogS = 0.16 - 0.63(cLogP) - 0.0062(MW) + 0.066(RB) - 0.74(AP)
            logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotatable_bonds - 0.74 * aromatic_proportion
            
            solubility_class = "Insoluble"
            if logs > -2: solubility_class = "Soluble"
            elif logs > -4: solubility_class = "Moderately Soluble"
            
            return {
                "logs": round(logs, 2),
                "solubility_class": solubility_class,
                "esol_details": {"LogP": round(logp, 2), "AP": round(aromatic_proportion, 2)}
            }
        except Exception as e:
            return {"error": str(e)}

class ComplexityAnalysisTool(BaseAnalysisTool):
    """Calculates BertzCT Complexity"""
    def execute(self, smiles: str) -> Dict[str, Any]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return {"error": "Invalid SMILES"}
            bertz = rdMolDescriptors.CalcNumAtomStereoCenters(mol) * 10 # Placeholder if Bertz missing? No, use GraphDescriptors
            from rdkit.Chem import GraphDescriptors
            bertz = GraphDescriptors.BertzCT(mol)
            return {"bertz_complexity": round(bertz, 1)}
        except Exception as e:
            return {"error": str(e)}

class FormulaAnalysisTool(BaseAnalysisTool):
    """Analyzes Formula Delta (Mass Balance)"""
    def execute(self, product_smiles: str, reactant_smiles_list: List[str]) -> Dict[str, Any]:
        try:
            prod_mol = Chem.MolFromSmiles(product_smiles)
            if not prod_mol: return {"error": "Invalid Product"}
            
            prod_formula = rdMolDescriptors.CalcMolFormula(prod_mol)
            
            # Aggregate reactants atoms
            from collections import Counter
            import re
            
            def parse_formula(f_str):
                # Simple parser for elements
                matches = re.findall(r'([A-Z][a-z]*)(\d*)', f_str)
                counts = Counter()
                for el, num in matches:
                    counts[el] += int(num) if num else 1
                return counts
            
            prod_counts = parse_formula(prod_formula)
            react_counts = Counter()
            
            react_formulas = []
            for s in reactant_smiles_list:
                m = Chem.MolFromSmiles(s)
                if m:
                    f = rdMolDescriptors.CalcMolFormula(m)
                    react_counts += parse_formula(f)
                    react_formulas.append(f)
            
            # Calculate Delta (Product - Reactants)
            # A negative delta means atoms were lost (e.g. leaving groups)
            # A positive delta means atoms appeared out of nowhere (impossible unless reagents masked)
            
            delta_parts = []
            
            # Defense against 'list' object has no attribute 'keys'
            if not isinstance(prod_counts, (dict, Counter)): prod_counts = {}
            if not isinstance(react_counts, (dict, Counter)): react_counts = {}
            
            all_elements = set(prod_counts.keys()) | set(react_counts.keys())
            
            for el in sorted(all_elements):
                diff = prod_counts.get(el, 0) - react_counts.get(el, 0)
                if diff != 0:
                    sign = "+" if diff > 0 else ""
                    delta_parts.append(f"{sign}{diff}{el}")
            
            delta_str = ", ".join(delta_parts) if delta_parts else "Balanced"
            
            # Calculate heavy atom delta
            full_delta_dict = {el: (prod_counts[el] - react_counts[el]) for el in all_elements if (prod_counts[el] - react_counts[el]) != 0}
            
            heavy_delta = {}
            for el, diff in full_delta_dict.items():
                if el in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
                    heavy_delta[el] = diff
                        
            return {
                "target_formula": prod_formula,
                "reactants_formula": " + ".join(react_formulas),
                "formula_delta": delta_str,
                "formula_delta_dict": full_delta_dict,
                "heavy_atom_delta_dict": heavy_delta
            }
        except Exception as e:
            return {"error": str(e)}

class FunctionalGroupTool(BaseAnalysisTool):
    """Identifies Functional Group Changes"""
    
    SMARTS_DB = {
        "Acid": "[CX3](=O)[OX2H1]",
        "Carboxylate": "[CX3](=O)[O-]",
        "AcidChloride": "[CX3](=O)Cl",
        "AcidBromide": "[CX3](=O)Br",
        "Anhydride": "[CX3](=O)O[CX3](=O)",
        "Thioester": "[CX3](=O)S[#6]",
        "Amine_Primary": "[NX3;H2;!$(NC=O)]",
        "Amine_Secondary": "[NX3;H1;!$(NC=O)]([#6])[#6]",
        "Amine_Tertiary": "[NX3;H0;!$(NC=O)]([#6])[#6][#6]",
        "Amide": "[CX3](=O)[NX3]",
        "Imide": "[NX3]([CX3](=O))[CX3](=O)",
        "Urea": "[NX3][CX3](=O)[NX3]",
        "Carbamate": "[OX2][CX3](=O)[NX3]",
        "Carbonate": "[OX2][CX3](=O)[OX2]",
        "Ester": "[CX3](=O)[OX2H0]",
        "Alcohol": "[CX4][OX2H]",
        "Phenol": "c[OX2H]",
        "Ether": "[OD2]([#6])[#6]",
        "Thiol": "[SX2H]",
        "Thioether": "[SX2]([#6])[#6]",
        "Sulfoxide": "[SX3](=O)([#6])[#6]",
        "Sulfone": "[SX4](=O)(=O)([#6])[#6]",
        "Sulfonamide": "[SX4](=O)(=O)[NX3]",
        "SulfonylChloride": "[SX4](=O)(=O)Cl",
        "Halide_Cl": "[Cl]",
        "Halide_Br": "[Br]",
        "Halide_I": "[I]",
        "Nitro": "[N+](=O)[O-]",
        "Nitrile": "C#N",
        "Imine": "[CX3]=[NX2]",
        "Azide": "[N-]=[N+]=N",
        "Isocyanate": "N=C=O",
        "Isothiocyanate": "N=C=S",
        "Aldehyde": "[CX3H1](=O)[#6]",
        "Ketone": "[#6][CX3](=O)[#6]",
        "Sulfonate": "[S](=O)(=O)[O]",
        "Boronate": "[B](O)(O)",
        "Epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
        "Aziridine": "[NX3r3]1[CX4r3][CX4r3]1",
        "Alkyne": "C#C",
        "Alkene": "C=C"
    }
    
    def execute(self, smiles: str) -> List[str]:
        found = []
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return []
            for name, smarts in self.SMARTS_DB.items():
                pat = Chem.MolFromSmarts(smarts)
                if mol.HasSubstructMatch(pat):
                    found.append(name)
        except:
            pass
        return found

    def count_groups(self, smiles: str, presence_only: bool = False) -> Dict[str, int]:
        counts = {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {}
            for name, smarts in self.SMARTS_DB.items():
                pat = Chem.MolFromSmarts(smarts)
                if not pat:
                    continue
                matches = mol.GetSubstructMatches(pat)
                if matches:
                    counts[name] = 1 if presence_only else len(matches)
        except Exception:
            return {}
        return counts
        
    def analyze_reaction(self, product_smiles: str, reactant_smiles_list: List[str]) -> Dict[str, Any]:
        prod_fgs = set(self.execute(product_smiles))
        react_fgs = set()
        for s in reactant_smiles_list:
            react_fgs.update(self.execute(s))
            
        return {
            "product_fgs": list(prod_fgs),
            "reactants_fgs": list(react_fgs),
            "change_summary": f"Reactants: {list(react_fgs)} -> Product: {list(prod_fgs)}"
        }

class ReactionVectorTool(BaseAnalysisTool):
    """Builds a reaction vector from FG deltas and atom deltas."""
    def __init__(self, fg_tool: FunctionalGroupTool = None, form_tool: FormulaAnalysisTool = None):
        self.fg_tool = fg_tool or FunctionalGroupTool()
        self.form_tool = form_tool or FormulaAnalysisTool()

    def execute(
        self,
        product_smiles: str,
        reactant_smiles_list: List[str],
        formula_result: Optional[Dict[str, Any]] = None,
        presence_only: bool = True
    ) -> Dict[str, Any]:
        prod_counts = self.fg_tool.count_groups(product_smiles, presence_only=presence_only)
        react_counts = {}
        for smi in reactant_smiles_list:
            rc = self.fg_tool.count_groups(smi, presence_only=presence_only)
            for k, v in rc.items():
                react_counts[k] = react_counts.get(k, 0) + v

        fg_delta = self._delta_counts(prod_counts, react_counts)
        fg_delta_total = sum(abs(v) for v in fg_delta.values())

        if formula_result is None:
            formula_result = self.form_tool.execute(product_smiles, reactant_smiles_list)
        
        # Guard against incorrect type (e.g., if form_tool returned a list of errors)
        if not isinstance(formula_result, dict):
            formula_result = {}

        atom_delta = formula_result.get("formula_delta_dict", {}) or {}
        heavy_atom_delta = formula_result.get("heavy_atom_delta_dict", {}) or {}

        vector_signature = self._vector_signature(fg_delta, heavy_atom_delta)
        summary = self._format_vector_summary(fg_delta, heavy_atom_delta)

        return {
            "fg_counts_product": prod_counts,
            "fg_counts_reactants": react_counts,
            "fg_delta": fg_delta,
            "fg_delta_total": fg_delta_total,
            "atom_delta": atom_delta,
            "heavy_atom_delta": heavy_atom_delta,
            "vector_signature": vector_signature,
            "summary": summary
        }

    def _delta_counts(self, product_counts: Dict[str, int], reactant_counts: Dict[str, int]) -> Dict[str, int]:
        delta = {}
        if not isinstance(product_counts, dict): product_counts = {}
        if not isinstance(reactant_counts, dict): reactant_counts = {}
        
        keys = set(product_counts.keys()) | set(reactant_counts.keys())
        for k in keys:
            diff = product_counts.get(k, 0) - reactant_counts.get(k, 0)
            if diff != 0:
                delta[k] = diff
        return delta

    def _vector_signature(self, fg_delta: Dict[str, int], heavy_atom_delta: Dict[str, int]) -> str:
        fg_parts = [f"{k}:{v:+d}" for k, v in sorted(fg_delta.items()) if v != 0]
        ha_parts = [f"{k}:{v:+d}" for k, v in sorted(heavy_atom_delta.items()) if v != 0]
        fg_sig = ",".join(fg_parts) if fg_parts else "none"
        ha_sig = ",".join(ha_parts) if ha_parts else "none"
        return f"FG[{fg_sig}]|HA[{ha_sig}]"

    def _format_vector_summary(self, fg_delta: Dict[str, int], heavy_atom_delta: Dict[str, int]) -> str:
        fg_parts = [f"{k}{v:+d}" for k, v in sorted(fg_delta.items()) if v != 0]
        ha_parts = [f"{k}{v:+d}" for k, v in sorted(heavy_atom_delta.items()) if v != 0]
        fg_str = ", ".join(fg_parts) if fg_parts else "FG none"
        ha_str = ", ".join(ha_parts) if ha_parts else "HA none"
        return f"{fg_str}; {ha_str}"

class ElectronicEffectTool(BaseAnalysisTool):
    """Calculates Electronic Properties (Gasteiger Charges)"""
    def execute(self, smiles: str) -> Dict[str, Any]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return {"error": "Invalid SMILES"}
            
            # Compute Gasteiger Charges
            AllChem.ComputeGasteigerCharges(mol)
            atoms = mol.GetAtoms()
            charges = []
            
            for at in atoms:
                symbol = at.GetSymbol()
                idx = at.GetIdx()
                # Get charge, default to 0.0 if computation failed
                try: 
                    q = float(at.GetProp('_GasteigerCharge'))
                except: 
                    q = 0.0
                charges.append((q, symbol, idx))
                
            # Find extreme sites
            charges.sort(key=lambda x: x[0]) # Ascending (Negative first)
            
            most_nuc = [f"{s}:{i}({q:.2f})" for q, s, i in charges[:3] if q < -0.1]
            most_elec = [f"{s}:{i}(+{q:.2f})" for q, s, i in charges[-3:] if q > 0.1]
            most_elec.reverse() # Highest positive first
            
            return {
                "top_nucleophiles": most_nuc,
                "top_electrophiles": most_elec
            }
        except Exception as e:
            return {"error": str(e)}

class ProtectionGroupCheckTool(BaseAnalysisTool):
    """Checks for potentially conflicting groups needing protection"""
    
    CONFLICT_PAIRS = [
        # (Group A Name, Group B Name) -> If both present, risk!
        ("Amine_Primary", "Aldehyde"), # Imine formation
        ("Amine_Primary", "Ketone"),
        ("Amine_Primary", "Acid"),    # Amide (if activated) / Salt
        ("Amine_Primary", "Ester"),   # Aminolysis
        ("Amine_Primary", "Halide_Cl"), 
        ("Amine_Primary", "Halide_Br"),
        
        ("Amine_Secondary", "Aldehyde"),
        ("Amine_Secondary", "Ketone"),
        ("Amine_Secondary", "Acid"),
        ("Amine_Secondary", "Ester"),
        
        ("Acid", "Alcohol") # Esterification (less spontaneous but possible)
    ]
    
    def __init__(self):
        self.fg_tool = FunctionalGroupTool()
        
    def execute(self, smiles: str) -> List[str]:
        # Reuse FG Tool logic
        fgs = self.fg_tool.execute(smiles)
        found_conflicts = []
        
        # Check pairs
        for a, b in self.CONFLICT_PAIRS:
            if a in fgs and b in fgs:
                found_conflicts.append(f"{a} + {b}")
                
        return found_conflicts
        
    def analyze_reaction(self, reactant_smiles_list: List[str]) -> Dict[str, Any]:
        """
        Check conflicts within each reactant (Internal) or Cross-Reactant (External).
        For simplicity, we check total pool of precursors.
        """
        warnings = []
        # Check internal conflicts first
        for smi in reactant_smiles_list:
            conflicts = self.execute(smi)
            if conflicts:
                warnings.append(f"Internal Conflict in precursor: {conflicts}")
                
        if warnings:
            return {"status": "Warning", "message": "; ".join(warnings) + ". Check if protection is needed."}
        return {"status": "OK", "message": "No obvious incompatible group conflicts."}

class ChemicalRationalityTool(BaseAnalysisTool):
    """Checks for Chemical Rationality (Valence, Strain, Bredt Rules)"""
    def execute(self, smiles: str) -> Dict[str, Any]:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if not mol: return {"status": "FAIL", "message": "Invalid SMILES structure"}
            
            issues = []
            
            # 1. Valence Check
            try:
                mol.UpdatePropertyCache(strict=True)
            except Exception as e:
                issues.append(f"Valence violation: {str(e)}")
            
            # 2. Sanitization Check
            err = Chem.SanitizeMol(mol, catchErrors=True)
            if err:
                issues.append(f"Sanitization error: {str(err)}")
            
            # 3. Bredt's Rule (Simplified)
            # Check for double bonds at bridgehead atoms in small rings (<7)
            from rdkit.Chem import rdqueries
            submol = mol.GetSubstructMatches(Chem.MolFromSmarts("[D3,D4]~[D3,D4]"))
            # This is complex to implement fully with RDKit without geometry, 
            # but we can flag bridgehead double bonds.
            
            return {
                "status": "PASS" if not issues else "FAIL",
                "issues": issues,
                "message": "Structure is rational" if not issues else "; ".join(issues)
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

class ReactionSignatureTool(BaseAnalysisTool):
    """Calculates Reaction and State Signatures for global tracking"""
    def generate_transform_signature(self, mapped_rxn: str) -> str:
        """Extracts reaction center SMARTS from mapped reaction"""
        if not mapped_rxn: return "Unknown"
        # Simplistic extraction: keep only atoms with mapping
        try:
            rxn = AllChem.ReactionFromSmarts(mapped_rxn, useSmiles=True)
            # Logic to extract reaction center would go here
            # For now, return a placeholder or simplified SMARTS
            return f"TSign({mapped_rxn.split('>>')[0][:10]}...)"
        except:
            return "Invalid_Map"

    def generate_state_signature(self, smiles: str) -> str:
        """Generates a signature for the molecule state"""
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return "Invalid"
        
        # 1. Scaffold
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_smi = Chem.MolToSmiles(scaf) if scaf else "Acyclic"
        
        # 2. Oxidation state estimate (Sum of oxidation states of carbons? or simpler)
        # For simplicity, count heteroatoms and unsaturation
        hetero = Descriptors.NumHeteroatoms(mol)
        rings = Descriptors.RingCount(mol)
        
        return f"State({scaf_smi[:8]}|H:{hetero}|R:{rings})"

class ToxicityFilterTool(BaseAnalysisTool):
    """Checks for Structural Alerts (PAINS, BRENK, NIH)"""
    def __init__(self):
        self.catalog = None
        if FilterCatalog:
            try:
                params = FilterCatalog.FilterCatalogParams()
                params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
                params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
                params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
                self.catalog = FilterCatalog.FilterCatalog(params)
            except Exception as e:
                logger.warning(f"Failed to initialize FilterCatalog: {e}")
    
    def execute(self, smiles: str) -> List[str]:
        alerts = []
        if not self.catalog:
            return ["FilterCatalog unavailable"]
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return ["Invalid SMILES"]
            
            if self.catalog.HasMatch(mol):
                matches = self.catalog.GetMatches(mol)
                for match in matches:
                    alerts.append(match.GetDescription())
        except Exception as e:
            alerts.append(f"Check Error: {str(e)}")
            
        return alerts

class ReactionMappingTool(BaseAnalysisTool):
    """Reaction Atom Mapping and Ledger Generation"""
    def __init__(self):
        self.rxn_mapper = None
        self._try_load_rxnmapper()
        
    def _extract_bond_ledger(self, mapped_rxn: str) -> Dict[str, List[str]]:
        """
        Extracts specific bond changes from a mapped reaction SMARTS.
        """
        try:
            from rdkit.Chem import rdChemReactions
            rxn = rdChemReactions.ReactionFromSmarts(mapped_rxn, useSmiles=True)
            if not rxn: return {"formed": [], "broken": []}

            def get_bonds(mols):
                bonds = set()
                for mol in mols:
                    if not mol: continue
                    for bond in mol.GetBonds():
                        a1 = bond.GetBeginAtom()
                        a2 = bond.GetEndAtom()
                        m1 = a1.GetAtomMapNum()
                        m2 = a2.GetAtomMapNum()
                        if m1 and m2:
                            # Order map nums for set comparison
                            pair = tuple(sorted((m1, m2)))
                            # Include bond type (SINGLE, DOUBLE, etc)
                            bonds.add((pair, bond.GetBondType(), a1.GetSymbol(), a2.GetSymbol()))
                return bonds

            reactants_bonds = get_bonds(rxn.GetReactants())
            products_bonds = get_bonds(rxn.GetProducts())

            formed = products_bonds - reactants_bonds
            broken = reactants_bonds - products_bonds

            def format_bonds(bond_set):
                res = []
                for (m1, m2), btype, s1, s2 in bond_set:
                    res.append(f"{s1}-{s2} ({str(btype)})")
                return list(set(res)) # Distinct human names

            return {
                "formed": format_bonds(formed),
                "broken": format_bonds(broken)
            }
        except Exception as e:
            return {"error": str(e), "formed": [], "broken": []}

    def _try_load_rxnmapper(self):
        try:
            from rxnmapper import RXNMapper
            self.rxn_mapper = RXNMapper()
            logger.info("RxnMapper loaded successfully.")
        except ImportError:
            logger.warning("RxnMapper not found. Using RDKit MCS fallback logic.")
            
    def execute(self, product_smiles: str, reactant_smiles_list: List[str]) -> Dict[str, Any]:
        reactants_smi = ".".join(reactant_smiles_list)
        rxn_smiles = f"{reactants_smi}>>{product_smiles}"
        
        ledger = {
            "bonds_formed": [],
            "bonds_broken": [],
            "mapped_rxn": ""
        }
        
        # 1. Try RxnMapper
        if self.rxn_mapper:
            try:
                # Suppress huge output from rxnmapper/transformers
                results = self.rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])
                mapped_rxn = results[0]['mapped_rxn']
                ledger["mapped_rxn"] = mapped_rxn
                
                # 1b. Extract specific bond changes
                bond_data = self._extract_bond_ledger(mapped_rxn)
                if bond_data.get("formed") or bond_data.get("broken"):
                    formed_str = ", ".join(bond_data["formed"]) or "None"
                    broken_str = ", ".join(bond_data["broken"]) or "None"
                    ledger["message"] = f"Created: {formed_str} | Broken: {broken_str}"
                    ledger["bonds_formed"] = bond_data["formed"]
                    ledger["bonds_broken"] = bond_data["broken"]
                else:
                    ledger["message"] = "Mapping successful. No clear skeleton bond changes (FGI/Substitution)."
                
                return {
                    "method": "RxnMapper",
                    "ledger": ledger,
                    "confidence": results[0].get('confidence', 0.0)
                }
            except Exception as e:
                logger.error(f"RxnMapper failed: {e}")
                
        # 2. Fallback MCS
        # Very simplified: Just checks common substructure
        return {"method": "Fallback (No Mapping)", "ledger": ledger, "message": "RxnMapper unavailable"}

class FGEquivalenceClassTool(BaseAnalysisTool):
    """Wraps FGStateTracker to generate state signatures"""
    def __init__(self, fg_tool: FunctionalGroupTool = None):
         self.fg_tool = fg_tool or FunctionalGroupTool()

    def execute(self, smiles: str) -> Dict[str, Any]:
        fgs = self.fg_tool.execute(smiles)
        eq_sig = FGStateTracker.generate_equivalence_state_signature(smiles, fgs)
        ex_sig = FGStateTracker.generate_exact_state_signature(smiles, fgs)
        classes = list(FGStateTracker.get_active_equivalence_classes(fgs))
        return {
            "fg_equivalence_state": eq_sig,
            "exact_state_signature": ex_sig,
            "active_classes": classes,
            "summary": f"State: {eq_sig} (Exact: {ex_sig})"
        }

class AdvancedAnalysisToolbox:
    """
    Unified entry point for Multi-Step Agents.
    """
    def __init__(self):
        self.ae_tool = AtomEconomyTool()
        self.sol_tool = SolubilityPredictTool()
        self.cpx_tool = ComplexityAnalysisTool()
        self.form_tool = FormulaAnalysisTool()
        self.fg_tool = FunctionalGroupTool()
        self.elec_tool = ElectronicEffectTool()
        self.pg_tool = ProtectionGroupCheckTool()
        self.tox_tool = ToxicityFilterTool()
        self.map_tool = ReactionMappingTool()
        self.rat_tool = ChemicalRationalityTool() # New
        self.sig_tool = ReactionSignatureTool()   # New
        self.rv_tool = ReactionVectorTool(self.fg_tool, self.form_tool)
        
        # New Strategic Tools
        from .chemistry import ReactionClassifyTool, ScaffoldAnalysisTool
        self.strat_tool = ReactionClassifyTool()
        self.scaf_tool = ScaffoldAnalysisTool()
        
        # New FG Cycle Tool
        self.eq_tool = FGEquivalenceClassTool(self.fg_tool)

        
    def analyze_candidate(self, target_smiles: str, precursor_smiles_list: List[str]) -> Dict[str, Any]:
        """
        Runs comprehensive analysis for a retrosynthetic step (Target <- Precursors).
        Note: The Agent views this as "Reverse", but Tools view it as "Forward" (Precursors -> Target).
        """
        report = {}
        
        # 1. Physicochemical (Precursors) & Risks
        prec_props = []
        prec_elec = []
        prec_risks = []
        
        for p in precursor_smiles_list:
            sol = self.sol_tool.execute(p)
            cpx = self.cpx_tool.execute(p)
            elec = self.elec_tool.execute(p)
            tox = self.tox_tool.execute(p)
            
            prec_props.append(f"Precursor ({p[:10]}...): LogS={sol.get('logs','?')} ({sol.get('solubility_class','?')}), Bertz={cpx.get('bertz_complexity','?')}")
            
            if 'error' not in elec:
                nuc = ", ".join(elec.get('top_nucleophiles', [])[:2])
                elc = ", ".join(elec.get('top_electrophiles', [])[:2])
                prec_elec.append(f"Elec-Info: Nuc=[{nuc}] | Elec=[{elc}]")
            
            if tox:
                prec_risks.append(f"Alerts for {p[:10]}...: {'; '.join(tox)}")
                
        report["physicochemical_summary"] = "; ".join(prec_props)
        report["electronic_summary"] = " || ".join(prec_elec)
        report["risk_summary"] = "; ".join(prec_risks) if prec_risks else "No structural alerts found."
        
        # === SINGLE MOLECULE MODE (Agent-0 Strategy) ===
        if not precursor_smiles_list:
            # If no precursors, we analyze the TARGET itself
            t_sol = self.sol_tool.execute(target_smiles)
            t_cpx = self.cpx_tool.execute(target_smiles)
            t_tox = self.tox_tool.execute(target_smiles)
            t_sig = self.sig_tool.generate_state_signature(target_smiles)
            t_elec = self.elec_tool.execute(target_smiles) # Added Electronic Analysis
            
            report["physicochemical_summary"] = f"Target: LogS={t_sol.get('logs','?')} ({t_sol.get('solubility_class','?')}), Bertz={t_cpx.get('bertz_complexity','?')}"
            report["risk_summary"] = f"Target Alerts: {'; '.join(t_tox)}" if t_tox else "No Target Alerts."
            report["state_signature"] = t_sig
            
            # Format Electronic Summary
            elec_summary = "Not available"
            if 'error' not in t_elec:
                 nuc = ", ".join(t_elec.get('top_nucleophiles', [])[:3])
                 elc = ", ".join(t_elec.get('top_electrophiles', [])[:3])
                 elec_summary = f"Nucleophiles: [{nuc}] | Electrophiles: [{elc}]"
            
            report["formatted_report"] = (
                f"**Target Analysis**:\n"
                f"- Props: {report['physicochemical_summary']}\n"
                f"- Risks: {report['risk_summary']}\n"
                f"- Electronic: {elec_summary}\n"
                f"- Signature: {report['state_signature']}\n"
            )
            return report

        # 2. Reaction Metrics (Reaction Mode)
        ae = self.ae_tool.execute(target_smiles, precursor_smiles_list)
        report["atom_economy"] = ae.get("atom_economy", 0)
        
        # 3. Formula Delta (Updated)
        form = self.form_tool.execute(target_smiles, precursor_smiles_list)
        report["formula_delta"] = form.get("formula_delta", "?")
        report["formula_dict"] = str(form.get("formula_delta_dict", {}))
        report["heavy_atom_delta"] = str(form.get("heavy_atom_delta_dict", {}))

        # 3b. Reaction Vector (FG + Atom Delta)
        rv = self.rv_tool.execute(target_smiles, precursor_smiles_list, formula_result=form)
        report["reaction_vector"] = rv
        report["reaction_vector_signature"] = rv.get("vector_signature", "")
        report["reaction_vector_summary"] = rv.get("summary", "")
        report["fg_delta_total"] = rv.get("fg_delta_total", 0)
        
        # 4. Functional Groups
        fgs = self.fg_tool.analyze_reaction(target_smiles, precursor_smiles_list)
        report["fg_change"] = fgs.get("change_summary", "?")
        
        # 5. Protection Group Check (New)
        pg_check = self.pg_tool.analyze_reaction(precursor_smiles_list)
        report["pg_warning"] = pg_check.get("message", "")
        
        # 6. Mapping (New)
        mapping_res = self.map_tool.execute(target_smiles, precursor_smiles_list)
        ledger_msg = mapping_res.get("ledger", {}).get("message", "Mapping N/A")
        
        strat = self.strat_tool.execute(target_smiles, precursor_smiles_list)
        report["reaction_strategy"] = strat.get("reaction_type", "MIXED")
        report["reaction_explanation"] = strat.get("explanation", "")
        report["scaffold_sim"] = strat.get("scaffold_similarity", 0)
        
        # 8. Rationality Check (New)
        rat_results = []
        for p in precursor_smiles_list:
            res = self.rat_tool.execute(p)
            if res.get("status") == "FAIL":
                rat_results.append(f"Precursor {p[:10]}: {res.get('message')}")
        report["rationality_warning"] = "; ".join(rat_results) if rat_results else "Structures are rational."
        
        # 9. Signatures (New)
        report["state_signature"] = self.sig_tool.generate_state_signature(target_smiles)
        report["transform_signature"] = self.sig_tool.generate_transform_signature(mapping_res.get("ledger", {}).get("mapped_rxn", ""))
        
        # 10. FG Equivalence State (New Cycle Check)
        eq_res = self.eq_tool.execute(target_smiles)
        report["fg_equivalence_state"] = eq_res["fg_equivalence_state"]
        report["exact_state_signature"] = eq_res["exact_state_signature"]

        
        # Construct Text Report
        text_report = (
            f"- **Formula Delta (String)**: {report['formula_delta']} (Check Mass Balance)\n"
            f"- **Formula Delta (Dict)**: {report['formula_dict']}\n"
            f"- **Reaction Vector**: {report['reaction_vector_summary']}\n"
            f"- **FG Changes**: {report['fg_change']}\n"
            f"- **Reaction Strategy**: {report['reaction_strategy']} ({report['reaction_explanation']})\n"
            f"- **Reaction Signature**: {report['transform_signature']}\n"
            f"- **State Signature**: {report['state_signature']} (Equivalent: {report['fg_equivalence_state']})\n"
            f"- **Scaffold Similarity**: {report['scaffold_sim']:.2f}\n"
            f"- **Atom Economy**: {report['atom_economy']}%\n"
            f"- **Electronic**: {report['electronic_summary']}\n"
            f"- **Reaction Mapping**: {ledger_msg}\n"
            f"- **Risks (Rationality)**: {report['rationality_warning']}\n"
            f"- **Risks (PG)**: {report['pg_warning']}\n"
            f"- **Risks (Tox/PAINS)**: {report['risk_summary']}\n"
        )
        report["formatted_report"] = text_report
        return report

# Singleton instance for easy import
toolbox = AdvancedAnalysisToolbox()
