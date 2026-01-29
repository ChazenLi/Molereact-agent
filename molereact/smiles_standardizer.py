# -*- coding: utf-8 -*-
"""
SMILES Standardizer for MoleReact
==================================

基于 RDKit 的分子 SMILES 标准化工具。
支持：
1. 规范化 (Canonicalization)
2. 语法校验 (Validation)
3. 中性化与去盐 (Neutralization & Salt Stripping)
"""

import sys
import argparse
import logging
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover

logger = logging.getLogger(__name__)

class Standardizer:
    def __init__(self):
        self.salt_remover = SaltRemover()

    def canonicalize(self, smiles: str) -> Optional[str]:
        """将 SMILES 转换为规范格式。如果无效则返回 None。"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.error(f"Canonicalize failed for {smiles}: {e}")
        return None

    def validate(self, smiles: str) -> Tuple[bool, str]:
        """校验 SMILES 是否合法，并返回错误说明。"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return True, "Success"
            else:
                return False, "Invalid SMILES structure (RDKit parsing failed)"
        except Exception as e:
            return False, str(e)

    def standardize_full(self, smiles: str, remove_salts: bool = True) -> Optional[str]:
        """
        全量标准化：规范化 + 去盐 + 中性化。
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            # 1. 去盐
            if remove_salts:
                mol = self.salt_remover.StripMol(mol)
            
            # 2. 中性化 (简单处理：移除电荷)
            for atom in mol.GetAtoms():
                atom.SetFormalCharge(0)
            
            # 3. 规范化输出
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.error(f"Standardize_full failed: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="MoleReact SMILES Standardizer CLI")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES string")
    parser.add_argument("--mode", type=str, choices=["canon", "full", "validate"], default="canon")
    
    args = parser.parse_args()
    standardizer = Standardizer()
    
    if args.mode == "canon":
        res = standardizer.canonicalize(args.smiles)
        print(res if res else "ERROR: Invalid SMILES")
    elif args.mode == "validate":
        ok, msg = standardizer.validate(args.smiles)
        print(f"Valid: {ok}, Message: {msg}")
    elif args.mode == "full":
        res = standardizer.standardize_full(args.smiles)
        print(res if res else "ERROR: Processing failed")

if __name__ == "__main__":
    main()
