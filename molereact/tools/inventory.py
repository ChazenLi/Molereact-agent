# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.inventory
Called By: agent.py, agent_run.py
Role: Supply Chain & Stock Management

Functionality:
    Manages checking of chemical databases to determine if a molecule is commercially available ("Buyable").
    
Key Classes:
    - StockCheckTool: Checks internal DB or external APIs (Mock/Real).
    - SupplyChainTool: (Mock) Analyzes lead times and suppliers.

Relations:
    - Crucial for terminating the retrosynthesis tree recursion (Leaves must be in stock).
"""
"""
Inventory Tools
===============
"""
from typing import Dict, Any, List
from .base import BaseTool

class StockCheckTool(BaseTool):
    """Encapsulates Molecule Stock Check."""
    def __init__(self, session=None):
        self._session = session
        
    @property
    def name(self) -> str:
        return "StockCheck"
    
    @property
    def description(self) -> str:
        return "Checks if molecules are commercially available in the database."
    
    def execute(self, smiles_list: List[str], **kwargs) -> Dict[str, Any]:
        if self._session is None:
            # Lazy load
            from multistep.aizynthsession import AizynthSession
            import os
            # Heuristic to find config... assuming standard location relative to this file
            # agent/tools/inventory.py -> agent/tools -> agent -> multistep
            # We need config.yml in multistep or root?
            # Config loader usually handles default.
            # Resolve config path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # multistep/
            config_path = os.path.join(base_dir, "config.yml")
            if not os.path.exists(config_path):
                 # Fallback to hardcoded commonly used path if relative fails
                 config_path = r"e:\Python\p1\MoleReact\multistep\config.yml"
            
            self._session = AizynthSession(config_path)
        
        results = []
        in_stock_count = 0
        
        for smi in smiles_list:
            is_stock = self._session.is_in_stock(smi)
            results.append({
                "smiles": smi,
                "in_stock": is_stock,
                "database": "ZINC" if is_stock else None
            })
            if is_stock:
                in_stock_count += 1
                
        return {
            "results": results,
            "stock_rate": in_stock_count / len(smiles_list) if smiles_list else 0,
            "in_stock_count": in_stock_count,
            "total_count": len(smiles_list)
        }

class SupplyChainTool(BaseTool):
    """Encapsulates Supply Chain Query."""
    @property
    def name(self) -> str:
        return "SupplyChainQuery"
    
    @property
    def description(self) -> str:
        return "Queries vendor database (simulated) for price and lead time."

    def execute(self, smiles_list: List[str], preferred_region: str = "china", **kwargs) -> Dict[str, Any]:
        # Using the logic from original skills_production.py
        import random
        
        VENDOR_DATABASE = {
            "Sigma-Aldrich": {"region": "global", "lead_time_factor": 1.0},
            "TCI": {"region": "asia", "lead_time_factor": 1.2},
            "Alfa Aesar": {"region": "global", "lead_time_factor": 1.1},
            "国药": {"region": "china", "lead_time_factor": 0.5},
            "百灵威": {"region": "china", "lead_time_factor": 0.6},
        }
        
        materials = []
        critical_path_items = []
        
        for smi in smiles_list:
            vendors = []
            base_price = random.randint(1000, 5000)
            base_lead_time = random.randint(3, 14)
            
            for vendor_name, info in VENDOR_DATABASE.items():
                price_adj = 1.0 if info["region"] == preferred_region else 1.3
                vendors.append({
                    "name": vendor_name,
                    "price_per_kg": int(base_price * price_adj),
                    "lead_time_days": int(base_lead_time * info["lead_time_factor"]),
                    "region": info["region"],
                })
            
            vendors.sort(key=lambda x: x["price_per_kg"])
            availability = random.choice(["IN_STOCK", "IN_STOCK", "MADE_TO_ORDER"])
            
            shortest_lead = min(v["lead_time_days"] for v in vendors) if vendors else 0
            
            material_info = {
                "smiles": smi,
                "name": f"Chemical_{smi[:8]}",
                "vendors": vendors[:3],
                "availability": availability,
                "best_price": vendors[0]["price_per_kg"] if vendors else None,
                "shortest_lead_time": shortest_lead
            }
            materials.append(material_info)
            
            if availability == "MADE_TO_ORDER" or shortest_lead > 21:
                critical_path_items.append(f"{smi[:20]}... 交货期较长")
        
        return {
            "materials": materials,
            "critical_path_items": critical_path_items,
            "total_materials": len(smiles_list),
            "preferred_region": preferred_region
        }
