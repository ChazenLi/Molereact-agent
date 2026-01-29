# -*- coding: utf-8 -*-
"""
Module: multistep.agent.knowledge_base
Description: Long-term memory knowledge base for high-quality synthesis routes.

Features:
    - SQLite-based persistent storage
    - Morgan Fingerprint similarity search  
    - Route quality scoring and retrieval
    - Integration with LLM context
    
Configuration:
    - SIMILARITY_THRESHOLD: 0.7 (Tanimoto)
    - MAX_ROUTES_IN_CONTEXT: 3
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# RDKit imports for fingerprint similarity
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    
logger = logging.getLogger(__name__)

# Configuration
SIMILARITY_THRESHOLD = 0.7  # Tanimoto similarity threshold
MAX_ROUTES_IN_CONTEXT = 3   # Maximum routes to include in LLM context


class SynthesisKnowledgeBase:
    """
    Persistent storage for high-quality synthesis routes.
    
    Stores successful retrosynthesis routes with:
    - Target SMILES and Morgan Fingerprint
    - Complete route data (precursors, reaction types, etc.)
    - LLM quality scores
    - Timestamps and session metadata
    
    Supports similarity-based retrieval for new targets.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize Knowledge Base.
        
        Args:
            db_path: Path to SQLite database. Defaults to output/agent_runs/knowledge_base.db
        """
        if db_path is None:
            # Default path
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _multistep_dir = os.path.dirname(_script_dir)
            db_path = os.path.join(_multistep_dir, "output", "agent_runs", "knowledge_base.db")
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._init_database()
        logger.info(f"SynthesisKnowledgeBase initialized: {db_path}")
        
    def _init_database(self):
        """Create database tables if not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Routes table - stores complete synthesis routes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_smiles TEXT NOT NULL,
                target_fingerprint BLOB,
                route_data TEXT NOT NULL,
                reaction_types TEXT,
                total_steps INTEGER,
                llm_score REAL,
                quality_notes TEXT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(target_smiles, session_id)
            )
        """)
        
        # Index for faster SMILES lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_target_smiles 
            ON routes(target_smiles)
        """)
        
        # Route steps table - stores individual steps for detailed retrieval
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS route_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER NOT NULL,
                step_number INTEGER NOT NULL,
                target_smiles TEXT NOT NULL,
                precursors TEXT NOT NULL,
                reaction_type TEXT,
                llm_analysis TEXT,
                analysis_metadata TEXT,
                FOREIGN KEY(route_id) REFERENCES routes(id)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _get_morgan_fingerprint(self, smiles: str) -> Optional[bytes]:
        """Generate Morgan fingerprint for a SMILES string."""
        if not RDKIT_AVAILABLE:
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            return fp.ToBitString().encode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to generate fingerprint for {smiles[:20]}...: {e}")
            return None
    
    def _calculate_tanimoto(self, fp1_bytes: bytes, fp2_bytes: bytes) -> float:
        """Calculate Tanimoto similarity between two fingerprints."""
        if not RDKIT_AVAILABLE or fp1_bytes is None or fp2_bytes is None:
            return 0.0
            
        try:
            from rdkit import DataStructs
            from rdkit.DataStructs import ExplicitBitVect
            
            fp1 = ExplicitBitVect(len(fp1_bytes.decode('utf-8')))
            fp2 = ExplicitBitVect(len(fp2_bytes.decode('utf-8')))
            
            for i, bit in enumerate(fp1_bytes.decode('utf-8')):
                if bit == '1':
                    fp1.SetBit(i)
            for i, bit in enumerate(fp2_bytes.decode('utf-8')):
                if bit == '1':
                    fp2.SetBit(i)
                    
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            logger.warning(f"Tanimoto calculation failed: {e}")
            return 0.0
    
    def store_route(
        self, 
        target_smiles: str, 
        route_data: Dict,
        llm_score: float = 0.0,
        quality_notes: str = "",
        session_id: str = ""
    ) -> int:
        """
        Store a successful synthesis route.
        
        Args:
            target_smiles: Target molecule SMILES
            route_data: Complete route dictionary (from cumulative_route)
            llm_score: Overall LLM quality score (0-10)
            quality_notes: Human or LLM notes on route quality
            session_id: Session identifier for tracking
            
        Returns:
            int: Route ID in database, or -1 if failed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate fingerprint
            fp = self._get_morgan_fingerprint(target_smiles)
            
            # Extract summary info
            stages = route_data.get("stages", [])
            total_steps = len(stages)
            reaction_types = []
            for s in stages:
                rt = s.get("reaction_type", "Unknown")
                if rt not in reaction_types:
                    reaction_types.append(rt)
            
            # Insert route
            cursor.execute("""
                INSERT OR REPLACE INTO routes 
                (target_smiles, target_fingerprint, route_data, reaction_types, 
                 total_steps, llm_score, quality_notes, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                target_smiles,
                fp,
                json.dumps(route_data, ensure_ascii=False),
                json.dumps(reaction_types, ensure_ascii=False),
                total_steps,
                llm_score,
                quality_notes,
                session_id
            ))
            
            route_id = cursor.lastrowid
            
            # Insert individual steps
            for i, stage in enumerate(stages, 1):
                cursor.execute("""
                    INSERT INTO route_steps
                    (route_id, step_number, target_smiles, precursors, 
                     reaction_type, llm_analysis, analysis_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    route_id,
                    i,
                    stage.get("target", ""),
                    json.dumps(stage.get("precursors", []), ensure_ascii=False),
                    stage.get("reaction_type", "Unknown"),
                    stage.get("reason", ""),
                    json.dumps(stage.get("analysis_metadata", {}), ensure_ascii=False)
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored route {route_id}: {target_smiles[:30]}... ({total_steps} steps, score={llm_score})")
            return route_id
            
        except Exception as e:
            logger.error(f"Failed to store route: {e}")
            return -1
    
    def retrieve_similar_routes(
        self, 
        target_smiles: str, 
        top_k: int = MAX_ROUTES_IN_CONTEXT,
        min_similarity: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Retrieve similar historical routes for a target molecule.
        
        Args:
            target_smiles: Target molecule to find similar routes for
            top_k: Maximum number of routes to return
            min_similarity: Minimum Tanimoto similarity threshold
            
        Returns:
            List of route dictionaries with similarity scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all routes with fingerprints
            cursor.execute("""
                SELECT id, target_smiles, target_fingerprint, route_data, 
                       llm_score, total_steps, reaction_types, created_at
                FROM routes
                ORDER BY llm_score DESC, created_at DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return []
            
            # Calculate target fingerprint
            target_fp = self._get_morgan_fingerprint(target_smiles)
            
            # Calculate similarities and filter
            similar_routes = []
            for row in rows:
                route_id, route_target, route_fp, route_data_str, score, steps, rtypes, created = row
                
                # Calculate similarity
                if target_fp and route_fp:
                    similarity = self._calculate_tanimoto(target_fp, route_fp)
                else:
                    # Fallback: exact match check
                    similarity = 1.0 if target_smiles == route_target else 0.0
                
                if similarity >= min_similarity:
                    similar_routes.append({
                        "id": route_id,
                        "target_smiles": route_target,
                        "similarity": similarity,
                        "route_data": json.loads(route_data_str),
                        "llm_score": score,
                        "total_steps": steps,
                        "reaction_types": json.loads(rtypes) if rtypes else [],
                        "created_at": created
                    })
            
            # Sort by similarity, then by score
            similar_routes.sort(key=lambda x: (x["similarity"], x["llm_score"]), reverse=True)
            
            return similar_routes[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar routes: {e}")
            return []
    
    def format_for_context(self, routes: List[Dict]) -> str:
        """
        Format retrieved routes for LLM context injection.
        
        Args:
            routes: List of route dictionaries from retrieve_similar_routes
            
        Returns:
            Formatted string for LLM prompt context
        """
        if not routes:
            return ""
        
        context = f"\n### 📚 Historical Success Routes (Knowledge Base, Top-{len(routes)}):\n"
        context += "> The following are successful synthesis routes for structurally similar molecules, provided for reference.\n\n"
        
        for i, route in enumerate(routes, 1):
            sim = route["similarity"]
            score = route["llm_score"]
            steps = route["total_steps"]
            rtypes = route["reaction_types"]
            target = route["target_smiles"]
            
            context += f"**[Reference Case {i}]** (Similarity: {sim:.2f}, Score: {score:.1f}, {steps} steps)\n"
            context += f"- Target: `{target[:50]}...`\n"
            context += f"- Reaction Types: {', '.join(rtypes[:3])}\n"
            
            # Add key steps summary
            stages = route["route_data"].get("stages", [])
            if stages:
                context += "- Key Steps:\n"
                for j, stage in enumerate(stages[:3], 1):  # Max 3 steps shown
                    precs = stage.get("precursors", [])
                    rtype = stage.get("reaction_type", "N/A")
                    context += f"  {j}. [{rtype}] => {', '.join([p[:20]+'...' for p in precs[:2]])}\n"
            
            context += "\n"
        
        return context
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM routes")
            total_routes = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(llm_score) FROM routes WHERE llm_score > 0")
            avg_score = cursor.fetchone()[0] or 0.0
            
            cursor.execute("SELECT AVG(total_steps) FROM routes")
            avg_steps = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            return {
                "total_routes": total_routes,
                "avg_score": round(avg_score, 2),
                "avg_steps": round(avg_steps, 1),
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}


# Singleton instance for easy import
_default_kb = None

def get_knowledge_base(db_path: str = None) -> SynthesisKnowledgeBase:
    """Get or create the default knowledge base instance."""
    global _default_kb
    if _default_kb is None:
        _default_kb = SynthesisKnowledgeBase(db_path)
    return _default_kb
