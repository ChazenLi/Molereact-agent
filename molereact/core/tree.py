# -*- coding: utf-8 -*-
"""
Module: multistep.agent.core.tree
Role: Global State Management for Autonomous Retrosynthesis

Functionality:
    Provides a graph/tree data structure (`SynthesisTree`) to track the entire synthesis plan.
    Unlike basic lists, this tracks lineage (parent-child), status (SOLVED/OPEN/DEAD),
    and provides text serialization for the LLM to "see" the current state.
"""

import json
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import uuid

class NodeStatus(Enum):
    OPEN = "OPEN"         # Needs expansion
    SOLVED = "SOLVED"     # Commercially available or solved by children
    DEAD = "DEAD"         # No routes or too risky
    VISITED = "VISITED"   # Expanded, children added

@dataclass
class SynthesisNode:
    smiles: str
    id: str  # Hierarchical ID (e.g., '1', '1.1', '1.2.1')
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    status: NodeStatus = NodeStatus.OPEN
    metadata: Dict[str, Any] = field(default_factory=dict) # scores, risk, etc.
    
    # Track the reaction that created this node (if purely retrosynthetic)
    reaction_rule: str = "" 
    source_type: str = "" # 'model' or 'template'
    
    def __repr__(self):
        return f"Node({self.id}, {self.status.value}, d={self.depth}, {self.smiles[:15]}...)"

class SynthesisTree:
    """
    Manages the global state of the retrosynthesis project.
    """
    def __init__(self, target_smiles: str):
        self.root_id = "1" # User preferred '1' as root
        self.nodes: Dict[str, SynthesisNode] = {}
        
        # Initialize root
        root_node = SynthesisNode(
            smiles=self._canonicalize(target_smiles),
            id=self.root_id,
            depth=0,
            status=NodeStatus.OPEN
        )
        self.nodes[self.root_id] = root_node
        self.cursor_id = self.root_id 

    def _canonicalize(self, smiles: str) -> str:
        """Internal helper to ensure SMILES are consistent."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return smiles

    def add_child(self, parent_id: str, smiles: str, metadata: Dict = None, reaction_rule: str = "", source_type: str = "") -> str:
        """Add a new precursor node with hierarchical ID and loop detection."""
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent {parent_id} not found")
            
        # Standardize SMILES for robust loop detection
        smiles = self._canonicalize(smiles)
        
        # Determine unique hierarchical ID: parent_id.child_index
        child_index = len(parent.children_ids) + 1
        new_id = f"{parent_id}.{child_index}"
            
        # --- Loop Detection (Ancestry Check) ---
        ancestors = self.get_lineage(parent_id)
        if smiles in [self.nodes[aid].smiles for aid in ancestors]:
            if metadata is None: metadata = {}
            metadata["is_loop"] = True
            metadata["loop_path"] = [self.nodes[aid].smiles for aid in ancestors] + [smiles]
            
        new_node = SynthesisNode(
            smiles=smiles,
            id=new_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            status=NodeStatus.OPEN,
            metadata=metadata or {},
            reaction_rule=reaction_rule,
            source_type=source_type
        )
        
        self.nodes[new_id] = new_node
        parent.children_ids.append(new_id)
        return new_id

    def get_lineage(self, node_id: str) -> List[str]:
        """Get the ID path from Root to the specified node."""
        path = []
        curr = node_id
        while curr:
            path.append(curr)
            curr = self.nodes[curr].parent_id
        return path[::-1]

    def get_node(self, node_id: str) -> Optional[SynthesisNode]:
        return self.nodes.get(node_id)

    def set_status(self, node_id: str, status: NodeStatus):
        """Set status and propagate if necessary (Recursive)."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        if node.status == status:
            return

        print(f"🔄 [Tree] Node {node_id} status changed: {node.status.value} -> {status.value}")
        node.status = status
        
        # --- Propagation Logic ---
        if status == NodeStatus.DEAD:
            # If any child of a parent is DEAD, the parent is DEAD (assuming mandatory precursors)
            if node.parent_id:
                self.set_status(node.parent_id, NodeStatus.DEAD)
        
        elif status == NodeStatus.SOLVED:
            # If a leaf is SOLVED (stock), check if its parent can now be SOLVED
            if node.parent_id:
                parent = self.nodes[node.parent_id]
                # If parent was VISITED and now ALL children are SOLVED, parent is SOLVED
                if parent.status == NodeStatus.VISITED:
                    if all(self.nodes[cid].status == NodeStatus.SOLVED for cid in parent.children_ids):
                        self.set_status(parent.id, NodeStatus.SOLVED)

    def get_open_nodes(self) -> List[SynthesisNode]:
        """Return all leaf nodes that are OPEN."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.OPEN]

    def serialize_state(self, focused: bool = True) -> str:
        """
        Convert the tree state into a text summary for the LLM.
        Aggressively collapses branches to save tokens.
        """
        summary = ["### Current Synthesis Status"]
        
        # 1. Global Metrics
        open_nodes = self.get_open_nodes()
        summary.append(f"- Unsolved Intermediate Molecules: {len(open_nodes)}")
        
        if self.cursor_id not in self.nodes:
            self.cursor_id = self.root_id
            
        summary.append(f"- Current Focus: Node {self.cursor_id} ({self.nodes[self.cursor_id].smiles})")
        
        # 2. Tree Structure
        summary.append("\n### Tree Structure (Simplified):")
        
        # Calculate active path (Root -> Cursor)
        active_path = set(self.get_lineage(self.cursor_id))

        def _has_open_descendant(nid):
            node = self.nodes[nid]
            if node.status == NodeStatus.OPEN:
                return True
            for cid in node.children_ids:
                if _has_open_descendant(cid):
                    return True
            return False

        def _render(nid, indent=0):
            node = self.nodes[nid]
            status_icon = {
                NodeStatus.OPEN: "⭕(Unsolved)",
                NodeStatus.SOLVED: "✅(Stock)",
                NodeStatus.DEAD: "❌(Dead)",
                NodeStatus.VISITED: "🔽(Expanded)"
            }.get(node.status, "?")
            
            is_focus = (nid == self.cursor_id)
            cursor_mark = "👈 CURRENT FOCUS" if is_focus else ""
            prefix = "  " * indent
            
            meta_str = ""
            if node.metadata.get("is_loop"):
                meta_str = " ⚠️ [LOOP DETECTED]"
            
            # --- AGGRESSIVE COLLAPSING LOGIC ---
            if focused:
                # If NOT on active path AND NOT an OPEN leaf AND doesn't lead to an OPEN leaf
                if nid not in active_path and node.status != NodeStatus.OPEN:
                    if not _has_open_descendant(nid):
                        # It's a completed or dead side branch. Summarize it.
                        return [f"{prefix}- [{nid}] {status_icon} (Side-branch completed/dead)"]

            line = f"{prefix}- [{nid}] {status_icon} {node.smiles}{meta_str} {cursor_mark}"
            lines = [line]

            if node.children_ids:
                should_recurse = True
                if focused and nid not in active_path and not _has_open_descendant(nid):
                    should_recurse = False
                
                if should_recurse:
                    for child_id in node.children_ids:
                        lines.extend(_render(child_id, indent + 1))
            return lines

        summary.extend(_render(self.root_id))
        return "\n".join(summary)

    def to_json(self) -> str:
        """Full dump for logging/resuming."""
        # Helper to convert Enum to str
        data = {k: {**v.__dict__, 'status': v.status.value} for k, v in self.nodes.items()}
        return json.dumps(data, indent=2)

if __name__ == "__main__":
    # Simple Test
    tree = SynthesisTree("C1=CC=CC=C1C(=O)O") # Benzoic acid
    print("Initialized Tree.")
    print(tree.serialize_state())
    
    print("\nSimulating Expansion...")
    tree.set_status("ROOT", NodeStatus.VISITED)
    c1 = tree.add_child("ROOT", "C1=CC=CC=C1Br") # Bromobenzene
    c2 = tree.add_child("ROOT", "CO") # CO2 equivalent
    
    tree.set_status(c2, NodeStatus.SOLVED) # Assume stock
    tree.cursor_id = c1 # Move focus
    
    print(tree.serialize_state())
