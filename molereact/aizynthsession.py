# -*- coding: utf-8 -*-
"""
AizynthSession - AiZynthFinder Python API 封装

替代 1.py 的 subprocess 调用，提供进程内 Python 接口。
一次加载，多次调用；支持 stock 查询、单步扩展、深度搜索。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """单步扩展结果"""
    reactants: Tuple[str, ...]
    template_smarts: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """深度搜索结果"""
    is_solved: bool
    num_routes: int
    top_score: float
    search_time: float
    trees: List[Dict]
    statistics: Dict[str, Any]


class AizynthSession:
    """
    AiZynthFinder 进程内会话。
    
    功能:
    - is_in_stock(smiles) -> bool: 检查分子是否可买
    - expand_once(smiles) -> List[ExpansionResult]: 单步模板扩展
    - search(smiles, ...) -> SearchResult: 完整深度搜索
    
    Example:
        session = AizynthSession("config.yml")
        
        # Stock check
        if session.is_in_stock("CCO"):
            print("Ethanol is in stock!")
        
        # Single-step expansion
        for result in session.expand_once("c1ccccc1Br"):
            print(f"Reactants: {result.reactants}")
        
        # Full search
        result = session.search("CC(=O)Oc1ccccc1C(=O)O")
        print(f"Solved: {result.is_solved}, Routes: {result.num_routes}")
    """
    
    def __init__(
        self,
        config_path: str,
        stock_names: List[str] = None,
        expansion_names: List[str] = None,
        filter_names: List[str] = None,
    ):
        """
        初始化会话。
        
        Args:
            config_path: AiZynthFinder 配置文件路径 (config.yml)
            stock_names: 要使用的 stock 名称列表 (默认使用配置中的所有)
            expansion_names: 要使用的 expansion policy 名称 (默认使用配置中的所有)
            filter_names: 要使用的 filter 名称 (默认使用配置中的所有)
        """
        self._config_path = Path(config_path).resolve()
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        
        # Lazy initialization
        self._finder = None
        self._expander = None
        self._Molecule = None
        
        self._stock_names = stock_names
        self._expansion_names = expansion_names
        self._filter_names = filter_names
        
        # Initialize on first use
        self._init_finder()
    
    def _init_finder(self) -> None:
        """Lazy initialization of AiZynthFinder"""
        if self._finder is not None:
            return
        
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder
            from aizynthfinder.chem import Molecule
        except ImportError as e:
            raise ImportError(
                "AiZynthFinder not installed. Install with: "
                "pip install aizynthfinder"
            ) from e
        
        self._Molecule = Molecule
        
        logger.info(f"Loading AiZynthFinder from {self._config_path}")
        
        # Change to config directory so relative paths in config work correctly
        original_cwd = os.getcwd()
        try:
            os.chdir(self._config_path.parent)
            self._finder = AiZynthFinder(configfile=str(self._config_path.name))
        except Exception as e:
            logger.error(f"Failed to initialize AiZynthFinder: {e}")
            raise
        finally:
            os.chdir(original_cwd)
        
        # Select expansion policies from config
        self._select_policies()
        
        logger.info("AiZynthFinder initialized successfully")
    
    def _select_policies(self) -> None:
        """Select expansion, stock, and filter policies"""
        if not self._finder:
            return

        # Helper to select multiple items
        def select_items(policy_wrapper, default_names, user_specified_names=None):
            names_to_select = []
            if user_specified_names:
                names_to_select = user_specified_names
            elif policy_wrapper.items:
                items = policy_wrapper.items
                
                # Extract names from items
                if isinstance(items, dict):
                    names_to_select = list(items.keys())
                elif isinstance(items, list):
                    # List of names or objects
                    for item in items:
                        if hasattr(item, 'name'):
                            names_to_select.append(item.name)
                        elif hasattr(item, 'key'):
                            names_to_select.append(item.key)
                        else:
                            names_to_select.append(str(item))
            
            # If no names found (empty), use defaults if provided
            if not names_to_select and default_names:
                names_to_select = default_names

            if names_to_select:
                try:
                    # Select all at once to avoid overwriting
                    policy_wrapper.select(names_to_select)
                    logger.info(f"Selected {names_to_select}")
                except Exception as e:
                    logger.warning(f"Failed to select {names_to_select}: {e}")
                    # Fallback: try one by one if list select fails
                    for name in names_to_select:
                        try:
                            policy_wrapper.select(name)
                        except: pass
            else:
                logger.debug(f"No items to select")

        # Select expansion policies
        logger.info("Selecting expansion policies...")
        select_items(self._finder.expansion_policy, ['uspto', 'ringbreaker'], self._expansion_names)

        # Select stock
        logger.info("Selecting stock...")
        select_items(self._finder.stock, ['zinc'], self._stock_names)
        
        # Select filter policies
        logger.info("Selecting filter policies...")
        select_items(self._finder.filter_policy, ['uspto'], self._filter_names)
        
        # Log selected policies
        logger.info(
            f"Policies selected: "
            f"expansion={list(self._finder.expansion_policy.selection)}, "
            f"stock={list(self._finder.stock.selection)}"
        )
        # Extra diagnostic for zero-result debugging
        if not self._finder.expansion_policy.selection:
            logger.warning("CRITICAL: No expansion policies selected! Retrosynthesis will fail.")
        print(f"  [AIZYNTH] Policies loaded: {list(self._finder.expansion_policy.selection)}")
    
    @property
    def finder(self):
        """Get the underlying AiZynthFinder instance"""
        self._init_finder()
        return self._finder
    
    def is_in_stock(self, smiles: str) -> bool:
        """
        检查分子是否可买。
        
        Args:
            smiles: 分子 SMILES
            
        Returns:
            True if in stock, False otherwise
            
        Note:
            AiZynthFinder stock 使用 Molecule 对象，基于 InChIKey 进行匹配。
        """
        self._init_finder()
        
        if not smiles or ">>" in smiles:
            return False

        try:
            mol = self._Molecule(smiles=smiles)
            # AiZynthFinder's stock query might raise errors on invalid mols during InChI generation
            return mol in self._finder.stock
        except Exception as e:
            # logger.debug(f"Failed to check stock for {smiles}: {e}")
            return False
    
    def expand_once(
        self,
        smiles: str,
        topk: int = 50
    ) -> List[ExpansionResult]:
        """
        单步模板扩展。
        
        Rewrite: uses tree_search(limit=1) to ensure robust execution
        instead of calling policy.get_actions() directly which has unstable API.
        
        Args:
            smiles: 产物 SMILES
            topk: 返回的最大候选数
            
        Returns:
            List of ExpansionResult
        """
        self._init_finder()
        
        # Save old config to restore later
        old_limit = self._finder.config.search.iteration_limit
        old_time = self._finder.config.search.time_limit
        
        results = []
        
        try:
            # Configure specifically for single-step expansion
            self._finder.target_smiles = smiles
            # Configure specifically for single-step expansion
            self._finder.target_smiles = smiles
            # V2.3: Increased iteration limit to 5 to ensure more templates are explored 
            # and actions are populated in the tree structure.
            self._finder.config.search.iteration_limit = 5
            self._finder.config.search.time_limit = 60 
            
            # Run search
            self._finder.tree_search()
            
            # Extract from root node
            if self._finder.tree and self._finder.tree.root:
                root = self._finder.tree.root
                
                # root.children are Reaction objects (edges)
                actions = root.children
                
                for rxn in actions[:topk]:
                    try:
                        # Extract reactants
                        reactants = None
                        
                        # Method 1: Standard attribute (Reaction object)
                        if hasattr(rxn, 'reactants'):
                            mols = rxn.reactants
                            reactants = tuple(m.smiles for m in mols)
                        
                        # Method 2: Node wrapper (MCTS Node wrapping action)
                        elif hasattr(rxn, 'action') and hasattr(rxn.action, 'reactants'):
                            mols = rxn.action.reactants
                            reactants = tuple(m.smiles for m in mols)
                            
                        # Method 3: Children check (MolNodes)
                        elif hasattr(rxn, 'children'):
                            r_list = []
                            for child in rxn.children:
                                if hasattr(child, 'smiles'):
                                    r_list.append(child.smiles)
                                elif hasattr(child, 'mol') and hasattr(child.mol, 'smiles'):
                                    r_list.append(child.mol.smiles)
                            if r_list:
                                reactants = tuple(r_list)
                        
                        if reactants is None:
                             if hasattr(rxn, 'to_dict'):
                                 try:
                                     d = rxn.to_dict()
                                     # children in dict are reactants
                                     if 'children' in d:
                                          reactants = tuple(c.get('smiles') for c in d['children'] if 'smiles' in c)
                                 except: pass
                        
                        if reactants is None:
                            # Method 5: MCTS Node state (rxn is the child node representing precursors)
                            if hasattr(rxn, 'state'):
                                if hasattr(rxn.state, 'mols'):
                                    reactants = tuple(rxn.state.mols)
                                elif hasattr(rxn.state, 'molecules'):
                                    # molecules might be objects
                                    mols_obj = rxn.state.molecules
                                    if mols_obj and hasattr(mols_obj[0], 'smiles'):
                                         reactants = tuple(m.smiles for m in mols_obj)
                                    else:
                                         reactants = tuple(str(m) for m in mols_obj)
                        
                        if reactants is None:
                            logger.warning(f"Reaction node structure unknown: {dir(rxn)}")
                            continue
                        
                        # Extract metadata
                        metadata = {}
                        template = ""
                        if hasattr(rxn, 'metadata'):
                             metadata = dict(rxn.metadata)
                             template = metadata.get('template_hash', '') or \
                                       metadata.get('template_code', '') or \
                                       str(metadata.get('policy_name', ''))
                        
                        results.append(ExpansionResult(
                            reactants=reactants,
                            template_smarts=template,
                            metadata=metadata
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to parse reaction node: {e}")
                        continue
            
            return results

        except Exception as e:
            logger.error(f"expand_once (via search) failed for {smiles}: {e}")
            return []
            
        finally:
            # Restore config
            try:
                self._finder.config.search.iteration_limit = old_limit
                self._finder.config.search.time_limit = old_time
            except:
                pass
    
    def search(
        self,
        smiles: str,
        break_bonds: List[Tuple[int, int]] = None,
        freeze_bonds: List[Tuple[int, int]] = None,
        max_transforms: int = None,
        time_limit: float = None,
        iteration_limit: int = None,
    ) -> SearchResult:
        """
        完整深度搜索。
        
        支持 break_bonds/freeze_bonds 约束 (用于 MO-MCTS)。
        
        Args:
            smiles: 目标分子 SMILES
            break_bonds: 必须断开的键 [(atom1_idx, atom2_idx), ...]
                         注意: 索引是 0-based (需验证)
            freeze_bonds: 不能断开的键 [(atom1_idx, atom2_idx), ...]
            max_transforms: 最大转化步数
            time_limit: 搜索时间限制 (秒)
            iteration_limit: 搜索迭代限制
            
        Returns:
            SearchResult with routes, statistics, and trees
        """
        self._init_finder()
        
        # Set target
        self._finder.target_smiles = smiles
        
        # Apply bond constraints if specified
        config = self._finder.config
        
        if break_bonds is not None:
            # Convert to list of lists format expected by AiZynthFinder
            config.search.break_bonds = [list(b) for b in break_bonds]
        else:
            config.search.break_bonds = []
        
        if freeze_bonds is not None:
            config.search.freeze_bonds = [list(b) for b in freeze_bonds]
        else:
            config.search.freeze_bonds = []
        
        # Apply search limits
        if max_transforms is not None:
            config.search.max_transforms = max_transforms
        if time_limit is not None:
            config.search.time_limit = time_limit
        if iteration_limit is not None:
            config.search.iteration_limit = iteration_limit
        
        # Run search
        try:
            self._finder.tree_search()
            self._finder.build_routes()
            stats = self._finder.extract_statistics()
        except Exception as e:
            logger.error(f"Search failed for {smiles}: {e}")
            return SearchResult(
                is_solved=False,
                num_routes=0,
                top_score=0.0,
                search_time=0.0,
                trees=[],
                statistics={"error": str(e)}
            )
        
        # Extract route trees
        # finder.routes returns dicts with structure: {'reaction_tree': ReactionTree, ...}
        trees = []
        num_routes = 0
        try:
            routes = list(self._finder.routes)
            num_routes = len(routes)
            
            for route in routes:
                try:
                    # Route is typically a dict with 'reaction_tree' key containing ReactionTree object
                    if isinstance(route, dict):
                        if 'reaction_tree' in route:
                            rt = route['reaction_tree']
                            if hasattr(rt, 'to_dict'):
                                trees.append(rt.to_dict())
                            elif isinstance(rt, dict):
                                trees.append(rt)
                            else:
                                logger.debug(f"reaction_tree is {type(rt)}, cannot convert")
                        else:
                            # Route dict itself might be a tree
                            trees.append(route)
                    elif hasattr(route, 'reaction_tree'):
                        rt = route.reaction_tree
                        if hasattr(rt, 'to_dict'):
                            trees.append(rt.to_dict())
                    elif hasattr(route, 'to_dict'):
                        trees.append(route.to_dict())
                    else:
                        logger.debug(f"Unknown route type: {type(route)}")
                except Exception as e:
                    logger.debug(f"Failed to extract route: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract route trees: {e}")
        
        # Parse statistics
        if isinstance(stats, dict):
            stats_dict = stats
        else:
            # stats might be a DataFrame row
            try:
                stats_dict = stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
            except Exception:
                stats_dict = {}
        
        return SearchResult(
            is_solved=stats_dict.get("is_solved", False),
            num_routes=stats_dict.get("number_of_routes", num_routes),
            top_score=stats_dict.get("top_score", 0.0),
            search_time=stats_dict.get("search_time", 0.0),
            trees=trees,
            statistics=stats_dict
        )
    
    def render_routes(
        self,
        output_dir: str,
        prefix: str = "route"
    ) -> List[str]:
        """
        渲染路线图为 PNG。
        
        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
            
        Returns:
            List of saved image paths
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved = []
        for i, route in enumerate(self._finder.routes):
            try:
                img_path = output_path / f"{prefix}{i:03d}.png"
                route.to_image().save(str(img_path))
                saved.append(str(img_path))
                logger.info(f"Saved route image: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to render route {i}: {e}")
        
        return saved


# =============================================================================
# CLI for quick testing
# =============================================================================

def main():
    """Quick test of AizynthSession"""
    import argparse
    import sys
    
    # Default config path: same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config.yml")
    
    parser = argparse.ArgumentParser(description="Test AizynthSession")
    parser.add_argument("--config", default=default_config, help="Config file path")
    parser.add_argument("--smiles", default="c1ccccc1Br", help="Target SMILES")
    parser.add_argument("--check-stock", action="store_true", help="Check if in stock")
    parser.add_argument("--expand", action="store_true", help="Run single-step expansion")
    parser.add_argument("--search", action="store_true", help="Run full search")
    parser.add_argument("--output-dir", default=".", help="Output directory for routes")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"Initializing AizynthSession with {args.config}")
    try:
        session = AizynthSession(args.config)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)
    
    print(f"Target: {args.smiles}")
    
    if args.check_stock:
        in_stock = session.is_in_stock(args.smiles)
        print(f"In stock: {in_stock}")
    
    if args.expand:
        print("\nSingle-step expansion:")
        results = session.expand_once(args.smiles)
        for i, r in enumerate(results[:10], 1):
            print(f"  {i}. {' + '.join(r.reactants)}")
    
    if args.search:
        print("\nFull search:")
        result = session.search(args.smiles)
        print(f"  Solved: {result.is_solved}")
        print(f"  Routes: {result.num_routes}")
        print(f"  Top score: {result.top_score:.3f}")
        print(f"  Search time: {result.search_time:.2f}s")
        
        if result.trees:
            import json
            print("\n--- DEBUG: First Tree Structure ---")
            print(json.dumps(result.trees[0], indent=2))
            print("-----------------------------------\n")
        
        if result.num_routes > 0:
            saved = session.render_routes(args.output_dir)
            print(f"  Saved {len(saved)} route images")
    
    if not (args.check_stock or args.expand or args.search):
        # Default: show all
        print(f"\nStock check: {session.is_in_stock(args.smiles)}")
        
        print("\nExpansion (top 5):")
        for i, r in enumerate(session.expand_once(args.smiles)[:5], 1):
            print(f"  {i}. {' + '.join(r.reactants)}")


if __name__ == "__main__":
    main()
