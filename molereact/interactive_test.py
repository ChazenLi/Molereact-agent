# -*- coding: utf-8 -*-
"""
交互式 Agent 测试脚本
=======================

用于调试和测试 RetroSynthesisAgent 的交互式脚本。

Usage:
    cd MoleReact/multistep
    python agent/interactive_test.py
    
    或带参数:
    python agent/interactive_test.py --smiles "CCO" --mode research
"""

import sys
import os
import argparse
import logging

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Default test molecule (更复杂的药物分子)
DEFAULT_TARGET = "OC12C(C(C3=CC=CC=C3)C(C(OC)=O)C2O)(C4=CC=C(OC)C=C4)CC5=CC(OCC6=CC=CC=C6)=CC(OC)=C51"


def print_banner():
    """打印欢迎信息"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         🧪 MoleReact Agent 交互式测试环境                       ║
╚══════════════════════════════════════════════════════════════════╝
""")


def test_config_interactive():
    """交互式测试配置"""
    print("\n" + "=" * 60)
    print("📋 配置模块测试")
    print("=" * 60)
    
    from agent.config import AgentConfig, AgentMode, InteractionMode
    
    # 显示可用模式
    print("\n可用运行模式:")
    for mode in AgentMode:
        print(f"  - {mode.value}")
    
    print("\n可用交互模式:")
    for mode in InteractionMode:
        print(f"  - {mode.value}")
    
    # 创建配置
    config_research = AgentConfig.for_research()
    config_production = AgentConfig.for_production()
    
    print("\n研究模式配置:")
    print(f"  mode: {config_research.mode.value}")
    print(f"  features.safety_check: {config_research.features.safety_check}")
    print(f"  features.cost_estimation: {config_research.features.cost_estimation}")
    
    print("\n生产模式配置:")
    print(f"  mode: {config_production.mode.value}")
    print(f"  features.safety_check: {config_production.features.safety_check}")
    print(f"  features.cost_estimation: {config_production.features.cost_estimation}")
    print(f"  features.supply_chain: {config_production.features.supply_chain}")
    
    return config_research, config_production


def test_production_skills_interactive(target_smiles: str):
    """交互式测试生产技能"""
    print("\n" + "=" * 60)
    print("🏭 生产技能测试")
    print("=" * 60)
    print(f"目标分子: {target_smiles[:50]}...")
    
    from agent.skills_production import (
        estimate_cost,
        safety_check,
        scale_up_analysis,
        supply_chain_query,
    )
    
    # 模拟路线数据
    mock_route = {
        "stages": [
            {"target": target_smiles, "precursors": ["CCN(CC)c1ccc(cc1)CCBr", "HN1CCN(C(=O)C2CCCCC2)C1=O"]},
            {"target": "CCN(CC)c1ccc(cc1)CCBr", "precursors": ["CCN(CC)c1ccc(cc1)CCO", "PBr3"]},
            {"target": "HN1CCN(C(=O)C2CCCCC2)C1=O", "precursors": ["哌嗪", "环己甲酰氯"]},
        ],
        "metadata": {"target": target_smiles}
    }
    
    # 1. 成本估算
    print("\n--- 💰 成本估算 ---")
    cost_result = estimate_cost(mock_route, scale="lab", target_quantity="10g")
    print(f"  总成本: ¥{cost_result['total_cost']:.0f}")
    print(f"  原料成本: ¥{cost_result['material_cost']['amount']:.0f}")
    print(f"  试剂成本: ¥{cost_result['reagent_cost']:.0f}")
    print(f"  人工工时: {cost_result['estimated_labor_hours']} 小时")
    if cost_result['cost_drivers']:
        print(f"  成本驱动因素:")
        for driver in cost_result['cost_drivers']:
            print(f"    - {driver}")
    
    # 2. 安全评估
    print("\n--- ⚠️ 安全评估 ---")
    reagents = ["NaH", "n-BuLi", "THF", "DMF", "Et3N", "PBr3"]
    safety_result = safety_check(mock_route, reagent_list=reagents)
    print(f"  风险等级: {safety_result['overall_risk_level']}")
    print(f"  识别危害: {safety_result['hazard_count']} 项")
    if safety_result['hazard_flags']:
        print("  危险试剂:")
        for hazard in safety_result['hazard_flags'][:3]:  # 只显示前3个
            print(f"    - {hazard['reagent']}: {hazard['hazard_type']}")
    print(f"  所需 PPE: {', '.join(safety_result['required_ppe'][:5])}")
    print(f"  所需设备: {', '.join(safety_result['required_equipment'])}")
    
    # 3. 放大分析
    print("\n--- 📈 放大分析 ---")
    scale_result = scale_up_analysis(mock_route, target_scale="pilot")
    print(f"  推荐规模: {scale_result['recommended_scale']}")
    print(f"  瓶颈步骤: {scale_result['bottleneck_steps']}")
    if scale_result['process_modifications']:
        print("  工艺改进建议:")
        for mod in scale_result['process_modifications'][:2]:
            print(f"    - {mod}")
    
    # 4. 供应链查询
    print("\n--- 🚚 供应链查询 ---")
    precursors = ["CCN(CC)c1ccc(cc1)CCO", "PBr3", "环己甲酰氯"]
    supply_result = supply_chain_query(precursors, preferred_region="china")
    print(f"  查询原料: {supply_result['total_materials']} 种")
    if supply_result['materials']:
        print("  供应商信息 (示例):")
        for mat in supply_result['materials'][:2]:
            print(f"    - {mat['name'][:20]}: ¥{mat['best_price']}/kg, {mat['shortest_lead_time']}天到货")
    if supply_result['critical_path_items']:
        print("  关键路径项:")
        for item in supply_result['critical_path_items']:
            print(f"    ⚠️ {item}")
    
    return {
        "cost": cost_result,
        "safety": safety_result,
        "scale": scale_result,
        "supply": supply_result,
    }


def test_heuristic_selection_interactive(target_smiles: str):
    """交互式测试启发式选择"""
    print("\n" + "=" * 60)
    print("🎯 启发式路线选择测试")
    print("=" * 60)
    
    # 模拟候选路线
    candidates = [
        {
            "precursors": ["CCN(CC)c1ccc(cc1)CCBr", "HN1CCN(C(=O)C2CCCCC2)C1=O"],
            "source": "template",
            "confidence": 0.85,
            "reaction_type": "N-烷基化",
        },
        {
            "precursors": ["CCN(CC)c1ccc(cc1)CH=O", "哌嗪酰胺"],
            "source": "model",
            "confidence": 0.72,
            "reaction_type": "还原胺化",
        },
        {
            "precursors": ["对乙氨基苯乙醇", "哌嗪-2-酮-环己酰胺"],
            "source": "both",
            "confidence": 0.90,
            "reaction_type": "Mitsunobu",
        },
    ]
    
    stock_map = {
        "CCN(CC)c1ccc(cc1)CCBr": True,
        "HN1CCN(C(=O)C2CCCCC2)C1=O": False,
        "CCN(CC)c1ccc(cc1)CH=O": True,
        "哌嗪酰胺": True,
        "对乙氨基苯乙醇": False,
        "哌嗪-2-酮-环己酰胺": False,
    }
    
    print(f"\n候选路线数: {len(candidates)}")
    print("\n候选详情:")
    for i, cand in enumerate(candidates, 1):
        precursors = cand['precursors']
        stock_status = [f"{'✅' if stock_map.get(p, False) else '❌'}{p[:20]}" for p in precursors]
        print(f"\n  路线 {i} [{cand['source']}] - {cand['reaction_type']}")
        print(f"    置信度: {cand['confidence']}")
        print(f"    前体: {' + '.join(stock_status)}")
    
    # 执行启发式选择
    def heuristic_select(candidates, stock_map, top_n=2):
        scored = []
        for i, cand in enumerate(candidates):
            score = 0
            precursors = cand.get("precursors", [])
            
            if cand.get("source") == "template":
                score += 10
            elif cand.get("source") == "both":
                score += 15
            
            stock_count = sum(1 for p in precursors if stock_map.get(p, False))
            score += stock_count * 20
            score += cand.get("confidence", 0) * 10
            
            scored.append((score, i, cand))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:top_n]
    
    print("\n--- 启发式评分结果 ---")
    selected = heuristic_select(candidates, stock_map, top_n=2)
    for rank, (score, idx, cand) in enumerate(selected, 1):
        print(f"\n  🏆 排名 {rank} (得分: {score:.1f})")
        print(f"     来源: {cand['source']}")
        print(f"     反应: {cand['reaction_type']}")
        print(f"     前体: {cand['precursors']}")
    
    return selected


def test_agent_structure_interactive():
    """交互式测试 Agent 结构"""
    print("\n" + "=" * 60)
    print("🤖 Agent 结构测试")
    print("=" * 60)
    
    from agent.agent import WorkModuleResult
    from agent.config import AgentConfig
    
    # 创建模拟结果
    result = WorkModuleResult(
        stage_number=1,
        target_smiles=DEFAULT_TARGET,
        top_n_routes=[
            {"rank": 1, "precursors": ["前体A", "前体B"], "reason": "可购买率高，反应条件温和"},
            {"rank": 2, "precursors": ["前体C", "前体D"], "reason": "步骤少，但需要保护基"},
        ],
        unsolved_leaves=["前体B"],
        stage_image_path=None,
        is_complete=False,
        audit_info={
            "timestamp": "2026-01-19T15:00:00",
            "config_mode": "research",
            "stock_rate": 0.5,
        },
    )
    
    print("\nWorkModuleResult 示例:")
    print(f"  阶段: {result.stage_number}")
    print(f"  目标: {result.target_smiles[:40]}...")
    print(f"  推荐路线数: {len(result.top_n_routes)}")
    print(f"  未解决分子: {result.unsolved_leaves}")
    print(f"  是否完成: {result.is_complete}")
    
    # 显示格式化输出
    print("\n--- 模拟 Agent 输出 ---")
    output = f"""
## 📊 阶段 {result.stage_number} 分析报告

### 🎯 目标分子
`{result.target_smiles}`

### 🔬 推荐路线
"""
    for route in result.top_n_routes:
        output += f"\n**路线 {route['rank']}**: {route['precursors']}\n"
        output += f"理由: {route['reason']}\n"
    
    if result.is_complete:
        output += "\n✅ 路线完成\n"
    else:
        output += f"\n⏳ 待继续: {result.unsolved_leaves}\n"
    
    print(output)
    
    return result


def interactive_menu():
    """交互式菜单"""
    print("\n" + "=" * 60)
    print("选择测试项目:")
    print("=" * 60)
    print("  1. 配置模块测试")
    print("  2. 生产技能测试 (成本/安全/放大/供应链)")
    print("  3. 启发式路线选择测试")
    print("  4. Agent 结构测试")
    print("  5. 运行全部测试")
    print("  0. 退出")
    print("-" * 60)
    
    choice = input("请输入选项 (0-5): ").strip()
    return choice


def main():
    parser = argparse.ArgumentParser(description="MoleReact Agent 交互式测试")
    parser.add_argument("--smiles", default=DEFAULT_TARGET, help="目标分子 SMILES")
    parser.add_argument("--mode", default="research", choices=["research", "production"], help="运行模式")
    parser.add_argument("--all", action="store_true", help="运行全部测试")
    
    args = parser.parse_args()
    target = args.smiles
    
    print_banner()
    print(f"目标分子: {target[:60]}{'...' if len(target) > 60 else ''}")
    print(f"运行模式: {args.mode}")
    
    if args.all:
        # 运行全部测试
        test_config_interactive()
        test_production_skills_interactive(target)
        test_heuristic_selection_interactive(target)
        test_agent_structure_interactive()
        print("\n" + "=" * 60)
        print("✅ 全部测试完成")
        print("=" * 60)
        return 0
    
    # 交互式菜单
    while True:
        choice = interactive_menu()
        
        if choice == "0":
            print("\n👋 再见!")
            break
        elif choice == "1":
            test_config_interactive()
        elif choice == "2":
            test_production_skills_interactive(target)
        elif choice == "3":
            test_heuristic_selection_interactive(target)
        elif choice == "4":
            test_agent_structure_interactive()
        elif choice == "5":
            test_config_interactive()
            test_production_skills_interactive(target)
            test_heuristic_selection_interactive(target)
            test_agent_structure_interactive()
            print("\n✅ 全部测试完成")
        else:
            print("无效选项，请重试")
        
        input("\n按 Enter 继续...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
