# -*- coding: utf-8 -*-
"""
Agent Configuration
====================

配置选项，支持不同使用场景 (研究/工艺开发/生产)。
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum


class AgentMode(Enum):
    """Agent 运行模式"""
    RESEARCH = "research"                    # 研究模式
    PROCESS_DEVELOPMENT = "process_dev"      # 工艺开发模式
    PRODUCTION = "production"                # 生产模式


class InteractionMode(Enum):
    """人机交互模式"""
    INTERACTIVE = "interactive"    # 每阶段交互
    AUTO = "auto"                  # 全自动
    MIXED = "mixed"                # 混合 (前 N 阶段交互)


@dataclass
class FeatureConfig:
    """功能模块配置"""
    cost_estimation: bool = False        # 成本估算
    safety_check: bool = True            # 安全评估 (始终建议开启)
    scale_up_analysis: bool = False      # 放大分析
    supply_chain: bool = False           # 供应链查询
    compliance_check: bool = False       # 合规检查


@dataclass
class InteractionConfig:
    """交互配置"""
    default_mode: InteractionMode = InteractionMode.MIXED
    interaction_stages: int = 2          # 混合模式下前 N 阶段交互
    timeout_seconds: int = 300           # 等待用户响应超时


@dataclass
class OutputConfig:
    """输出配置"""
    detail_level: Literal["brief", "standard", "detailed"] = "standard"
    include_literature: bool = True
    include_mechanism: bool = True
    include_audit_info: bool = True


@dataclass
class AgentConfig:
    """Agent 总配置"""
    mode: AgentMode = AgentMode.RESEARCH
    features: FeatureConfig = field(default_factory=FeatureConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # 模型路径 (可选覆盖)
    retro_model_path: Optional[str] = None
    forward_model_path: Optional[str] = None
    vocab_path: Optional[str] = None
    config_path: Optional[str] = None
    
    @classmethod
    def for_research(cls) -> "AgentConfig":
        """研究模式预设"""
        return cls(
            mode=AgentMode.RESEARCH,
            features=FeatureConfig(
                cost_estimation=False,
                safety_check=True,
                scale_up_analysis=False,
                supply_chain=False,
            ),
            interaction=InteractionConfig(
                default_mode=InteractionMode.MIXED,
                interaction_stages=2,
            ),
            output=OutputConfig(
                detail_level="standard",
            ),
        )
    
    @classmethod
    def for_production(cls) -> "AgentConfig":
        """生产模式预设"""
        return cls(
            mode=AgentMode.PRODUCTION,
            features=FeatureConfig(
                cost_estimation=True,
                safety_check=True,
                scale_up_analysis=True,
                supply_chain=True,
                compliance_check=True,
            ),
            interaction=InteractionConfig(
                default_mode=InteractionMode.INTERACTIVE,
                interaction_stages=99,  # 始终交互
            ),
            output=OutputConfig(
                detail_level="detailed",
                include_audit_info=True,
            ),
        )
