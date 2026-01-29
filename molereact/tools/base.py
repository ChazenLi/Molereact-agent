# -*- coding: utf-8 -*-
"""
Module: multistep.agent.tools.base
Called By: agent.py, tools/*, core/react.py
Role: Tool Interface Definition & Registry

Functionality:
    Defines the abstract base class `BaseTool` that all agent tools must inherit from.
    Implements `ToolRegistry` for centralized management, lookup, and description generation of tools.

Key Classes:
    - BaseTool: Abstract base class enforcing `execute` method.
    - ToolRegistry: Manager for tool instances.

Relations:
    - Parent of all tools in `multistep.agent.tools.*`.
"""
"""
Base Tool Infrastructure
========================
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """Abstract base class for all Agent Tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of the tool."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool logic."""
        pass

class ToolRegistry:
    """Registry to manage available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool):
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> BaseTool:
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_tools_description_block(self) -> str:
        lines = ["## Available Components (Tools)"]
        for tool in self._tools.values():
            lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)
