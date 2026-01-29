
import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to path (MoleReact root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multistep.agent.core.react import ReActSession
from multistep.agent.tools.base import ToolRegistry, BaseTool

class MockTool(BaseTool):
    def __init__(self, name):
        self._name = name
        
    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return f"Mock tool {self._name}"
        
    def execute(self, args):
        return f"Executed {self._name} with {args}"

class TestReActParallel(unittest.TestCase):
    def test_parallel_actions(self):
        # Setup
        registry = ToolRegistry()
        registry.register(MockTool("ToolA"))
        registry.register(MockTool("ToolB"))
        
        mock_llm = MagicMock()
        # Mock LLM response with parallel actions
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].delta.content = """
Plan: I need to call both tools.
Action: ToolA(arg1)
Action: ToolB(arg2)
"""
        # Second call returns Final Answer
        mock_response_2 = MagicMock()
        mock_response_2.choices = [MagicMock()]
        mock_response_2.choices[0].delta.content = "Final Answer: Done"
        
        # Generator for streaming response
        def stream_response(mock_resp):
            yield mock_resp

        # Mock create method
        mock_llm.chat.completions.create.side_effect = [
            stream_response(mock_response),
            stream_response(mock_response_2)
        ]
        
        session = ReActSession(mock_llm, registry, max_steps=3)
        
        # Run
        result = session.run("Test Goal")
        
        # Verify
        print("History:", session.history)
        
        # Check if both observations are in history
        obs_entry = None
        for entry in session.history:
            if "Observation from ToolA" in entry["content"] and "Observation from ToolB" in entry["content"]:
                obs_entry = entry
                break
        
        self.assertIsNotNone(obs_entry, "Combined observation not found")
        self.assertIn("Executed ToolA with arg1", obs_entry["content"])
        self.assertIn("Executed ToolB with arg2", obs_entry["content"])
        self.assertEqual(result, "Done")

if __name__ == '__main__':
    unittest.main()
