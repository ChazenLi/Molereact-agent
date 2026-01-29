
import unittest
import sys
import os
from unittest.mock import MagicMock
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multistep.agent.agent_run import CompleteWorkModuleRunner

class TestDeepScanBatch(unittest.TestCase):
    def test_deep_scan_batch_logic(self):
        # Setup Runner
        runner = CompleteWorkModuleRunner(use_llm=True)
        
        # Mock LLM Client
        mock_llm = MagicMock()
        runner.llm_client = mock_llm
        runner.enable_deep_scan = True
        
        # Mock Engine (Required for StockCheckTool in _build_react_registry)
        mock_engine = MagicMock()
        mock_engine.template_engine.is_in_stock.return_value = True
        runner.engine = mock_engine
        
        # Mock ReActSession.run to return a JSON string
        # We need to mock the ReActSession class usage inside the method.
        # Since we can't easily patch the class inside the method without patching the module, 
        # we will rely on checking if the parsing logic handles a valid string if we can mock the session.
        
        # Actually, best way is to mock valid candidates and see if the method tries to run ReAct.
        # But `_deep_scan_candidates` instantiates `ReActSession` locally.
        # Let's patch `multistep.agent.agent_run.ReActSession`.
        
        from unittest.mock import patch
        
        with patch('multistep.agent.agent_run.ReActSession') as MockSessionCls:
            mock_session_instance = MockSessionCls.return_value
            
            # Simulated LLM output as Plain Text (Testing Fallback)
            mock_session_instance.run.return_value = """
Here is the analysis:
1. Route is safe and precursors are available.
2. Warning: Toxic intermediate detected.
"""
            
            # Input Candidates
            candidates = [
                {"source": "model", "precursors": ["A", "B"], "reason": "Initial reason 1"},
                {"source": "template", "precursors": ["C", "D"], "reason": "Initial reason 2"}
            ]
            
            # Execute
            results = runner._deep_scan_candidates(candidates, "TargetMol")
            
            # Verify
            print("Modified Candidates:", results)
            
            self.assertIn("[DeepScan: Route is safe", results[0]["reason"])
            self.assertIn("[DeepScan: Warning: Toxic", results[1]["reason"])
            
            # Verify Mock was called
            mock_session_instance.run.assert_called_once()
            args, _ = mock_session_instance.run.call_args
            self.assertIn("Analyze the following 2", args[0])

if __name__ == '__main__':
    unittest.main()
