# -*- coding: utf-8 -*-
"""
Module: multistep.agent.core.react
Called By: multistep.agent.agent_run (Interaction Block)
Role: Reasoning Engine / Dynamic Tool User

Functionality:
    Implements the ReAct (Reasoning + Acting) loop capability for the Agent.
    Allows the LLM to dynamically select and execute tools based on "Thoughts".

Classes:
    - ReActSession: Manages the turn-based interaction loop.
"""

import re
import json
from typing import List, Dict, Any, Optional
from ..tools.base import ToolRegistry

class ReActSession:
    """
    ReAct Session Manager
    
    Encapsulates the logic for "Thought -> Action -> Observation -> Thought" loops.
    Designed to be instantiated on-demand (e.g., during Verification phase).
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry, max_steps: int = 5):
        """
        Initialize ReAct Session.
        
        Args:
            llm_client: Client for LLM generation (must support chat completions).
            tool_registry: Registry containing available tools.
            max_steps: Maximum reasoning steps to prevent infinite loops.
        """
        self.llm_client = llm_client
        self.registry = tool_registry
        self.max_steps = max_steps
        self.history = []
        
    def _get_system_prompt(self) -> str:
        """Generate ReAct system prompt."""
        tools_desc = self.registry.get_tools_description_block()
        return f"""You are an autonomous chemical reasoning agent.
You have access to the following tools:

{tools_desc}

Use the following strict format for your reasoning loop:

Plan: [Analyze context, choose tool strategy. You can plan multiple actions.]
Action: [Function call: ToolName(Args)]
Action: [Optional: Another function call: ToolName(Args)]
...
Observation: [Result from tool execution]
Analysis: [Analyze the observation result, connect to context, and form intermediate conclusion]

... (Repeat the above loop until you have a final answer)

Plan: [Summary and final reasoning]
Final Answer: [The comprehensive final answer to the user's task]

Begin!"""

    def run(self, goal: str, context: str = "") -> str:
        """
        Run the ReAct loop for a specific goal.
        
        Args:
            goal (str): The specific question or analysis task.
            context (str): Background context (e.g., current route info).
            
        Returns:
            str: Final answer or conclusion.
        """
        self.history = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Context:\n{context}\n\nTask: {goal}"}
        ]
        
        print(f"\n🧠 [ReAct] Starting structured reasoning session for: '{goal}'")
        
        step = 0
        while step < self.max_steps:
            # 1. Generate Plan & Action
            response = self._call_llm(self.history)
            output = response.content
            
            # Print thoughts (Plan section)
            plan_match = re.search(r"Plan:\s*(.+?)(?=\nAction|\nFinal Answer|$)", output, re.DOTALL)
            if plan_match:
                plan_text = plan_match.group(1).strip()
                print(f"\n  🧠 [PLAN]  {plan_text[:200]}..." if len(plan_text) > 200 else f"\n  🧠 [PLAN]  {plan_text}")
            
            self.history.append({"role": "assistant", "content": output})
            
            # 2. Check for Final Answer
            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            # 3. Parse Actions (Support Parallel)
            # Find all "Action: Tool(Args)" lines
            action_matches = list(re.finditer(r"Action:\s*([a-zA-Z0-9_]+)\s*\((.*?)\)(?=\n|$)", output, re.DOTALL))
            
            if not action_matches:
                if step == self.max_steps - 1:
                    return output
                step += 1
                continue

            # 4. Execute Actions (Parallel)
            observations = []
            for match in action_matches:
                tool_name = match.group(1).strip()
                tool_args_str = match.group(2).strip()
                
                print(f"  🛠️  [TOOL]  {tool_name}({tool_args_str})")
                
                tool_result = self._execute_tool(tool_name, tool_args_str)
                
                # Truncate for display
                obs_disp = str(tool_result)[:100] + "..." if len(str(tool_result)) > 100 else str(tool_result)
                print(f"  👀 [DATA]  {obs_disp}")
                
                observations.append(f"Observation from {tool_name}: {tool_result}")
            
            # 5. Append Combined Observation
            combined_obs = "\n".join(observations)
            self.history.append({"role": "user", "content": combined_obs})
            
            step += 1
            
        return "Max execution steps reached without Final Answer."

    def _call_llm(self, messages: List[Dict[str, str]]) -> Any:
        """Call usage of LLM client with Robust Retry (Exponential Backoff)"""
        import time
        import random
        
        max_retries = 5
        base_delay = 5.0  # Start with 5 seconds (slower than SDK)
        max_delay = 60.0 # Cap at 60s
        
        for attempt in range(max_retries + 1):
            try:
                # Check if client is valid
                if not self.llm_client:
                     class MockMsg:
                        content = "Final Answer: Error - No LLM Client disconnected."
                     return MockMsg()

                response = self.llm_client.chat.completions.create(
                    model="glm-4.7", # Or config
                    messages=messages,
                    temperature=0.1,
                    stream=True
                )
                
                # Simple Stream Handler for ReAct
                full_content = []
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        c = chunk.choices[0].delta.content
                        full_content.append(c)
                
                # Construct Mock Message object to match existing interface
                class MockMessage:
                    content = "".join(full_content)
                return MockMessage()
                
            except Exception as e:
                # Check for 429 or Rate Limit in string
                error_str = str(e)
                is_rate_limit = "429" in error_str or "Too Many Requests" in error_str
                
                if attempt < max_retries:
                    # Exponential Backoff with Jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, 1.0)
                    total_wait = delay + jitter
                    
                    print(f"\n  ⚠️ [API WARNING] Request failed ({e}). Retrying in {total_wait:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(total_wait)
                else:
                    print(f"\n  ❌ [API FATAL] Max retries reached. Error: {e}")
                    # Mock return for error to prevent crash
                    class MockMsg:
                        content = f"Error calling LLM after retries: {e}. Final Answer: Error."
                    return MockMsg()

    def _execute_tool(self, tool_name: str, args_str: str) -> str:
        """Execute a tool from the registry."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."
            
        try:
            # Parse args - naïve splitting or strict JSON?
            # Prompt says "Tool(Args)". Let's assume single string arg or comma split.
            # Most current tools take list or string.
            # Let's try to interpret args.
            if "," in args_str and tool_name not in ["MoleculeAnalysisTool", "ChemicalAnalysis"]: # Heuristic
                args = [a.strip().strip('"').strip("'") for a in args_str.split(",")]
            else:
                args = args_str.strip().strip('"').strip("'")
            
            # Execute
            # Some tools expect list (StockCheck), some str (Analysis).
            # We can check tool signature or just try.
            # StockCheck expects list.
            if tool_name == "StockCheck" and isinstance(args, str):
                args = [args]
                
            result = tool.execute(args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
