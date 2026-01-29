# MoleReact Workflow & Lifecycle

This document visualizes the complete execution flow of the MoleReact Agent (V3.6), from initialization to final report generation.

## 🔄 High-Level Life Cycle

```mermaid
graph TD
    Start((🚀 Start)) --> Init[Initialize Agent & Tools]
    AutoAgent[autoagent/*] --> Init
    Init --> GlobalPlan[Agent-0: Global Strategy Blueprint]
    GlobalPlan --> KB[Knowledge Base (knowledge_base.py)]
    
    subgraph "Main Planning Loop"
        Queue{Queue Empty?} -- No --> Pop[Pop Current Node]
        Pop --> CheckLoop{Path Loop?}
        CheckLoop -- Yes --> Warn[Warning / Stop Branch]
        CheckLoop -- No --> Expand[Atomic Expansion]
        
        Expand --> Retro[Retrosynthesis Engine]
        Retro --> Analyze[Chemical Analysis & Stock Check]
        Analyze --> DeepScan[ReAct Deep Scan (Audit)]
        KB --> Select
        DeepScan --> Select[LLM Selector (5D Scoring)]
        
        Select --> Decision{Auto or Human?}
        Decision -- Auto --> UpdateTree[Update Tree & Push Children]
        Decision -- Human --> HITL[Interactive CLI]
        
        HITL -- "reopen" --> Prune[Prune Ghost Nodes]
        HITL -- "switch" --> Reorder[Reorder Queue]
        HITL -- "expert" --> Inject[Inject Manual Step]
        HITL -- "select" --> UpdateTree
        
        Prune --> Queue
        Reorder --> Queue
        Inject --> UpdateTree
        UpdateTree --> Queue
    end
    
    Queue -- Yes --> Finalize[Finalize Session]
    Finalize --> Visualize[Generate Tree PNG]
    Visualize --> Report[Write JSON/Markdown Reports]
    Report --> End((🏁 End))
```

---

## Flowchart (Decision-Oriented)

```mermaid
flowchart TB
    Start((Start)) --> Init[Init Engine + LLM + Tools]
    AutoAgent[autoagent/*] --> Init
    Init --> Strategy[Agent-0 Global Strategy]
    Strategy --> KB[Knowledge Base (knowledge_base.py)]
    Strategy --> Queue[Seed global_unsolved_queue]

    subgraph Loop["Main Planning Loop"]
        Queue --> Pop[Pop current_node]
        Pop --> LoopCheck{Loop in Lineage?}
        LoopCheck -- Yes --> Skip[Skip or confirm]
        LoopCheck -- No --> Expand[Run retrosynthesis engine]
        Expand --> Analyze[Analysis + StockCheck]
        Analyze --> Rank[LLMSelectionHandler ranks routes]
        KB --> Rank
        Rank --> Decide{Auto or Human}
        Decide -- Auto --> Update[Append stage + push children]
        Decide -- Human --> CLI[Interactive CLI]
        CLI -- select --> Update
        CLI -- switch --> Switch[Reorder queue]
        CLI -- reopen --> Prune[Prune subtree queue + rollback stages]
        CLI -- expert --> Inject[Manual expert step]
        Switch --> Queue
        Prune --> Queue
        Inject --> Queue
        Update --> Queue
        Update --> KB
    end

    Queue -->|empty| Finalize[Finalize + Visualize + Report]
    Finalize --> End((End))
```

## 🚦 Detailed Execution Steps

### 1. Initialization Phase
*   **Input**: Target SMILES, Configuration (API Keys, Stage Limit).
*   **Action**: 
    - Load `SingleStepEngine` (AI models/Templates).
    - Initialize `SessionLogger` (JSON/MD persistence).
    - **Agent-0 (V2)**: Generates a global "Blueprint" (Strategic cut points, sensitive groups) to guide downstream decisions.

### 2. The Main Loop (Breadth/Depth Mixed Search)
The agent operates on a **Global Unsolved Queue** (`global_unsolved_queue`).

#### A. Node Processing
1.  **Pop**: Retrieve the next molecule with its lineage (history).
2.  **Safety Check**: Verify the molecule hasn't appeared earlier in its own ancestry (Loop Detection).
3.  **Expansion**: Call the Retrosynthesis Engine to propose 50+ precursor combinations.

#### B. The "Glass-Box" Audit
before making a choice, the agent performs a deep audit:
1.  **Stock Check**: Which precursors are chemically available?
2.  **Deep Scan**: A ReAct sub-agent "thinks" about the toxicity, stability, and feasibility of the top candidates.

#### C. Decision Making (The Selector)
The `LLMSelectionHandler` ranks routes based on:
-   **Strategic Fit**: Does this match Agent-0's blueprint?
-   **5D Score**: Complexity, Reactivity, Selectivity, Efficiency, PG-Cost.
-   **Result**: Top 1-3 routes are selected.

### 3. Human-in-the-Loop (Interactive Mode)
At this stage, the user can intervene via the CLI:

-   **`select [N]`**: Approve the agent's recommendation.
-   **`switch Q[N]`**: "This branch is boring, let's look at that other molecule first."
-   **`reopen [ID]`**: "Wait, Step 1.2 was a mistake. Undo it." (Triggers **Ghost Node Pruning**).
-   **`expert [A] >> [B]`**: "I know better. Force this reaction."

### 4. Tree Update & Serialization
*   **Persistence**: Every decision is immediately saved to `JSON` snapshots.
*   **Queue**: New unsolved precursors are added to the front (Depth-First bias) or back of the queue.

### 5. Finalization
*   **Visualization**: Converts the final `cumulative_route` into an AiZynthFinder-style tree graph (`.png`).
*   **Reporting**: Outputs a detailed markdown report (`ACADEMIC_REPORT.md` compatible) and a machine-readable JSON dump.
