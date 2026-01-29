# MoleReact Agent V3.6 User Guide

This guide details the interactive commands available in the `agent_run.py` CLI interface.

## 🕹️ Interactive CLI Commands

When running in manual mode (`python agent_run.py --smiles ...`), the agent pauses at each decision point. You can enter the following commands:

### 1. Basic Navigation

*   **`[Enter]`**: Select **Route 1** (Top recommendation).
*   **`[1-N]`**: Select a specific route index (e.g., `2` for Route 2).
*   **`list` / `方案`**: View the current synthesis tree, pending queue, and route details.
*   **`q` / `quit` / `stop`**: Terminate the session immediately and generate reports.
*   **`verify`**: Mark the current node as "Needs Experimental Verification" (triggering a ReAct audit log).

### 2. Complex Control Flow

#### `switch Q[n]`
Switch focus to a different branch in the synthesis tree.
*   **Usage**: `switch Q2`
*   **Effect**: Moves the pending molecule at Queue Index 2 to the top. The current node is pushed back into the queue.

#### `reopen [PathID]`
**Undo & Retry**. Reverts a previous decision for a specific node ID.
*   **Usage**: `reopen 1.1`
*   **Logic**:
    1.  Deletes the decision record for Node 1.1.
    2.  **Prunes all downstream descendants** (e.g., 1.1.1, 1.1.2) from the execution queue to preventing ghost nodes.
    3.  Restores Node 1.1 to the top of the queue for re-planning.

#### `expert [Target] >> [Precursors]`
**Manual Strategy Injection**. Bypass the AI and force a specific chemical step.
*   **Usage**: `expert C1=CC=CC=C1 >> C1=CC=C(Br)C=C1.Mg`
*   **Effect**:
    1.  Forces the agent to accept this specific disconnection.
    2.  Auto-updates the inventory check for the new precursors.
    3.  Dynamically generates new children nodes for the manual precursors.

## 📊 Outputs & Reports

All outputs are saved to `multistep/output/agent_runs/{session_id}/`.

| File | Description |
|------|-------------|
| `session_log.md` | Human-readable chronological log of the session. |
| `tree_full_....png` | Visual retrosynthesis tree (AiZynthFinder style). |
| `report_....json` | Machine-readable full state dump (JSON-First persistence). |

---

> [!TIP]
> **Pro Tip**: Use `reopen` if you realize 3 steps down the line that a protecting group strategy was invalid. Retaining the global context while fixing a local error is the agent's superpower.
