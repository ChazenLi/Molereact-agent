import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class SessionLogger:
    """
    Session Logger (Session Memory)
    
    Responsible for recording the Agent's execution process, LLM analysis, 
    user decisions, etc., into Markdown files in real-time.
    Supports breakpoint resume and historical context reconstruction.
    """
    
    def __init__(self, output_dir: str, session_id: str = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if session_id:
            self.session_id = session_id
            self.log_path = os.path.join(self.output_dir, f"session_{session_id}.md")
            # If resuming session, do not overwrite; append instead
            if not os.path.exists(self.log_path):
                self._init_log_file()
        else:
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_path = os.path.join(self.output_dir, f"session_{self.session_id}.md")
            self._init_log_file()
            
    def _init_log_file(self):
        """Initialize log file"""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# MoleReact Session Log\n\n")
            f.write(f"- **Session ID**: `{self.session_id}`\n")
            f.write(f"- **Start Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Status**: Active\n\n")
            f.write("--- \n\n")

    def log_stage_start(self, stage: int, target_smiles: str):
        """Record stage start"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"## Stage {stage}\n\n")
            f.write(f"- **Target**: `{target_smiles}`\n")
            f.write(f"- **Time**: {datetime.now().strftime('%H:%M:%S')}\n\n")

    def log_llm_analysis(self, analysis_text: str):
        """Record LLM analysis"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"### 🧠 LLM Analysis\n\n")
            f.write(f"{analysis_text}\n\n")

    def log_candidates_summary(self, top_candidates: List[Dict]):
        """Record candidates summary"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"### 📋 Candidates Overview\n\n")
            f.write("| Rank | Source | Precursors | Scores | Reason |\n")
            f.write("|------|--------|------------|--------|--------|\n")
            for cand in top_candidates:
                rank = cand.get('rank', '-')
                source = cand.get('source', 'unknown')
                p_str = "<br>".join([f"`{p}`" for p in cand.get('precursors', [])])
                
                scores = cand.get('scores', {})
                if scores:
                    score_str = f"S:{scores.get('strategic','-')} F:{scores.get('feasibility','-')}"
                else:
                    score_str = "-"
                    
                reason = cand.get('reason', '').replace('\n', ' ')
                f.write(f"| {rank} | {source} | {p_str} | {score_str} | {reason} |\n")
            f.write("\n")

    def log_event(self, title: str, content: str, level: str = "INFO"):
        """Record general event (e.g. branch switch, jump, etc.)"""
        icon = "ℹ️"
        if level == "WARNING": icon = "⚠️"
        elif level == "SUCCESS": icon = "✅"
        elif level == "DANGER": icon = "🚨"
        elif level == "REOPEN": icon = "♻️"
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"### {icon} {title}\n\n")
            f.write(f"{content}\n\n")
            f.write(f"- **Time**: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write("\n---\n\n")

    def log_reopen(self, path_id: str, target_smiles: str, reason: str = "User chose to backtrack"):
        """Record node reopen (Backtrack) event"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"## ♻️ Node Reopened: {path_id}\n\n")
            f.write(f"- **Target Molecule**: `{target_smiles}`\n")
            f.write(f"- **Reason**: {reason}\n")
            f.write(f"- **Impact**: This node and all its derived branches have been removed. Restarting decision process.\n")
            f.write(f"- **Time**: {datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write("---\n\n")

    def log_decision(self, stage: int, route_idx: int, chosen_route: Dict, user_note: str = "", global_unsolved_queue: List[str] = None, path_id: str = None):
        """Record user/system decision"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"### 👤 Decision\n\n")
            if path_id:
                f.write(f"- **Path ID**: `{path_id}`\n")
            
            source = chosen_route.get('source', 'unknown')
            f.write(f"- **Action**: Selected Route {route_idx + 1} ({source})\n")
            
            p_list = chosen_route.get('precursors', [])
            f.write(f"- **Next Targets**: `{p_list}`\n")
            
            # 保存额外的化学元数据供恢复使用
            reaction_type = chosen_route.get('reaction_type', 'Unknown')
            reason = chosen_route.get('reason', '').replace('\n', ' ')
            f.write(f"- **Reaction Type**: {reaction_type}\n")
            f.write(f"- **Rationale**: {reason}\n")
            
            if global_unsolved_queue is not None:
                f.write(f"- **Global Unsolved Queue**: `{global_unsolved_queue}`\n")
            
            if user_note:
                f.write(f"- **User Note**: {user_note}\n")
            else:
                f.write(f"- **User Note**: (None)\n")
            
            f.write("\n---\n\n")

    def get_latest_context(self) -> Dict:
        """Get the state of the most recent session for recovery (excluding current session)"""
        import glob
        # 查找所有 session_*.md 文件
        pattern = os.path.join(self.output_dir, "session_*.md")
        files = glob.glob(pattern)
        
        # 过滤掉当前正在使用的日志文件 (通过绝对路径比较避免符号链接等问题)
        current_abs = os.path.abspath(self.log_path)
        other_files = [f for f in files if os.path.abspath(f) != current_abs]
        
        if not other_files:
            return {"exists": False}
            
        # 按名称排序 (session_YYYYMMDD_HHMMSS.md 格式天然支持按时间排序)
        other_files.sort(reverse=True)
        latest_file = other_files[0]
        
        # 提取 session_id (例如从 session_20260120_203416.md 提取 20260120_203416)
        session_id = os.path.basename(latest_file).replace("session_", "").replace(".md", "")
        
        return {
            "session_id": session_id,
            "exists": True,
            "path": latest_file
        }

    def load_history_context(self, file_path: str = None) -> str:
        """
        Read log file and reconstruct historical context for LLM Prompt.
        Only extract decision parts to avoid excessive Token usage.
        """
        path = file_path if file_path else self.log_path
        if not os.path.exists(path):
            return ""
            
        history_text = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            current_stage = 0
            capture = False
            
            for line in lines:
                if line.startswith("## Stage"):
                    try:
                        current_stage = int(line.replace("## Stage", "").strip())
                    except:
                        pass
                
                if line.startswith("### 👤 Decision"):
                    capture = True
                    history_text += f"\n[Stage {current_stage} Decision]\n"
                    continue
                
                if line.startswith("---") and capture:
                    capture = False
                    continue
                    
                if capture and line.strip():
                    if line.startswith("- **Action**"):
                        history_text += f"- {line.strip().replace('**Action**: ', '')}\n"
                    elif line.startswith("- **User Note**"):
                        note = line.strip().replace('**User Note**: ', '')
                        if note and note != "(None)":
                            history_text += f"- User Feedback: {note}\n"
                            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            
        return history_text

    def save_json_snapshot(self, cumulative_route: Dict, extra_data: Dict = None):
        """
        保存会话状态的 JSON 快照，用于精准恢复。
        
        Args:
            cumulative_route: Full synthesis tree state dictionary
            extra_data: Extra metadata (e.g. global strategy)
        """
        snapshot_path = self.log_path.replace(".md", ".json")
        try:
            data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "cumulative_route": cumulative_route,
                "extra_data": extra_data or {}
            }
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # logger.info(f"JSON Snapshot saved: {snapshot_path}")
        except Exception as e:
            logger.warning(f"Failed to save JSON snapshot: {e}")

    def load_json_snapshot(self, log_path: str) -> Optional[Dict]:
        """
        读取对应的 JSON 快照文件。
        
        Args:
            log_path: Path to the Markdown log file
        """
        snapshot_path = log_path.replace(".md", ".json")
        if not os.path.exists(snapshot_path):
            return None
            
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load JSON snapshot: {e}")
            return None



    def restore_session_state(self, file_path: str = None) -> Dict:
        """
        Full state reconstruction: Priority recovery from JSON snapshot, 
        fallback to MD log parsing if failure.
        
        V3.6: Robustness recovery - JSON first strategy
        """
        path = file_path if file_path else self.log_path
        
        # ========== Strategy 1: Attempt JSON snapshot first ==========
        json_snapshot = self.load_json_snapshot(path)
        if json_snapshot and json_snapshot.get("cumulative_route"):
            logger.info("Using JSON snapshot for recovery (more reliable)")
            cumulative_route = json_snapshot["cumulative_route"]
            # Recover extra data
            if "extra_data" in json_snapshot:
                extra = json_snapshot["extra_data"]
                if "global_strategy" in extra:
                    cumulative_route["global_strategy"] = extra["global_strategy"]
            return cumulative_route
        
        # ========== Strategy 2: Fallback to Markdown parsing ==========
        logger.info("JSON snapshot unavailable, attempting Markdown parsing...")
        
        if not os.path.exists(path):
            return {}

        import re
        
        cumulative_route = {"stages": []}

        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 1. 解析 Session Info
            # 2. 按 Stage 切分
            stages = re.split(r'## Stage (\d+)', content)
            
            # stages[0] 是 header, 之后是 (stage_num, content) 对
            for i in range(1, len(stages), 2):
                stage_num = int(stages[i])
                block = stages[i+1]
                
                # 提取 Target
                target_match = re.search(r'- \*\*Target\*\*: `([^`]+)`', block)
                target = target_match.group(1) if target_match else "Unknown"
                if i == 1:
                    cumulative_route["target"] = target # Initial Target
                
                # 尝试提取 Path ID (如果日志中有记录)
                # 目前 log_decision 没有显式记录 current path_id, 但通常 target 对应某个 path_id
                # 为了支持恢复，我们需要推断或修改 log_decision 记录 ID
                # 暂时假设按顺序恢复，或者从 decision context 中提取
                # 检查是否有 "- **Path ID**: `1.1`" 这样的字段
                path_id_match = re.search(r'- \*\*Path ID\*\*: `([^`]+)`', block)
                path_id = path_id_match.group(1) if path_id_match else f"{cumulative_route.get('stages', [])[-1]['stage'] + 1 if cumulative_route.get('stages') else 1}"
                
                # 提取 Decision
                decision_match = re.search(r'### 👤 Decision[\s\S]*?- \*\*Action\*\*: (.*?)\r?\n', block)
                action = decision_match.group(1).strip() if decision_match else "Unknown"
                
                # 提取 Precursors / Next Targets
                next_targets_match = re.search(r'- \*\*Next Targets\*\*: `(.*?)`', block)
                next_targets_str = next_targets_match.group(1) if next_targets_match else "[]"
                try:
                    # 尝试处理为列表
                    precursors_text = next_targets_str.replace("'", '"')
                    precursors = json.loads(precursors_text) if precursors_text.startswith('[') else []
                except:
                    precursors = []
                
                # 提取 Reaction Type 和 Rationale
                rtype_match = re.search(r'- \*\*Reaction Type\*\*: (.*?)\r?\n', block)
                reaction_type = rtype_match.group(1).strip() if rtype_match else "Unknown"
                
                reason_match = re.search(r'- \*\*Rationale\*\*: (.*?)\r?\n', block)
                reason = reason_match.group(1).strip() if reason_match else ""
                
                # 提取 Global Unsolved Queue
                queue_match = re.search(r'- \*\*Global Unsolved Queue\*\*: `(.*?)`', block)
                if queue_match:
                    try:
                        # 使用 json.loads 替代 eval 更安全
                        raw_queue = queue_match.group(1).replace("'", '"')
                        cumulative_route["global_unsolved_queue"] = json.loads(raw_queue)
                    except:
                        pass

                stage_data = {
                    "stage": stage_num,
                    "target": target,
                    "path_id": path_id, # Added path_id
                    "action": action,
                    "precursors": precursors,
                    "unsolved_leaves": precursors,
                    "reaction_type": reaction_type,
                    "reason": reason
                }
                
                cumulative_route["stages"].append(stage_data)
                
            logger.info(f"Reconstructed {len(cumulative_route['stages'])} stages")
            return cumulative_route
            
        except Exception as e:
            logger.error(f"State reconstruction failed: {e}")
            return {}

    def log_session_summary(self, cumulative_route: Dict, image_path: str = None):
        """Record session summary (Final Report)"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("## 🏁 Session Summary\n\n")
            f.write(f"- **End Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 1. 完整逆合成路线图 (Image)
            if image_path:
                # 使用相对路径以确保可移植性
                rel_path = os.path.basename(image_path)
                # 假设 image_path 在 output/agent_stages/ 下，而 log 在 output/ 下
                # 这里的路径处理需要根据实际目录结构调整
                # 简单起见，直接引用绝对路径或相对路径
                f.write(f"\n### 🌳 Retrosynthesis Tree\n\n")
                f.write(f"![Retrosynthesis Tree]({image_path})\n")
            
            # 2. 完整路线数据 (SMILES)
            f.write(f"\n### 🧬 Cumulative Route Data\n\n")
            f.write("```json\n")
            f.write(json.dumps(cumulative_route, indent=2, ensure_ascii=False))
            f.write("\n```\n")
            
            f.write("\n---\n*Session Closed.*\n")
