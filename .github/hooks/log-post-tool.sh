#!/usr/bin/env python3
"""Log tool results AFTER execution"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

LOGDIR = Path(os.environ.get("AGENTIC_LOG_DIR", Path.home() / ".config" / "agentic" / "logs"))
LOGDIR.mkdir(parents=True, exist_ok=True)

try:
    data = json.loads(sys.stdin.read())
except:
    data = {}

timestamp = data.get("timestamp", "")
tool_name = data.get("toolName", "")
tool_args = json.dumps(data.get("toolArgs", ""))
tool_result = data.get("toolResult", {})
result_type = tool_result.get("resultType", "unknown") if isinstance(tool_result, dict) else "unknown"
result_text = tool_result.get("textResultForLlm", str(tool_result)) if isinstance(tool_result, dict) else str(tool_result)

session_file = LOGDIR / ".current_session"
session_id = session_file.read_text().strip() if session_file.exists() else "unknown"

# Detect if this is an MCP tool
is_mcp = tool_name.startswith("mcp_")

# Write structured log entry
entry = {
    "event": "POST_TOOL_USE",
    "timestamp": timestamp,
    "sessionId": session_id,
    "toolName": tool_name,
    "toolArgs": tool_args,
    "resultType": result_type,
    "resultText": result_text[:1000],
    "isMcpTool": is_mcp
}
with open(LOGDIR / "agent-timeline.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

# Write human-readable log
with open(LOGDIR / "agent-debug.log", "a") as f:
    f.write(f"\n--- POST-TOOL @ {datetime.now()} ---\n")
    f.write(f"Tool: {tool_name} (MCP: {is_mcp})\n")
    f.write(f"Result Type: {result_type}\n")
    f.write(f"Result: {result_text[:500]}\n")

# Alert on MCP failures
if is_mcp:
    if result_type not in ("success", "unknown"):
        with open(LOGDIR / "agent-debug.log", "a") as f:
            f.write(f"[MCP FAILURE] {tool_name}\n")
            f.write(f"  Result Type: {result_type}\n")
            f.write(f"  Output: {result_text[:300]}\n")
        with open(LOGDIR / "mcp-failures.csv", "a") as f:
            f.write(f"{timestamp},{tool_name},{result_type},{result_text[:200]}\n")
    else:
        with open(LOGDIR / "agent-debug.log", "a") as f:
            f.write(f"[MCP SUCCESS] {tool_name} completed\n")
