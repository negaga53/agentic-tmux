#!/usr/bin/env python3
"""Log session end and generate summary report"""
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LOGDIR = Path(os.environ.get("AGENTIC_LOG_DIR", Path.home() / ".config" / "agentic" / "logs"))
LOGDIR.mkdir(parents=True, exist_ok=True)

try:
    data = json.loads(sys.stdin.read())
except:
    data = {}

timestamp = data.get("timestamp", "")
reason = data.get("reason", "unknown")

session_file = LOGDIR / ".current_session"
session_id = session_file.read_text().strip() if session_file.exists() else "unknown"

# Write structured log entry
entry = {
    "event": "SESSION_END",
    "timestamp": timestamp,
    "sessionId": session_id,
    "reason": reason
}
with open(LOGDIR / "agent-timeline.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

# Generate session summary
mcp_calls_file = LOGDIR / "mcp-calls.csv"
mcp_failures_file = LOGDIR / "mcp-failures.csv"
timeline_file = LOGDIR / "agent-timeline.jsonl"

with open(LOGDIR / "agent-debug.log", "a") as f:
    f.write(f"\n========================================\n")
    f.write(f"SESSION END: {session_id}\n")
    f.write(f"Time: {datetime.now()}\n")
    f.write(f"Reason: {reason}\n")
    f.write(f"========================================\n\n")
    f.write(f"--- SESSION SUMMARY ---\n")
    
    # Count MCP tool calls
    if mcp_calls_file.exists():
        lines = mcp_calls_file.read_text().strip().split("\n")
        mcp_count = len([l for l in lines if l])
        f.write(f"MCP Tool Calls: {mcp_count}\n\n")
        f.write(f"MCP Tools Used:\n")
        
        tools = [line.split(",")[1] for line in lines if line and "," in line]
        for tool, count in Counter(tools).most_common():
            f.write(f"  {count:3d} {tool}\n")
        f.write("\n")
        
        # Check for specific tools
        tool_set = set(tools)
        if any("list_agents" in t for t in tool_set):
            f.write("[OK] list_agents() was called\n")
        else:
            f.write("[WARNING] list_agents() was NEVER called!\n")
        
        if any("send_to_agent" in t for t in tool_set):
            f.write("[OK] send_to_agent() was called\n")
        else:
            f.write("[WARNING] send_to_agent() was NEVER called - results not reported!\n")
        
        if any("receive_message" in t for t in tool_set):
            f.write("[OK] receive_message() was called\n")
        else:
            f.write("[WARNING] receive_message() was NEVER called - not in polling loop!\n")
    else:
        f.write("MCP Tool Calls: 0\n\n")
        f.write("[CRITICAL] NO MCP TOOLS WERE CALLED THIS SESSION!\n")
        f.write("Agent completely ignored communication protocol.\n")
    
    # Count failures
    if mcp_failures_file.exists():
        fail_lines = mcp_failures_file.read_text().strip().split("\n")
        fail_count = len([l for l in fail_lines if l])
        f.write(f"\nMCP Failures: {fail_count}\n")
        if fail_count > 0:
            f.write("Failed tools:\n")
            fail_tools = [line.split(",")[1] for line in fail_lines if line and "," in line]
            for tool, count in Counter(fail_tools).most_common():
                f.write(f"  {count:3d} {tool}\n")
    
    # Count total tool calls
    total_tools = 0
    error_count = 0
    if timeline_file.exists():
        for line in timeline_file.read_text().strip().split("\n"):
            if line:
                try:
                    ev = json.loads(line)
                    if ev.get("event") == "PRE_TOOL_USE":
                        total_tools += 1
                    elif ev.get("event") == "ERROR_OCCURRED":
                        error_count += 1
                except:
                    pass
    
    f.write(f"\nTotal Tool Calls: {total_tools}\n")
    f.write(f"Errors: {error_count}\n\n")
    f.write(f"Full logs: {LOGDIR}/agent-debug.log\n")
    f.write(f"Timeline: {LOGDIR}/agent-timeline.jsonl\n")
    f.write(f"========================================\n")

# Also write summary to separate file for easy access
with open(LOGDIR / "last-session-summary.txt", "w") as f:
    f.write(f"Session: {session_id}\n")
    f.write(f"End Time: {datetime.now()}\n")
    f.write(f"Reason: {reason}\n")
    if mcp_calls_file.exists():
        lines = mcp_calls_file.read_text().strip().split("\n")
        f.write(f"MCP Calls: {len([l for l in lines if l])}\n")
    else:
        f.write(f"MCP Calls: 0 [PROBLEM!]\n")

# Clean up per-session files for next session
for cleanup_file in [mcp_calls_file, mcp_failures_file]:
    if cleanup_file.exists():
        cleanup_file.unlink()
