#!/usr/bin/env python3
"""Log tool calls BEFORE execution - critical for MCP debugging"""
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

session_file = LOGDIR / ".current_session"
session_id = session_file.read_text().strip() if session_file.exists() else "unknown"

# Detect if this is an MCP tool (prefixed with mcp_)
is_mcp = tool_name.startswith("mcp_")
mcp_server = ""
mcp_tool = ""
if is_mcp:
    parts = tool_name[4:].split("_", 1)  # Remove "mcp_" prefix and split
    mcp_server = parts[0] if parts else ""
    mcp_tool = parts[1] if len(parts) > 1 else ""

# Write structured log entry
entry = {
    "event": "PRE_TOOL_USE",
    "timestamp": timestamp,
    "sessionId": session_id,
    "toolName": tool_name,
    "toolArgs": tool_args,
    "isMcpTool": is_mcp,
    "mcpServer": mcp_server,
    "mcpTool": mcp_tool
}
with open(LOGDIR / "agent-timeline.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

# Write human-readable log
with open(LOGDIR / "agent-debug.log", "a") as f:
    f.write(f"\n+++ PRE-TOOL @ {datetime.now()} +++\n")
    f.write(f"Tool: {tool_name}\n")
    if is_mcp:
        f.write(f"*** MCP TOOL DETECTED ***\n")
        f.write(f"MCP Server: {mcp_server}\n")
        f.write(f"MCP Tool: {mcp_tool}\n")
    f.write(f"Args: {tool_args}\n")

# Track MCP tool calls specifically
if is_mcp:
    with open(LOGDIR / "mcp-calls.csv", "a") as f:
        f.write(f"{timestamp},{tool_name},{mcp_server},{mcp_tool}\n")
    with open(LOGDIR / "agent-debug.log", "a") as f:
        f.write(f"[MCP CALL] {tool_name} with args: {tool_args[:200]}\n")
