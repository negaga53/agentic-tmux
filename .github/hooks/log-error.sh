#!/usr/bin/env python3
"""Log errors during agent execution"""
import json
import os
import re
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
error = data.get("error", {})
if isinstance(error, dict):
    error_msg = error.get("message", "unknown error")
    error_name = error.get("name", "Error")
    error_stack = error.get("stack", "no stack trace")
else:
    error_msg = str(error)
    error_name = "Error"
    error_stack = "no stack trace"

session_file = LOGDIR / ".current_session"
session_id = session_file.read_text().strip() if session_file.exists() else "unknown"

# Write structured log entry
entry = {
    "event": "ERROR_OCCURRED",
    "timestamp": timestamp,
    "sessionId": session_id,
    "errorName": error_name,
    "errorMessage": error_msg,
    "errorStack": error_stack
}
with open(LOGDIR / "agent-timeline.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

# Write human-readable log
with open(LOGDIR / "agent-debug.log", "a") as f:
    f.write(f"\n!!! ERROR @ {datetime.now()} !!!\n")
    f.write(f"Type: {error_name}\n")
    f.write(f"Message: {error_msg}\n")
    f.write(f"Stack: {error_stack}\n")

# Check if MCP-related error
if re.search(r'mcp|agentic|connection|timeout|worker', f"{error_msg}{error_stack}", re.I):
    with open(LOGDIR / "mcp-errors.log", "a") as f:
        f.write(f"[POSSIBLE MCP ERROR]\n")
        f.write(f"  Type: {error_name}\n")
        f.write(f"  Message: {error_msg}\n")
