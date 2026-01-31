#!/usr/bin/env python3
"""Log user prompts submitted to agents"""
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
prompt = data.get("prompt", "")

session_file = LOGDIR / ".current_session"
session_id = session_file.read_text().strip() if session_file.exists() else "unknown"

# Write structured log entry
entry = {
    "event": "PROMPT_SUBMITTED",
    "timestamp": timestamp,
    "sessionId": session_id,
    "prompt": prompt
}
with open(LOGDIR / "agent-timeline.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

# Write human-readable log
with open(LOGDIR / "agent-debug.log", "a") as f:
    f.write(f"\n--- PROMPT @ {datetime.now()} ---\n")
    f.write(f"{prompt}\n\n")

# Check if prompt mentions MCP tools - important for debugging
if re.search(r'list_agents|send_to_agent|receive_message|mcp|orchestrator', prompt, re.I):
    with open(LOGDIR / "agent-debug.log", "a") as f:
        f.write("[PROMPT CHECK] Prompt mentions MCP/communication tools - agent SHOULD call them\n")
