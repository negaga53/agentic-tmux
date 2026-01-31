#!/usr/bin/env python3
"""Log session start events for debugging agent behavior"""
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
cwd = data.get("cwd", "")
source = data.get("source", "unknown")

# Generate session ID from timestamp
import time
session_id = f"session_{int(time.time())}"
(LOGDIR / ".current_session").write_text(session_id)

# Write structured log entry
entry = {
    "event": "SESSION_START",
    "timestamp": timestamp,
    "sessionId": session_id,
    "cwd": cwd,
    "source": source
}
with open(LOGDIR / "agent-timeline.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

# Write human-readable log
with open(LOGDIR / "agent-debug.log", "a") as f:
    f.write(f"\n========================================\n")
    f.write(f"SESSION START: {session_id}\n")
    f.write(f"Time: {datetime.now()}\n")
    f.write(f"CWD: {cwd}\n")
    f.write(f"Source: {source}\n")
    f.write(f"========================================\n")
