#!/bin/bash
# Analyze agent behavior from hook logs
# Usage: ./analyze-agent-logs.sh [log_dir]

LOGDIR="${1:-$HOME/.config/agentic/logs}"

if [ ! -d "$LOGDIR" ]; then
    echo "Log directory not found: $LOGDIR"
    echo "Run an agent session first to generate logs."
    exit 1
fi

echo "========================================"
echo "AGENT BEHAVIOR ANALYSIS"
echo "Log directory: $LOGDIR"
echo "========================================"
echo ""

# Check if timeline exists
if [ ! -f "$LOGDIR/agent-timeline.jsonl" ]; then
    echo "No timeline file found. Run an agent session first."
    exit 1
fi

# Event counts
echo "Event Summary:"
echo "--------------"
python3 -c "
import json, sys
from collections import Counter
events = Counter()
for line in open('$LOGDIR/agent-timeline.jsonl'):
    try:
        events[json.loads(line).get('event', '')] += 1
    except: pass
for event, count in sorted(events.items(), key=lambda x: -x[1]):
    print(f'  {count:4d} {event}')
" 2>/dev/null

echo ""
echo "MCP Tool Calls:"
echo "---------------"
MCP_CALLS=$(python3 -c "
import json, sys
from collections import Counter
tools = Counter()
for line in open('$LOGDIR/agent-timeline.jsonl'):
    try:
        d = json.loads(line)
        if d.get('isMcpTool') == True and d.get('toolName'):
            tools[d['toolName']] += 1
    except: pass
for tool, count in sorted(tools.items(), key=lambda x: -x[1]):
    print(f'  {count:4d} {tool}')
" 2>/dev/null)
if [ -n "$MCP_CALLS" ]; then
    echo "$MCP_CALLS"
else
    echo "  [PROBLEM] No MCP tools were called!"
    echo "  Agent ignored communication protocol."
fi

echo ""
echo "Non-MCP Tool Calls:"
echo "-------------------"
python3 -c "
import json, sys
from collections import Counter
tools = Counter()
for line in open('$LOGDIR/agent-timeline.jsonl'):
    try:
        d = json.loads(line)
        if d.get('event') == 'PRE_TOOL_USE' and d.get('isMcpTool') == False and d.get('toolName'):
            tools[d['toolName']] += 1
    except: pass
for tool, count in sorted(tools.items(), key=lambda x: -x[1]):
    print(f'  {count:4d} {tool}')
" 2>/dev/null

echo ""
echo "Protocol Compliance Check:"
echo "--------------------------"
LIST_AGENTS=$(grep -c '"list_agents"' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null || echo 0)
SEND_TO=$(grep -c '"send_to_agent"' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null || echo 0)
RECEIVE=$(grep -c '"receive_message"' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null || echo 0)

if [ "$LIST_AGENTS" -gt 0 ]; then
    echo "  [OK] list_agents() called ($LIST_AGENTS times)"
else
    echo "  [FAIL] list_agents() was NEVER called (Phase 1 skipped)"
fi

if [ "$SEND_TO" -gt 0 ]; then
    echo "  [OK] send_to_agent() called ($SEND_TO times)"
else
    echo "  [FAIL] send_to_agent() was NEVER called (Phase 3 skipped - results lost!)"
fi

if [ "$RECEIVE" -gt 0 ]; then
    echo "  [OK] receive_message() called ($RECEIVE times)"
else
    echo "  [FAIL] receive_message() was NEVER called (Phase 4 skipped - no polling loop)"
fi

echo ""
echo "Errors:"
echo "-------"
ERROR_COUNT=$(grep -c '"ERROR_OCCURRED"' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null || echo 0)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "  $ERROR_COUNT errors occurred:"
    python3 -c "
import json
for line in open('$LOGDIR/agent-timeline.jsonl'):
    try:
        d = json.loads(line)
        if d.get('event') == 'ERROR_OCCURRED':
            print(f\"  - {d.get('errorName', 'Unknown')}: {d.get('errorMessage', 'No message')}\")
    except: pass
" 2>/dev/null
else
    echo "  No errors recorded"
fi

echo ""
echo "Timeline (last 20 events):"
echo "--------------------------"
python3 -c "
import json
lines = open('$LOGDIR/agent-timeline.jsonl').readlines()
for line in lines[-20:]:
    try:
        d = json.loads(line)
        event = d.get('event', '')
        detail = d.get('toolName') or d.get('reason') or ''
        if detail:
            print(f'  {event}: {detail}')
        else:
            print(f'  {event}')
    except: pass
" 2>/dev/null

echo ""
echo "========================================"
echo "Full logs: $LOGDIR/agent-debug.log"
echo "========================================"
