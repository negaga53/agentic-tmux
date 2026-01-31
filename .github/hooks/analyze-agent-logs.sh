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
jq -r '.event' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | sort | uniq -c | sort -rn

echo ""
echo "MCP Tool Calls:"
echo "---------------"
MCP_CALLS=$(jq -r 'select(.isMcpTool == true) | .toolName' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | sort | uniq -c | sort -rn)
if [ -n "$MCP_CALLS" ]; then
    echo "$MCP_CALLS"
else
    echo "  [PROBLEM] No MCP tools were called!"
    echo "  Agent ignored communication protocol."
fi

echo ""
echo "Non-MCP Tool Calls:"
echo "-------------------"
jq -r 'select(.event == "PRE_TOOL_USE" and .isMcpTool == false) | .toolName' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | sort | uniq -c | sort -rn

echo ""
echo "Protocol Compliance Check:"
echo "--------------------------"
LIST_AGENTS=$(jq -r 'select(.toolName | test("list_agents")) | .toolName' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | wc -l)
SEND_TO=$(jq -r 'select(.toolName | test("send_to_agent")) | .toolName' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | wc -l)
RECEIVE=$(jq -r 'select(.toolName | test("receive_message")) | .toolName' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | wc -l)

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
ERROR_COUNT=$(jq -r 'select(.event == "ERROR_OCCURRED")' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "  $ERROR_COUNT errors occurred:"
    jq -r 'select(.event == "ERROR_OCCURRED") | "  - \(.errorName): \(.errorMessage)"' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null
else
    echo "  No errors recorded"
fi

echo ""
echo "Timeline (last 20 events):"
echo "--------------------------"
jq -r '[.event, .toolName // .reason // ""] | @tsv' "$LOGDIR/agent-timeline.jsonl" 2>/dev/null | tail -20 | while read event tool; do
    if [ -n "$tool" ]; then
        echo "  $event: $tool"
    else
        echo "  $event"
    fi
done

echo ""
echo "========================================"
echo "Full logs: $LOGDIR/agent-debug.log"
echo "========================================"
