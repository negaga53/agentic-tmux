#!/usr/bin/env bash
# Worker post-tool hook  
# Called after worker pane runs a tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Read hook input
HOOK_DATA=$(parse_hook_input)

TOOL_NAME=$(json_field "$HOOK_DATA" "tool")
RESULT=$(json_field "$HOOK_DATA" "result")

# Extract file path from tool input
FILE_PATH=$(json_field "$HOOK_DATA" "input.path")
if [[ -z "$FILE_PATH" ]]; then
    FILE_PATH=$(json_field "$HOOK_DATA" "input.filePath")
fi

# Update heartbeat
update_heartbeat

# Log tool completion
log_action "tool_complete_$TOOL_NAME" "$FILE_PATH" "$TOOL_NAME"

# Publish action summary to agent's log stream for other agents to read
if [[ -n "$FILE_PATH" ]]; then
    redis_xadd "$(agent_key log)" \
        "action" "modified" \
        "file" "$FILE_PATH" \
        "tool" "$TOOL_NAME" \
        "timestamp" "$(date +%s.%N)"
fi

# After tool completes, check for pending tasks in queue
# This implements the task polling behavior

# Non-blocking check for tasks
QUEUE_LEN=$(redis_cmd LLEN "$(agent_key queue)" 2>/dev/null || echo "0")

if [[ "$QUEUE_LEN" -gt 0 ]]; then
    log_info "Tasks waiting in queue: $QUEUE_LEN"
fi

# Set status back to idle (will be set to working again on next tool)
set_status "idle"

# No output needed
