#!/usr/bin/env bash
# Worker task polling script
# Can be called to poll for new tasks from the queue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TIMEOUT="${1:-5}"

log_info "Polling for tasks (timeout: ${TIMEOUT}s)"

# Update heartbeat
update_heartbeat

# Check if session is done
if is_session_done; then
    log_info "Session is done, no more tasks"
    echo '{"status": "done"}'
    exit 0
fi

# Blocking pop from queue
TASK_DATA=$(pop_task "$TIMEOUT")

if [[ -z "$TASK_DATA" ]]; then
    log_info "No tasks available"
    echo '{"status": "empty"}'
    exit 0
fi

# Parse the task (BRPOP returns "key" "value")
# Extract just the JSON value
TASK_JSON=$(echo "$TASK_DATA" | python3 -c "
import sys, json
data = sys.stdin.read()
# BRPOP returns: ['key', 'json_string']
try:
    parts = data.strip().split('\n')
    if len(parts) >= 2:
        # The JSON is the second part
        print(parts[1])
    else:
        print(data)
except:
    print(data)
" 2>/dev/null || echo "$TASK_DATA")

# Check for done signal
if echo "$TASK_JSON" | grep -q '"task":"done"'; then
    log_info "Received done signal"
    set_status "done"
    echo '{"status": "done"}'
    exit 0
fi

# Log task receipt
TASK_TITLE=$(json_field "$TASK_JSON" "task")
FROM_AGENT=$(json_field "$TASK_JSON" "from")

log_info "Received task '$TASK_TITLE' from $FROM_AGENT"
log_action "task_received" "" ""

# Update status
set_status "working"

# Output the task for the CLI to process
echo "$TASK_JSON"
