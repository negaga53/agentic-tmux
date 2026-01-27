#!/usr/bin/env bash
# Worker session end hook
# Called when worker pane CLI session ends

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Read hook input
HOOK_DATA=$(parse_hook_input)

log_info "Worker session ending: $AGENT_ROLE"

# Check if there are pending tasks
if [[ -n "$SESSION_ID" && -n "$AGENT_ID" ]]; then
    QUEUE_LEN=$(redis_cmd LLEN "$(agent_key queue)" 2>/dev/null || echo "0")
    
    if [[ "$QUEUE_LEN" -gt 0 ]]; then
        # There are pending tasks - notify admin
        log_info "Session ending with $QUEUE_LEN pending tasks"
        
        send_to_admin "{
            \"type\": \"agent_session_end_with_tasks\",
            \"agent_id\": \"$AGENT_ID\",
            \"pending_tasks\": $QUEUE_LEN,
            \"timestamp\": $(date +%s)
        }"
    fi
    
    # Check if session is done
    if is_session_done; then
        log_info "Session marked as done, exiting cleanly"
        set_status "done"
    else
        # Session not done but worker ending - might need respawn
        set_status "ended"
        
        send_to_admin "{
            \"type\": \"agent_session_ended\",
            \"agent_id\": \"$AGENT_ID\",
            \"timestamp\": $(date +%s)
        }"
    fi
    
    # Final log entry
    log_action "session_ended" "" ""
fi

# No output needed
