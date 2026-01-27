#!/usr/bin/env bash
# Worker session start hook
# Called when worker pane CLI session starts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Read hook input
HOOK_DATA=$(parse_hook_input)

log_info "Worker session started: $AGENT_ROLE"

# Register with Redis
if [[ -n "$SESSION_ID" && -n "$AGENT_ID" ]]; then
    # Update status
    set_status "idle"
    
    # Initial heartbeat
    update_heartbeat
    
    # Log start
    log_action "session_started" "" ""
fi

# No output needed - allow session to proceed
