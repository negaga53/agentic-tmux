#!/usr/bin/env bash
# Admin session start hook
# Called when admin pane CLI session starts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Read hook input
HOOK_DATA=$(parse_hook_input)

log_info "Admin session started"

# Initialize session state if needed
if [[ -n "$SESSION_ID" ]]; then
    redis_hset "$(session_key config)" "admin_started" "$(date +%s)"
fi

# No output needed - allow session to proceed
