#!/usr/bin/env bash
# Admin post-tool hook
# Called after admin pane runs a tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Read hook input
HOOK_DATA=$(parse_hook_input)

TOOL_NAME=$(json_field "$HOOK_DATA" "tool")
RESULT=$(json_field "$HOOK_DATA" "result")

# Log the action
log_action "admin_tool_$TOOL_NAME" "" "$TOOL_NAME"

# Check if we need to dispatch more tasks
# This could trigger when a review is complete, etc.
