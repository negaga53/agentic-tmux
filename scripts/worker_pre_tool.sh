#!/usr/bin/env bash
# Worker pre-tool hook
# Called before worker pane runs a tool
# Can deny tool execution if file is outside scope

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Read hook input
HOOK_DATA=$(parse_hook_input)

TOOL_NAME=$(json_field "$HOOK_DATA" "tool")

# Extract file path from tool input (varies by tool)
FILE_PATH=$(json_field "$HOOK_DATA" "input.path")
if [[ -z "$FILE_PATH" ]]; then
    FILE_PATH=$(json_field "$HOOK_DATA" "input.filePath")
fi
if [[ -z "$FILE_PATH" ]]; then
    FILE_PATH=$(json_field "$HOOK_DATA" "input.file")
fi

# Update heartbeat
update_heartbeat

# Update status to working
set_status "working"

# If there's a file path, check it against scope
if [[ -n "$FILE_PATH" ]]; then
    # Get scope patterns from Redis
    SCOPE_JSON=$(redis_hget "$(agent_key config)" "scope")
    
    if [[ -n "$SCOPE_JSON" ]]; then
        # Extract patterns from JSON
        PATTERNS=$(echo "$SCOPE_JSON" | python3 -c "
import sys, json, fnmatch
data = json.load(sys.stdin)
patterns = data.get('patterns', ['**/*'])
read_only = data.get('read_only', False)
file_path = '$FILE_PATH'
tool = '$TOOL_NAME'

# Check if file matches any pattern
matched = False
for pattern in patterns:
    if fnmatch.fnmatch(file_path, pattern):
        matched = True
        break

if not matched:
    print('DENY:File outside agent scope')
elif read_only and tool in ['write_file', 'edit_file', 'replace_string_in_file', 'create_file']:
    print('DENY:Agent has read-only access')
else:
    print('ALLOW')
" 2>/dev/null || echo "ALLOW")
        
        if [[ "$PATTERNS" == DENY:* ]]; then
            REASON="${PATTERNS#DENY:}"
            log_action "denied_$TOOL_NAME" "$FILE_PATH" "$TOOL_NAME"
            log_error "Denied: $REASON for $FILE_PATH"
            deny_tool "$REASON: $FILE_PATH"
            exit 0
        fi
    fi
    
    log_action "accessing_file" "$FILE_PATH" "$TOOL_NAME"
fi

# Log tool start
log_action "tool_start_$TOOL_NAME" "$FILE_PATH" "$TOOL_NAME"

# Allow the tool to proceed (no output)
