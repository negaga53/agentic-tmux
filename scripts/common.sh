#!/usr/bin/env bash
# Common utilities for agentic hooks
# Source this file in other hook scripts

set -euo pipefail

# Redis configuration
REDIS_HOST="${AGENTIC_REDIS_HOST:-localhost}"
REDIS_PORT="${AGENTIC_REDIS_PORT:-6379}"
REDIS_DB="${AGENTIC_REDIS_DB:-0}"

# Session/Agent info from environment
SESSION_ID="${AGENTIC_SESSION_ID:-}"
AGENT_ID="${AGENTIC_AGENT_ID:-}"
AGENT_ROLE="${AGENTIC_AGENT_ROLE:-}"
PANE_ID="${AGENTIC_PANE_ID:-${TMUX_PANE:-}}"

# Redis CLI wrapper
redis_cmd() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" "$@"
}

# Get a key
redis_get() {
    redis_cmd GET "$1"
}

# Set a key
redis_set() {
    redis_cmd SET "$1" "$2"
}

# Hash operations
redis_hget() {
    redis_cmd HGET "$1" "$2"
}

redis_hset() {
    redis_cmd HSET "$1" "$2" "$3"
}

# List operations
redis_lpush() {
    redis_cmd LPUSH "$1" "$2"
}

redis_brpop() {
    redis_cmd --no-raw BRPOP "$1" "$2"
}

# Stream operations
redis_xadd() {
    local key="$1"
    shift
    redis_cmd XADD "$key" "*" "$@"
}

# Key builders
session_key() {
    echo "session:${SESSION_ID}:$1"
}

agent_key() {
    echo "agent:${SESSION_ID}:${AGENT_ID}:$1"
}

# Update heartbeat
update_heartbeat() {
    if [[ -n "$SESSION_ID" && -n "$AGENT_ID" ]]; then
        redis_set "$(agent_key heartbeat)" "$(date +%s.%N)"
    fi
}

# Set agent status
set_status() {
    local status="$1"
    if [[ -n "$SESSION_ID" && -n "$AGENT_ID" ]]; then
        redis_set "$(agent_key status)" "$status"
    fi
}

# Log an action
log_action() {
    local action="$1"
    local file="${2:-}"
    local tool="${3:-}"
    
    if [[ -n "$SESSION_ID" && -n "$AGENT_ID" ]]; then
        redis_xadd "$(agent_key log)" \
            "action" "$action" \
            "file" "$file" \
            "tool" "$tool" \
            "timestamp" "$(date +%s.%N)"
    fi
}

# Push task to another agent
push_task() {
    local target_agent="$1"
    local task_json="$2"
    
    if [[ -n "$SESSION_ID" ]]; then
        redis_lpush "agent:${SESSION_ID}:${target_agent}:queue" "$task_json"
    fi
}

# Pop task from queue (blocking)
pop_task() {
    local timeout="${1:-5}"
    
    if [[ -n "$SESSION_ID" && -n "$AGENT_ID" ]]; then
        redis_brpop "$(agent_key queue)" "$timeout"
    fi
}

# Check if session is done
is_session_done() {
    if [[ -n "$SESSION_ID" ]]; then
        local done=$(redis_get "$(session_key done)")
        [[ "$done" == "1" ]]
    else
        return 1
    fi
}

# Send message to admin
send_to_admin() {
    local message="$1"
    
    if [[ -n "$SESSION_ID" ]]; then
        redis_lpush "bus:${SESSION_ID}:admin" "$message"
    fi
}

# Parse JSON from stdin (for hook data)
parse_hook_input() {
    cat
}

# Extract field from JSON
json_field() {
    local json="$1"
    local field="$2"
    echo "$json" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('$field', ''))" 2>/dev/null || echo ""
}

# Check if a file path matches the agent's scope
check_scope() {
    local file_path="$1"
    local scope_patterns="${AGENTIC_SCOPE_PATTERNS:-**/*}"
    
    # Simple glob matching
    for pattern in $scope_patterns; do
        if [[ "$file_path" == $pattern ]]; then
            return 0
        fi
    done
    
    return 1
}

# Deny a tool use (return JSON to stdout)
deny_tool() {
    local reason="$1"
    echo "{\"permissionDecision\": \"deny\", \"permissionDecisionReason\": \"$reason\"}"
}

# Allow a tool use (return empty or nothing)
allow_tool() {
    echo ""
}

# Log with prefix
log_info() {
    echo "[agentic:${AGENT_ID:-admin}] $*" >&2
}

log_error() {
    echo "[agentic:${AGENT_ID:-admin}] ERROR: $*" >&2
}
