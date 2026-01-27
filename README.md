# Agentic TMUX

Multi-agent orchestration for CLI coding assistants via tmux panes.

## Overview

Agentic TMUX allows you to spawn multiple AI coding agents (GitHub Copilot CLI, Claude, etc.) in separate tmux panes that can:

- Execute tasks in parallel
- Communicate with each other via message queues
- Stay within defined file scopes
- Report progress to a central admin pane

**No external dependencies required** - runs with SQLite storage by default (for cross-process communication), or use Redis for additional features.

## Features

- **Multi-agent orchestration** - Spawn and coordinate multiple AI agents
- **Interactive planning** - LLM generates task DAG, you approve/modify
- **File scope enforcement** - Pre-hooks validate file access per agent
- **Real-time communication** - Message queues for agent-to-agent messaging
- **Deadlock prevention** - Orchestrator daemon monitors for circular waits
- **Session resume** - Reuse existing agents for new prompts
- **Failure recovery** - Automatic retry with exponential backoff
- **MCP Server interface** - Integrate with VS Code, Claude Desktop, or any MCP client
- **Optional Redis** - Works with in-memory storage or Redis for persistence

## Prerequisites

- Python 3.11+
- tmux
- GitHub Copilot CLI (`copilot`) or Claude CLI
- Redis server (optional, for persistence)

## Installation

```bash
pip install agentic-tmux
```

Or from source:

```bash
git clone https://github.com/agentic-cli/agentic-tmux
cd agentic-tmux
pip install -e .
```

## Quick Start

### Using MCP (Recommended)

The MCP server is the primary interface. Configure it in your MCP client:

1. **Add to VS Code** (`.vscode/mcp.json`):
   ```json
   {
     "servers": {
       "agentic": {
         "command": "agentic-mcp"
       }
     }
   }
   ```

2. **Use MCP tools from your AI assistant**:
   - `plan_tasks("Refactor auth module")` - Auto-starts session
   - `create_plan(...)` - Create plan from your analysis
   - `execute_plan(plan_id)` - Spawn agents and dispatch tasks
   - `get_status()` - Monitor progress
   - `stop_session()` - Clean up when done

### Manual CLI

For debugging or manual control:

1. **Start Redis** (optional):
   ```bash
   redis-server
   ```
   *Without Redis, data is stored in SQLite (`~/.config/agentic/agentic.db`) which persists across restarts.*

2. **Monitor with CLI** (after MCP creates session):
   ```bash
   agentic status --watch
   ```

3. **View agent logs**:
   ```bash
   agentic logs W1 -f
   ```

## CLI Commands

The CLI is primarily for debugging and monitoring. Use MCP for orchestration.

| Command | Description |
|---------|-------------|
| `agentic mcp` | **Start MCP server** (primary interface) |
| `agentic status` | Show status of all agents |
| `agentic logs <agent_id>` | View logs for an agent |
| `agentic send <agent_id> "task"` | Send a task to an agent |
| `agentic stop` | Stop the current session |
| `agentic clear` | Clear all workers |
| `agentic export` | Export session transcript |
| `agentic init` | Initialize hooks in current repo |

## MCP Server

Agentic TMUX can be used as a Model Context Protocol (MCP) server, allowing integration with VS Code, Claude Desktop, and other MCP-compatible clients.

### VS Code Integration

Add to your VS Code settings (`.vscode/mcp.json`):

```json
{
  "servers": {
    "agentic": {
      "command": "agentic-mcp",
      "env": {
        "AGENTIC_REDIS_HOST": "localhost"
      }
    }
  }
}
```

Or start manually:

```bash
agentic mcp
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agentic": {
      "command": "agentic-mcp",
      "env": {
        "AGENTIC_REDIS_HOST": "localhost"
      }
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `plan_tasks` | Get context and template for planning (auto-starts session) |
| `create_plan` | Create execution plan from structured input |
| `execute_plan` | Execute an approved plan |
| `resume_session` | Resume an existing session |
| `stop_session` | Stop the current session |
| `spawn_agent` | Spawn a single new agent |
| `send_task` | Send a task to a specific agent |
| `get_status` | Get status of all agents and tasks |
| `get_agent_logs` | Get logs for a specific agent |
| `clear_agents` | Clear all workers, keep session |
| `init_hooks` | Initialize hooks in a repository |

### MCP Resources

| Resource | Description |
|----------|-------------|
| `session://status` | Current session state (JSON) |
| `dag://current` | Task DAG visualization |
| `agents://list` | List of all agents |

### MCP Prompts

| Prompt | Description |
|--------|-------------|
| `orchestrate_task` | Guide for orchestrating a multi-agent task |
| `review_agent_work` | Guide for reviewing an agent's work |

### Example MCP Workflow

From within an MCP client (like VS Code Copilot or Claude Desktop):

```
1. plan_tasks(prompt="Add authentication", working_dir="/path/to/project")
   → Auto-starts session, returns project context

2. create_plan(
     prompt="Add authentication",
     agents=[{id: "W1", role: "Developer", scope_patterns: ["src/**"]}],
     tasks=[{id: "t1", title: "Implement auth", agent_id: "W1", dependencies: []}]
   )
   → Returns plan_id

3. execute_plan(plan_id="...")
   → Spawns agents in tmux, dispatches tasks

4. get_status()
   → Monitor progress

5. stop_session()
   → Clean up when done
```

**Resume a session:**
```
resume_session()  → Returns session state and active agents
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTIC_REDIS_HOST` | localhost | Redis host (if using Redis) |
| `AGENTIC_REDIS_PORT` | 6379 | Redis port |
| `AGENTIC_REDIS_DB` | 0 | Redis database |

> **Note:** Redis environment variables are only used if Redis is available. Without Redis, data is stored in-memory.

### Hooks

Agentic uses hooks to intercept CLI events. Install hooks in your repo:

```bash
agentic init
```

This creates `.github/hooks/` with:
- `sessionStart.json` - Agent registration
- `preToolUse.json` - File scope validation
- `postToolUse.json` - Action logging
- `sessionEnd.json` - Cleanup

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MCP CLIENT (VS Code, Claude Desktop, etc.)       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ MCP Protocol (stdio)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MCP SERVER (agentic-mcp)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Tools: plan_tasks, execute_plan, get_status, send_task, ...            │
│  Resources: session://status, dag://current, agents://list              │
│  Prompts: orchestrate_task, review_agent_work                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              TMUX SESSION                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │   ADMIN PANE    │   │  WORKER PANE 1  │   │  WORKER PANE 2  │  ...  │
│  │                 │   │                 │   │                 │       │
│  │  copilot     │   │  copilot     │   │  copilot     │       │
│  │  + admin hooks  │   │  + worker hooks │   │  + worker hooks │       │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘       │
│           │                     │                     │                 │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              REDIS                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  session:{id}:config    │  Session state                                │
│  session:{id}:dag       │  Task dependency graph                        │
│  agent:{id}:queue       │  Per-agent task queue                         │
│  agent:{id}:status      │  Agent status                                 │
│  agent:{id}:log         │  Action stream                                │
└─────────────────────────────────────────────────────────────────────────┘
            ▲
            │
┌───────────┴───────────┐
│   ORCHESTRATOR DAEMON  │
├────────────────────────┤
│  - Heartbeat monitor   │
│  - Deadlock detection  │
│  - Task completion     │
└────────────────────────┘
```

## Example Plan Output

```
╭─────────────────────────────────────────────────────────────────────╮
│                     📋 EXECUTION PLAN                                │
│                                                                      │
│  Prompt: "Refactor auth module and add comprehensive tests"          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AGENTS TO SPAWN: 3                                                  │
│                                                                      │
│  ┌─────┬────────────────────┬─────────────────────────────────────┐ │
│  │ ID  │ Role               │ File Scope                          │ │
│  ├─────┼────────────────────┼─────────────────────────────────────┤ │
│  │ W1  │ Auth Refactorer    │ src/auth/**, src/utils/crypto.ts    │ │
│  │ W2  │ Test Author        │ tests/auth/**, tests/fixtures/**    │ │
│  │ W3  │ Code Reviewer      │ READ-ONLY: src/**, tests/**         │ │
│  └─────┴────────────────────┴─────────────────────────────────────┘ │
│                                                                      │
│  TASK FLOW:                                                          │
│                                                                      │
│      ┌──────────┐                                                    │
│      │ W1: Auth │                                                    │
│      │ Refactor │                                                    │
│      └────┬─────┘                                                    │
│           │                                                          │
│           ▼                                                          │
│      ┌──────────┐      ┌──────────┐                                 │
│      │ W2: Test │ ───► │ W3: Rev  │                                 │
│      └──────────┘      └──────────┘                                 │
│                                                                      │
╰─────────────────────────────────────────────────────────────────────╯
```

## Inter-Agent Communication

Agents communicate via Redis queues. To send a task to another agent:

```bash
# From within a hook or script
redis-cli LPUSH agent:$SESSION_ID:W2:queue '{"task":"review changes","from":"W1","files":["src/auth/login.ts"]}'
```

Agents poll their queues and process tasks sequentially. The `postToolUse` hook automatically checks for pending tasks.

## Deadlock Prevention

The orchestrator daemon runs checks every 30 seconds:

1. **Heartbeat monitoring** - Respawns panes that haven't sent a heartbeat in 2 minutes
2. **Circular wait detection** - Breaks deadlocks by unblocking one agent
3. **Task stuck detection** - Escalates if an agent is working on one task for too long

## Error Handling

When an agent fails:
1. Automatic retry with exponential backoff (5s → 15s → 45s)
2. After 3 failures, escalate to admin
3. Admin can choose to: retry with different model, skip task, or abort session

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black agentic/
ruff agentic/

# Type check
mypy agentic/
```

## License

MIT

## Credits

Inspired by discussions in the AI coding community about multi-agent orchestration patterns.
