# Agentic TMUX

Multi-agent orchestration for CLI coding assistants via tmux panes.

## Overview

Agentic TMUX allows you to spawn multiple AI coding agents (GitHub Copilot CLI, Claude, etc.) in separate tmux panes that can:

- Execute tasks in parallel
- Communicate with each other via Redis message queues
- Stay within defined file scopes
- Report progress to a central admin pane

## Features

- **Multi-agent orchestration** - Spawn and coordinate multiple AI agents
- **Interactive planning** - LLM generates task DAG, you approve/modify
- **File scope enforcement** - Pre-hooks validate file access per agent
- **Real-time communication** - Redis streams for agent-to-agent messaging
- **Deadlock prevention** - Orchestrator daemon monitors for circular waits
- **Session resume** - Reuse existing agents for new prompts
- **Failure recovery** - Automatic retry with exponential backoff

## Prerequisites

- Python 3.11+
- tmux
- Redis server
- GitHub Copilot CLI (`gh copilot`) or Claude CLI

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

1. **Start Redis** (if not already running):
   ```bash
   redis-server
   ```

2. **Start a session**:
   ```bash
   agentic start --working-dir /path/to/your/project
   ```

3. **Create a plan**:
   ```bash
   agentic plan "Refactor the auth module and add comprehensive tests"
   ```

4. **Monitor progress**:
   ```bash
   agentic status --watch
   ```

5. **Stop when done**:
   ```bash
   agentic stop
   ```

## CLI Commands

| Command | Description |
|---------|-------------|
| `agentic start` | Start a new session |
| `agentic stop` | Stop the current session |
| `agentic plan "prompt"` | Create and execute an execution plan |
| `agentic status` | Show status of all agents |
| `agentic logs <agent_id>` | View logs for an agent |
| `agentic send <agent_id> "task"` | Send a task to an agent |
| `agentic resume` | Resume with existing agents |
| `agentic clear` | Clear all workers |
| `agentic export` | Export session transcript |
| `agentic init` | Initialize hooks in current repo |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTIC_REDIS_HOST` | localhost | Redis host |
| `AGENTIC_REDIS_PORT` | 6379 | Redis port |
| `AGENTIC_REDIS_DB` | 0 | Redis database |
| `OPENAI_API_KEY` | - | OpenAI API key for LLM planning |

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
│                              TMUX SESSION                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │   ADMIN PANE    │   │  WORKER PANE 1  │   │  WORKER PANE 2  │  ...  │
│  │                 │   │                 │   │                 │       │
│  │  gh copilot     │   │  gh copilot     │   │  gh copilot     │       │
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
