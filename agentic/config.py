"""Configuration and path management for agentic-tmux.

This module provides per-repository config directories (.agentic/) instead of
a centralized ~/.config/agentic/ folder. Each repo gets isolated storage.

Also provides shared session/storage helpers used across CLI, MCP server,
and monitor modules to avoid code duplication.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Default config folder name (hidden)
CONFIG_FOLDER_NAME = ".agentic"


def get_config_dir(working_dir: str | Path | None = None) -> Path:
    """Get the config directory for the given working directory.
    
    Args:
        working_dir: The working directory (repo root). If None, uses CWD.
        
    Returns:
        Path to the .agentic folder within the working directory.
    """
    if working_dir is None:
        working_dir = os.getcwd()
    return Path(working_dir) / CONFIG_FOLDER_NAME


def get_db_path(working_dir: str | Path | None = None) -> Path:
    """Get the SQLite database path for the given working directory."""
    return get_config_dir(working_dir) / "agentic.db"


def get_log_dir(working_dir: str | Path | None = None) -> Path:
    """Get the logs directory for the given working directory."""
    return get_config_dir(working_dir) / "logs"


def get_pid_file(working_dir: str | Path | None = None) -> Path:
    """Get the orchestrator PID file path."""
    return get_config_dir(working_dir) / "orchestrator.pid"


def get_session_file(working_dir: str | Path | None = None) -> Path:
    """Get the current session file path."""
    return get_config_dir(working_dir) / "current_session"


def get_activity_log(working_dir: str | Path | None = None) -> Path:
    """Get the activity log file path."""
    return get_config_dir(working_dir) / "activity.log"


def get_debug_log(working_dir: str | Path | None = None, name: str = "debug") -> Path:
    """Get a debug log file path."""
    return get_log_dir(working_dir) / f"{name}.log"


def ensure_config_dir(working_dir: str | Path | None = None) -> Path:
    """Ensure the config directory and subdirectories exist and return the config path."""
    config_dir = get_config_dir(working_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir = config_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def cleanup_session_data(working_dir: str | Path | None = None) -> dict:
    """Clean up all session data (logs, database, etc.) for a fresh start.
    
    This is called when starting a new session to ensure a clean slate.
    
    Args:
        working_dir: The working directory containing .agentic folder.
        
    Returns:
        Dict with cleanup statistics.
    """
    config_dir = get_config_dir(working_dir)
    stats = {
        "removed_files": [],
        "removed_dirs": [],
        "errors": [],
    }
    
    if not config_dir.exists():
        return stats
    
    # Files to remove
    files_to_remove = [
        config_dir / "agentic.db",
        config_dir / "agentic.db-journal",
        config_dir / "agentic.db-wal",
        config_dir / "agentic.db-shm",
        config_dir / "activity.log",
        config_dir / "current_session",
        config_dir / "orchestrator.pid",
        config_dir / "sqlite_debug.log",
        config_dir / "worker_mcp_debug.log",
    ]
    
    # Also remove any orchestrator log files
    for f in config_dir.glob("orchestrator_*.log"):
        files_to_remove.append(f)
    
    for file_path in files_to_remove:
        if file_path.exists():
            try:
                file_path.unlink()
                stats["removed_files"].append(str(file_path))
            except Exception as e:
                stats["errors"].append(f"Failed to remove {file_path}: {e}")
    
    # Remove logs directory
    logs_dir = config_dir / "logs"
    if logs_dir.exists():
        try:
            shutil.rmtree(logs_dir)
            stats["removed_dirs"].append(str(logs_dir))
        except Exception as e:
            stats["errors"].append(f"Failed to remove {logs_dir}: {e}")
    
    return stats


# Environment variable for passing working dir to worker processes
WORKING_DIR_ENV_VAR = "AGENTIC_WORKING_DIR"


def get_working_dir_from_env() -> str | None:
    """Get working directory from environment variable (used by workers)."""
    return os.environ.get(WORKING_DIR_ENV_VAR)


def resolve_working_dir(working_dir: str | None = None) -> str:
    """Resolve working directory from argument, env var, or CWD.

    Priority: explicit arg > AGENTIC_WORKING_DIR env var > os.getcwd().
    """
    if working_dir is not None:
        return working_dir
    return os.environ.get(WORKING_DIR_ENV_VAR) or os.getcwd()


def get_current_session_id(working_dir: str | None = None) -> str | None:
    """Get the current session ID if one exists.

    Checks the per-repo session file first, then the environment variable.

    Args:
        working_dir: Working directory to check. If None, resolved automatically.
    """
    working_dir = resolve_working_dir(working_dir)
    session_file = get_session_file(working_dir)
    if session_file.exists():
        return session_file.read_text().strip()
    return os.environ.get("AGENTIC_SESSION_ID")


def save_current_session_id(session_id: str, working_dir: str | None = None) -> None:
    """Save the current session ID to the per-repo session file.

    Args:
        session_id: The session ID to save.
        working_dir: Working directory for per-repo storage. If None, resolved automatically.
    """
    working_dir = resolve_working_dir(working_dir)
    ensure_config_dir(working_dir)
    session_file = get_session_file(working_dir)
    session_file.write_text(session_id)


def clear_current_session(working_dir: str | None = None) -> None:
    """Clear the current session ID.

    Args:
        working_dir: Working directory for per-repo storage. If None, resolved automatically.
    """
    working_dir = resolve_working_dir(working_dir)
    session_file = get_session_file(working_dir)
    if session_file.exists():
        session_file.unlink()


def get_storage_client(working_dir: str | None = None):
    """Get storage client (Redis preferred, SQLite fallback).

    Lazily imports redis_client to avoid circular imports.

    Args:
        working_dir: Working directory for per-repo storage. If None, resolved automatically.
    """
    working_dir = resolve_working_dir(working_dir)
    from agentic.redis_client import get_client

    return get_client(
        host=os.environ.get("AGENTIC_REDIS_HOST", "localhost"),
        port=int(os.environ.get("AGENTIC_REDIS_PORT", "6379")),
        db=int(os.environ.get("AGENTIC_REDIS_DB", "0")),
        working_dir=working_dir,
    )
