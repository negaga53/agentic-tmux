"""Configuration and path management for agentic-tmux.

This module provides per-repository config directories (.agentic/) instead of
a centralized ~/.config/agentic/ folder. Each repo gets isolated storage.
"""

from pathlib import Path
import shutil
import os

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
    """Ensure the config directory exists and return its path."""
    config_dir = get_config_dir(working_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
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
