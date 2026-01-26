"""Hook installation module for agentic-tmux."""

from pathlib import Path

from agentic.hooks import install_hooks as _install_hooks


def install_hooks(repo_path: Path) -> None:
    """Install hooks to a repository."""
    _install_hooks(repo_path, agent_type="worker")
