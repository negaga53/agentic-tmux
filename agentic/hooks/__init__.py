"""Hook installer for agentic-tmux."""

from __future__ import annotations

import shutil
from pathlib import Path


# Get the hooks directory from package data
_PACKAGE_DIR = Path(__file__).parent.parent  # agentic/
PACKAGE_HOOKS_DIR = _PACKAGE_DIR / "data" / "hooks"
PACKAGE_SCRIPTS_DIR = _PACKAGE_DIR / "data" / "scripts"


def install_hooks(repo_path: Path, agent_type: str = "worker") -> None:
    """
    Install agentic hooks into a repository.
    
    Args:
        repo_path: Path to the repository root
        agent_type: Type of hooks to install ("admin" or "worker")
    """
    hooks_dir = repo_path / ".github" / "hooks"
    scripts_dir = hooks_dir / "scripts"
    
    # Create directories
    hooks_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy hook JSON files
    source_hooks = PACKAGE_HOOKS_DIR / agent_type
    if source_hooks.exists():
        for hook_file in source_hooks.glob("*.json"):
            dest_file = hooks_dir / hook_file.name
            shutil.copy2(hook_file, dest_file)
    
    # Copy all scripts
    if PACKAGE_SCRIPTS_DIR.exists():
        for script_file in PACKAGE_SCRIPTS_DIR.glob("*.sh"):
            dest_file = scripts_dir / script_file.name
            shutil.copy2(script_file, dest_file)
            # Make executable
            dest_file.chmod(0o755)


def uninstall_hooks(repo_path: Path) -> None:
    """
    Remove agentic hooks from a repository.
    
    Args:
        repo_path: Path to the repository root
    """
    hooks_dir = repo_path / ".github" / "hooks"
    
    if hooks_dir.exists():
        # Remove hook files
        for hook_file in hooks_dir.glob("*.json"):
            hook_file.unlink()
        
        # Remove scripts directory
        scripts_dir = hooks_dir / "scripts"
        if scripts_dir.exists():
            shutil.rmtree(scripts_dir)
        
        # Remove hooks dir if empty
        if not any(hooks_dir.iterdir()):
            hooks_dir.rmdir()


def get_installed_hooks(repo_path: Path) -> list[str]:
    """
    Get list of installed hooks in a repository.
    
    Args:
        repo_path: Path to the repository root
    
    Returns:
        List of hook names that are installed.
    """
    hooks_dir = repo_path / ".github" / "hooks"
    
    if not hooks_dir.exists():
        return []
    
    return [f.stem for f in hooks_dir.glob("*.json")]
