"""Tmux pane management for agentic-tmux."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import libtmux
from libtmux.exc import LibTmuxException

from agentic.models import Agent


@dataclass
class PaneInfo:
    """Information about a tmux pane."""

    pane_id: str
    window_id: str
    session_name: str
    width: int
    height: int
    active: bool
    current_command: str | None = None


class TmuxManager:
    """Manages tmux sessions, windows, and panes for agentic orchestration."""

    def __init__(self, session_name: str = "agentic", use_current_session: bool = True):
        self.server = libtmux.Server()
        self._session: libtmux.Session | None = None
        
        # If inside tmux and use_current_session is enabled, use the current session
        if use_current_session and self._is_inside_tmux():
            self.session_name = self._get_current_session_name()
        else:
            self.session_name = session_name

    def _is_inside_tmux(self) -> bool:
        """Check if we're running inside a tmux session (internal helper)."""
        return "TMUX" in os.environ

    def _get_current_session_name(self) -> str:
        """Get the name of the current tmux session we're inside."""
        try:
            result = subprocess.run(
                ["tmux", "display-message", "-p", "#S"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return "agentic"  # fallback

    @property
    def session(self) -> libtmux.Session:
        """Get or create the tmux session."""
        if self._session is None:
            self._session = self._get_or_create_session()
        return self._session

    def _get_or_create_session(self) -> libtmux.Session:
        """Get existing session or create a new one."""
        try:
            session = self.server.sessions.get(session_name=self.session_name, default=None)
            if session:
                return session
        except LibTmuxException:
            pass

        # Create new session (suppress the "already exists" error gracefully)
        try:
            return self.server.new_session(
                session_name=self.session_name,
                window_name="admin",
                attach=False,
            )
        except LibTmuxException:
            # Session was created by another process in the meantime
            session = self.server.sessions.get(session_name=self.session_name, default=None)
            if session:
                return session
            raise

    def is_inside_tmux(self) -> bool:
        """Check if we're running inside a tmux session."""
        return self._is_inside_tmux()

    def get_current_pane_id(self) -> str | None:
        """Get the ID of the current pane (if inside tmux)."""
        if not self.is_inside_tmux():
            return None
        return os.environ.get("TMUX_PANE")

    def session_exists(self) -> bool:
        """Check if the agentic session exists."""
        try:
            return self.server.sessions.get(session_name=self.session_name, default=None) is not None
        except LibTmuxException:
            return False

    def create_admin_pane(self, working_dir: str = ".") -> str:
        """
        Create or get the admin pane.
        
        Returns:
            Pane ID of the admin pane.
        """
        window = self.session.active_window
        if window.name != "admin":
            window = self.session.new_window(window_name="admin")
        
        pane = window.active_pane
        if working_dir != ".":
            pane.send_keys(f"cd {working_dir}")
        
        return pane.id

    def spawn_worker_pane(
        self,
        agent: Agent,
        working_dir: str = ".",
        cli_command: str = "copilot -i",
        session_id: str = "",
        initial_task: str | None = None,
    ) -> str:
        """
        Spawn a new pane for a worker agent.
        
        Args:
            agent: The agent configuration
            working_dir: Working directory for the pane
            cli_command: The CLI command to run (e.g., "copilot -i", "claude")
            session_id: The agentic session ID
            initial_task: Optional initial task to send to the agent
        
        Returns:
            Pane ID of the new worker pane.
        """
        # Find the admin window or create worker window
        worker_window = None
        for window in self.session.windows:
            if window.name == "workers":
                worker_window = window
                break
        
        if worker_window is None:
            worker_window = self.session.new_window(window_name="workers")
            # The new window creates a pane, use it for the first worker
            pane = worker_window.active_pane
        else:
            # Split the window to create a new pane
            pane = worker_window.split()
        
        # Wait for shell to be ready
        time.sleep(0.3)
        
        # Set up environment variables one at a time with small delays
        pane.send_keys(f'export AGENTIC_SESSION_ID="{session_id}"', enter=True)
        time.sleep(0.1)
        pane.send_keys(f'export AGENTIC_AGENT_ID="{agent.id}"', enter=True)
        time.sleep(0.1)
        # Don't export full role (too long, contains special chars)
        # Store a shortened version for reference
        short_role = agent.role[:50].replace('"', "'").replace('\n', ' ')
        pane.send_keys(f'export AGENTIC_AGENT_ROLE="{short_role}"', enter=True)
        time.sleep(0.1)
        pane.send_keys(f'export AGENTIC_PANE_ID="{pane.id}"', enter=True)
        time.sleep(0.1)
        pane.send_keys(f"cd {working_dir}", enter=True)
        time.sleep(0.2)  # Longer delay for cd
        
        # Start the CLI first, then send the initial task separately
        # Extract base command without -i flag arguments
        # For copilot: "copilot -i" => "copilot", "copilot" => "copilot"
        # For other CLIs like claude/aider, just use as-is
        base_cli = cli_command
        if " -i" in cli_command:
            # Remove the -i flag as we'll send the prompt separately
            base_cli = cli_command.replace(" -i", "")
        
        # Start the CLI with the initial task
        # For Copilot, use -i flag with properly escaped prompt
        # For other CLIs, start them and send prompt separately (may not work for all)
        base_cli = cli_command
        
        if initial_task:
            # Clean the task text
            clean_task = initial_task.replace('\n', ' ').strip()
            # Limit length to avoid issues  
            if len(clean_task) > 1500:
                clean_task = clean_task[:1500] + "..."
            
            # Debug: Log what we're sending
            from pathlib import Path
            debug_file = Path.home() / ".config" / "agentic" / "spawn_debug.log"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "a") as f:
                f.write(f"\n=== {time.time()} ===\n")
                f.write(f"Pane: {pane.id}\n")
                f.write(f"Agent: {agent.id}\n")
                f.write(f"Task length: {len(clean_task)}\n")
                f.write(f"Task preview: {clean_task[:200]}...\n")
            
            # Check if this is a copilot command (handles "copilot", "copilot -i", etc.)
            if "copilot" in cli_command.lower():
                # Use copilot -i with the prompt
                # Escape single quotes in the prompt for shell
                escaped_task = clean_task.replace("'", "'\"'\"'")
                
                # Build MCP config for worker MCP tools
                # The worker MCP server needs AGENTIC_SESSION_ID and AGENTIC_AGENT_ID env vars
                # Format matches ~/.copilot/mcp-config.json structure
                import json
                mcp_config = {
                    "mcpServers": {
                        "agentic-worker": {
                            "type": "local",
                            "tools": ["*"],
                            "command": "agentic-worker-mcp",
                            "args": [],
                            "env": {
                                "AGENTIC_SESSION_ID": session_id,
                                "AGENTIC_AGENT_ID": agent.id,
                            }
                        }
                    }
                }
                # Escape the JSON for shell (single quotes escape, double quotes are fine inside)
                mcp_json = json.dumps(mcp_config)
                escaped_mcp = mcp_json.replace("'", "'\"'\"'")
                
                # Build full command with MCP config and allow-all for non-interactive
                full_cmd = f"copilot --additional-mcp-config '{escaped_mcp}' --allow-all -i '{escaped_task}'"
                
                with open(debug_file, "a") as f:
                    f.write(f"Using copilot -i with escaped prompt and MCP config\n")
                    f.write(f"Full cmd length: {len(full_cmd)}\n")
                    f.write(f"MCP config: {mcp_json}\n")
                pane.send_keys(full_cmd, enter=True)
            else:
                # For other CLIs, start them then send prompt separately
                # (This may not work for all TUI-based CLIs)
                if " -i" in base_cli:
                    base_cli = base_cli.replace(" -i", "")
                pane.send_keys(base_cli, enter=True)
                time.sleep(5.0)  # Wait for CLI to start
                # Try to send the prompt (may not work for all CLIs)
                import subprocess
                subprocess.run(
                    ["tmux", "send-keys", "-t", pane.id, "-l", clean_task],
                    capture_output=True,
                    check=False,
                )
                time.sleep(0.3)
                subprocess.run(
                    ["tmux", "send-keys", "-t", pane.id, "C-m"],
                    capture_output=True,
                    check=False,
                )
        else:
            # No initial task, just start the CLI
            if " -i" in base_cli:
                base_cli = base_cli.replace(" -i", "")
            pane.send_keys(base_cli, enter=True)
        
        time.sleep(0.5)
        return pane.id

    def spawn_multiple_workers(
        self,
        agents: list[Agent],
        working_dir: str = ".",
        cli_command: str = "copilot -i",
        session_id: str = "",
        layout: str = "tiled",
    ) -> dict[str, str]:
        """
        Spawn multiple worker panes and arrange them.
        
        Args:
            agents: List of agent configurations
            working_dir: Working directory for panes
            cli_command: The CLI command to run
            session_id: The agentic session ID
            layout: Tmux layout (tiled, even-horizontal, even-vertical, main-horizontal, main-vertical)
        
        Returns:
            Dict mapping agent_id to pane_id.
        """
        pane_mapping: dict[str, str] = {}
        
        for agent in agents:
            pane_id = self.spawn_worker_pane(
                agent=agent,
                working_dir=working_dir,
                cli_command=cli_command,
                session_id=session_id,
            )
            pane_mapping[agent.id] = pane_id
            agent.pane_id = pane_id
        
        # Apply layout
        self._apply_layout(layout)
        
        return pane_mapping

    def _apply_layout(self, layout: str = "tiled") -> None:
        """Apply a layout to the workers window."""
        for window in self.session.windows:
            if window.name == "workers":
                try:
                    window.select_layout(layout)
                except LibTmuxException:
                    pass
                break

    def send_keys_to_pane(self, pane_id: str, keys: str, enter: bool = True) -> bool:
        """
        Send keys to a specific pane.
        
        Args:
            pane_id: Target pane ID
            keys: Keys/text to send
            enter: Whether to press Enter after
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                pane.send_keys(keys, enter=enter)
                return True
        except LibTmuxException:
            pass
        return False

    def send_prompt_to_worker(
        self,
        pane_id: str,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send a prompt to a worker pane's CLI.
        
        Args:
            pane_id: Target pane ID
            prompt: The prompt to send
            context: Additional context to include
        
        Returns:
            True if successful.
        """
        # Build the full prompt with context if provided
        full_prompt = prompt
        if context:
            ctx_str = " ".join(f"[{k}: {v}]" for k, v in context.items())
            full_prompt = f"{ctx_str} {prompt}"
        
        return self.send_keys_to_pane(pane_id, full_prompt, enter=True)

    def capture_pane_output(self, pane_id: str, lines: int = 100) -> str:
        """
        Capture recent output from a pane.
        
        Args:
            pane_id: Target pane ID
            lines: Number of lines to capture
        
        Returns:
            Captured text.
        """
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                return "\n".join(pane.capture_pane(start=-lines))
        except LibTmuxException:
            pass
        return ""

    def get_pane_info(self, pane_id: str) -> PaneInfo | None:
        """Get information about a pane."""
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                return PaneInfo(
                    pane_id=pane.id,
                    window_id=pane.window.id,
                    session_name=self.session_name,
                    width=int(pane.width),
                    height=int(pane.height),
                    active=pane == pane.window.active_pane,
                    current_command=pane.current_command,
                )
        except (LibTmuxException, AttributeError):
            pass
        return None

    def kill_pane(self, pane_id: str) -> bool:
        """Kill a specific pane."""
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                pane.kill()
                return True
        except LibTmuxException:
            pass
        return False

    def respawn_pane(self, pane_id: str, command: str | None = None) -> bool:
        """
        Respawn a pane (kill and restart).
        
        Args:
            pane_id: Target pane ID
            command: Command to run in respawned pane (optional)
        
        Returns:
            True if successful.
        """
        try:
            # Use tmux command directly for respawn-pane
            cmd = ["tmux", "respawn-pane", "-k", "-t", pane_id]
            if command:
                cmd.append(command)
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def kill_all_workers(self) -> int:
        """
        Kill all worker panes.
        
        Returns:
            Number of panes killed.
        """
        killed = 0
        for window in self.session.windows:
            if window.name == "workers":
                for pane in window.panes:
                    try:
                        pane.kill()
                        killed += 1
                    except LibTmuxException:
                        pass
        return killed

    def list_all_panes(self) -> list[PaneInfo]:
        """List all panes in the session."""
        panes = []
        try:
            for window in self.session.windows:
                for pane in window.panes:
                    panes.append(
                        PaneInfo(
                            pane_id=pane.id,
                            window_id=window.id,
                            session_name=self.session_name,
                            width=int(pane.width),
                            height=int(pane.height),
                            active=pane == window.active_pane,
                            current_command=pane.current_command,
                        )
                    )
        except LibTmuxException:
            pass
        return panes

    def focus_pane(self, pane_id: str) -> bool:
        """Focus (select) a specific pane."""
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                pane.select()
                return True
        except LibTmuxException:
            pass
        return False

    def attach_session(self) -> None:
        """Attach to the agentic session (blocks until detached)."""
        self.session.attach()

    def kill_session(self) -> bool:
        """Kill the entire agentic session."""
        try:
            self.session.kill()
            self._session = None
            return True
        except LibTmuxException:
            return False

    def _get_pane_by_id(self, pane_id: str) -> libtmux.Pane | None:
        """Find a pane by its ID."""
        try:
            for window in self.session.windows:
                for pane in window.panes:
                    if pane.id == pane_id:
                        return pane
        except LibTmuxException:
            pass
        return None

    def set_pane_title(self, pane_id: str, title: str) -> bool:
        """Set a pane's title for easier identification."""
        try:
            cmd = ["tmux", "select-pane", "-t", pane_id, "-T", title]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False


def get_tmux_manager(session_name: str = "agentic") -> TmuxManager:
    """Factory function to get a TmuxManager instance."""
    return TmuxManager(session_name=session_name)


def check_tmux_available() -> bool:
    """Check if tmux is available on the system."""
    try:
        result = subprocess.run(
            ["tmux", "-V"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
