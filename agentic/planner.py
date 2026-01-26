"""LLM-based task planning and decomposition for agentic-tmux."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from agentic.models import Agent, ExecutionPlan, FileScope, Task, TaskDAG


PLANNING_SYSTEM_PROMPT = """You are an expert software engineering planner. Your job is to decompose complex development tasks into smaller, parallelizable subtasks that can be executed by multiple AI coding agents.

Each agent will run in its own tmux pane and can:
- Read and modify files within their assigned scope
- Execute shell commands
- Communicate with other agents by sending tasks to their queues

Guidelines for creating execution plans:
1. Identify natural boundaries in the work (by module, by concern, by file type)
2. Maximize parallelism where tasks are independent
3. Create explicit dependencies where order matters
4. Assign clear file scopes to prevent conflicts
5. Include review/validation tasks when appropriate
6. Consider communication points where agents need to coordinate

Output your plan as JSON with this structure:
{
  "agents": [
    {
      "id": "W1",
      "role": "Descriptive role name",
      "scope": {
        "patterns": ["src/auth/**", "src/utils/crypto.ts"],
        "read_only": false
      }
    }
  ],
  "tasks": [
    {
      "id": "t1",
      "title": "Short task title",
      "description": "Detailed description of what to do",
      "agent_id": "W1",
      "dependencies": [],
      "files": ["src/auth/login.ts"]
    }
  ],
  "communications": [
    "W1 → W2: When auth refactoring is complete",
    "W2 → W3: When tests are written"
  ]
}"""


PLANNING_USER_TEMPLATE = """Project context:
- Working directory: {working_dir}
- Files in scope: {files_summary}

User request:
{prompt}

Create an execution plan with {suggested_agents} agents (you can suggest more or fewer if appropriate).

Requirements:
- Each agent must have a clear, non-overlapping file scope
- Dependencies must form a DAG (no cycles)
- Include at least one communication point between agents
- Final task should aggregate/validate results"""


class Planner:
    """LLM-based planner for task decomposition."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def create_plan(
        self,
        prompt: str,
        working_dir: str = ".",
        suggested_agents: int = 3,
        file_context: list[str] | None = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan from a user prompt.
        
        Args:
            prompt: User's task description
            working_dir: Working directory path
            suggested_agents: Suggested number of agents
            file_context: List of relevant files in the project
        
        Returns:
            ExecutionPlan with agents, tasks, and communications.
        """
        # Build file summary
        files_summary = "Not provided"
        if file_context:
            if len(file_context) > 50:
                files_summary = f"{len(file_context)} files. Key paths: {', '.join(file_context[:20])}"
            else:
                files_summary = ", ".join(file_context)

        user_message = PLANNING_USER_TEMPLATE.format(
            working_dir=working_dir,
            files_summary=files_summary,
            prompt=prompt,
            suggested_agents=suggested_agents,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        plan_json = json.loads(response.choices[0].message.content or "{}")
        return self._parse_plan(prompt, plan_json)

    def _parse_plan(self, prompt: str, plan_data: dict[str, Any]) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan."""
        # Parse agents
        agents = []
        for agent_data in plan_data.get("agents", []):
            scope_data = agent_data.get("scope", {})
            agent = Agent(
                id=agent_data.get("id", f"W{len(agents)+1}"),
                role=agent_data.get("role", "Worker"),
                scope=FileScope(
                    patterns=scope_data.get("patterns", ["**/*"]),
                    read_only=scope_data.get("read_only", False),
                ),
            )
            agents.append(agent)

        # Parse tasks into DAG
        dag = TaskDAG()
        for task_data in plan_data.get("tasks", []):
            task = Task(
                id=task_data.get("id", ""),
                title=task_data.get("title", ""),
                description=task_data.get("description", ""),
                agent_id=task_data.get("agent_id"),
                dependencies=task_data.get("dependencies", []),
                files=task_data.get("files", []),
            )
            dag.add_task(task)

        # Parse communications
        communications = plan_data.get("communications", [])

        return ExecutionPlan(
            prompt=prompt,
            agents=agents,
            dag=dag,
            estimated_communications=communications,
        )

    def refine_plan(
        self,
        plan: ExecutionPlan,
        feedback: str,
    ) -> ExecutionPlan:
        """
        Refine an existing plan based on user feedback.
        
        Args:
            plan: Current execution plan
            feedback: User's feedback/modifications
        
        Returns:
            Updated ExecutionPlan.
        """
        current_plan_json = {
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "scope": {"patterns": a.scope.patterns, "read_only": a.scope.read_only},
                }
                for a in plan.agents
            ],
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "agent_id": t.agent_id,
                    "dependencies": t.dependencies,
                    "files": t.files,
                }
                for t in plan.dag.tasks.values()
            ],
            "communications": plan.estimated_communications,
        }

        refine_prompt = f"""Current plan:
```json
{json.dumps(current_plan_json, indent=2)}
```

User feedback:
{feedback}

Please update the plan according to the feedback. Output the complete updated plan in the same JSON format."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": refine_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        plan_json = json.loads(response.choices[0].message.content or "{}")
        return self._parse_plan(plan.prompt, plan_json)


class SimplePlanner:
    """Simple rule-based planner for when LLM is not available."""

    def create_plan(
        self,
        prompt: str,
        working_dir: str = ".",
        suggested_agents: int = 3,
        file_context: list[str] | None = None,
    ) -> ExecutionPlan:
        """Create a simple plan based on keywords and file structure."""
        agents = []
        dag = TaskDAG()
        communications = []

        # Analyze prompt for common patterns
        prompt_lower = prompt.lower()
        
        # Default: create a main worker and a reviewer
        main_agent = Agent(
            id="W1",
            role="Main Developer",
            scope=FileScope(patterns=["src/**", "lib/**"]),
        )
        agents.append(main_agent)

        # Check for test-related keywords
        if any(kw in prompt_lower for kw in ["test", "spec", "coverage"]):
            test_agent = Agent(
                id="W2",
                role="Test Writer",
                scope=FileScope(patterns=["tests/**", "test/**", "spec/**", "__tests__/**"]),
            )
            agents.append(test_agent)
            
            # Create tasks
            main_task = Task(
                id="t1",
                title="Implement main changes",
                description=prompt,
                agent_id="W1",
            )
            dag.add_task(main_task)
            
            test_task = Task(
                id="t2",
                title="Write tests",
                description=f"Write tests for: {prompt}",
                agent_id="W2",
                dependencies=["t1"],
            )
            dag.add_task(test_task)
            
            communications.append("W1 → W2: Implementation complete, ready for tests")
        
        # Check for refactor-related keywords
        elif any(kw in prompt_lower for kw in ["refactor", "restructure", "reorganize"]):
            review_agent = Agent(
                id="W2",
                role="Code Reviewer",
                scope=FileScope(patterns=["**/*"], read_only=True),
            )
            agents.append(review_agent)
            
            refactor_task = Task(
                id="t1",
                title="Perform refactoring",
                description=prompt,
                agent_id="W1",
            )
            dag.add_task(refactor_task)
            
            review_task = Task(
                id="t2",
                title="Review changes",
                description="Review the refactoring for correctness and style",
                agent_id="W2",
                dependencies=["t1"],
            )
            dag.add_task(review_task)
            
            communications.append("W1 → W2: Refactoring complete, please review")
        
        # Default: single task
        else:
            task = Task(
                id="t1",
                title="Execute task",
                description=prompt,
                agent_id="W1",
            )
            dag.add_task(task)

        return ExecutionPlan(
            prompt=prompt,
            agents=agents,
            dag=dag,
            estimated_communications=communications,
        )


def get_planner(use_llm: bool = True, **kwargs) -> Planner | SimplePlanner:
    """
    Get a planner instance.
    
    Args:
        use_llm: Whether to use LLM-based planning
        **kwargs: Additional arguments for Planner
    
    Returns:
        Planner or SimplePlanner instance.
    """
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        return Planner(**kwargs)
    return SimplePlanner()
