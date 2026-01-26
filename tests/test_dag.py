"""Tests for models and DAG operations."""

import pytest

from agentic.models import Agent, FileScope, Task, TaskDAG, TaskStatus
from agentic.dag import (
    validate_dag,
    topological_sort,
    get_parallel_groups,
    detect_wait_cycle,
    get_critical_path,
)


class TestFileScope:
    def test_matches_exact(self):
        scope = FileScope(patterns=["src/auth/login.ts"])
        assert scope.matches("src/auth/login.ts")
        assert not scope.matches("src/auth/logout.ts")

    def test_matches_glob(self):
        scope = FileScope(patterns=["src/auth/**"])
        assert scope.matches("src/auth/login.ts")
        assert scope.matches("src/auth/utils/hash.ts")
        assert not scope.matches("src/utils/helper.ts")

    def test_matches_multiple_patterns(self):
        scope = FileScope(patterns=["src/auth/**", "tests/auth/**"])
        assert scope.matches("src/auth/login.ts")
        assert scope.matches("tests/auth/test_login.py")
        assert not scope.matches("src/utils/helper.ts")


class TestTaskDAG:
    def test_add_task(self):
        dag = TaskDAG()
        task = Task(id="t1", title="Test task")
        dag.add_task(task)
        assert "t1" in dag.tasks

    def test_get_ready_tasks_no_deps(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2"))
        
        ready = dag.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_tasks_with_deps(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        
        ready = dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_is_complete(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", status=TaskStatus.COMPLETED))
        dag.add_task(Task(id="t2", title="Task 2", status=TaskStatus.COMPLETED))
        
        assert dag.is_complete()

    def test_is_not_complete(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", status=TaskStatus.COMPLETED))
        dag.add_task(Task(id="t2", title="Task 2", status=TaskStatus.PENDING))
        
        assert not dag.is_complete()

    def test_has_cycle_false(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t2"]))
        
        assert not dag.has_cycle()

    def test_has_cycle_true(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", dependencies=["t3"]))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t2"]))
        
        assert dag.has_cycle()

    def test_get_completion_progress(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", status=TaskStatus.COMPLETED))
        dag.add_task(Task(id="t2", title="Task 2", status=TaskStatus.PENDING))
        dag.add_task(Task(id="t3", title="Task 3", status=TaskStatus.IN_PROGRESS))
        
        completed, total = dag.get_completion_progress()
        assert completed == 1
        assert total == 3


class TestValidateDAG:
    def test_valid_dag(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", agent_id="W1"))
        dag.add_task(Task(id="t2", title="Task 2", agent_id="W2", dependencies=["t1"]))
        
        valid, errors = validate_dag(dag)
        assert valid
        assert len(errors) == 0

    def test_cycle_detection(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", agent_id="W1", dependencies=["t2"]))
        dag.add_task(Task(id="t2", title="Task 2", agent_id="W2", dependencies=["t1"]))
        
        valid, errors = validate_dag(dag)
        assert not valid
        assert any("cycle" in e.lower() for e in errors)

    def test_missing_dependency(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1", agent_id="W1", dependencies=["t_nonexistent"]))
        
        valid, errors = validate_dag(dag)
        assert not valid
        assert any("non-existent" in e.lower() for e in errors)


class TestTopologicalSort:
    def test_simple_chain(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t2"]))
        
        sorted_ids = topological_sort(dag)
        assert sorted_ids.index("t1") < sorted_ids.index("t2")
        assert sorted_ids.index("t2") < sorted_ids.index("t3")

    def test_parallel_tasks(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2"))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t1", "t2"]))
        
        sorted_ids = topological_sort(dag)
        assert sorted_ids.index("t1") < sorted_ids.index("t3")
        assert sorted_ids.index("t2") < sorted_ids.index("t3")


class TestGetParallelGroups:
    def test_all_parallel(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2"))
        dag.add_task(Task(id="t3", title="Task 3"))
        
        groups = get_parallel_groups(dag)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_sequential(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t2"]))
        
        groups = get_parallel_groups(dag)
        assert len(groups) == 3
        assert groups[0] == ["t1"]
        assert groups[1] == ["t2"]
        assert groups[2] == ["t3"]

    def test_diamond(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t1"]))
        dag.add_task(Task(id="t4", title="Task 4", dependencies=["t2", "t3"]))
        
        groups = get_parallel_groups(dag)
        assert len(groups) == 3
        assert groups[0] == ["t1"]
        assert set(groups[1]) == {"t2", "t3"}
        assert groups[2] == ["t4"]


class TestDetectWaitCycle:
    def test_no_cycle(self):
        agents = [
            Agent(id="W1", role="Worker 1", status="idle"),
            Agent(id="W2", role="Worker 2", status="waiting"),
        ]
        
        def get_waiting_for(agent):
            if agent.id == "W2":
                return "W1"
            return None
        
        cycle = detect_wait_cycle(agents, get_waiting_for)
        assert cycle is None

    def test_with_cycle(self):
        agents = [
            Agent(id="W1", role="Worker 1", status="waiting"),
            Agent(id="W2", role="Worker 2", status="waiting"),
        ]
        
        def get_waiting_for(agent):
            if agent.id == "W1":
                return "W2"
            if agent.id == "W2":
                return "W1"
            return None
        
        cycle = detect_wait_cycle(agents, get_waiting_for)
        assert cycle is not None
        assert set(cycle[:2]) == {"W1", "W2"}


class TestGetCriticalPath:
    def test_single_path(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t2"]))
        
        path = get_critical_path(dag)
        assert path == ["t1", "t2", "t3"]

    def test_parallel_paths(self):
        dag = TaskDAG()
        dag.add_task(Task(id="t1", title="Task 1"))
        dag.add_task(Task(id="t2", title="Task 2", dependencies=["t1"]))
        dag.add_task(Task(id="t3", title="Task 3", dependencies=["t1"]))
        dag.add_task(Task(id="t4", title="Task 4", dependencies=["t2"]))
        dag.add_task(Task(id="t5", title="Task 5", dependencies=["t3", "t4"]))
        
        path = get_critical_path(dag)
        # Critical path is t1 -> t2 -> t4 -> t5 (length 4)
        assert len(path) == 4
        assert path[0] == "t1"
        assert path[-1] == "t5"
