"""
Project Briareus — Entry Point
Run the multi-agent orchestrator from the command line.

Usage:
    python main.py "Your task description here"
    python main.py                                  # interactive mode
"""

from __future__ import annotations

import sys

from graph.builder import get_graph
from state.schemas import OrchestratorState


def run(task: str) -> str:
    """Execute a task through the Briareus orchestrator."""
    graph = get_graph()

    initial_state: OrchestratorState = {
        "messages": [],
        "task": task,
        "plan": None,
        "current_step": 0,
        "next_agent": "researcher",
        "current_task_brief": "",
        "agent_outputs": {},
        "iteration_count": 0,
        "retry_count": 0,
        "last_review": None,
        "final_output": "",
    }

    print(f"\n{'='*60}")
    print("🦾 BRIAREUS — Multi-Agent Orchestrator")
    print(f"{'='*60}")
    print(f"📌 Task: {task}\n")

    final_state = graph.invoke(initial_state)

    return final_state.get("final_output", "No output generated.")


def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("🦾 Briareus — Enter your task (or 'quit' to exit):\n")
        task = input(">>> ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return

    result = run(task)
    print("\n📋 FINAL OUTPUT:")
    print("-" * 60)
    print(result)


if __name__ == "__main__":
    main()