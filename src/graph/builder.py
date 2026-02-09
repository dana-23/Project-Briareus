"""
Project Briareus — Graph Builder
Constructs and compiles the LangGraph orchestration graph.

Flow:
    START → plan → route → [researcher|coder|writer] → review → route → ... → synthesize → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from agents import get_agent
from state.schemas import OrchestratorState
from supervisor.planner import plan_node
from supervisor.router import route_node, route_conditional_edge
from supervisor.reviewer import review_node
from supervisor.synthesizer import synthesize_node


# ──────────────────────────────────────────────
# Agent Wrapper Nodes
# ──────────────────────────────────────────────
# Each agent needs a thin wrapper that reads from / writes to OrchestratorState.

def _make_agent_node(agent_name: str):
    """
    Factory: creates a LangGraph node function for a specialist agent.
    The returned function reads the task brief from state, runs the agent,
    and stores the output back in state.
    """
    agent = get_agent(agent_name)

    def agent_node(state: OrchestratorState) -> dict:
        task_brief = state.get("current_task_brief", "")
        iteration_count = state.get("iteration_count", 0)

        # Build context from prior agent outputs (concise summaries)
        context_parts = []
        for name, outputs in state.get("agent_outputs", {}).items():
            if outputs:
                last = outputs[-1]
                context_parts.append(f"[{name}]: {agent._summarize_output(last)}")
                
        context = "\n\n".join(context_parts) if context_parts else None

        # Run the agent
        result = agent.run(task_brief, context)

        # Append to agent_outputs
        existing_outputs = dict(state.get("agent_outputs", {}))
        if agent_name not in existing_outputs:
            existing_outputs[agent_name] = []
        existing_outputs[agent_name] = existing_outputs[agent_name] + [result]

        return {
            "agent_outputs": existing_outputs,
            "iteration_count": iteration_count + 1,
        }

    agent_node.__name__ = agent_name  # for LangGraph display
    return agent_node


# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the Briareus orchestration graph.

    Returns a compiled LangGraph ready for .invoke() or .stream().
    """
    workflow = StateGraph(OrchestratorState)

    # ── Add nodes ─────────────────────────────
    workflow.add_node("plan", plan_node)
    workflow.add_node("route", route_node)
    workflow.add_node("researcher", _make_agent_node("researcher"))
    workflow.add_node("coder", _make_agent_node("coder"))
    workflow.add_node("writer", _make_agent_node("writer"))
    workflow.add_node("review", review_node)
    workflow.add_node("synthesize", synthesize_node)

    # ── Add edges ─────────────────────────────
    #  START → plan → route
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "route")

    # route → conditional branch to agent or synthesize
    workflow.add_conditional_edges(
        "route",
        route_conditional_edge,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "synthesize": "synthesize",
            "__end__": END,
        },
    )

    # Each agent → review
    workflow.add_edge("researcher", "review")
    workflow.add_edge("coder", "review")
    workflow.add_edge("writer", "review")

    # review → route (loop back for next decision)
    workflow.add_edge("review", "route")

    # synthesize → END
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# ── Convenience ───────────────────────────────

_compiled_graph = None

def get_graph():
    """Singleton accessor for the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph