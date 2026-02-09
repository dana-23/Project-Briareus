"""
Project Briareus — Agents Package
Registry of all specialist agents for easy lookup by the supervisor.
"""

from agents.base import BaseAgent
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.writer import WriterAgent


# ── Agent Registry ────────────────────────────
# The supervisor uses this to instantiate agents by name.

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
    "writer": WriterAgent,
}


def get_agent(name: str, **kwargs) -> BaseAgent:
    """Instantiate an agent by name from the registry."""
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[name](**kwargs)


__all__ = [
    "BaseAgent",
    "ResearcherAgent",
    "CoderAgent",
    "WriterAgent",
    "AGENT_REGISTRY",
    "get_agent",
]