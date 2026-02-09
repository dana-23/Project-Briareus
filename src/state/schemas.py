"""
Project Briareus — State Schemas
Centralized state definitions and Pydantic models for structured agent I/O.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ──────────────────────────────────────────────
# Agent Output Models (Pydantic — structured LLM output)
# ──────────────────────────────────────────────

class ResearchOutput(BaseModel):
    """Structured output from the Researcher agent."""
    findings: list[str] = Field(description="Key findings from the research")
    sources: list[str] = Field(default_factory=list, description="URLs or references")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence level in the findings (0-1)"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps that couldn't be resolved"
    )


class CoderOutput(BaseModel):
    """Structured output from the Coder agent."""
    code: str = Field(description="The generated code")
    language: str = Field(default="python", description="Programming language used")
    explanation: str = Field(description="What the code does and key design decisions")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required packages or libraries"
    )


class WriterOutput(BaseModel):
    """Structured output from the Writer agent."""
    content: str = Field(description="The written content")
    format: str = Field(default="markdown", description="Output format: markdown, report, etc.")
    summary: str = Field(description="One-line summary of what was written")


# ──────────────────────────────────────────────
# Supervisor Models
# ──────────────────────────────────────────────

class SubTask(BaseModel):
    """A single subtask in the supervisor's plan."""
    id: int = Field(description="Step number (1-indexed)")
    description: str = Field(description="What this step accomplishes")
    agent: Literal["researcher", "coder", "writer"] = Field(
        description="Which agent handles this step"
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="IDs of steps that must complete before this one"
    )


class Plan(BaseModel):
    """The supervisor's execution plan."""
    goal: str = Field(description="High-level interpretation of the user's task")
    subtasks: list[SubTask] = Field(description="Ordered list of subtasks")
    reasoning: str = Field(description="Why this plan structure was chosen")


class RouteDecision(BaseModel):
    """The supervisor's routing decision after reviewing progress."""
    next_agent: Literal["researcher", "coder", "writer", "synthesize"] = Field(
        description="Which agent to invoke next, or 'synthesize' to finish"
    )
    task_brief: str = Field(
        description="Concise instructions for the next agent (or synthesis prompt)"
    )
    reasoning: str = Field(description="Why this routing decision was made")


class ReviewResult(BaseModel):
    """The supervisor's evaluation of an agent's output."""
    quality: Literal["good", "acceptable", "needs_retry"] = Field(
        description="Quality assessment of the agent output"
    )
    feedback: str = Field(description="What was good or what needs improvement")
    should_retry: bool = Field(
        default=False, description="Whether to retry the same agent"
    )
    retry_instructions: Optional[str] = Field(
        default=None, description="Revised instructions if retrying"
    )


# ──────────────────────────────────────────────
# LangGraph Orchestrator State
# ──────────────────────────────────────────────

AgentName = Literal["researcher", "coder", "writer"]

class OrchestratorState(TypedDict):
    """Shared state flowing through the LangGraph orchestrator."""
    messages: Annotated[list, add_messages]

    # Task & Planning
    task: str                                      # original user request
    plan: Optional[Plan]                           # supervisor's plan
    current_step: int                              # index into plan.subtasks

    # Routing & Execution
    next_agent: Literal["researcher", "coder", "writer", "synthesize", "__end__"]
    current_task_brief: str                        # scoped instructions for current agent
    agent_outputs: dict[str, list[dict]]           # keyed by agent name → list of outputs

    # Control Flow
    iteration_count: int                           # total agent calls (loop safety)
    retry_count: int                               # consecutive retries for current step
    last_review: Optional[ReviewResult]            # most recent review

    # Final
    final_output: str                              # synthesized result