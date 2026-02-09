# src/state/schemas.py

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]                                         # conversation history
    task: str                                                                       # original user request
    plan: list[str]                                                                 # supervisor's current plan
    current_step: int                                                               # where we are in the plan
    agent_outputs: dict[str, str]                                                   # keyed by agent name
    next_agent: Literal["researcher", "coder", "writer", "synthesize", "END"]
    iteration_count: int                                                            # loop safety
    final_output: str                                                               # synthesized result

# Each agent should return a well-typed response using Pydantic

class ResearchOutput(BaseModel):
    findings: list[str]
    sources: list[str]
    confidence: float
    gaps: list[str]           # what it couldn't find

class CoderOutput(BaseModel):
    code: str
    language: str
    explanation: str
    tested: bool

class WriterOutput(BaseModel):
    content: str
    format: str               # "markdown", "report", etc.
    word_count: int