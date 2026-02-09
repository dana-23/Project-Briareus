"""
Project Briareus — Writer Agent
Specialist agent for producing polished written content.
Handles reports, documentation, summaries, and creative writing.
Returns structured WriterOutput.
"""

from __future__ import annotations

from agents.base import BaseAgent
from config.prompts import WRITER_SYSTEM
from state.schemas import WriterOutput


class WriterAgent(BaseAgent):
    """
    The writer is the simplest agent — no tools, pure LLM generation.
    It relies on context from prior agents (research findings, code) to
    produce integrated written content.
    """

    name = "writer"
    system_prompt = WRITER_SYSTEM
    output_schema = WriterOutput

    # No tools needed — uses the default _execute from BaseAgent
    # which sends the task brief + context to the structured LLM.