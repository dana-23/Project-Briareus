"""
Project Briareus — Researcher Agent
Specialist agent for web research, fact gathering, and source verification.
Uses Tavily search for web lookups and returns structured ResearchOutput.
"""

from __future__ import annotations

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.base import BaseAgent
from config.prompts import RESEARCHER_SYSTEM, build_agent_prompt
from config.settings import TAVILY_MAX_RESULTS
from state.schemas import ResearchOutput


class ResearcherAgent(BaseAgent):
    name = "researcher"
    system_prompt = RESEARCHER_SYSTEM
    output_schema = ResearchOutput

    def _get_tools(self) -> list:
        return [
            TavilySearchResults(
                max_results=TAVILY_MAX_RESULTS,
                name="web_search",
                description="Search the web for current information. Input should be a search query string.",
            )
        ]

    def _execute(self, task_brief: str, context: str | None = None) -> ResearchOutput:
        """
        Two-phase execution:
        1. Tool phase — search the web using tools to gather raw information.
        2. Synthesis phase — parse raw findings into structured ResearchOutput.
        """
        # ── Phase 1: Search with tools ────────────
        user_prompt = build_agent_prompt(task_brief, context)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=(
                f"{user_prompt}\n\n"
                "Use the web_search tool to find relevant information. "
                "Search multiple queries if needed for thoroughness. "
                "After gathering information, provide a comprehensive summary of your findings."
            )),
        ]

        raw_findings = self._call_with_tools(messages)

        # ── Phase 2: Structure the output ─────────
        structure_messages = [
            SystemMessage(content=(
                "You are a research analyst. Convert the following raw research findings "
                "into a structured format with: key findings (list), sources (URLs), "
                "confidence score (0-1), and any information gaps."
            )),
            HumanMessage(content=(
                f"ORIGINAL TASK: {task_brief}\n\n"
                f"RAW FINDINGS:\n{raw_findings}\n\n"
                "Structure these findings."
            )),
        ]

        result = self.llm_structured.invoke(structure_messages)
        return result