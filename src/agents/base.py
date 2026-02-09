"""
Project Briareus — Base Agent
Abstract base class defining the shared interface for all specialist agents.
Every sub-agent inherits from this and implements `_execute`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from config.prompts import build_agent_prompt
from config.settings import AGENT_MODEL, AGENT_TEMPERATURE


class BaseAgent(ABC):
    """
    Abstract specialist agent.

    Subclasses must:
        1. Set `name` and `system_prompt` class attributes.
        2. Define `output_schema` (a Pydantic model class).
        3. Optionally override `_get_tools()` to bind tools to the LLM.
        4. Optionally override `_execute()` for custom logic.
    """

    name: str = "base"
    system_prompt: str = ""
    output_schema: type[BaseModel] = BaseModel

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # Very cheap, INPUT: $0.10 / 1M tokens, OUTPUT: $0.40 / 1M tokens
        )
        self._bind_tools()

    # ── Tool Binding ──────────────────────────

    def _get_tools(self) -> list:
        """Override to return LangChain-compatible tools for this agent."""
        return []

    def _bind_tools(self) -> None:
        """Bind tools and structured output to the LLM."""
        tools = self._get_tools()
        if tools:
            self.llm_with_tools = self.llm.bind_tools(tools)
        else:
            self.llm_with_tools = None

        # Separate LLM instance for final structured output (no tools)
        self.llm_structured = self.llm.with_structured_output(self.output_schema)

    # ── Execution ─────────────────────────────

    def run(self, task_brief: str, context: str | None = None) -> dict[str, Any]:
        """
        Public entry point. Runs the agent and returns a dict of its structured output.

        Args:
            task_brief: Scoped instructions from the supervisor.
            context: Optional context from prior agent outputs.

        Returns:
            Dict representation of the agent's Pydantic output model.
        """
        print(f"  🔧 [{self.name.upper()}] Starting...")
        result = self._execute(task_brief, context)
        print(f"  ✅ [{self.name.upper()}] Complete.")
        return result.model_dump() if isinstance(result, BaseModel) else result

    def _execute(self, task_brief: str, context: str | None = None) -> BaseModel:
        """
        Default execution: send task to LLM with structured output parsing.

        Override this in subclasses that need tool-calling loops or
        multi-step reasoning.
        """
        user_prompt = build_agent_prompt(task_brief, context)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm_structured.invoke(messages)
        return response

    # ── Helpers ────────────────────────────────

    def _call_with_tools(self, messages: list) -> str:
        """
        Run a tool-augmented conversation loop.

        Sends messages to the LLM with tools bound. If the LLM requests
        tool calls, executes them and feeds results back until the LLM
        produces a final text response.
        """
        if not self.llm_with_tools:
            raise RuntimeError(f"Agent '{self.name}' has no tools bound.")

        tools_by_name = {t.name: t for t in self._get_tools()}
        max_tool_rounds = 6

        for _ in range(max_tool_rounds):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, we're done — return the text content
            if not response.tool_calls:
                return response.content

            # Execute each tool call and append results
            for tool_call in response.tool_calls:
                tool = tools_by_name.get(tool_call["name"])
                if tool is None:
                    tool_result = f"Error: Unknown tool '{tool_call['name']}'"
                else:
                    try:
                        tool_result = tool.invoke(tool_call["args"])
                    except Exception as e:
                        tool_result = f"Error executing {tool_call['name']}: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result),
                })

        return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    def _summarize_output(self, output: dict[str, Any], max_chars: int = 500) -> str:
        """Create a concise summary of agent output for the supervisor."""
        text = json.dumps(output, indent=2)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"