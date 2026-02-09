"""
Project Briareus — Coder Agent
Specialist agent for code generation, debugging, and technical implementation.
Returns structured CoderOutput with code, explanation, and dependencies.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from agents.base import BaseAgent
from config.prompts import CODER_SYSTEM, build_agent_prompt
from state.schemas import CoderOutput


# ── Optional Tools ────────────────────────────

@tool
def execute_python(code: str) -> str:
    """
    Execute a Python code snippet in a sandboxed environment and return stdout/stderr.
    Use this to test code before finalizing your output.
    Only use for short validation snippets — not full programs.
    """
    import io
    import contextlib

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, {"__builtins__": __builtins__}, {})

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        output_parts = []
        if stdout:
            output_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")
        return "\n".join(output_parts) if output_parts else "Code executed successfully (no output)."

    except Exception as e:
        return f"EXECUTION ERROR: {type(e).__name__}: {e}"


class CoderAgent(BaseAgent):
    name = "coder"
    system_prompt = CODER_SYSTEM
    output_schema = CoderOutput

    def __init__(self, enable_execution: bool = True, **kwargs):
        self.enable_execution = enable_execution
        super().__init__(**kwargs)

    def _get_tools(self) -> list:
        if self.enable_execution:
            return [execute_python]
        return []

    def _execute(self, task_brief: str, context: str | None = None) -> CoderOutput:
        """
        Two-phase execution:
        1. If tools enabled — draft and optionally test code via execute_python.
        2. Parse final output into structured CoderOutput.
        """
        user_prompt = build_agent_prompt(task_brief, context)

        if self.enable_execution and self.llm_with_tools:
            # ── Phase 1: Draft with optional testing ──
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=(
                    f"{user_prompt}\n\n"
                    "You can use the execute_python tool to test small code snippets "
                    "before finalizing. After you're confident in the code, provide "
                    "your complete solution with explanation."
                )),
            ]
            raw_response = self._call_with_tools(messages)

            # ── Phase 2: Structure the output ─────────
            structure_messages = [
                SystemMessage(content=(
                    "Extract the final code solution from the following response. "
                    "Return structured output with: the complete code, language, "
                    "explanation of design decisions, and list of dependencies."
                )),
                HumanMessage(content=(
                    f"ORIGINAL TASK: {task_brief}\n\n"
                    f"RESPONSE:\n{raw_response}"
                )),
            ]
            return self.llm_structured.invoke(structure_messages)

        else:
            # ── Direct structured generation (no tools) ──
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
            return self.llm_structured.invoke(messages)