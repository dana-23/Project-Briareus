"""
Project Briareus — Supervisor: Reviewer Node
Evaluates the quality of the most recent agent output and decides
whether to proceed, accept with caveats, or retry.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.prompts import REVIEWER_SYSTEM, build_reviewer_prompt
from config.settings import (
    SUPERVISOR_MODEL,
    SUPERVISOR_TEMPERATURE,
    MAX_ITERATIONS,
    MAX_RETRIES_PER_STEP,
)
from state.schemas import OrchestratorState, ReviewResult


def _get_reviewer_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
    ).with_structured_output(ReviewResult)


def _get_last_agent_output(state: OrchestratorState) -> tuple[str, str]:
    """
    Get the name and stringified output of the most recent agent call.
    Returns: (agent_name, output_string)
    """
    outputs = state.get("agent_outputs", {})
    for agent_name in reversed(list(outputs.keys())):
        if outputs[agent_name]:
            last = outputs[agent_name][-1]
            return agent_name, json.dumps(last, indent=2)
    return "unknown", "No output available."


def review_node(state: OrchestratorState) -> dict:
    """
    LangGraph node: Review the most recent agent output.

    Reads:  state["agent_outputs"], state["current_task_brief"],
            state["iteration_count"], state["retry_count"]
    Writes: state["last_review"], state["retry_count"], state["current_step"]
    """
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    current_step = state.get("current_step", 0)

    agent_name, agent_output = _get_last_agent_output(state)
    task_brief = state.get("current_task_brief", "")

    print(f"\n🔍 [REVIEWER] Evaluating {agent_name} output...")

    # ── Fast-path: skip review if budget is nearly exhausted ──
    if iteration_count >= MAX_ITERATIONS - 1:
        print(f"  ⚡ Budget exhausted — auto-accepting output.")
        return {
            "last_review": ReviewResult(
                quality="acceptable",
                feedback="Auto-accepted due to iteration budget.",
                should_retry=False,
            ),
            "retry_count": 0,
            "current_step": current_step + 1,
        }

    # ── LLM-based review ─────────────────────
    llm = _get_reviewer_llm()

    user_prompt = build_reviewer_prompt(
        task_brief=task_brief,
        agent_name=agent_name,
        agent_output=agent_output,
        iteration_count=iteration_count,
        max_iterations=MAX_ITERATIONS,
    )

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    review: ReviewResult = llm.invoke(messages)

    # ── Enforce retry limits ──────────────────
    if review.should_retry and retry_count >= MAX_RETRIES_PER_STEP:
        print(f"\t⚠️ Max retries ({MAX_RETRIES_PER_STEP}) hit for this step — accepting as-is.")
        review.should_retry = False
        review.quality = "acceptable"
        review.feedback += f" [Accepted after {retry_count} retries — retry limit reached.]"

    # ── Update state ──────────────────────────
    new_retry_count = retry_count + 1 if review.should_retry else 0
    new_step = current_step if review.should_retry else current_step + 1

    emoji = {"good": "✅", "acceptable": "⚠️", "needs_retry": "🔄"}
    print(f"\t{emoji.get(review.quality, '❓')} Quality: {review.quality}")
    print(f"\t📝 Feedback: {review.feedback[:120]}...")
    if review.should_retry:
        print(f"\t🔄 Retrying (attempt {new_retry_count}/{MAX_RETRIES_PER_STEP})...")

    return {
        "last_review": review,
        "retry_count": new_retry_count,
        "current_step": new_step,
    }