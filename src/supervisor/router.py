"""
Project Briareus — Supervisor: Router Node
Evaluates current progress and decides which agent to invoke next,
or whether to proceed to synthesis.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.prompts import ROUTER_SYSTEM, build_router_prompt
from config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE, MAX_ITERATIONS
from state.schemas import OrchestratorState, Plan, RouteDecision


def _get_router_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
    ).with_structured_output(RouteDecision)


def _summarize_plan(plan: Plan) -> str:
    """Create a concise string summary of the plan for the router prompt."""
    lines = [f"Goal: {plan.goal}"]
    for st in plan.subtasks:
        lines.append(f"\tStep {st.id}: [{st.agent}] {st.description}")
    return "\n".join(lines)


def _summarize_progress(state: OrchestratorState) -> str:
    """Summarize what has been accomplished so far."""
    outputs = state.get("agent_outputs", {})
    if not outputs:
        return "No steps completed yet."

    lines = []
    for agent_name, output_list in outputs.items():
        for i, output in enumerate(output_list):
            # Truncate each output summary to keep the prompt manageable
            summary = json.dumps(output, indent=2)
            if len(summary) > 300:
                summary = summary[:300] + "..."
            lines.append(f"[{agent_name} #{i+1}]: {summary}")

    return "\n".join(lines)


def _get_last_output_summary(state: OrchestratorState) -> str | None:
    """Get a summary of the most recent agent output."""
    outputs = state.get("agent_outputs", {})
    if not outputs:
        return None

    # Find the last output across all agents
    last_agent = None
    last_output = None
    for agent_name, output_list in outputs.items():
        if output_list:
            last_agent = agent_name
            last_output = output_list[-1]

    if last_output is None:
        return None

    summary = json.dumps(last_output, indent=2)
    if len(summary) > 400:
        summary = summary[:400] + "..."
    return f"[{last_agent}]: {summary}"


def route_node(state: OrchestratorState) -> dict:
    """
    LangGraph node: Decide the next routing step.

    Reads:  state["task"], state["plan"], state["agent_outputs"],
            state["iteration_count"], state["last_review"], state["current_step"]
    Writes: state["next_agent"], state["current_task_brief"], state["iteration_count"]
    """
    plan = state.get("plan")
    iteration_count = state.get("iteration_count", 0)
    current_step = state.get("current_step", 0)
    last_review = state.get("last_review")

    # ── Safety valve: max iterations ──────────
    if iteration_count >= MAX_ITERATIONS:
        print(f"\t⛔ [ROUTER] Max iterations ({MAX_ITERATIONS}) reached — forcing synthesis.")
        return {
            "next_agent": "synthesize",
            "current_task_brief": "Synthesize all available outputs into a final response.",
            "iteration_count": iteration_count,
        }

    # ── Check if plan is complete ─────────────
    if plan and current_step >= len(plan.subtasks):
        print(f"\t🏁 [ROUTER] All plan steps complete — routing to synthesis.")
        return {
            "next_agent": "synthesize",
            "current_task_brief": "All planned steps are complete. Synthesize the results.",
            "iteration_count": iteration_count,
        }

    # ── Handle retry from review ──────────────
    if last_review and last_review.should_retry:
        print(f"\t🔄 [ROUTER] Retrying last step with feedback...")
        # Re-route to the same agent from the current step
        subtask = plan.subtasks[max(0, current_step - 1)]
        return {
            "next_agent": subtask.agent,
            "current_task_brief": last_review.retry_instructions or subtask.description,
            "iteration_count": iteration_count,
        }

    # ── LLM-based routing for complex decisions ──
    print(f"\n🧭 [ROUTER] Deciding next step (iteration {iteration_count + 1}/{MAX_ITERATIONS})...")

    llm = _get_router_llm()

    plan_summary = _summarize_plan(plan) if plan else "No plan generated."
    progress_summary = _summarize_progress(state)
    last_output_summary = _get_last_output_summary(state)
    last_review_summary = None

    if last_review:
        last_review_summary = f"Quality: {last_review.quality} | Feedback: {last_review.feedback}"

    user_prompt = build_router_prompt(
        task=state["task"],
        plan_summary=plan_summary,
        progress_summary=progress_summary,
        last_output_summary=last_output_summary,
        last_review_summary=last_review_summary,
    )

    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    decision: RouteDecision = llm.invoke(messages)

    print(f"\t→ Next: [{decision.next_agent.upper()}]")
    print(f"\t→ Brief: {decision.task_brief[:200]}...")
    print(f"\t→ Reason: {decision.reasoning[:200]}...")

    return {
        "next_agent": decision.next_agent,
        "current_task_brief": decision.task_brief,
        "iteration_count": iteration_count,  # incremented in agent execution
    }


def route_conditional_edge(state: OrchestratorState) -> str:
    """
    LangGraph conditional edge function.
    Maps state["next_agent"] to graph node names.

    Returns one of: "researcher", "coder", "writer", "synthesize", "__end__"
    """
    next_agent = state.get("next_agent", "synthesize")

    if next_agent in ("researcher", "coder", "writer"):
        return next_agent
    elif next_agent == "synthesize":
        return "synthesize"
    else:
        return "__end__"