"""
Project Briareus — Supervisor: Planner Node
Takes the user's raw task and produces a structured execution plan
with subtasks assigned to specialist agents.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.prompts import PLANNER_SYSTEM, build_planner_prompt
from config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE, MAX_PLAN_STEPS
from state.schemas import OrchestratorState, Plan


def _get_planner_llm() -> ChatOpenAI:
    return ChatGoogleGenerativeAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
    ).with_structured_output(Plan)


def plan_node(state: OrchestratorState) -> dict:
    """
    LangGraph node: Generate an execution plan from the user's task.

    Reads:  state["task"]
    Writes: state["plan"], state["current_step"], state["iteration_count"],
            state["retry_count"], state["agent_outputs"]
    """
    task = state["task"]
    print(f"\n📋 [PLANNER] Decomposing task...")

    llm = _get_planner_llm()
    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=build_planner_prompt(task)),
    ]

    plan: Plan = llm.invoke(messages)

    # Enforce max plan steps
    if len(plan.subtasks) > MAX_PLAN_STEPS:
        print(f"  ⚠️ Plan had {len(plan.subtasks)} steps, truncating to {MAX_PLAN_STEPS}")
        plan.subtasks = plan.subtasks[:MAX_PLAN_STEPS]

    # Log the plan
    print(f"  🎯 Goal: {plan.goal}")
    for st in plan.subtasks:
        deps = f" (after step {st.depends_on})" if st.depends_on else ""
        print(f"  Step {st.id}: [{st.agent}] {st.description}{deps}")
    print(f"  💡 Reasoning: {plan.reasoning}\n")

    return {
        "plan": plan,
        "current_step": 0,
        "iteration_count": 0,
        "retry_count": 0,
        "agent_outputs": {},
    }