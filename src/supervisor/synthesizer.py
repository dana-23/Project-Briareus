"""
Project Briareus — Supervisor: Synthesizer Node
Combines all specialist agent outputs into a single, coherent
response that directly addresses the user's original task.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.prompts import SYNTHESIZER_SYSTEM, build_synthesizer_prompt
from config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE
from state.schemas import OrchestratorState


def _get_synthesizer_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=SUPERVISOR_MODEL,
        temperature=0.3,  # slightly creative for polished prose
    )


def _format_all_outputs(state: OrchestratorState) -> str:
    """
    Format all agent outputs into a readable block for the synthesizer.
    Groups by agent and presents them in execution order.
    """
    outputs = state.get("agent_outputs", {})
    if not outputs:
        return "No agent outputs were collected."

    sections = []
    agent_order = ["researcher", "coder", "writer"]

    for agent_name in agent_order:
        if agent_name not in outputs or not outputs[agent_name]:
            continue

        sections.append(f"{'='*60}")
        sections.append(f"AGENT: {agent_name.upper()}")
        sections.append(f"{'='*60}")

        for i, output in enumerate(outputs[agent_name]):
            if len(outputs[agent_name]) > 1:
                sections.append(f"\n--- Output #{i+1} ---")

            # Format based on agent type for readability
            if agent_name == "researcher":
                findings = output.get("findings", [])
                sources = output.get("sources", [])
                gaps = output.get("gaps", [])
                confidence = output.get("confidence", "N/A")

                sections.append(f"Confidence: {confidence}")
                sections.append("Findings:")
                for f in findings:
                    sections.append(f"  • {f}")
                if sources:
                    sections.append("Sources:")
                    for s in sources:
                        sections.append(f"  - {s}")
                if gaps:
                    sections.append("Gaps:")
                    for g in gaps:
                        sections.append(f"  ⚠ {g}")

            elif agent_name == "coder":
                sections.append(f"Language: {output.get('language', 'N/A')}")
                sections.append(f"Explanation: {output.get('explanation', 'N/A')}")
                code = output.get("code", "")
                sections.append(f"Code:\n```\n{code}\n```")
                deps = output.get("dependencies", [])
                if deps:
                    sections.append(f"Dependencies: {', '.join(deps)}")

            elif agent_name == "writer":
                sections.append(f"Format: {output.get('format', 'N/A')}")
                sections.append(f"Summary: {output.get('summary', 'N/A')}")
                sections.append(f"Content:\n{output.get('content', 'N/A')}")

            else:
                sections.append(json.dumps(output, indent=2))

        sections.append("")  # blank line between agents

    return "\n".join(sections)


def synthesize_node(state: OrchestratorState) -> dict:
    """
    LangGraph node: Synthesize all agent outputs into a final response.

    Reads:  state["task"], state["agent_outputs"], state["plan"]
    Writes: state["final_output"], state["messages"]
    """
    print(f"\n🧬 [SYNTHESIZER] Combining outputs into final response...")

    task = state["task"]
    all_outputs = _format_all_outputs(state)

    llm = _get_synthesizer_llm()
    user_prompt = build_synthesizer_prompt(task, all_outputs)

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    final_output = response.content

    # Log a preview
    preview = final_output[:200].replace("\n", " ")
    print(f"  📄 Preview: {preview}...")
    print(f"\n{'='*60}")
    print("✨ BRIAREUS — Task Complete")
    print(f"{'='*60}\n")

    return {
        "final_output": final_output,
        "messages": [AIMessage(content=final_output)],
    }