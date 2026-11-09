"""
Example LangGraph workflow node stubs.

NOTE: The node functions below (e.g. elicitation, recommendation, guardrails)
are provided purely as illustrative examples. You are encouraged to extend,
rename, split, replace, or add new node functions to match their specific
agentic architecture and workflow design.
"""

from typing import Any, Dict
from src.agents.state import AgentState


def guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    Example node stub: Executes input/output guardrail checks on the current conversation state.

    NOTE: This is an example node. You can rename, remove, split, or adapt this function.
    TODO: Implement guardrail execution and routing logic.
    """
    raise NotImplementedError("TODO: Implement guardrail_node (or replace with custom node)")


def preference_elicitation_node(state: AgentState) -> Dict[str, Any]:
    """
    Example node stub: Analyzes conversation history, elicits missing preferences, or asks clarifying questions.

    NOTE: This is an example node. You can rename, remove, split, or adapt this function.
    TODO: Implement active preference elicitation logic.
    """
    raise NotImplementedError("TODO: Implement elicitation_node (or replace with custom node)")


def recommendation_node(state: AgentState) -> Dict[str, Any]:
    """
    Example node stub: Retrieves grounded catalog items via RAG and formats recommendations with explanations.

    NOTE: This is an example node. You can rename, remove, split, or adapt this function.
    TODO: Implement catalog retrieval and recommendation generation logic.
    """
    raise NotImplementedError("TODO: Implement recommendation_node (or replace with custom node)")
