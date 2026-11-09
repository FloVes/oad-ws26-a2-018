"""
Agent state schema definition for the LangGraph recommender workflow.

NOTE: The state definition below is an illustrative example. You are free
to extend, rename, or customize the fields in AgentState (or use custom Pydantic/dataclass models)
to suit your specific multi-turn dialogue, retrieval, and guardrail requirements.
"""

from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Represents an example conversational recommender state.

    NOTE: Students can add, rename, or modify fields as needed for their workflow.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_preferences: Dict[str, Any]
    retrieved_items: List[Dict[str, Any]]
