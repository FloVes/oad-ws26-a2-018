"""
Pydantic schemas for structured LLM outputs, user preferences, and recommendation payloads.

NOTE: The schemas below are provided as examples of structured output modeling.
You can modify, rename, extend, or replace these models to fit your domain data
and recommendation strategy.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """
    Example schema: Extracted and refined user preferences across conversation turns.

    NOTE: You can adapt fields to their specific domain attributes.
    """
    categories: List[str] = Field(default_factory=list, description="Preferred categories")
    key_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific attributes (e.g., ingredients, release year, price range, brand)"
    )


class RecommendationItem(BaseModel):
    """
    Example schema: Grounded recommendation item with rationale and transparent explanation.
    """
    item_id: Optional[str] = Field(default=None, description="Catalog item identifier if available")
    title: str = Field(description="Name or title of recommended item")
    description: Optional[str] = Field(default=None, description="Brief item summary or attributes")
    explanation: str = Field(description="Transparent explanation of why this item matches user preferences")


class RecommendationResponse(BaseModel):
    """
    Example schema: Structured output returned by recommendation generation.
    """
    recommendations: List[RecommendationItem] = Field(default_factory=list, description="List of recommended items")
    summary: str = Field(description="Summary narrative explaining the overall recommendation set")
