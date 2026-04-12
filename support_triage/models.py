"""Pydantic data models for support ticket triage environment.

Defines type-safe schemas for observations, actions, rewards, and state tracking.
All models use ConfigDict(extra="forbid") to catch unexpected fields during validation.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Difficulty = Literal["easy", "medium", "hard"]


class TicketMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_tier: Literal["free", "pro", "enterprise"]
    channel: Literal["email", "chat", "voice"]
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]
    age_minutes: int = Field(..., ge=0)
    language: str = "en"
    order_value_usd: float = Field(..., ge=0)


class SupportTicketAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Optional[str] = None
    priority: Optional[str] = None
    route_to: Optional[str] = None
    summary: Optional[str] = None
    response_draft: Optional[str] = None
    escalate: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SupportTicketObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    task_title: str
    difficulty: Difficulty
    step_count: int
    remaining_steps: int
    max_steps: int
    ticket_id: str
    title: str
    customer_message: str
    metadata: TicketMetadata
    action_schema: str
    task_instruction: str
    allowed_actions: list[str]
    progress: Dict[str, float] = Field(default_factory=dict)
    last_feedback: Optional[str] = None


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float = 0.0
    progress_delta: float = 0.0
    bonus: float = 0.0
    penalty: float = 0.0
    component_scores: Dict[str, float] = Field(default_factory=dict)
    score: float = 0.0


class SupportTicketState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    task_title: str
    difficulty: Difficulty
    step_count: int
    max_steps: int
    score: float
    done: bool
    observation: SupportTicketObservation
    history: list[dict[str, str]] = Field(default_factory=list)
    progress: Dict[str, float] = Field(default_factory=dict)
