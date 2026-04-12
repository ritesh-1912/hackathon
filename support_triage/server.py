"""FastAPI HTTP server for the support ticket triage OpenEnv.

Provides REST endpoints for the OpenEnv interface:
  - GET  /       : Health check
  - POST /reset  : Initialize episode (with optional query params)
  - GET  /reset  : Initialize episode via GET (compatibility)
  - POST /step   : Execute action
  - GET  /state  : Get current episode state

Thread-safe with a global lock protecting environment state.
Supports both query parameters and JSON request bodies for flexibility.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .environment import SupportTicketEnv
from .models import SupportTicketAction

app = FastAPI(title="Support Ticket Triage OpenEnv", version="1.0.0")
_env = SupportTicketEnv()
_lock = Lock()


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: dict[str, Any]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    seed: int | None = None


def _reset_env(task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
    with _lock:
        try:
            observation = _env.reset(task_id=task_id, seed=seed)
            state = _env.state()
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "state": state.model_dump(),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "support-ticket-triage-openenv",
        "status": "ok",
        "message": "OpenEnv support-ticket triage service is running.",
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, Any]:
    """POST /reset: Initialize a new episode.
    
    Request body (optional):
        task_id: str | None - Specific task to run
        seed: int | None - Random seed for task selection
    
    Returns:
        observation: Initial observation for the episode
        state: Complete episode state
    """
    payload = request or ResetRequest()
    return _reset_env(task_id=payload.task_id, seed=payload.seed)


@app.get("/reset")
def reset_get(task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
    """GET /reset: Initialize a new episode via query parameters.
    
    Query parameters (optional):
        task_id: str | None - Specific task to run
        seed: int | None - Random seed for task selection
    
    Returns: Same as POST /reset
    """
    return _reset_env(task_id=task_id, seed=seed)


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    """POST /step: Execute an action in the current episode.
    
    Request body:
        action: dict with keys from {classification, priority, route_to, summary, response_draft, escalate, confidence}
    
    Returns:
        observation: Updated observation after action
        reward: RewardBreakdown with component scores and episode score
        done: Whether episode is complete
        info: Metadata (task_id, score, done, step_count, progress, history)
    """
    with _lock:
        try:
            observation, reward, done, info = _env.step(SupportTicketAction.model_validate(request.action))
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    """GET /state: Retrieve the current episode state.
    
    Returns:
        episode_id: Unique ID for this episode
        task_id: The active task (billing_double_charge, etc.)
        task_title: Human-readable task description
        difficulty: easy | medium | hard
        score: Current cumulative score [0.0, 1.0]
        done: Whether episode is complete
        progress: Dict of component_name → score
        history: List of (step, action, feedback) dicts
    """
    with _lock:
        try:
            current_state = _env.state()
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return current_state.model_dump()


def main() -> None:
    parser = ArgumentParser(description="Run the support ticket triage OpenEnv server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
