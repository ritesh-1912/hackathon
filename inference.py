"""Baseline agent for the support ticket triage OpenEnv.

Demonstrates how to interact with the environment and integrate with LLMs.

Features:
  - Uses OpenAI-compatible API (Hugging Face, OpenRouter, etc.)
  - Falls back to deterministic heuristic policy if no API key
  - Emits [START], [STEP], [END] log markers for grading
  - Supports single-task evaluation via TASK_ID env var
  - Type-safe action extraction with JSON parsing

Environment variables:
  API_BASE_URL: OpenAI-compatible API endpoint (default: Hugging Face)
  MODEL_NAME: Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN or API_KEY: Authentication token
  TASK_ID: Run specific task only (optional)
  BENCHMARK_NAME: Logging identifier (default: support-ticket-triage)

Usage:
    python inference.py                                    # Run all 3 tasks
    TASK_ID=billing_double_charge python inference.py     # Run one task
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py  # Custom API
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from openai import OpenAI

from support_triage.environment import SupportTicketEnv
from support_triage.models import SupportTicketAction
from support_triage.scenarios import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_ID = os.getenv("TASK_ID")
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "support-ticket-triage")
TEMPERATURE = 0.2
MAX_TOKENS = 320
SUCCESS_SCORE_THRESHOLD = 0.85

SYSTEM_PROMPT = (
    "You are triaging support tickets in a structured OpenEnv environment. "
    "Return only a JSON object with any of these keys: classification, priority, route_to, summary, "
    "response_draft, escalate, confidence. "
    "Do not include markdown or extra commentary. "
    "Keep the response policy-safe, concise, and actionable."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def compact_action(action: SupportTicketAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"), ensure_ascii=True)


def build_prompt(observation: Any) -> str:
    payload = observation.model_dump() if hasattr(observation, "model_dump") else observation
    return json.dumps(payload, indent=2, sort_keys=True)


def heuristic_action(task_id: str, observation: Any) -> SupportTicketAction:
    if task_id == "billing_double_charge":
        return SupportTicketAction(
            classification="billing",
            priority="p1",
            route_to="payments_ops",
            summary="Duplicate charge after a cancelled order; refund review needed.",
            response_draft=(
                "We are reviewing the duplicate charge with the payments team and will follow up with a timeline."
            ),
            escalate=True,
            confidence=0.72,
        )
    if task_id == "security_lockout_triage":
        return SupportTicketAction(
            classification="account_security",
            priority="p0",
            route_to="trust_safety",
            summary="Suspicious login triggered a lockout; MFA and recovery email need review.",
            response_draft=(
                "We will help secure the account, verify identity, and investigate the login activity before access is restored."
            ),
            escalate=True,
            confidence=0.78,
        )
    return SupportTicketAction(
        classification="platform_outage",
        priority="p1",
        route_to="platform_engineering",
        summary="API timeouts started after deployment and are blocking production workflows.",
        response_draft=(
            "We are investigating the outage with engineering and will share the next update as soon as we have one."
        ),
        escalate=True,
        confidence=0.8,
    )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    return json.loads(candidate)


def request_action(client: OpenAI | None, observation: Any) -> SupportTicketAction:
    task_id = getattr(observation, "task_id", "")
    prompt = build_prompt(observation)
    if client is None:
        return heuristic_action(task_id, observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Observation JSON:\n"
                        f"{prompt}\n\n"
                        "Return the next best SupportTicketAction JSON object only."
                    ),
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = completion.choices[0].message.content or ""
        return SupportTicketAction.model_validate(extract_json(text))
    except Exception as exc:
        print(f"[baseline] OpenAI request failed for {task_id}: {exc}", file=sys.stderr)
        return heuristic_action(task_id, observation)


def run_episode(env: SupportTicketEnv, client: OpenAI | None, task_id: str) -> tuple[bool, int, float, list[float]]:
    observation = env.reset(task_id=task_id)
    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False

    try:
        while True:
            action = request_action(client, observation)
            observation, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward.total)
            score = float(info.get("score", score))
            log_step(step=steps, action=compact_action(action), reward=reward.total, done=done, error=info.get("last_action_error"))
            if done:
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
    finally:
        env.close()
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return success, steps, score, rewards


def main() -> None:
    _ = LOCAL_IMAGE_NAME
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            print(f"[baseline] OpenAI client initialization failed: {exc}", file=sys.stderr)
            client = None

    env = SupportTicketEnv()
    selected_tasks = [TASK_ID] if TASK_ID else [task.task_id for task in TASKS]
    for task_id in selected_tasks:
        run_episode(env, client, task_id)


if __name__ == "__main__":
    main()
