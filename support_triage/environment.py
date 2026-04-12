"""Support Ticket Triage Environment.

Core environment logic for the OpenEnv support ticket triage benchmark.
Implements the standard OpenEnv interface (reset, step, state) with deterministic
grading, shaped rewards, and policy-based penalties.

Key features:
  - Deterministic component-based scoring
  - Incremental rewards for partial progress
  - Policy violation penalties
  - Type-safe Pydantic models

Example:
    >>> env = SupportTicketEnv()
    >>> obs = env.reset(task_id="billing_double_charge")
    >>> action = SupportTicketAction(classification="billing", priority="p1", ...)
    >>> obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4

from .models import RewardBreakdown, SupportTicketAction, SupportTicketObservation, SupportTicketState
from .scenarios import ComponentSpec, TASKS, TASK_BY_ID, TaskScenario


@dataclass
class EpisodeState:
    """Mutable episode state tracking.
    
    Attributes:
        task: The active TaskScenario for this episode.
        episode_id: Unique episode identifier (hex UUID).
        step_count: Number of steps taken so far.
        score: Cumulative episode score [0.0, 1.0].
        done: Whether episode is complete.
        progress: Per-component scores (ComponentSpec.name → [0.0, 1.0]).
        history: List of (step, action, feedback) tuples.
        last_feedback: Human-readable feedback from last step.
    """
    task: TaskScenario
    episode_id: str = field(default_factory=lambda: uuid4().hex)
    step_count: int = 0
    score: float = 0.0
    done: bool = False
    progress: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    last_feedback: str | None = None


class SupportTicketEnv:
    """A compact OpenEnv-style environment for support-ticket triage.
    
    Implements the standard OpenEnv interface:
      - reset(task_id, seed) → Observation
      - step(action) → (Observation, RewardBreakdown, done: bool, info: dict)
      - state() → SupportTicketState
      - close() → None
    
    Features:
      - 3 tasks: easy (billing), medium (security), hard (enterprise)
      - Deterministic component grading (exact, keywords, policy, boolean)
      - Shaped rewards: incremental credit for partial progress
      - Policy penalties: forbidden words reduce reward
      - Max 3 steps per episode
    
    Args:
        tasks: Optional list of TaskScenarios. Defaults to all 3 tasks.
    
    Raises:
        ValueError: If no tasks are provided.
    """

    def __init__(self, tasks: Iterable[TaskScenario] | None = None) -> None:
        self._tasks = tuple(tasks or TASKS)
        if not self._tasks:
            raise ValueError("SupportTicketEnv requires at least one task scenario")
        self._state: EpisodeState | None = None

    def reset(self, task_id: str | None = None, seed: int | None = None) -> SupportTicketObservation:
        """Reset the environment and start a new episode.
        
        Args:
            task_id: Specific task to run, or None to sample based on seed.
            seed: Random seed for task selection (ignored if task_id specified).
        
        Returns:
            Initial SupportTicketObservation for the episode.
        
        Raises:
            KeyError: If task_id is not found.
        """
        if task_id is not None:
            task = TASK_BY_ID.get(task_id)
            if task is None:
                raise KeyError(f"Unknown task_id: {task_id}")
        else:
            index = 0 if seed is None else seed % len(self._tasks)
            task = self._tasks[index]

        self._state = EpisodeState(task=task)
        return self._build_observation()

    def state(self) -> SupportTicketState:
        state = self._require_state()
        observation = self._build_observation()
        return SupportTicketState(
            episode_id=state.episode_id,
            task_id=state.task.task_id,
            task_title=state.task.task_title,
            difficulty=state.task.difficulty,
            step_count=state.step_count,
            max_steps=state.task.max_steps,
            score=state.score,
            done=state.done,
            observation=observation,
            history=state.history,
            progress=dict(state.progress),
        )

    def close(self) -> None:
        self._state = None

    def step(self, action: SupportTicketAction | dict[str, Any]) -> tuple[SupportTicketObservation, RewardBreakdown, bool, dict[str, Any]]:
        """Execute an action in the environment.
        
        Args:
            action: SupportTicketAction or dict with keys from action_schema.
        
        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Updated SupportTicketObservation
            - reward: RewardBreakdown with component scores and penalties
            - done: Whether episode is complete (score >= 0.999 or max_steps reached)
            - info: Dict with task_id, score, done, step_count, progress, history
        
        Raises:
            RuntimeError: If called before reset() or after done.
            ValueError: If action is invalid (see SupportTicketAction schema).
        """
        state = self._require_state()
        if state.done:
            raise RuntimeError("Episode already finished; call reset() before stepping again.")

        parsed_action = action if isinstance(action, SupportTicketAction) else SupportTicketAction.model_validate(action)
        state.step_count += 1

        component_scores = self._score_components(state.task.components, parsed_action, state.step_count)
        previous_score = state.score
        for name, value in component_scores.items():
            state.progress[name] = max(state.progress.get(name, 0.0), value)

        raw_score = self._weighted_score(state.task.components, state.progress)
        penalty = self._policy_penalty(state.task.components, parsed_action)
        state.score = self._clamp(raw_score + penalty)
        progress_delta = state.score - previous_score

        state.done = state.score >= 0.999 or state.step_count >= state.task.max_steps
        state.last_feedback = self._build_feedback(state.task, parsed_action, component_scores, penalty)
        state.history.append(
            {
                "step": str(state.step_count),
                "action": self._compact_action(parsed_action),
                "feedback": state.last_feedback,
            }
        )

        observation = self._build_observation()
        reward = RewardBreakdown(
            total=progress_delta,
            progress_delta=progress_delta,
            bonus=max(state.score - previous_score, 0.0),
            penalty=penalty,
            component_scores=component_scores,
            score=state.score,
        )
        info = {
            "task_id": state.task.task_id,
            "difficulty": state.task.difficulty,
            "score": state.score,
            "done": state.done,
            "step_count": state.step_count,
            "last_action_error": None,
            "progress": dict(state.progress),
            "history": list(state.history),
        }
        return observation, reward, state.done, info

    def _require_state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._state

    def _build_observation(self) -> SupportTicketObservation:
        state = self._require_state()
        return SupportTicketObservation(
            episode_id=state.episode_id,
            task_id=state.task.task_id,
            task_title=state.task.task_title,
            difficulty=state.task.difficulty,
            step_count=state.step_count,
            remaining_steps=max(state.task.max_steps - state.step_count, 0),
            max_steps=state.task.max_steps,
            ticket_id=state.task.ticket_id,
            title=state.task.title,
            customer_message=state.task.customer_message,
            metadata=state.task.metadata,
            action_schema=self._action_schema(),
            task_instruction=state.task.task_instruction,
            allowed_actions=list(state.task.allowed_actions),
            progress=dict(state.progress),
            last_feedback=state.last_feedback,
        )

    def _action_schema(self) -> str:
        return (
            "Return a SupportTicketAction JSON object with any of these keys: "
            "classification, priority, route_to, summary, response_draft, escalate, confidence. "
            "Use compact, policy-safe language and avoid unsupported claims."
        )

    def _score_components(
        self,
        components: tuple[ComponentSpec, ...],
        action: SupportTicketAction,
        step_count: int,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for spec in components:
            if step_count < spec.unlock_step:
                continue
            value = self._score_component(spec, action)
            scores[spec.name] = value
        return scores

    def _score_component(self, spec: ComponentSpec, action: SupportTicketAction) -> float:
        value = getattr(action, spec.field, None)
        if spec.kind == "exact":
            return 1.0 if self._normalize(value) == self._normalize(spec.expected) else 0.0
        if spec.kind == "boolean":
            return 1.0 if value is spec.expected else 0.0
        if spec.kind == "keywords":
            text = self._normalize(value)
            if not text:
                return 0.0
            hits = sum(1 for keyword in spec.keywords if self._normalize(keyword) in text)
            return hits / max(len(spec.keywords), 1)
        if spec.kind == "policy":
            text = self._normalize(value)
            if not text:
                return 0.0
            good_hits = sum(1 for keyword in spec.keywords if self._normalize(keyword) in text)
            good_score = good_hits / max(len(spec.keywords), 1)
            bad_hits = sum(1 for keyword in spec.forbidden_keywords if self._normalize(keyword) in text)
            bad_penalty = bad_hits / max(len(spec.forbidden_keywords), 1) if spec.forbidden_keywords else 0.0
            return self._clamp(good_score - bad_penalty)
        return 0.0

    def _weighted_score(self, components: tuple[ComponentSpec, ...], progress: dict[str, float]) -> float:
        total = 0.0
        for spec in components:
            total += spec.weight * progress.get(spec.name, 0.0)
        return self._clamp(total)

    def _policy_penalty(self, components: tuple[ComponentSpec, ...], action: SupportTicketAction) -> float:
        penalty = 0.0
        response_text = self._normalize(action.response_draft)
        summary_text = self._normalize(action.summary)
        for spec in components:
            if spec.kind != "policy":
                continue
            for keyword in spec.forbidden_keywords:
                if self._normalize(keyword) in response_text or self._normalize(keyword) in summary_text:
                    penalty -= 0.08
        if action.confidence is not None and action.confidence < 0.25:
            penalty -= 0.02
        return penalty

    def _build_feedback(
        self,
        task: TaskScenario,
        action: SupportTicketAction,
        component_scores: dict[str, float],
        penalty: float,
    ) -> str:
        completed = [name for name, score in component_scores.items() if score >= 0.99]
        partial = [f"{name}:{score:.2f}" for name, score in component_scores.items() if 0.0 < score < 0.99]
        if penalty < 0:
            penalty_text = f" penalty={penalty:.2f}"
        else:
            penalty_text = ""
        if completed:
            return f"{task.task_id} progress={','.join(completed)}{' ' if partial else ''}{' '.join(partial)}{penalty_text}".strip()
        if partial:
            return f"{task.task_id} partial={' '.join(partial)}{penalty_text}".strip()
        return f"{task.task_id} no_valid_progress{penalty_text}".strip()

    def _compact_action(self, action: SupportTicketAction) -> str:
        data = action.model_dump(exclude_none=True)
        pairs = [f"{key}={self._normalize(value)}" for key, value in data.items()]
        return "{" + ",".join(pairs) + "}"

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        for token in ("\n", "\r", "\t"):
            text = text.replace(token, " ")
        return " ".join(text.split())

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
