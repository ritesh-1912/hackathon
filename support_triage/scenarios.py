from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import TicketMetadata

ComponentKind = Literal["exact", "keywords", "boolean", "policy"]


@dataclass(frozen=True)
class ComponentSpec:
    """Grading specification for a single action component.
    
    Attributes:
        name: Identifier (e.g., \"classification\", \"routing\")
        field: Action field to grade (e.g., \"classification\")
        kind: Grading method (exact|keywords|boolean|policy)
        weight: Contribution to total score [0.0, 1.0]
        expected: Expected value (for exact/boolean)
        keywords: Required/bonus keywords (for keywords/policy)
        forbidden_keywords: Words that incur penalty (for policy)
        unlock_step: First step this component is scored (default: 1)
    \"\"\"\n    name: str
    field: str
    kind: ComponentKind
    weight: float
    expected: str | bool | None = None
    keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()
    unlock_step: int = 1


@dataclass(frozen=True)
class TaskScenario:
    task_id: str
    task_title: str
    difficulty: Literal["easy", "medium", "hard"]
    ticket_id: str
    title: str
    customer_message: str
    metadata: TicketMetadata
    task_instruction: str
    allowed_actions: tuple[str, ...]
    components: tuple[ComponentSpec, ...]
    max_steps: int = 3


TASKS: tuple[TaskScenario, ...] = (
    TaskScenario(
        task_id="billing_double_charge",
        task_title="Resolve a duplicate charge refund request",
        difficulty="easy",
        ticket_id="TCK-1042",
        title="Customer was charged twice after canceling an order",
        customer_message=(
            "I cancelled order 18814 yesterday and my card was charged twice anyway. "
            "I need the duplicate charge reviewed and a refund started today."
        ),
        metadata=TicketMetadata(
            customer_tier="pro",
            channel="email",
            sentiment="negative",
            urgency="high",
            age_minutes=120,
            language="en",
            order_value_usd=248.50,
        ),
        task_instruction=(
            "Classify the issue, assign the correct priority, and route it to the payments team. "
            "Then write a short summary and a safe response that acknowledges the duplicate charge "
            "without promising an instant refund."
        ),
        allowed_actions=(
            "classification",
            "priority",
            "route_to",
            "summary",
            "response_draft",
            "escalate",
            "confidence",
        ),
        components=(
            ComponentSpec("classification", "classification", "exact", 0.22, expected="billing", unlock_step=1),
            ComponentSpec("priority", "priority", "exact", 0.18, expected="p1", unlock_step=1),
            ComponentSpec("routing", "route_to", "exact", 0.18, expected="payments_ops", unlock_step=2),
            ComponentSpec(
                "summary",
                "summary",
                "keywords",
                0.20,
                keywords=("duplicate charge", "cancelled order", "refund"),
                unlock_step=2,
            ),
            ComponentSpec(
                "response",
                "response_draft",
                "keywords",
                0.14,
                keywords=("review", "payment team", "timeline"),
                unlock_step=3,
            ),
            ComponentSpec("escalation", "escalate", "boolean", 0.08, expected=True, unlock_step=3),
        ),
    ),
    TaskScenario(
        task_id="security_lockout_triage",
        task_title="Investigate a suspicious account lockout",
        difficulty="medium",
        ticket_id="TCK-2077",
        title="User locked out after suspicious login attempt",
        customer_message=(
            "My account was locked after a login from another country. The MFA code stopped working "
            "and my recovery email is old. Please secure the account and help me regain access."
        ),
        metadata=TicketMetadata(
            customer_tier="enterprise",
            channel="chat",
            sentiment="negative",
            urgency="high",
            age_minutes=35,
            language="en",
            order_value_usd=0.0,
        ),
        task_instruction=(
            "Treat this as a security-sensitive ticket. Classify it correctly, prioritize it as urgent, "
            "route it to trust and safety, and write a policy-safe response that asks for the right next steps "
            "without exposing sensitive account details."
        ),
        allowed_actions=(
            "classification",
            "priority",
            "route_to",
            "summary",
            "response_draft",
            "escalate",
            "confidence",
        ),
        components=(
            ComponentSpec("classification", "classification", "exact", 0.20, expected="account_security", unlock_step=1),
            ComponentSpec("priority", "priority", "exact", 0.16, expected="p0", unlock_step=1),
            ComponentSpec("routing", "route_to", "exact", 0.18, expected="trust_safety", unlock_step=2),
            ComponentSpec(
                "summary",
                "summary",
                "keywords",
                0.18,
                keywords=("suspicious login", "mfa", "recovery email"),
                unlock_step=2,
            ),
            ComponentSpec(
                "response",
                "response_draft",
                "keywords",
                0.16,
                keywords=("secure", "reset", "investigate"),
                unlock_step=3,
            ),
            ComponentSpec(
                "policy",
                "response_draft",
                "policy",
                0.12,
                keywords=("verify identity", "support can help"),
                forbidden_keywords=("password", "guarantee", "immediately unlock"),
                unlock_step=3,
            ),
        ),
    ),
    TaskScenario(
        task_id="enterprise_api_degradation",
        task_title="Draft a response for an enterprise API outage",
        difficulty="hard",
        ticket_id="TCK-3319",
        title="Enterprise customer reports timeouts after a deployment",
        customer_message=(
            "The API started timing out right after yesterday's deployment. We need an RCA, a status update, "
            "and clear next steps because our production workflow is blocked."
        ),
        metadata=TicketMetadata(
            customer_tier="enterprise",
            channel="email",
            sentiment="negative",
            urgency="high",
            age_minutes=18,
            language="en",
            order_value_usd=12000.0,
        ),
        task_instruction=(
            "Handle this as a platform incident. Classify it, assign the correct priority, route it to platform "
            "engineering, then produce a short summary and a careful customer-facing response that acknowledges "
            "the outage without claiming a root cause or promising a fix time."
        ),
        allowed_actions=(
            "classification",
            "priority",
            "route_to",
            "summary",
            "response_draft",
            "escalate",
            "confidence",
        ),
        components=(
            ComponentSpec("classification", "classification", "exact", 0.18, expected="platform_outage", unlock_step=1),
            ComponentSpec("priority", "priority", "exact", 0.16, expected="p1", unlock_step=1),
            ComponentSpec("routing", "route_to", "exact", 0.16, expected="platform_engineering", unlock_step=2),
            ComponentSpec(
                "summary",
                "summary",
                "keywords",
                0.18,
                keywords=("timeouts", "deployment", "production workflow"),
                unlock_step=2,
            ),
            ComponentSpec(
                "response",
                "response_draft",
                "keywords",
                0.16,
                keywords=("apologize", "investigating", "status"),
                unlock_step=3,
            ),
            ComponentSpec(
                "policy",
                "response_draft",
                "policy",
                0.16,
                keywords=("we are investigating", "next update"),
                forbidden_keywords=("root cause", "fix in 1 hour", "RCA complete"),
                unlock_step=3,
            ),
        ),
    ),
)

TASK_BY_ID = {task.task_id: task for task in TASKS}
