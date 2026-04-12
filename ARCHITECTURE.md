# 🏗️ Architecture & Design

A deep dive into the support ticket triage benchmark's design philosophy, grading system, and extensibility patterns.

---

## Core Design Principles

### 1. **Deterministic Grading**
Every evaluation is reproducible and explainable.
- No randomness in scoring (except episode sampling)
- Component scores are deterministic functions of action values
- Agents can learn patterns from feedback

### 2. **Shaped Rewards (Incremental Progress)**
Agents earn partial credit for progress toward completeness.
- Early steps reward "broad strokes" (correct classification, priority, routing)
- Later steps enable "fine-tuning" (careful language, policy compliance)
- Components unlock at specific steps (e.g., routing at step 2)

**Why:** Encourages systematic problem-solving. Bad agents won't get stuck at step 1.

### 3. **Policy Constraints (Real-World Safety)**
Realistic guardrails on agent language and reasoning.
- Forbidden words in responses incur direct penalties
- Encourages agents to avoid common failure modes
- Examples: "guaranteed", "immediately", "fixed in 1 hour"

**Why:** Mirrors real support ops where policy violations have consequences.

### 4. **Type-Safe Models**
All data is validated via Pydantic.
- Catches invalid actions early
- HTTP layer rejects malformed requests
- Client SDKs have IDE autocomplete support

---

## Grading System Deep Dive

### Component Scoring Methods

Each task has 4-6 **components** (like a rubric), scored independently:

#### **Exact Match** (binary: 0.0 or 1.0)
Example: `classification == "billing"` → 1.0, else 0.0
- Used for objective decisions (classification, routing, priority)
- High weight because correctness is clear-cut

#### **Keywords** (partial credit)
Example: Response must mention "refund", "review", "timeline"
- Score = (# matched keywords) / (# required keywords)
- Allows flexibility in phrasing
- Used for summaries and response drafts

#### **Boolean** (binary: 0.0 or 1.0)
Example: `escalate == true` → 1.0, else 0.0
- Simple yes/no decisions
- Low weight (0.08–0.12)

#### **Policy** (keywords + forbidden words)
Example: Good words: ["verify identity", "support can help"] | Forbidden: ["password", "immediately unlock"]
- Score = (good hits / total good) - (bad hits / total bad)
- Penalties for policy violations
- Most complex scoring method
- Used for high-stakes components (security, enterprise responses)

### Example: billing_double_charge Task

| Component | Kind | Field | Weight | Evaluation |
|-----------|------|-------|--------|------------|
| classification | exact | classification | 0.22 | Must be "billing" |
| priority | exact | priority | 0.18 | Must be "p1" |
| routing | exact | route_to | 0.18 | Must be "payments_ops" |
| summary | keywords | summary | 0.20 | Mention: ["duplicate charge", "cancelled order", "refund"] |
| response | keywords | response_draft | 0.14 | Mention: ["review", "payment team", "timeline"] |
| escalation | boolean | escalate | 0.08 | Must be true |

**Total weight**: 1.0 (weighted sum of all components)

---

## Episode Lifecycle

```
Agent                           Environment
  │                                │
  ├── POST /reset ───────────────>│
  │   { task_id?, seed? }         │
  │                               ├─ Sample task (or use task_id)
  │                               ├─ Initialize EpisodeState
  │<────────────────── Observation │
  │   (ticket, task_id, allowed_actions, ...)
  │
  ├── POST /step ───────────────>│
  │   { action: {...} }           │
  │                               ├─ Validate action schema
  │                               ├─ Score each component
  │                               ├─ Apply policy penalties
  │                               ├─ Calculate reward (Δ score)
  │                               ├─ Check done (score >= 0.999 or step >= 3)
  │<────────────────── (Obs, Reward, done, info)
  │
  ├── [If not done, repeat STEP]
  │
  └── GET /state
      (retrieve full history)
```

---

## Reward Structure

```python
RewardBreakdown {
    "total": 0.35,              # Incremental reward (Δ score from this step)
    "progress_delta": 0.35,     # Same as total
    "bonus": 0.35,              # Max(Δ, 0) - reward for moving forward
    "penalty": -0.02,           # Negative for policy violations
    "component_scores": {       # Breakdown of all component scores
        "classification": 1.0,
        "priority": 1.0,
        "routing": 1.0,
        "summary": 0.67,
        "response": 0.25,
        "escalation": 1.0
    },
    "score": 0.35               # Cumulative episode score [0.0, 1.0]
}
```

---

## Task Difficulty Progression

### 🟢 Easy: billing_double_charge

**Why easy:**
- Clear, objective decisions (classification, priority, routing)
- Straightforward domain knowledge (payments team)
- Minimal policy constraints
- Baseline score: 0.95

**Challenge:**
- Must avoid promising "instant refund"
- Requires understanding refund timelines

---

### 🟡 Medium: security_lockout_triage

**Why medium:**
- Introduces urgency escalation (P0 instead of P1)
- Adds policy constraints (careful security language)
- Requires domain understanding (MFA, verification)
- Baseline score: 0.89

**Challenge:**
- Can't promise "immediately unlock" or "guarantee"
- Must balance customer empathy with security rigor
- Forbidden words: ["password", "guarantee", "immediately unlock"]

---

### 🔴 Hard: enterprise_api_degradation

**Why hard:**
- Enterprise customer = high stakes
- Production workflow disruption = urgency
- Highly constrained language (RCA, fix time promises forbidden)
- Baseline score: 0.89

**Challenge:**
- Must *not* claim to know root cause
- Must *not* promise a fix timeline
- Must appear professional and transparent
- Forbidden words: ["root cause", "fix in 1 hour", "RCA complete"]

---

## Extension Points

### Adding a New Task

1. **Define scenario** in `support_triage/scenarios.py`:
   ```python
   TaskScenario(
       task_id="my_task_id",
       task_title="...",
       difficulty="medium",
       ticket_id="TCK-9999",
       title="...",
       customer_message="...",
       metadata=TicketMetadata(...),
       task_instruction="...",
       allowed_actions=(...),
       components=(
           ComponentSpec("name1", "field1", "exact", 0.25, expected="value"),
           ComponentSpec("name2", "field2", "keywords", 0.30, keywords=("word1", "word2")),
           # ...
       ),
   )
   ```

2. **Add to TASKS tuple** (keep in order: easy, medium, hard)

3. **Register in openenv.yaml**:
   ```yaml
   tasks:
     - task_id: my_task_id
       task_title: ...
       difficulty: medium
   ```

4. **Test locally**:
   ```bash
   TASK_ID=my_task_id python inference.py
   ```

### Adding New Evaluation Metrics

Edit `support_triage/environment.py`:
```python
def _compute_custom_metric(self, action: SupportTicketAction) -> float:
    """Custom metric for research."""
    # e.g., sentiment analysis of response
    return score

# In step():
custom_metric = self._compute_custom_metric(action)
reward.custom_metrics = {"sentiment": custom_metric}
```

---

## Performance Characteristics

### Baseline Agent (deterministic fallback)

Runs without an API key using hand-crafted heuristics.

| Task | Strategy | Score |
|------|----------|-------|
| billing_double_charge | Exact heuristic match | 0.95 |
| security_lockout_triage | Exact heuristic match | 0.89 |
| enterprise_api_degradation | Exact heuristic match | 0.89 |

### LLM Agent (with API)

Depends on model quality:
- Qwen/Qwen2.5-72B: ~0.85–0.92 (good reasoning)
- GPT-4: ~0.90–0.95 (excellent)
- GPT-3.5: ~0.75–0.85 (reasonable)
- Smaller models: ~0.60–0.75 (inconsistent policy compliance)

---

## API Reference Summary

### `POST /reset`
Initialize episode.
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "billing_double_charge"}'
```

### `POST /step`
Execute action.
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "classification": "billing",
      "priority": "p1",
      "route_to": "payments_ops",
      "summary": "...",
      "response_draft": "...",
      "escalate": true,
      "confidence": 0.92
    }
  }'
```

### `GET /state`
Retrieve episode state.
```bash
curl http://localhost:7860/state
```

---

## Code Quality Standards

### Type Safety
- All public functions use type hints
- No `Any` in critical paths
- Pydantic for external IO validation

### Testability
- Deterministic (same seed → same score)
- No sleep/network in core logic
- Mock-friendly HTTP layer

### Documentation
- Module docstrings explain purpose
- Class docstrings include examples
- Method docstrings describe contracts

---

## Next Steps for Judges

1. **Run locally**: `docker run -p 7860:7860 support-triage-openenv`
2. **Test via HTTP**: POST /reset, POST /step, GET /state
3. **Try baseline**: `TASK_ID=billing_double_charge python inference.py`
4. **Extend**: Add a custom task or new evaluation metric
5. **Deploy**: Push to Hugging Face Spaces for live evaluation

---

**Design decisions prioritize clarity, reproducibility, and educational value.**
