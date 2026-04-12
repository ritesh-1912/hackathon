---
title: Support Ticket Triage OpenEnv
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blueviolet)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

---

## 🚀 Why This Benchmark?

Support ticket triage is **mission-critical, real-world work** that tests multiple AI capabilities simultaneously:

- **Decision-making**: Agents must classify issues, assign priority, and route to correct teams
- **Domain knowledge**: Understanding support operations, compliance, and best practices
- **Safety & policy**: Drafting responses that avoid unsupported claims and protect customers
- **Multi-step reasoning**: Completing interconnected triage steps with partial credit

This environment surfaces common failure modes:

- ❌ Misclassification (e.g., security issues → billing)
- ❌ Incorrect priority (routine → P0, or vice versa)
- ❌ Policy violations (("guaranteed instant refund", "definitely fixed in 1 hour")
- ❌ Poor routing (enterprise issue → frontline team)

**Perfect for evaluating both LLM agents and rule-based baselines.**

---

## 🏗️ Architecture

### Environment Design

```
┌──────────────────────────────────────────────────────────────┐
│                    OpenEnv Interface                         │
│  reset() → Observation  │  step(Action) → (Obs, Reward...)   │
│  state() → Episode State │  health() → {"status": "ok"}     │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│         Support Ticket Triage Environment                    │
│  • Episode state management & scoring                        │
│  • Deterministic component grading                            │
│  • Policy violation penalty system                           │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│            FastAPI Server (Port 7860)                        │
│  POST /reset    → Initialize episode, return observation     │
│  GET  /reset     → Reset via query params, return observation│
│  POST /step      → Execute action, return (reward, done...)  │
│  GET  /state     → Retrieve current episode state            │
│  GET  /health    → Liveness check                            │
│  GET  /          → Root status check                         │
└──────────────────────────────────────────────────────────────┘
```

### Observation & Action Flow

**Observation** (after reset/step):

- Episode metadata (ID, task type, difficulty, step count)
- Ticket details (title, customer message, metadata)
- Action schema & allowed fields
- Component scores for the current step
- Current progress & last feedback

**Action** (agent must decide):

```json
{
  "classification": "billing",
  "priority": "p1",
  "route_to": "payments_ops",
  "summary": "Duplicate charge after order cancellation",
  "response_draft": "We are reviewing with the payments team...",
  "escalate": true,
  "confidence": 0.92
}
```

**Reward Breakdown**:

- ✅ Component scores (per grading rubric)
- ✅ Total incremental reward delta
- ✅ Policy penalty (if violated)
- ✅ Episode score (cumulative)

---

## 📋 The Three Tasks

### Task 1: **Billing Double Charge** 🟢 EASY

**Scenario**: Customer charged twice after canceling an order; needs immediate triage.

| Objective          | Details                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| **Classification** | Must be: `billing`                                                              |
| **Priority**       | Must be: `p1` (urgent but not emergency)                                        |
| **Routing**        | Must go to: `payments_ops` team                                                 |
| **Summary**        | Must mention: duplicate charge, cancelled order, refund                         |
| **Response**       | Must include: acknowledge issue, mention timeline, avoid instant refund promise |
| **Escalation**     | Should escalate: `true`                                                         |

**Baseline score**: 0.95

---

### Task 2: **Security Account Lockout** 🟡 MEDIUM

**Scenario**: Suspicious login from another country triggers account lockout; customer needs help.

| Objective          | Details                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| **Classification** | Must be: `account_security`                                                                    |
| **Priority**       | Must be: `p0` (emergency - security incident)                                                  |
| **Routing**        | Must go to: `trust_safety` team                                                                |
| **Summary**        | Must mention: suspicious login, MFA, recovery email                                            |
| **Response**       | Policy-safe language: verify identity, reset MFA, investigate (no password/guarantee promises) |
| **Escalation**     | Should escalate: `true`                                                                        |

**Baseline score**: 0.887

---

### Task 3: **Enterprise API Degradation** 🔴 HARD

**Scenario**: Enterprise customer's production workflow blocked by API timeouts after deployment.

| Objective          | Details                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Classification** | Must be: `platform_outage`                                                                      |
| **Priority**       | Must be: `p1` (critical but investigate first)                                                  |
| **Routing**        | Must go to: `platform_engineering` team                                                         |
| **Summary**        | Must mention: timeouts, deployment, production workflow                                         |
| **Response**       | Careful language: apologize, investigating status, next update (no "root cause", "fixed in 1h") |
| **Escalation**     | Should escalate: `true`                                                                         |

**Baseline score**: 0.872

### Step Budget

- Easy task: 5 steps max
- Medium task: 5 steps max
- Hard task: 8 steps max

Episodes also end early if the cumulative score reaches 0.95 or higher.

---

## 📊 Grading & Rewards

Each component is scored deterministically:

- **Exact match**: 1.0 if correct value, 0.0 otherwise
- **Keywords**: Partial credit based on # matching keywords
- **Policy**: Good keywords + penalty for forbidden words
- **Boolean**: 1.0 or 0.0

**Incremental rewards**: Agents get partial credit as they complete components, encouraging step-by-step progress.

**Policy penalties**: Forbidden phrases like "guaranteed", "definitely fixed", "immediate access" reduce reward.

---

## 🎮 Quick Start

### 1️⃣ Local Setup

```bash
# Clone and activate environment
git clone https://github.com/ritesh-1912/hackathon.git
cd hackathon
python3 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run the Server

```bash
# Terminal 1: Start the environment server
python -m server.app --host 0.0.0.0 --port 7860

# Or directly with uvicorn
uvicorn support_triage.server:app --host 0.0.0.0 --port 7860
```

### 3️⃣ Run Baseline Agent

```bash
# Terminal 2: Run baseline (uses deterministic fallback policy)
python inference.py

# Run specific task only
TASK_ID=billing_double_charge python inference.py

# Run with Hugging Face API
API_BASE_URL=https://api-inference.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your_token_here \
python inference.py
```

### 📝 Environment Variables

| Variable                | Default                            | Purpose                                                                                        |
| ----------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `API_BASE_URL`          | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint                                                                     |
| `MODEL_NAME`            | `Qwen/Qwen2.5-72B-Instruct`        | LLM to query                                                                                   |
| `HF_TOKEN` or `API_KEY` | (required)                         | Authentication token                                                                           |
| `TASK_ID`               | (random)                           | Run specific task (billing_double_charge, security_lockout_triage, enterprise_api_degradation) |
| `BENCHMARK_NAME`        | `support-ticket-triage`            | Logging identifier                                                                             |

---

## 🐳 Docker Deployment

```bash
# Build container
docker build -t support-triage-openenv .

# Run locally
docker run -p 7860:7860 support-triage-openenv

# Or on Hugging Face Spaces: https://huggingface.co/spaces/ritesh1912/hackathon
```

---

## 📈 Validation Checklist

```bash
# Verify setup
✅ docker build .
✅ docker run -p 7860:7860 <image>
✅ openenv validate
✅ curl -X POST http://localhost:7860/reset
```

All checks must pass for submission.

---

## 🔍 Project Structure

```
hackathon/
├── README.md                           # This file
├── Dockerfile                          # Container image definition
├── pyproject.toml                      # Project metadata & dependencies
├── requirements.txt                    # Pip installable packages
├── uv.lock                             # Locked dependency versions
├── openenv.yaml                        # OpenEnv environment manifest
│
├── inference.py                        # Baseline agent + OpenAI integration
│
├── server/
│   ├── __init__.py
│   └── app.py                          # Entrypoint wrapper (calls support_triage.server:main)
│
└── support_triage/
    ├── __init__.py
    ├── server.py                       # FastAPI HTTP server + endpoints
    ├── environment.py                  # Core environment logic (step, reset, state)
    ├── models.py                       # Pydantic data models
    └── scenarios.py                    # Task definitions & component specs
```

---

## 💡 Key Features

### ✨ For Researchers

- **Deterministic grading**: Reproducible scoring across runs
- **Shaped rewards**: Incremental feedback for multi-step triage
- **Policy constraints**: Realistic safety guardrails

### 🎯 For Practitioners

- **OpenEnv standard**: Compatible with industry-standard agentic eval frameworks
- **Production-like scenarios**: Real-world support operations challenges
- **Extensible design**: Add new tasks, metrics, or difficulty levels

### 🧠 For AI/ML Engineers

- **Type-safe models**: Pydantic schemas for actions, observations, rewards
- **HTTP-first**: Easy integration with any agent framework (LangChain, CrewAI, etc.)
- **Docker-ready**: One-line deployment to Hugging Face Spaces

---

## 📌 API Reference

### `POST /reset`

Initialize environment and return the first observation directly.

**Request**:

```json
{
  "task_id": null,
  "seed": null
}
```

**Response**:

```json
{ ...observation fields... }
```

---

### `POST /step`

Execute an action and get feedback.

**Request**:

```json
{
  "action": {
    "classification": "billing",
    "priority": "p1",
    "route_to": "payments_ops",
    "summary": "...",
    "response_draft": "...",
    "escalate": true,
    "confidence": 0.92
  }
}
```

**Response**:

```json
{
  "observation": { ... },
  "reward": { "total": 0.25, "score": 0.25, ... },
  "done": false,
  "info": { "score": 0.25, "task_id": "billing_double_charge", ... }
}
```

---

### `GET /state`

Retrieve current episode state.

**Response**:

```json
{
  "episode_id": "abc123",
  "task_id": "billing_double_charge",
  "score": 0.5,
  "done": false,
  "progress": { "classification": 1.0, "priority": 1.0, ... },
  "history": [ ... ]
}
```

### `GET /health`

Liveness check used by Docker and the Hugging Face runtime.

**Response**:

```json
{ "status": "ok" }
```

---

## 🚀 Baseline Performance

| Task                       | Difficulty | Fallback Score | Notes                           |
| -------------------------- | ---------- | -------------- | ------------------------------- |
| billing_double_charge      | 🟢 Easy    | **0.953**      | Clear routing, good response    |
| security_lockout_triage    | 🟡 Medium  | **0.887**      | Handles policy constraints well |
| enterprise_api_degradation | 🔴 Hard    | **0.872**      | Harder keyword + policy target   |

**Fallback policy** (runs without API key):

- Uses heuristic rules to fill each action field
- Suitable for testing infrastructure and baseline comparisons
- Deterministic: same score on every run

---

## 🎓 Extending the Environment

### Add a New Task

1. Define scenario in `support_triage/scenarios.py`:

```python
TaskScenario(
    task_id="my_new_task",
    difficulty="hard",
    ...
)
```

2. Add component rubrics (classification, priority, etc.)

3. Register in `TASKS` tuple

4. Update `openenv.yaml`

---

## 🏆 Evaluation Metrics

Judges will assess:

- ✅ **Code quality**: Clean, documented, type-safe
- ✅ **Completeness**: All required tasks, endpoints, validation
- ✅ **Realism**: Believable support scenarios and grading
- ✅ **Extensibility**: Easy to add tasks, metrics, or scenarios
- ✅ **Deployment**: Docker works, Hugging Face integration smooth
- ✅ **Documentation**: Clear README, API docs, examples

---

## 📝 License

Open source benchmark for the Meta OpenEnv Hackathon.

---

## 🤝 Support

For issues or questions:

- 📧 Open an issue on GitHub
- 🚀 check deployment at: https://huggingface.co/spaces/ritesh1912/hackathon
- 💬 See examples in `inference.py`

---

**Built with ❤️ for the Meta Hackathon. Let's make support AI better.** 🎯
