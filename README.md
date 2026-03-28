---
<<<<<<< HEAD
title: Incident Response Env
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
title: Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - incident-response
  - sre
  - devops
  - real-world
  - multi-step-reasoning
---

# 🚨 Incident Response OpenEnv

> An OpenEnv environment where an AI agent acts as an on-call Site Reliability Engineer,
> diagnosing and resolving production incidents across a simulated microservices system.

---

## Why This Environment?

Every company running production software faces incidents — services going down, latency
spiking, cascading failures rippling through a dependency graph. Incident response is one
of the most high-stakes, time-pressured reasoning tasks in software engineering.

This environment captures that task faithfully:

- The agent reads real-looking monitoring dashboards (service metrics, PagerDuty-style alerts)
- It runs diagnostic commands (inspecting log tails from failing services)
- It must identify the **root cause** — not just the symptom — through a dependency graph
- It applies the **correct remediation**: rollback a bad deployment, scale up an overloaded
  service, or restart one stuck in a crash loop
- All under a ticking SLA clock

This is exactly what SRE teams at Google, Meta, Netflix, and Stripe do every day.

---

## Environment Design

### System Topology

Each episode presents a snapshot of a microservices system. Services have:

| Metric | Description |
|---|---|
| `status` | `healthy` / `degraded` / `down` |
| `error_rate` | Fraction of requests failing (0.0–1.0) |
| `latency_p99_ms` | 99th-percentile response time |
| `cpu_utilisation` | CPU usage (0.0–1.0) |
| `memory_utilisation` | Memory usage (0.0–1.0) |
| `recent_deployment` | Whether a deploy happened in the last 30 minutes |
| `dependencies` | Upstream services this service calls |

### Action Space

The agent submits one action per step — mirroring real SRE operations:

| Action | Description | Real-world analogue |
|---|---|---|
| `investigate` | Read logs + diagnostics for a service | `kubectl logs` / APM trace |
| `rollback` | Revert last deployment | `helm rollback` / git revert |
| `restart` | Restart a service | `kubectl rollout restart` |
| `scale_up` | Add replicas | `kubectl scale --replicas=N` |
| `escalate` | Page a human engineer | PagerDuty escalation policy |
| `resolve` | Declare incident resolved | Incident closure |

Every action requires a `rationale` — the agent's reasoning. This is scored for
specificity (mentioning service names, metrics, root-cause keywords).

### Observation Space

```json
{
  "incident_title": "[P1] Site-wide degradation — multiple services failing",
  "incident_severity": "P1",
  "step_count": 2,
  "max_steps": 10,
  "time_to_resolve_budget": 50,
  "services": [
    {
      "name": "auth-service",
      "status": "degraded",
      "error_rate": 0.30,
      "latency_p99_ms": 3200,
      "cpu_utilisation": 0.97,
      "memory_utilisation": 0.85,
      "recent_deployment": false,
      "dependencies": ["redis-cache"]
    }
  ],
  "active_alerts": [
    {
      "alert_id": "ALT-020",
      "service": "api-gateway",
      "severity": "P1",
      "message": "api-gateway: error rate 45% — multiple upstream failures"
    }
  ],
  "last_diagnostic": {
    "service": "auth-service",
    "log_tail": ["2026-03-27T16:00:05Z [ERROR] Token validation timeout — queue full"],
    "error_summary": "auth-service is CPU-saturated due to a 2x traffic spike...",
    "suggested_action": "SCALE_UP auth-service"
  },
  "action_history": [
    "Step 1: [INVESTIGATE] → api-gateway | Starting at user-facing entry point..."
  ]
}
```

---

## Tasks

### Easy — Single Service Failure
**Scenario**: Payment service crashes immediately after a bad deployment.
API Gateway begins returning 502s. Alerts fire on both services.

The agent must:
1. Investigate the API Gateway (upstream errors point to payment-service)
2. Investigate payment-service (crash logs show failed DB migration)
3. Rollback payment-service
4. Resolve

**What makes it hard enough**: the agent must read the diagnostic logs to
determine whether to `restart` or `rollback` (wrong choice = partial credit only).

---

### Medium — Cascading Failure (Multi-hop)
**Scenario**: Checkout latency exceeds 10 seconds. The root cause is
inventory-service stuck in a crash loop — but it's three hops from the
user-facing alert.

The agent must trace: `checkout → order → inventory`.

**What makes it hard**: all three services show elevated metrics. The agent
must investigate in dependency order to find the actual root cause.

---

### Hard — Site-wide Degradation with Red Herrings
**Scenario**: Four services are simultaneously degraded with P1 alerts firing.
The root cause is auth-service CPU saturation from a 2x traffic spike — but:

- `notification-service` had a recent deployment (red herring)
- `user-service` has a *worse* error rate than `auth-service`
- `auth-service` has a *lower* severity alert (P2) than the symptoms (P1)
- The fix is `scale_up` — not `restart` or `rollback`

A frontier model that naively follows alert severity will fail this scenario.

---

## Reward Function

Reward is **dense** — partial progress signals fire every step.

### Episode-end reward (90% of final score)

| Component | Weight | Measures |
|---|---|---|
| Root cause identification | 35% | Did the agent investigate the root-cause service? |
| Correct remediation | 30% | Was the right fix applied to the right service? |
| Resolution speed | 20% | Fewer steps + time budget remaining |
| Reasoning quality | 15% | Rationale specificity (keywords, service names, metrics) |

### Per-step reward (10% of final score, dense feedback)

| Trigger | Reward |
|---|---|
| Investigating root-cause service | +0.10 |
| Investigating a dependency of root cause | +0.05 |
| Correct fix on root-cause service | +0.12 |
| Any investigation (exploration) | +0.02 |

### Penalties

| Trigger | Penalty |
|---|---|
| Escalate without investigating root cause | −0.10 |
| Destructive action on a healthy service | −0.05 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns 200 |
| `GET` | `/schema` | Action + observation JSON schemas |
| `GET` | `/tasks` | All tasks with difficulty + action schema |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Current observation (no advance) |
| `POST` | `/grader` | Run grader for a task |
| `POST` | `/baseline` | Run baseline agent on all tasks |
| `WS` | `/ws` | WebSocket for persistent sessions |

### Quick Start (Python)

```python
import requests

BASE = "https://YOUR_HF_USERNAME-incident-response-env.hf.space"

# 1. Start episode (hard scenario)
obs = requests.post(f"{BASE}/reset", json={
    "difficulty": "hard",
    "seed": 42
}).json()

print(obs["incident_title"])
# → [P1] Site-wide degradation — multiple services failing simultaneously

# 2. Investigate the most suspicious service
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "target_service": "api-gateway",
    "rationale": "Starting at the user-facing entry point to understand which upstream is failing."
}).json()

print(result["last_diagnostic"]["error_summary"])
# → API Gateway is seeing failures across all upstreams — all auth-related...

# 3. Follow the signal
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "target_service": "auth-service",
    "rationale": "Gateway logs show all failures are auth token timeouts. auth-service CPU is 97%."
}).json()

# 4. Apply the correct fix
result = requests.post(f"{BASE}/step", json={
    "action_type": "scale_up",
    "target_service": "auth-service",
    "rationale": "auth-service is CPU-saturated with 2x traffic. 2 replicas insufficient. Scaling to 6."
}).json()

# 5. Resolve
result = requests.post(f"{BASE}/step", json={
    "action_type": "resolve",
    "target_service": "auth-service",
    "rationale": "auth-service scaled. Token validation recovering. All downstream services stabilising."
}).json()

print(f"Final reward: {result['reward']}")  # → ~0.82
print(f"Done: {result['done']}")            # → True
```

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Mean |
|---|---|---|---|---|
| Random (immediate resolve) | 0.08 | 0.07 | 0.06 | 0.07 |
| Rule-based heuristic | 0.7999 | 0.7247 | 0.7606 | 0.7617 |
| Optimal action sequence | 0.8244 | 0.8100 | 0.8231 | 0.8192 |

The gap between random and heuristic confirms the environment rewards reasoning.
The gap between heuristic and optimal confirms there is headroom for LLM agents
to improve through reinforcement learning.

---

## Local Setup

### Option 1 — Python (fastest)

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/incident-response-env
cd incident-response-env
pip install openenv-core fastapi "uvicorn[standard]" pydantic openai
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860/docs` for the interactive Swagger UI.

### Option 2 — Docker

```bash
docker build -t incident-response-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-...  \  # Optional — enables GPT-4o-mini baseline
  incident-response-env
```

### Run tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
# Expected: 34 passed
```

### Run baseline

```bash
python -m baseline.baseline
# With GPT-4o-mini: export OPENAI_API_KEY=sk-...
```

---

## Deployment to Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login

# Initialise git and push
git init
git add .
git commit -m "Initial submission"
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/incident-response-env
git push hf main
```

Then paste your Space URL into the competition submission form before **7 April 2026, 11:59 PM IST**.

---

## Project Structure

```
incident-response-env/
├── server/
│   ├── app.py           # FastAPI server (openenv-core create_app)
│   ├── environment.py   # IncidentResponseEnv — core environment class
│   ├── models.py        # Typed Pydantic models: Action, Observation, State
│   ├── scenarios.py     # Incident scenario definitions (easy/medium/hard)
│   ├── reward.py        # Dense multi-component reward function
│   └── graders.py       # Deterministic graders for all 3 tasks
├── baseline/
│   └── baseline.py      # LLM + rule-based baseline agent
├── tests/
│   └── test_environment.py  # 34-test suite
├── openenv.yaml         # OpenEnv metadata spec
├── Dockerfile           # Multi-stage production build
├── pyproject.toml       # Package configuration
└── README.md            # This file
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
>>>>>>> a041aa6 (Initial submission — incident-response-env v1.0.0)
