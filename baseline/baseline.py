"""
baseline.py
===========
Baseline inference script for the Incident Response environment.

Strategy
--------
The baseline agent uses an LLM (GPT-4o-mini via the OpenAI API) to reason
about each incident and take actions. If OPENAI_API_KEY is not set, it falls
back to a deterministic rule-based heuristic that still produces valid,
reproducible scores.

The rule-based heuristic implements:
  1. Investigate the service with the highest error_rate
  2. Investigate the root-cause service (determined from signals)
  3. Apply the statistically most-likely fix based on signals
  4. Resolve

This ensures the /baseline endpoint always returns scores without requiring
any API credentials.

Usage
-----
    # Run directly (produces console output)
    python -m baseline.baseline

    # Called by the /baseline API endpoint
    from baseline.baseline import run_baseline_agent
    scores = run_baseline_agent()
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

# Allow running directly from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import IncidentResponseEnv
from server.models import ActionType, IncidentAction, IncidentObservation


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

You will receive a JSON description of the current system state including:
- Active alerts with severity levels
- Per-service metrics (error_rate, latency_p99_ms, cpu_utilisation, memory_utilisation)
- Which services had recent deployments
- Results from any investigations you have already run

Your job: take ONE action per turn to diagnose and resolve the incident as quickly as possible.

Available actions:
  - investigate  : Inspect logs and diagnostics for a specific service
  - restart      : Restart a service (clears transient faults / crash loops)
  - rollback     : Roll back to the previous deployment (for bad deployments)
  - scale_up     : Add more replicas (for resource exhaustion / traffic spikes)
  - escalate     : Page a human (last resort)
  - resolve      : Declare incident resolved (only when you are confident)

Respond ONLY with a JSON object — no preamble, no markdown:
{
  "action_type": "<one of: investigate | restart | rollback | scale_up | escalate | resolve>",
  "target_service": "<exact service name from the topology>",
  "rationale": "<your detailed reasoning — be specific about metrics and log findings>"
}"""


# ---------------------------------------------------------------------------
# LLM-powered agent
# ---------------------------------------------------------------------------


def _call_llm(observation: IncidentObservation) -> IncidentAction:
    """
    Call the OpenAI API to decide the next action.
    Falls back to heuristic if API key is not available.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return _heuristic_action(observation)

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        # Serialise observation to a compact JSON prompt
        obs_dict = {
            "incident_title": observation.incident_title,
            "incident_severity": observation.incident_severity,
            "step": observation.step_count,
            "time_remaining_minutes": observation.time_to_resolve_budget,
            "active_alerts": [
                {
                    "service": a.service,
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in observation.active_alerts
            ],
            "services": [
                {
                    "name": s.name,
                    "status": s.status,
                    "error_rate": s.error_rate,
                    "latency_p99_ms": s.latency_p99_ms,
                    "cpu_utilisation": s.cpu_utilisation,
                    "memory_utilisation": s.memory_utilisation,
                    "recent_deployment": s.recent_deployment,
                    "dependencies": s.dependencies,
                }
                for s in observation.services
            ],
            "last_diagnostic": (
                {
                    "service": observation.last_diagnostic.service,
                    "error_summary": observation.last_diagnostic.error_summary,
                    "logs": observation.last_diagnostic.log_tail,
                    "suggested_action": observation.last_diagnostic.suggested_action,
                }
                if observation.last_diagnostic
                else None
            ),
            "action_history": observation.action_history,
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs_dict, indent=2)},
            ],
            temperature=0,
            seed=42,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return IncidentAction(**data)

    except Exception as exc:
        print(f"[baseline] LLM call failed ({exc}). Falling back to heuristic.")
        return _heuristic_action(observation)


# ---------------------------------------------------------------------------
# Rule-based heuristic fallback
# ---------------------------------------------------------------------------


def _heuristic_action(observation: IncidentObservation) -> IncidentAction:
    """
    Deterministic rule-based SRE agent.

    Decision logic (in priority order):
      1. If a diagnostic suggests an action → take it
      2. If a service is DOWN and not yet investigated → investigate it
      3. If a service has a recent deployment and high error rate → investigate it
      4. If already investigated the worst service → apply fix based on signals
      5. Investigate the service with the highest error rate
      6. If all services investigated → resolve
    """
    investigated_in_history = set()
    for entry in observation.action_history:
        if "[INVESTIGATE]" in entry:
            # Extract service name from history log format
            parts = entry.split("→")
            if len(parts) > 1:
                svc = parts[1].split("|")[0].strip()
                investigated_in_history.add(svc)

    # If last diagnostic has a suggested action → parse and act
    if (
        observation.last_diagnostic
        and observation.last_diagnostic.suggested_action
    ):
        hint = observation.last_diagnostic.suggested_action.lower()
        target = observation.last_diagnostic.service

        if "rollback" in hint:
            return IncidentAction(
                action_type=ActionType.ROLLBACK,
                target_service=target,
                rationale=(
                    f"Diagnostic for {target} explicitly recommends rollback: "
                    f"{observation.last_diagnostic.suggested_action}"
                ),
            )
        if "scale" in hint or "replica" in hint:
            return IncidentAction(
                action_type=ActionType.SCALE_UP,
                target_service=target,
                rationale=(
                    f"Diagnostic for {target} recommends scale-up: "
                    f"{observation.last_diagnostic.suggested_action}"
                ),
            )
        if "restart" in hint:
            return IncidentAction(
                action_type=ActionType.RESTART,
                target_service=target,
                rationale=(
                    f"Diagnostic for {target} recommends restart: "
                    f"{observation.last_diagnostic.suggested_action}"
                ),
            )
        if "resolve" in hint:
            return IncidentAction(
                action_type=ActionType.RESOLVE,
                target_service=target,
                rationale="Diagnostic confirms resolution is appropriate.",
            )

    # Find down services not yet investigated
    down_services = [
        s for s in observation.services
        if s.status == "down" and s.name not in investigated_in_history
    ]
    if down_services:
        target = down_services[0].name
        return IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service=target,
            rationale=(
                f"{target} is DOWN with error_rate={down_services[0].error_rate:.0%}. "
                "Investigating as priority target."
            ),
        )

    # Find recent-deployment services with high error rates
    deployment_services = [
        s for s in observation.services
        if s.recent_deployment
        and s.error_rate > 0.20
        and s.name not in investigated_in_history
    ]
    if deployment_services:
        target = deployment_services[0].name
        return IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service=target,
            rationale=(
                f"{target} had a recent deployment and has error_rate="
                f"{deployment_services[0].error_rate:.0%}. Investigating for bad deploy."
            ),
        )

    # Investigate highest-error-rate service not yet seen
    not_investigated = [
        s for s in observation.services
        if s.name not in investigated_in_history
    ]
    if not_investigated:
        target = max(not_investigated, key=lambda s: s.error_rate)
        return IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service=target.name,
            rationale=(
                f"{target.name} has the highest error rate among uninvestigated services: "
                f"error_rate={target.error_rate:.0%}, latency={target.latency_p99_ms:.0f}ms"
            ),
        )

    # All services investigated — resolve
    return IncidentAction(
        action_type=ActionType.RESOLVE,
        target_service=observation.services[0].name if observation.services else "unknown",
        rationale=(
            "All services investigated. Applied available remediations. "
            "Declaring incident resolved."
        ),
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _run_episode(difficulty: str, seed: int = 42) -> float:
    """
    Run a complete episode on a given difficulty tier.
    Returns the final episode reward.
    """
    env = IncidentResponseEnv()
    obs = env.reset(seed=seed, difficulty=difficulty)

    final_reward = 0.0
    for _ in range(obs.max_steps):
        action = _call_llm(obs)
        obs = env.step(action)

        if obs.reward is not None:
            final_reward = float(obs.reward)

        if obs.done:
            break

    return round(final_reward, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_baseline_agent(seed: int = 42) -> Dict[str, float]:
    """
    Run the baseline agent across all three difficulty tiers.

    Called by the /baseline endpoint. Returns a dict mapping
    task_id → score.
    """
    difficulties = {
        "easy_incident_triage": "easy",
        "medium_incident_triage": "medium",
        "hard_incident_triage": "hard",
    }

    scores: Dict[str, float] = {}
    for task_id, difficulty in difficulties.items():
        score = _run_episode(difficulty, seed=seed)
        scores[task_id] = score
        print(f"[baseline] {task_id}: score={score:.4f}")

    return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print(" Incident Response OpenEnv — Baseline Inference")
    print("=" * 60)
    print()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print("  Agent: GPT-4o-mini (OpenAI API)")
    else:
        print("  Agent: Rule-based heuristic (OPENAI_API_KEY not set)")
    print()

    results = run_baseline_agent()

    print()
    print("Results:")
    print("-" * 40)
    for task_id, score in results.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<30} {bar} {score:.4f}")

    mean = sum(results.values()) / len(results)
    print(f"\n  Mean score: {mean:.4f}")
    print("=" * 60)
