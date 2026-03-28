"""
inference.py
============
OpenEnv inference entry point for the Incident Response environment.

This file is required at the repo root by the OpenEnv submission validator.
It exposes a standard `run_inference` function that:
  1. Starts a fresh episode for a given task
  2. Runs the rule-based (or LLM-powered, if OPENAI_API_KEY is set) agent
  3. Returns the final score

The function signature and return type match the OpenEnv inference spec:
  - Input : task_id (str), optional base_url (str)
  - Output: dict with keys  score (float), actions (list), metadata (dict)

Usage
-----
    # Run inference from the command line
    python inference.py

    # Import in tests / evaluation harness
    from inference import run_inference
    result = run_inference("easy_incident_triage")
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")

TASK_IDS = [
    "easy_incident_triage",
    "medium_incident_triage",
    "hard_incident_triage",
]

# ---------------------------------------------------------------------------
# Lightweight rule-based agent (no external dependencies)
# ---------------------------------------------------------------------------


def _rule_based_action(obs: Dict[str, Any], investigated: List[str]) -> Dict[str, str]:
    """
    Deterministic SRE heuristic:
      1. Start at the most-alerting service.
      2. Investigate the service with the highest error_rate not yet seen.
      3. If last diagnostic suggests a fix, apply it.
      4. Resolve.

    Returns an IncidentAction-compatible dict.
    """
    services = obs.get("services", [])
    last_diag = obs.get("last_diagnostic")
    action_history = obs.get("action_history", [])

    # Phase 1 — if we have a diagnostic hint, trust it
    if last_diag and last_diag.get("suggested_action"):
        suggestion = last_diag["suggested_action"].upper()
        svc = last_diag["service"]

        if "ROLLBACK" in suggestion:
            return {
                "action_type": "rollback",
                "target_service": svc,
                "rationale": (
                    f"Diagnostic on {svc} suggests ROLLBACK. "
                    f"Error summary: {last_diag.get('error_summary', '')[:120]}"
                ),
            }
        if "SCALE_UP" in suggestion or "SCALE" in suggestion:
            return {
                "action_type": "scale_up",
                "target_service": svc,
                "rationale": (
                    f"Diagnostic on {svc} recommends SCALE_UP due to resource exhaustion. "
                    f"Error summary: {last_diag.get('error_summary', '')[:120]}"
                ),
            }
        if "RESTART" in suggestion:
            return {
                "action_type": "restart",
                "target_service": svc,
                "rationale": (
                    f"Diagnostic on {svc} suggests RESTART to clear crash loop. "
                    f"Error summary: {last_diag.get('error_summary', '')[:120]}"
                ),
            }

    # Phase 2 — investigate the unhealthiest un-investigated service
    # Sort: down > degraded > healthy, then by error_rate desc
    status_rank = {"down": 0, "degraded": 1, "healthy": 2}
    candidates = [s for s in services if s["name"] not in investigated]

    if candidates:
        target = sorted(
            candidates,
            key=lambda s: (status_rank.get(s["status"], 2), -s.get("error_rate", 0)),
        )[0]
        return {
            "action_type": "investigate",
            "target_service": target["name"],
            "rationale": (
                f"Investigating {target['name']}: status={target['status']}, "
                f"error_rate={target.get('error_rate', 0):.0%}, "
                f"cpu={target.get('cpu_utilisation', 0):.0%}. "
                "Tracing root cause through dependency graph."
            ),
        }

    # Phase 3 — all services investigated, no clear hint → resolve
    return {
        "action_type": "resolve",
        "target_service": services[0]["name"] if services else "unknown",
        "rationale": (
            "All services investigated and remediation applied. "
            "Declaring incident resolved."
        ),
    }


# ---------------------------------------------------------------------------
# LLM-powered agent (optional, requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


def _llm_action(obs: Dict[str, Any], conversation: List[Dict]) -> Optional[Dict[str, str]]:
    """
    Call OpenAI (gpt-4o-mini) to pick the next action.
    Returns None if the API key is absent or the call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        import openai  # type: ignore

        client = openai.OpenAI(api_key=api_key)

        system_prompt = (
            "You are an expert Site Reliability Engineer responding to a production incident.\n"
            "You receive a JSON system state and must take ONE action per turn.\n"
            "Available action_type values: investigate, restart, rollback, scale_up, escalate, resolve.\n"
            "Respond ONLY with a JSON object — no markdown, no preamble:\n"
            '{"action_type": "...", "target_service": "...", "rationale": "..."}'
        )

        messages = [{"role": "system", "content": system_prompt}] + conversation
        messages.append({"role": "user", "content": json.dumps(obs)})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        action = json.loads(raw)
        conversation.append({"role": "user", "content": json.dumps(obs)})
        conversation.append({"role": "assistant", "content": raw})
        return action

    except Exception as exc:  # noqa: BLE001
        print(f"[inference] LLM call failed ({exc}), falling back to rule-based agent.")
        return None


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------


def run_inference(
    task_id: str,
    base_url: str = BASE_URL,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """
    Run one full episode for *task_id* and return results.

    Parameters
    ----------
    task_id : str
        One of: easy_incident_triage | medium_incident_triage | hard_incident_triage
    base_url : str
        Base URL of the running OpenEnv server. Defaults to localhost:7860.
    max_steps : int
        Hard safety cap; the environment also enforces its own limit.

    Returns
    -------
    dict with keys:
        task_id   : str
        score     : float   — final episode reward [0.0, 1.0]
        actions   : list    — sequence of actions taken
        steps     : int     — total steps used
        metadata  : dict    — raw final observation
    """
    session = requests.Session()

    # --- 1. Reset episode ---
    reset_resp = session.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    actions_taken: List[Dict[str, str]] = []
    investigated: List[str] = []
    llm_conversation: List[Dict] = []
    final_score = 0.0

    # --- 2. Step loop ---
    for step in range(max_steps):
        # Try LLM first; fall back to rule-based
        action = _llm_action(obs, llm_conversation)
        if action is None:
            action = _rule_based_action(obs, investigated)

        actions_taken.append(action)

        if action["action_type"] == "investigate":
            investigated.append(action["target_service"])

        # Submit action
        step_resp = session.post(f"{base_url}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result.get("observation", result)
        final_score = result.get("reward", 0.0) or 0.0
        done = result.get("done", False)

        if done:
            break

    return {
        "task_id": task_id,
        "score": round(float(final_score), 4),
        "actions": actions_taken,
        "steps": len(actions_taken),
        "metadata": obs,
    }


# ---------------------------------------------------------------------------
# Convenience: run all tasks
# ---------------------------------------------------------------------------


def run_all_tasks(base_url: str = BASE_URL) -> Dict[str, float]:
    """Run inference on every task and return {task_id: score}."""
    scores: Dict[str, float] = {}
    for task_id in TASK_IDS:
        result = run_inference(task_id, base_url=base_url)
        scores[task_id] = result["score"]
        print(f"  {task_id}: {result['score']:.4f}  ({result['steps']} steps)")
    return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenEnv inference for incident-response-env")
    parser.add_argument(
        "--task",
        default="all",
        choices=TASK_IDS + ["all"],
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"Server base URL (default: {BASE_URL})",
    )
    args = parser.parse_args()

    if args.task == "all":
        print(f"\nRunning inference on all tasks against {args.base_url}\n")
        scores = run_all_tasks(base_url=args.base_url)
        mean = sum(scores.values()) / len(scores)
        print(f"\nMean score: {mean:.4f}")
    else:
        print(f"\nRunning inference for '{args.task}' against {args.base_url}\n")
        result = run_inference(args.task, base_url=args.base_url)
        print(json.dumps(result, indent=2))
