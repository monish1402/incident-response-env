"""
reward.py
=========
Reward computation for the Incident Response environment.

Design Philosophy
-----------------
Rewards must be DENSE — not binary end-of-episode. An agent that
investigates correctly, narrows down the root cause, and takes
near-optimal actions must receive meaningful partial credit even if
the final resolution is imperfect.

Reward Components (total sums to 1.0)
--------------------------------------
1. Root-cause identification (0.35 weight)
   - Full credit if the agent investigated the root-cause service.
   - Partial credit if the agent investigated a direct dependency.
   - Zero if the agent never investigated the root cause.

2. Correct remediation action (0.30 weight)
   - Full credit if the agent applied the correct fix action
     (e.g., rollback for bad_deployment, scale_up for traffic_spike).
   - Partial credit for a plausible-but-suboptimal action.

3. Resolution speed (0.20 weight)
   - Rewards finishing the incident quickly relative to the max steps
     and remaining time budget.

4. Reasoning quality (0.15 weight)
   - Scores the rationale strings for specificity — mentions of service
     names, metric values, or root-cause keywords give partial credit.
   - Penalises empty or single-word rationales.

Penalties
---------
- Escalating without investigating: -0.10
- Applying a destructive action to a healthy service: -0.05 per occurrence
- Exceeding max_steps (auto-terminate): -0.05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from server.models import IncidentAction, IncidentState
from server.scenarios import Scenario


# ---------------------------------------------------------------------------
# Maps root-cause type to the correct remediation action
# ---------------------------------------------------------------------------

CORRECT_REMEDIATION: dict[str, str] = {
    "bad_deployment": "rollback",
    "resource_exhaustion": "scale_up",
    "dependency_failure": "restart",
    "traffic_spike": "scale_up",
}

PLAUSIBLE_REMEDIATION: dict[str, list[str]] = {
    "bad_deployment": ["restart"],
    "resource_exhaustion": ["restart"],
    "dependency_failure": ["rollback", "scale_up"],
    "traffic_spike": ["restart"],
}

# Keywords associated with each root-cause type — checked in rationale strings
ROOT_CAUSE_KEYWORDS: dict[str, list[str]] = {
    "bad_deployment": [
        "deploy", "rollback", "version", "crash", "startup", "migration",
    ],
    "resource_exhaustion": [
        "memory", "cpu", "oom", "limit", "heap", "scale", "replica",
    ],
    "dependency_failure": [
        "connection", "dependency", "upstream", "restart", "crash loop",
        "reconnect",
    ],
    "traffic_spike": [
        "traffic", "scale", "replica", "load", "queue", "throttl", "cpu",
    ],
}


# ---------------------------------------------------------------------------
# Reward result container
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown — returned in the info dict."""

    root_cause_score: float
    remediation_score: float
    speed_score: float
    reasoning_score: float
    penalty: float
    total: float
    explanation: str

    def to_dict(self) -> dict:
        return {
            "root_cause_identification": round(self.root_cause_score, 4),
            "correct_remediation": round(self.remediation_score, 4),
            "resolution_speed": round(self.speed_score, 4),
            "reasoning_quality": round(self.reasoning_score, 4),
            "penalty": round(self.penalty, 4),
            "total": round(self.total, 4),
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_episode_reward(
    episode_state: IncidentState,
    scenario: Scenario,
    final_step: int,
    max_steps: int,
    time_budget_remaining: int,
) -> RewardBreakdown:
    """
    Compute the final episode reward from the complete action history.

    Called when the episode terminates (RESOLVE, ESCALATE, or step limit).
    """
    actions_taken = episode_state.actions_taken
    investigated = set(episode_state.investigated_services)
    root_cause = scenario.root_cause_service
    root_cause_type = scenario.root_cause_type
    all_service_names = {s.name for s in scenario.services}

    # ── 1. Root-cause identification ────────────────────────────────────────
    root_cause_score = _score_root_cause(investigated, root_cause, scenario)

    # ── 2. Correct remediation ──────────────────────────────────────────────
    remediation_score = _score_remediation(actions_taken, root_cause_type, root_cause)

    # ── 3. Speed ────────────────────────────────────────────────────────────
    speed_score = _score_speed(
        final_step, max_steps, time_budget_remaining, episode_state.resolved
    )

    # ── 4. Reasoning quality ─────────────────────────────────────────────────
    reasoning_score = _score_reasoning(actions_taken, root_cause_type, all_service_names)

    # ── Penalties ────────────────────────────────────────────────────────────
    penalty = _compute_penalties(
        episode_state, scenario, all_service_names
    )

    # ── Weighted total ───────────────────────────────────────────────────────
    weights = {
        "root_cause": 0.35,
        "remediation": 0.30,
        "speed": 0.20,
        "reasoning": 0.15,
    }
    raw_total = (
        weights["root_cause"] * root_cause_score
        + weights["remediation"] * remediation_score
        + weights["speed"] * speed_score
        + weights["reasoning"] * reasoning_score
        + penalty  # penalty is already negative
    )
    total = round(min(max(raw_total, 0.0), 1.0), 4)

    explanation = (
        f"RootCause={root_cause_score:.2f} | "
        f"Remediation={remediation_score:.2f} | "
        f"Speed={speed_score:.2f} | "
        f"Reasoning={reasoning_score:.2f} | "
        f"Penalty={penalty:.2f} → Total={total:.2f}"
    )

    return RewardBreakdown(
        root_cause_score=root_cause_score,
        remediation_score=remediation_score,
        speed_score=speed_score,
        reasoning_score=reasoning_score,
        penalty=penalty,
        total=total,
        explanation=explanation,
    )


def compute_step_reward(
    action: IncidentAction,
    episode_state: IncidentState,
    scenario: Scenario,
) -> float:
    """
    Compute a small per-step reward signal (dense feedback).

    This fires every step so the agent receives immediate feedback,
    not just at episode end. Values are small (0.0–0.15) to avoid
    dominating the episode-end reward.
    """
    step_reward = 0.0
    root_cause = scenario.root_cause_service
    root_cause_type = scenario.root_cause_type
    action_type = action.action_type.value
    target = action.target_service

    # Bonus: investigating the root-cause service
    if action_type == "investigate" and target == root_cause:
        step_reward += 0.10

    # Small bonus: investigating a direct dependency of root cause
    rc_spec = next((s for s in scenario.services if s.name == root_cause), None)
    if (
        action_type == "investigate"
        and rc_spec
        and target in rc_spec.dependencies
    ):
        step_reward += 0.05

    # Bonus: correct remediation on the root-cause service
    correct_fix = CORRECT_REMEDIATION.get(root_cause_type, "")
    if action_type == correct_fix and target == root_cause:
        step_reward += 0.12

    # Small bonus: any investigate (exploration is good)
    if action_type == "investigate":
        step_reward += 0.02

    return round(min(step_reward, 0.15), 4)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _score_root_cause(
    investigated: set[str], root_cause: str, scenario: Scenario
) -> float:
    if root_cause in investigated:
        return 1.0

    # Partial credit: investigated a service that depends on root cause
    rc_spec = next((s for s in scenario.services if s.name == root_cause), None)
    if rc_spec:
        dependents = [
            s for s in scenario.services
            if root_cause in s.dependencies and s.name in investigated
        ]
        if dependents:
            return 0.40

    return 0.0


def _score_remediation(
    actions_taken: list[dict],
    root_cause_type: str,
    root_cause_service: str,
) -> float:
    correct_fix = CORRECT_REMEDIATION.get(root_cause_type, "")
    plausible = PLAUSIBLE_REMEDIATION.get(root_cause_type, [])

    for action in actions_taken:
        action_type = action.get("action_type", "")
        target = action.get("target_service", "")

        if action_type == correct_fix and target == root_cause_service:
            return 1.0
        if action_type == correct_fix and target != root_cause_service:
            return 0.30  # Right action, wrong service
        if action_type in plausible and target == root_cause_service:
            return 0.50  # Plausible action on right service

    return 0.0


def _score_speed(
    final_step: int,
    max_steps: int,
    time_budget_remaining: int,
    resolved: bool,
) -> float:
    if not resolved:
        return 0.10  # Small credit for any attempt

    # Fewer steps → higher speed score
    step_ratio = final_step / max(max_steps, 1)
    speed = 1.0 - (step_ratio * 0.8)  # Never zero even if slow

    # Bonus if time budget was not breached
    if time_budget_remaining > 0:
        speed = min(speed + 0.10, 1.0)

    return round(max(speed, 0.10), 4)


def _score_reasoning(
    actions_taken: list[dict],
    root_cause_type: str,
    all_service_names: set[str],
) -> float:
    keywords = ROOT_CAUSE_KEYWORDS.get(root_cause_type, [])
    all_rationales = " ".join(
        a.get("rationale", "").lower() for a in actions_taken
    )

    if not all_rationales.strip():
        return 0.0

    # Keyword hit rate
    keyword_hits = sum(1 for kw in keywords if kw in all_rationales)
    keyword_score = keyword_hits / max(len(keywords), 1)

    # Service name mentions (shows agent is referencing actual signals)
    service_mentions = sum(1 for svc in all_service_names if svc in all_rationales)
    mention_score = min(service_mentions / 3.0, 1.0)  # Cap at 3 distinct mentions

    # Length quality (rewards substantive rationales)
    avg_len = len(all_rationales) / max(len(actions_taken), 1)
    length_score = min(avg_len / 100.0, 1.0)

    return round((keyword_score * 0.5 + mention_score * 0.3 + length_score * 0.2), 4)


def _compute_penalties(
    episode_state: IncidentState,
    scenario: Scenario,
    all_service_names: set[str],
) -> float:
    penalty = 0.0
    investigated = set(episode_state.investigated_services)

    # Penalty: escalated without investigating the root cause
    if episode_state.escalated and scenario.root_cause_service not in investigated:
        penalty -= 0.10

    # Penalty: destructive actions on healthy services
    healthy_services = {s.name for s in scenario.services if s.status == "healthy"}
    for action in episode_state.actions_taken:
        action_type = action.get("action_type", "")
        target = action.get("target_service", "")
        if action_type in ("restart", "rollback") and target in healthy_services:
            penalty -= 0.05

    return round(penalty, 4)
