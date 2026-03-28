"""
graders.py
==========
Deterministic, reproducible graders for all three task difficulty levels.

Each grader creates a fresh environment, runs a fixed optimal-ish action
sequence, and returns a score in [0.0, 1.0].

Graders are deterministic because:
  - They use a fixed seed (default: 42)
  - They use a fixed scenario_id (pinned to a specific scenario)
  - Action sequences are hard-coded

The /grader endpoint uses these graders to score submitted agent responses.
The /baseline endpoint runs all three and reports the mean.
"""

from __future__ import annotations

from typing import Dict, Optional

from server.environment import IncidentResponseEnv
from server.models import ActionType, IncidentAction


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _run_episode(
    scenario_id: str,
    actions: list[IncidentAction],
    seed: int = 42,
) -> float:
    """
    Run a deterministic episode with a fixed action sequence.

    Returns the final reward from the last step (the episode-end reward).
    """
    env = IncidentResponseEnv()
    env.reset(seed=seed, scenario_id=scenario_id)

    final_reward = 0.0
    for action in actions:
        obs = env.step(action)
        if obs.reward is not None:
            final_reward = float(obs.reward)
        if obs.done:
            break

    return round(final_reward, 4)


# ---------------------------------------------------------------------------
# Grader 1 — Easy: bad deployment (payment-service crash loop)
# ---------------------------------------------------------------------------

def grade_easy(custom_actions: Optional[list[IncidentAction]] = None) -> float:
    """
    Grade the easy_001 scenario: payment-service bad deployment.

    Optimal sequence:
      1. INVESTIGATE api-gateway  (see upstream errors pointing to payment)
      2. INVESTIGATE payment-service  (find deployment crash logs)
      3. ROLLBACK payment-service  (correct fix for bad_deployment)
      4. RESOLVE

    If custom_actions provided, run those instead (for grading agent submissions).
    """
    default_actions = [
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="api-gateway",
            rationale=(
                "Starting with api-gateway as it is the user-facing entry point "
                "and all alerts point to 502 errors there."
            ),
        ),
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="payment-service",
            rationale=(
                "API Gateway logs show upstream payment-service refusing connections. "
                "Payment-service had a recent deployment — investigating for crash logs."
            ),
        ),
        IncidentAction(
            action_type=ActionType.ROLLBACK,
            target_service="payment-service",
            rationale=(
                "Payment-service crashed immediately after deployment v2.4.2 due to a "
                "failed DB migration. Rolling back to v2.4.1 will restore the service."
            ),
        ),
        IncidentAction(
            action_type=ActionType.RESOLVE,
            target_service="payment-service",
            rationale=(
                "Rollback applied. Payment-service should be recovering. "
                "Monitoring shows error rate dropping. Resolving incident."
            ),
        ),
    ]

    actions = custom_actions or default_actions
    return _run_episode("easy_001", actions)


# ---------------------------------------------------------------------------
# Grader 2 — Medium: inventory-service crash loop (dependency_failure)
# ---------------------------------------------------------------------------

def grade_medium(custom_actions: Optional[list[IncidentAction]] = None) -> float:
    """
    Grade the medium_001 scenario: cascading failure from inventory-service.

    Optimal sequence:
      1. INVESTIGATE checkout-service  (see order-service slowness)
      2. INVESTIGATE order-service  (see inventory-service timeouts)
      3. INVESTIGATE inventory-service  (find crash loop root cause)
      4. RESTART inventory-service  (correct fix for dependency_failure)
      5. RESOLVE
    """
    default_actions = [
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="checkout-service",
            rationale=(
                "Checkout is the user-facing service with the P1 latency alert. "
                "Starting investigation here to understand the blast radius."
            ),
        ),
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="order-service",
            rationale=(
                "Checkout logs show slow responses from order-service. "
                "Investigating order-service to trace the root cause deeper."
            ),
        ),
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="inventory-service",
            rationale=(
                "Order-service logs show inventory-service timing out repeatedly. "
                "Inventory-service appears to be the root cause — investigating now."
            ),
        ),
        IncidentAction(
            action_type=ActionType.RESTART,
            target_service="inventory-service",
            rationale=(
                "inventory-service entered a crash loop after losing its MongoDB "
                "connection. MongoDB has since recovered but inventory is stuck. "
                "Restart will allow it to reconnect and restore service."
            ),
        ),
        IncidentAction(
            action_type=ActionType.RESOLVE,
            target_service="inventory-service",
            rationale=(
                "inventory-service restarted successfully and reconnected to MongoDB. "
                "Checkout and order service latency returning to normal. Resolving."
            ),
        ),
    ]

    actions = custom_actions or default_actions
    return _run_episode("medium_001", actions)


# ---------------------------------------------------------------------------
# Grader 3 — Hard: auth-service traffic spike (site-wide cascading failure)
# ---------------------------------------------------------------------------

def grade_hard(custom_actions: Optional[list[IncidentAction]] = None) -> float:
    """
    Grade the hard_001 scenario: auth-service CPU saturation causing site-wide degradation.

    Optimal sequence:
      1. INVESTIGATE api-gateway  (see all failures are auth-related)
      2. INVESTIGATE auth-service  (CPU saturated, queue depth 60x normal)
      3. SCALE_UP auth-service  (correct fix for traffic_spike)
      4. RESOLVE

    Hard because:
      - notification-service has a recent deployment (red herring)
      - user-service and notification-service alerts look worse than auth-service
      - The P2 auth-service alert is rated lower severity than the P1 alerts
    """
    default_actions = [
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="api-gateway",
            rationale=(
                "Starting with api-gateway as it has the P1 alert and touches all "
                "upstreams. Need to understand which dependency is causing the cascade."
            ),
        ),
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="auth-service",
            rationale=(
                "API Gateway logs show all 502s are auth-related — token validation "
                "timeouts. auth-service CPU is 97% and queue depth is 60x normal. "
                "This is the root cause, not the deployment on notification-service."
            ),
        ),
        IncidentAction(
            action_type=ActionType.SCALE_UP,
            target_service="auth-service",
            rationale=(
                "auth-service is CPU-saturated due to 2x traffic spike with only "
                "2 replicas. Scaling up to 6+ replicas will drain the validation "
                "queue and restore token validation latency across all services."
            ),
        ),
        IncidentAction(
            action_type=ActionType.RESOLVE,
            target_service="auth-service",
            rationale=(
                "auth-service replicas scaled. Token validation latency dropping. "
                "user-service 401s clearing. Checkout and notification recovering. "
                "Declaring incident resolved — root cause was traffic spike on auth."
            ),
        ),
    ]

    actions = custom_actions or default_actions
    return _run_episode("hard_001", actions)


# ---------------------------------------------------------------------------
# Convenience: grade all tasks
# ---------------------------------------------------------------------------

GRADERS: Dict[str, callable] = {
    "easy_incident_triage": grade_easy,
    "medium_incident_triage": grade_medium,
    "hard_incident_triage": grade_hard,
}


def grade_all() -> Dict[str, float]:
    """Run all three default graders and return per-task scores."""
    return {task_id: grader() for task_id, grader in GRADERS.items()}
