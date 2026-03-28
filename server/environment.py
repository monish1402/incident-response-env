"""
environment.py
==============
IncidentResponseEnv — the core OpenEnv environment class.

Implements the full OpenEnv specification:
  - reset(seed, episode_id)  → IncidentObservation
  - step(action)             → IncidentObservation  (reward + done encoded inside)
  - state()                  → IncidentObservation

Architecture
------------
The environment is stateful per-session. Each call to reset() starts a fresh
incident. The Environment class is instantiated once per HTTP session by the
OpenEnv server infrastructure (SUPPORTS_CONCURRENT_SESSIONS = True).

Episode Termination
-------------------
An episode ends when:
  1. The agent calls RESOLVE or ESCALATE (explicit termination)
  2. max_steps is reached (implicit termination, penalty applied)

Reward is computed at episode end (dense step-level signals also emitted).
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from server.models import (
    ActionType,
    Alert,
    DiagnosticResult,
    IncidentAction,
    IncidentObservation,
    IncidentState,
    ServiceNode,
    ServiceStatus,
)
from server.reward import compute_episode_reward, compute_step_reward
from server.scenarios import Scenario, get_scenario


class IncidentResponseEnv(Environment[IncidentAction, IncidentObservation, IncidentState]):
    """
    Production Incident Response environment.

    The agent plays the role of an on-call Site Reliability Engineer.
    It receives a simulated production incident — degraded or failing
    microservices — and must:

      1. Investigate services to gather diagnostic information
      2. Identify the root cause service and failure type
      3. Apply the correct remediation (restart / rollback / scale_up)
      4. Declare the incident resolved

    Three difficulty tiers challenge agents at progressively harder
    reasoning tasks:
      - Easy   : Single failing service, obvious signals, direct fix
      - Medium : Multi-hop failure, requires 2+ investigate steps
      - Hard   : Cascading failures with misleading signals (red herrings)
    """

    # Allow concurrent sessions — each WebSocket gets its own instance
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Time cost per step (minutes in the incident timeline) ───────────────
    _MINUTES_PER_STEP: int = 5

    def __init__(self) -> None:
        super().__init__()
        self._scenario: Optional[Scenario] = None
        self._state: Optional[IncidentState] = None
        self._time_remaining: int = 0
        self._rng: random.Random = random.Random()

    # ────────────────────────────────────────────────────────────────────────
    # OpenEnv interface
    # ────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """
        Start a new incident episode.

        Args:
            seed        : Random seed for reproducible episode selection.
            episode_id  : Optional episode identifier (auto-generated if None).
            scenario_id : Pin to a specific scenario (useful for testing/grading).
            difficulty  : Constrain to a difficulty tier: 'easy' | 'medium' | 'hard'.

        Returns:
            Initial observation — the incident report the on-call agent sees.
        """
        self._rng = random.Random(seed)
        self._scenario = get_scenario(scenario_id, difficulty, self._rng)
        self._time_remaining = self._scenario.time_budget_minutes

        self._state = IncidentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            incident_id=self._scenario.scenario_id,
            root_cause_service=self._scenario.root_cause_service,
            root_cause_type=self._scenario.root_cause_type,
            correct_action_sequence=self._scenario.correct_action_sequence,
        )

        return self._build_observation(reward=None, done=False)

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """
        Execute one agent action and advance the episode.

        Args:
            action : The IncidentAction submitted by the agent.

        Returns:
            Updated observation with reward and done flag embedded.

        Raises:
            RuntimeError : If called before reset() or after episode termination.
        """
        self._assert_active_episode()

        self._state.step_count += 1
        self._time_remaining = max(
            0, self._time_remaining - self._MINUTES_PER_STEP
        )

        # Record the action in history
        self._state.actions_taken.append(
            {
                "action_type": action.action_type.value,
                "target_service": action.target_service,
                "rationale": action.rationale,
            }
        )

        # Process action side-effects
        diagnostic = self._process_action(action)

        # Compute per-step dense reward
        step_reward = compute_step_reward(action, self._state, self._scenario)

        # Check episode termination
        is_terminal = action.action_type in (
            ActionType.RESOLVE,
            ActionType.ESCALATE,
        ) or self._state.step_count >= self._scenario.max_steps

        episode_reward: Optional[float] = None
        if is_terminal:
            breakdown = compute_episode_reward(
                episode_state=self._state,
                scenario=self._scenario,
                final_step=self._state.step_count,
                max_steps=self._scenario.max_steps,
                time_budget_remaining=self._time_remaining,
            )
            episode_reward = breakdown.total
            # Annotate state with breakdown for info dict
            self._state.actions_taken[-1]["reward_breakdown"] = breakdown.to_dict()

        # Combine: step reward contributes 10%, episode reward 90%
        final_reward: float
        if episode_reward is not None:
            final_reward = round(0.10 * step_reward + 0.90 * episode_reward, 4)
        else:
            final_reward = step_reward

        return self._build_observation(
            reward=final_reward,
            done=is_terminal,
            last_diagnostic=diagnostic,
        )

    def state(self) -> IncidentObservation:
        """Return the current observation without advancing the episode."""
        self._assert_active_episode()
        return self._build_observation(reward=None, done=False)

    # ────────────────────────────────────────────────────────────────────────
    # Task metadata (exposed via /tasks endpoint)
    # ────────────────────────────────────────────────────────────────────────

    def get_tasks(self) -> list[dict]:
        """Return all available tasks with difficulty and action schema."""
        return [
            {
                "id": "easy_incident_triage",
                "description": (
                    "Single-service failure with obvious signals. "
                    "Investigate, apply the correct fix, resolve."
                ),
                "difficulty": "easy",
                "action_schema": IncidentAction.model_json_schema(),
            },
            {
                "id": "medium_incident_triage",
                "description": (
                    "Multi-hop cascading failure. Requires 2+ investigate steps "
                    "to trace the root cause through the dependency chain."
                ),
                "difficulty": "medium",
                "action_schema": IncidentAction.model_json_schema(),
            },
            {
                "id": "hard_incident_triage",
                "description": (
                    "Site-wide degradation with multiple misleading signals. "
                    "Agent must distinguish red herrings from the true root cause."
                ),
                "difficulty": "hard",
                "action_schema": IncidentAction.model_json_schema(),
            },
        ]

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────────────

    def _process_action(
        self, action: IncidentAction
    ) -> Optional[DiagnosticResult]:
        """
        Apply the action's side-effects to internal state and return any
        diagnostic output (only for INVESTIGATE actions).
        """
        action_type = action.action_type
        target = action.target_service

        if action_type == ActionType.INVESTIGATE:
            self._state.investigated_services.append(target)
            return self._get_diagnostic(target)

        if action_type == ActionType.RESOLVE:
            self._state.resolved = True

        if action_type == ActionType.ESCALATE:
            self._state.escalated = True

        return None

    def _get_diagnostic(self, service_name: str) -> DiagnosticResult:
        """
        Return the diagnostic output for investigating a named service.

        If the service exists in the scenario, return its log lines and
        error summary. Otherwise return a "service not found" diagnostic.
        """
        spec = next(
            (s for s in self._scenario.services if s.name == service_name),
            None,
        )

        if spec is None:
            return DiagnosticResult(
                service=service_name,
                log_tail=["ERROR: Service not found in current topology"],
                error_summary=f"'{service_name}' is not a known service in this incident.",
                suggested_action="Check the service list in the observation.",
            )

        return DiagnosticResult(
            service=spec.name,
            log_tail=spec.log_lines,  # log_lines is the dataclass field name in ScenarioSpec
            error_summary=spec.error_summary,
            suggested_action=spec.suggested_action,
        )

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        last_diagnostic: Optional[DiagnosticResult] = None,
    ) -> IncidentObservation:
        """Construct the full IncidentObservation from current state."""
        assert self._scenario is not None
        assert self._state is not None

        service_nodes = [
            ServiceNode(
                name=s.name,
                status=ServiceStatus(s.status),
                error_rate=s.error_rate,
                latency_p99_ms=s.latency_p99_ms,
                cpu_utilisation=s.cpu_utilisation,
                memory_utilisation=s.memory_utilisation,
                recent_deployment=s.recent_deployment,
                dependencies=s.dependencies,
            )
            for s in self._scenario.services
        ]

        alerts = [
            Alert(
                alert_id=a.alert_id,
                service=a.service,
                severity=a.severity,
                message=a.message,
                fired_at_step=0,
            )
            for a in self._scenario.alerts
        ]

        action_log = [
            f"Step {i + 1}: [{a['action_type'].upper()}] → {a['target_service']} "
            f"| {a['rationale'][:80]}"
            for i, a in enumerate(self._state.actions_taken)
        ]

        return IncidentObservation(
            # OpenEnv base fields
            reward=reward,
            done=done,
            # System snapshot
            services=service_nodes,
            active_alerts=alerts,
            last_diagnostic=last_diagnostic,
            # Episode context
            incident_title=self._scenario.incident_title,
            incident_severity=self._scenario.incident_severity,
            step_count=self._state.step_count,
            max_steps=self._scenario.max_steps,
            time_to_resolve_budget=self._time_remaining,
            action_history=action_log,
        )

    def _assert_active_episode(self) -> None:
        """Raise if no active episode exists."""
        if self._scenario is None or self._state is None:
            raise RuntimeError(
                "No active episode. Call reset() before step() or state()."
            )
