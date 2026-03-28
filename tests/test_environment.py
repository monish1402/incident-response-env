"""
test_environment.py
===================
Comprehensive test suite for the Incident Response OpenEnv environment.

Tests are organised into four classes:

  TestOpenEnvSpec         — Verifies compliance with the OpenEnv interface spec
  TestRewardFunction      — Validates reward properties (range, density, ordering)
  TestGraders             — Confirms graders are deterministic and correctly ordered
  TestEdgeCases           — Boundary conditions and failure modes

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest

from server.environment import IncidentResponseEnv
from server.graders import grade_all, grade_easy, grade_hard, grade_medium
from server.models import (
    ActionType,
    AlertSeverity,
    IncidentAction,
    IncidentObservation,
    ServiceStatus,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def easy_env() -> IncidentResponseEnv:
    """Fresh environment pinned to the easy scenario."""
    env = IncidentResponseEnv()
    env.reset(seed=42, scenario_id="easy_001")
    return env


@pytest.fixture
def medium_env() -> IncidentResponseEnv:
    """Fresh environment pinned to the medium scenario."""
    env = IncidentResponseEnv()
    env.reset(seed=42, scenario_id="medium_001")
    return env


@pytest.fixture
def hard_env() -> IncidentResponseEnv:
    """Fresh environment pinned to the hard scenario."""
    env = IncidentResponseEnv()
    env.reset(seed=42, scenario_id="hard_001")
    return env


@pytest.fixture
def investigate_gateway(easy_env) -> tuple[IncidentResponseEnv, IncidentObservation]:
    """Easy env after one INVESTIGATE on api-gateway."""
    obs = easy_env.step(
        IncidentAction(
            action_type=ActionType.INVESTIGATE,
            target_service="api-gateway",
            rationale="Starting at the user-facing entry point to understand blast radius.",
        )
    )
    return easy_env, obs


def _resolve_action(target: str = "payment-service") -> IncidentAction:
    return IncidentAction(
        action_type=ActionType.RESOLVE,
        target_service=target,
        rationale="Incident appears resolved — error rates dropping. Declaring resolution.",
    )


def _investigate_action(target: str, rationale: str = "") -> IncidentAction:
    return IncidentAction(
        action_type=ActionType.INVESTIGATE,
        target_service=target,
        rationale=rationale or f"Investigating {target} to understand its current state.",
    )


# ---------------------------------------------------------------------------
# TestOpenEnvSpec — interface compliance
# ---------------------------------------------------------------------------


class TestOpenEnvSpec:
    """Verify the environment implements the full OpenEnv spec correctly."""

    def test_reset_returns_observation(self) -> None:
        env = IncidentResponseEnv()
        obs = env.reset(seed=42, difficulty="easy")
        assert isinstance(obs, IncidentObservation)

    def test_reset_observation_has_required_fields(self) -> None:
        env = IncidentResponseEnv()
        obs = env.reset(seed=42, difficulty="easy")
        assert isinstance(obs.incident_title, str) and len(obs.incident_title) > 0
        assert len(obs.services) >= 2
        assert len(obs.active_alerts) >= 1
        assert obs.step_count == 0
        assert obs.max_steps > 0
        assert obs.time_to_resolve_budget > 0
        assert obs.done is False
        assert obs.reward is None  # No reward on reset

    def test_step_returns_observation(self, easy_env) -> None:
        obs = easy_env.step(_investigate_action("api-gateway"))
        assert isinstance(obs, IncidentObservation)

    def test_step_reward_is_float_in_range(self, easy_env) -> None:
        obs = easy_env.step(_investigate_action("api-gateway"))
        assert obs.reward is not None
        assert isinstance(obs.reward, float)
        assert 0.0 <= obs.reward <= 1.0

    def test_step_advances_step_count(self, easy_env) -> None:
        obs = easy_env.step(_investigate_action("api-gateway"))
        assert obs.step_count == 1

    def test_step_done_false_mid_episode(self, easy_env) -> None:
        obs = easy_env.step(_investigate_action("api-gateway"))
        assert obs.done is False

    def test_resolve_terminates_episode(self, easy_env) -> None:
        obs = easy_env.step(_resolve_action())
        assert obs.done is True

    def test_escalate_terminates_episode(self, easy_env) -> None:
        obs = easy_env.step(
            IncidentAction(
                action_type=ActionType.ESCALATE,
                target_service="api-gateway",
                rationale="Unable to determine root cause — escalating to senior engineer.",
            )
        )
        assert obs.done is True

    def test_max_steps_terminates_episode(self) -> None:
        env = IncidentResponseEnv()
        obs = env.reset(seed=42, scenario_id="easy_001")
        max_steps = obs.max_steps
        for i in range(max_steps):
            obs = env.step(_investigate_action("api-gateway"))
            if obs.done:
                break
        assert obs.done is True

    def test_state_returns_current_observation(self, easy_env) -> None:
        state_obs = easy_env.state()
        assert isinstance(state_obs, IncidentObservation)
        assert state_obs.step_count == 0

    def test_state_after_step_reflects_updated_step_count(
        self, investigate_gateway
    ) -> None:
        env, _ = investigate_gateway
        state_obs = env.state()
        assert state_obs.step_count == 1

    def test_state_does_not_advance_episode(self, easy_env) -> None:
        easy_env.state()
        easy_env.state()
        assert easy_env.state().step_count == 0

    def test_reset_before_step_raises_runtime_error(self) -> None:
        env = IncidentResponseEnv()
        with pytest.raises(RuntimeError):
            env.step(_investigate_action("api-gateway"))

    def test_reset_before_state_raises_runtime_error(self) -> None:
        env = IncidentResponseEnv()
        with pytest.raises(RuntimeError):
            env.state()

    def test_get_tasks_returns_three_tasks(self) -> None:
        env = IncidentResponseEnv()
        tasks = env.get_tasks()
        assert len(tasks) == 3
        ids = {t["id"] for t in tasks}
        assert "easy_incident_triage" in ids
        assert "medium_incident_triage" in ids
        assert "hard_incident_triage" in ids

    def test_get_tasks_includes_action_schema(self) -> None:
        env = IncidentResponseEnv()
        for task in env.get_tasks():
            assert "action_schema" in task
            schema = task["action_schema"]
            assert "properties" in schema

    def test_observation_services_have_valid_statuses(self) -> None:
        env = IncidentResponseEnv()
        obs = env.reset(seed=42, difficulty="hard")
        for svc in obs.services:
            assert svc.status in {s.value for s in ServiceStatus}

    def test_observation_alerts_have_valid_severity(self) -> None:
        env = IncidentResponseEnv()
        obs = env.reset(seed=42, difficulty="hard")
        for alert in obs.active_alerts:
            assert alert.severity in {s.value for s in AlertSeverity}

    def test_second_reset_clears_state(self) -> None:
        env = IncidentResponseEnv()
        env.reset(seed=42, difficulty="easy")
        env.step(_investigate_action("api-gateway"))
        # Second reset must clear step count and history
        obs = env.reset(seed=42, difficulty="easy")
        assert obs.step_count == 0
        assert obs.action_history == []
        assert obs.last_diagnostic is None

    def test_investigate_returns_diagnostic(self, investigate_gateway) -> None:
        _, obs = investigate_gateway
        assert obs.last_diagnostic is not None
        assert obs.last_diagnostic.service == "api-gateway"
        assert len(obs.last_diagnostic.log_tail) > 0
        assert isinstance(obs.last_diagnostic.error_summary, str)

    def test_non_investigate_action_has_no_diagnostic(self, easy_env) -> None:
        obs = easy_env.step(
            IncidentAction(
                action_type=ActionType.SCALE_UP,
                target_service="api-gateway",
                rationale="Attempting scale-up as a first response.",
            )
        )
        assert obs.last_diagnostic is None

    def test_action_history_grows_each_step(self, easy_env) -> None:
        easy_env.step(_investigate_action("api-gateway"))
        obs = easy_env.step(_investigate_action("payment-service"))
        assert len(obs.action_history) == 2

    def test_time_budget_decreases_each_step(self, easy_env) -> None:
        initial = easy_env.state().time_to_resolve_budget
        obs = easy_env.step(_investigate_action("api-gateway"))
        assert obs.time_to_resolve_budget < initial

    def test_concurrent_sessions_flag(self) -> None:
        assert IncidentResponseEnv.SUPPORTS_CONCURRENT_SESSIONS is True


# ---------------------------------------------------------------------------
# TestRewardFunction — reward properties
# ---------------------------------------------------------------------------


class TestRewardFunction:
    """Validate reward is dense, bounded, and correctly ordered."""

    def test_reward_range_zero_to_one(self) -> None:
        env = IncidentResponseEnv()
        for difficulty in ("easy", "medium", "hard"):
            obs = env.reset(seed=42, difficulty=difficulty)
            obs = env.step(_resolve_action())
            assert 0.0 <= float(obs.reward) <= 1.0

    def test_optimal_sequence_beats_random_resolve(self) -> None:
        """Optimal actions must outperform immediately resolving without investigation."""
        # Immediate resolve (no investigation)
        env_random = IncidentResponseEnv()
        env_random.reset(seed=42, scenario_id="easy_001")
        obs_random = env_random.step(_resolve_action())
        random_reward = float(obs_random.reward)

        # Optimal sequence
        env_opt = IncidentResponseEnv()
        env_opt.reset(seed=42, scenario_id="easy_001")
        env_opt.step(_investigate_action("api-gateway", "Checking user-facing gateway first."))
        env_opt.step(_investigate_action("payment-service", "Gateway logs point to payment-service. Recent deployment."))
        env_opt.step(IncidentAction(
            action_type=ActionType.ROLLBACK,
            target_service="payment-service",
            rationale="Bad deployment v2.4.2. DB migration failed. Rollback to v2.4.1.",
        ))
        obs_opt = env_opt.step(_resolve_action())
        optimal_reward = float(obs_opt.reward)

        assert optimal_reward > random_reward, (
            f"Optimal ({optimal_reward:.4f}) must beat immediate resolve ({random_reward:.4f})"
        )

    def test_investigating_root_cause_gives_step_bonus(self) -> None:
        """Investigating the root-cause service earns a higher step reward."""
        env_root = IncidentResponseEnv()
        env_root.reset(seed=42, scenario_id="easy_001")
        obs_root = env_root.step(_investigate_action("payment-service",
            "Payment-service is the root cause — investigating directly."))

        env_other = IncidentResponseEnv()
        env_other.reset(seed=42, scenario_id="easy_001")
        obs_other = env_other.step(_investigate_action("postgres-db",
            "Investigating the database as a starting point."))

        assert float(obs_root.reward) > float(obs_other.reward)

    def test_escalate_without_investigation_penalised(self) -> None:
        """Escalating without investigating root cause should score below optimal."""
        env_esc = IncidentResponseEnv()
        env_esc.reset(seed=42, scenario_id="easy_001")
        obs_esc = env_esc.step(IncidentAction(
            action_type=ActionType.ESCALATE,
            target_service="api-gateway",
            rationale="Unsure of root cause — escalating immediately.",
        ))

        env_opt = IncidentResponseEnv()
        env_opt.reset(seed=42, scenario_id="easy_001")
        env_opt.step(_investigate_action("api-gateway", "Checking gateway logs."))
        env_opt.step(_investigate_action("payment-service", "Found root cause: bad deploy."))
        env_opt.step(IncidentAction(
            action_type=ActionType.ROLLBACK,
            target_service="payment-service",
            rationale="Rollback bad deployment v2.4.2.",
        ))
        obs_opt = env_opt.step(_resolve_action())

        assert float(obs_opt.reward) > float(obs_esc.reward)

    def test_step_reward_is_non_zero_for_investigate(self, easy_env) -> None:
        obs = easy_env.step(_investigate_action("api-gateway"))
        assert float(obs.reward) > 0.0


# ---------------------------------------------------------------------------
# TestGraders — determinism and ordering
# ---------------------------------------------------------------------------


class TestGraders:
    """Graders must be deterministic and correctly ordered by difficulty."""

    def test_graders_return_float_in_range(self) -> None:
        for grader in (grade_easy, grade_medium, grade_hard):
            score = grader()
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_graders_are_deterministic(self) -> None:
        for grader in (grade_easy, grade_medium, grade_hard):
            assert grader() == grader()

    def test_grade_all_returns_all_task_ids(self) -> None:
        results = grade_all()
        assert set(results.keys()) == {
            "easy_incident_triage",
            "medium_incident_triage",
            "hard_incident_triage",
        }

    def test_grade_all_scores_in_range(self) -> None:
        for task_id, score in grade_all().items():
            assert 0.0 <= score <= 1.0, f"{task_id} score {score} out of range"

    def test_default_grader_scores_above_baseline_threshold(self) -> None:
        """The optimal action sequence must score well above a random agent."""
        for grader in (grade_easy, grade_medium, grade_hard):
            assert grader() > 0.6, "Optimal sequence should score > 0.60"
