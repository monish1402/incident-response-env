"""
models.py
=========
Typed Pydantic models for the Incident Response OpenEnv environment.

All models inherit from the official openenv-core base classes, ensuring
full compliance with the OpenEnv specification.

Domain Overview
---------------
An AI agent acts as an on-call Site Reliability Engineer (SRE).
It receives a simulated production incident — a degraded or failing
distributed system — and must diagnose the root cause, take corrective
actions, and restore the system to a healthy state.

This mirrors what Incident.io, PagerDuty, and internal SRE teams do
at companies like Google, Meta, and Netflix every day.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------


class ServiceStatus(str, Enum):
    """Health status of a single service in the dependency graph."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class ActionType(str, Enum):
    """
    The set of actions an on-call agent can take.

    Each action maps to a real SRE operation:
      - INVESTIGATE  : run diagnostics / read logs for a named service
      - RESTART      : restart a named service (may clear transient faults)
      - ROLLBACK     : roll back the last deployment for a service
      - SCALE_UP     : increase replica count for a service
      - ESCALATE     : page a secondary human or team (ends episode, partial credit)
      - RESOLVE      : declare the incident resolved (ends episode)
    """

    INVESTIGATE = "investigate"
    RESTART = "restart"
    ROLLBACK = "rollback"
    SCALE_UP = "scale_up"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


class AlertSeverity(str, Enum):
    """PagerDuty-style severity for triggered alerts."""

    P1 = "P1"  # Critical — user-facing outage
    P2 = "P2"  # High — significant degradation
    P3 = "P3"  # Medium — partial impact
    P4 = "P4"  # Low — informational


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ServiceNode(Observation):
    """
    Snapshot of a single service in the system topology.

    Extends Observation so it can be embedded in the top-level observation.
    """

    name: str = Field(..., description="Service identifier, e.g. 'api-gateway'")
    status: ServiceStatus = Field(..., description="Current health status")
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Request error rate (0.0–1.0)"
    )
    latency_p99_ms: float = Field(
        ..., ge=0.0, description="99th-percentile latency in milliseconds"
    )
    cpu_utilisation: float = Field(
        ..., ge=0.0, le=1.0, description="CPU utilisation (0.0–1.0)"
    )
    memory_utilisation: float = Field(
        ..., ge=0.0, le=1.0, description="Memory utilisation (0.0–1.0)"
    )
    recent_deployment: bool = Field(
        default=False,
        description="Whether a deployment was pushed in the last 30 minutes",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Names of upstream services this service depends on",
    )


class Alert(Observation):
    """A triggered monitoring alert, mirroring a PagerDuty / Datadog alert."""

    alert_id: str = Field(..., description="Unique alert identifier")
    service: str = Field(..., description="Service that triggered the alert")
    severity: AlertSeverity = Field(..., description="Alert severity tier")
    message: str = Field(..., description="Human-readable alert description")
    fired_at_step: int = Field(..., description="Episode step when alert fired")


class DiagnosticResult(Observation):
    """
    Output of an INVESTIGATE action.
    Mimics the output of `kubectl logs`, `journalctl`, or an APM trace.
    """

    service: str = Field(..., description="Service that was investigated")
    log_tail: List[str] = Field(
        ..., description="Last N log lines from the service"
    )
    error_summary: str = Field(
        ..., description="Natural-language summary of observed errors"
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description="Hinted corrective action (may be absent to increase difficulty)",
    )


# ---------------------------------------------------------------------------
# Top-level OpenEnv models
# ---------------------------------------------------------------------------


class IncidentObservation(Observation):
    """
    Full observation returned by reset() and step().

    The agent sees:
      - The current health of every service in the topology
      - All active alerts
      - The result of the most recent INVESTIGATE action (if any)
      - Contextual metadata (step count, time budget, incident severity)
    """

    # System state
    services: List[ServiceNode] = Field(
        ..., description="Current snapshot of all services in the topology"
    )
    active_alerts: List[Alert] = Field(
        ..., description="All currently firing alerts"
    )

    # Diagnostic output from the last INVESTIGATE (None until first investigate)
    last_diagnostic: Optional[DiagnosticResult] = Field(
        default=None,
        description="Output of the most recent INVESTIGATE action",
    )

    # Episode context
    incident_title: str = Field(
        ..., description="One-line incident description shown to the on-call agent"
    )
    incident_severity: AlertSeverity = Field(
        ..., description="Overall incident severity"
    )
    step_count: int = Field(
        default=0, description="Number of steps taken in this episode"
    )
    max_steps: int = Field(
        default=10, description="Maximum steps before episode auto-terminates"
    )
    time_to_resolve_budget: int = Field(
        ...,
        description=(
            "Notional minutes remaining to resolve the incident before SLA breach. "
            "Decreases by ~5 each step."
        ),
    )
    action_history: List[str] = Field(
        default_factory=list,
        description="Human-readable log of all actions taken this episode",
    )


class IncidentAction(Action):
    """
    Action submitted by the agent each step.

    The agent declares:
      - What type of action it wants to take (ActionType)
      - Which service it is targeting
      - A brief rationale (used for partial credit in grader)
    """

    action_type: ActionType = Field(
        ..., description="The operation the agent wants to perform"
    )
    target_service: str = Field(
        ..., description="Name of the service to act upon"
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description=(
            "Agent's reasoning for taking this action. "
            "Must be at least 10 characters. Used for partial-credit scoring."
        ),
    )


class IncidentState(State):
    """
    Internal episode state (not fully exposed to the agent).

    Tracks ground-truth root causes and resolution bookkeeping.
    """

    incident_id: str = Field(..., description="Unique identifier for this incident")
    root_cause_service: str = Field(
        ..., description="Ground-truth root cause service"
    )
    root_cause_type: str = Field(
        ...,
        description=(
            "Type of root cause: 'bad_deployment' | 'resource_exhaustion' | "
            "'dependency_failure' | 'traffic_spike'"
        ),
    )
    correct_action_sequence: List[str] = Field(
        ...,
        description="Optimal action sequence to resolve this incident",
    )
    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full log of all (action_type, target_service, rationale) tuples",
    )
    investigated_services: List[str] = Field(
        default_factory=list,
        description="Services the agent has run INVESTIGATE on",
    )
    resolved: bool = Field(
        default=False, description="Whether the agent has declared resolution"
    )
    escalated: bool = Field(
        default=False, description="Whether the agent escalated to a human"
    )
