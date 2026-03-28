"""
scenarios.py
============
Production incident scenarios for the Incident Response environment.

Each scenario is a complete, self-contained incident definition including:
  - The system topology (services and their dependencies)
  - Initial metric values that make the root cause diagnosable
  - Ground-truth root cause information
  - The optimal action sequence to resolve it
  - Alert definitions
  - Diagnostic logs returned when the agent investigates each service

Design Philosophy
-----------------
Scenarios are designed so that:
  1. A random agent performs poorly (< 0.3 average score)
  2. A rule-based heuristic achieves moderate performance (0.4–0.6)
  3. A reasoning LLM can achieve high performance (0.8+) with good prompting
  4. The hard scenario genuinely challenges frontier models (cascading failures,
     misleading signals, multi-hop root cause)

All values are deterministic given the scenario definition — no random noise
is added to the scenario data itself, only to episode selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes (pure Python — no Pydantic, fast to instantiate)
# ---------------------------------------------------------------------------


@dataclass
class ServiceSpec:
    """Blueprint for a service node within a scenario."""

    name: str
    status: str  # "healthy" | "degraded" | "down"
    error_rate: float
    latency_p99_ms: float
    cpu_utilisation: float
    memory_utilisation: float
    recent_deployment: bool
    dependencies: List[str]

    # Diagnostic output returned when the agent investigates this service
    log_lines: List[str] = field(default_factory=list)
    error_summary: str = ""
    suggested_action: Optional[str] = None  # May be None for harder scenarios


@dataclass
class ScenarioAlert:
    """Alert definition for a scenario."""

    alert_id: str
    service: str
    severity: str  # "P1" | "P2" | "P3" | "P4"
    message: str


@dataclass
class Scenario:
    """A complete incident scenario definition."""

    scenario_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    incident_title: str
    incident_severity: str  # "P1" | "P2"
    root_cause_service: str
    root_cause_type: str  # "bad_deployment" | "resource_exhaustion" | ...
    correct_action_sequence: List[str]  # Ordered list of action_type strings
    services: List[ServiceSpec]
    alerts: List[ScenarioAlert]
    time_budget_minutes: int = 60
    max_steps: int = 10


# ---------------------------------------------------------------------------
# EASY scenarios — single service failure, obvious signals
# ---------------------------------------------------------------------------

EASY_SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id="easy_001",
        difficulty="easy",
        incident_title="[P2] API Gateway returning 502s — checkout flow impacted",
        incident_severity="P2",
        root_cause_service="payment-service",
        root_cause_type="bad_deployment",
        correct_action_sequence=["investigate", "rollback", "resolve"],
        time_budget_minutes=45,
        alerts=[
            ScenarioAlert(
                alert_id="ALT-001",
                service="api-gateway",
                severity="P2",
                message="API Gateway: 502 error rate > 15% for 5 minutes",
            ),
            ScenarioAlert(
                alert_id="ALT-002",
                service="payment-service",
                severity="P2",
                message="Payment Service: error rate spike to 78% post-deployment",
            ),
        ],
        services=[
            ServiceSpec(
                name="api-gateway",
                status="degraded",
                error_rate=0.18,
                latency_p99_ms=2400,
                cpu_utilisation=0.45,
                memory_utilisation=0.50,
                recent_deployment=False,
                dependencies=[],
                log_lines=[
                    "2026-03-27T10:01:12Z [ERROR] Upstream payment-service: connection refused",
                    "2026-03-27T10:01:13Z [ERROR] Upstream payment-service: connection refused",
                    "2026-03-27T10:01:14Z [WARN]  Circuit breaker OPEN for payment-service",
                    "2026-03-27T10:01:20Z [ERROR] 502 Bad Gateway returned to client",
                ],
                error_summary=(
                    "API Gateway is returning 502s because payment-service is refusing "
                    "connections. Circuit breaker has opened. All other upstreams healthy."
                ),
                suggested_action="Investigate payment-service — it appears to be the root cause.",
            ),
            ServiceSpec(
                name="payment-service",
                status="down",
                error_rate=0.78,
                latency_p99_ms=30000,
                cpu_utilisation=0.15,
                memory_utilisation=0.20,
                recent_deployment=True,  # ← key signal
                dependencies=["postgres-db"],
                log_lines=[
                    "2026-03-27T09:55:01Z [INFO]  Deploying v2.4.1 → v2.4.2",
                    "2026-03-27T09:55:03Z [ERROR] Failed to apply DB migration: column 'amount_cents' does not exist",
                    "2026-03-27T09:55:03Z [FATAL] Application startup failed — exiting",
                    "2026-03-27T09:55:05Z [ERROR] Health check failed: service not responding",
                ],
                error_summary=(
                    "Payment service crashed immediately after deployment v2.4.2. "
                    "A database migration attempted to reference a non-existent column. "
                    "The service never completed startup. Rollback to v2.4.1 will restore service."
                ),
                suggested_action="ROLLBACK payment-service to the previous deployment.",
            ),
            ServiceSpec(
                name="postgres-db",
                status="healthy",
                error_rate=0.0,
                latency_p99_ms=8,
                cpu_utilisation=0.30,
                memory_utilisation=0.55,
                recent_deployment=False,
                dependencies=[],
                log_lines=[
                    "2026-03-27T10:01:00Z [INFO]  All connections healthy",
                    "2026-03-27T10:01:00Z [INFO]  Active connections: 42/200",
                ],
                error_summary="PostgreSQL is healthy. No errors detected.",
                suggested_action=None,
            ),
        ],
    ),
    Scenario(
        scenario_id="easy_002",
        difficulty="easy",
        incident_title="[P2] User-service OOMKilled — login failures reported",
        incident_severity="P2",
        root_cause_service="user-service",
        root_cause_type="resource_exhaustion",
        correct_action_sequence=["investigate", "scale_up", "resolve"],
        time_budget_minutes=45,
        alerts=[
            ScenarioAlert(
                alert_id="ALT-003",
                service="user-service",
                severity="P2",
                message="user-service: OOMKilled — pod restarted 4 times in 10 minutes",
            ),
        ],
        services=[
            ServiceSpec(
                name="user-service",
                status="degraded",
                error_rate=0.35,
                latency_p99_ms=8000,
                cpu_utilisation=0.60,
                memory_utilisation=0.98,  # ← near-OOM signal
                recent_deployment=False,
                dependencies=["redis-cache"],
                log_lines=[
                    "2026-03-27T11:20:00Z [WARN]  Memory usage at 94% of limit",
                    "2026-03-27T11:21:00Z [WARN]  Memory usage at 98% of limit",
                    "2026-03-27T11:21:30Z [ERROR] java.lang.OutOfMemoryError: Java heap space",
                    "2026-03-27T11:21:31Z [FATAL] Process killed by OOM killer",
                ],
                error_summary=(
                    "user-service is being OOMKilled repeatedly due to heap exhaustion. "
                    "Memory usage climbed steadily — likely a memory leak or a traffic spike "
                    "without corresponding resource limits. Scaling up replicas will distribute "
                    "the load and reduce per-instance memory pressure."
                ),
                suggested_action="SCALE_UP user-service to add more replicas.",
            ),
            ServiceSpec(
                name="redis-cache",
                status="healthy",
                error_rate=0.0,
                latency_p99_ms=2,
                cpu_utilisation=0.20,
                memory_utilisation=0.40,
                recent_deployment=False,
                dependencies=[],
                log_lines=["2026-03-27T11:21:00Z [INFO] All clients connected. Hit rate: 94%"],
                error_summary="Redis is healthy.",
                suggested_action=None,
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# MEDIUM scenarios — multi-service impact, less obvious root cause
# ---------------------------------------------------------------------------

MEDIUM_SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id="medium_001",
        difficulty="medium",
        incident_title="[P1] Checkout latency > 10s — revenue impact ongoing",
        incident_severity="P1",
        root_cause_service="inventory-service",
        root_cause_type="dependency_failure",
        correct_action_sequence=["investigate", "investigate", "restart", "resolve"],
        time_budget_minutes=30,
        alerts=[
            ScenarioAlert(
                alert_id="ALT-010",
                service="checkout-service",
                severity="P1",
                message="checkout-service: p99 latency > 10,000ms for 8 minutes",
            ),
            ScenarioAlert(
                alert_id="ALT-011",
                service="order-service",
                severity="P2",
                message="order-service: error rate elevated to 22%",
            ),
        ],
        services=[
            ServiceSpec(
                name="checkout-service",
                status="degraded",
                error_rate=0.08,
                latency_p99_ms=11500,
                cpu_utilisation=0.65,
                memory_utilisation=0.60,
                recent_deployment=False,
                dependencies=["order-service", "payment-service"],
                log_lines=[
                    "2026-03-27T14:00:01Z [WARN]  Slow response from order-service: 9800ms",
                    "2026-03-27T14:00:05Z [WARN]  Slow response from order-service: 11200ms",
                    "2026-03-27T14:00:10Z [ERROR] Timeout waiting for order-service (>10000ms)",
                ],
                error_summary=(
                    "Checkout is slow because order-service is responding very slowly. "
                    "Payment-service appears healthy. Investigate order-service next."
                ),
                suggested_action=None,  # Agent must figure this out
            ),
            ServiceSpec(
                name="order-service",
                status="degraded",
                error_rate=0.22,
                latency_p99_ms=9800,
                cpu_utilisation=0.70,
                memory_utilisation=0.68,
                recent_deployment=False,
                dependencies=["inventory-service", "postgres-db"],
                log_lines=[
                    "2026-03-27T13:58:00Z [WARN]  inventory-service call timed out after 5000ms",
                    "2026-03-27T13:58:02Z [WARN]  Retrying inventory-service (attempt 2/3)",
                    "2026-03-27T13:58:07Z [ERROR] inventory-service unavailable after 3 retries",
                    "2026-03-27T13:58:07Z [ERROR] Cannot confirm stock — order rejected",
                ],
                error_summary=(
                    "Order service is failing because inventory-service is not responding. "
                    "Every order requires an inventory check. This is the bottleneck. "
                    "Investigate or restart inventory-service."
                ),
                suggested_action="Investigate or restart inventory-service.",
            ),
            ServiceSpec(
                name="inventory-service",
                status="down",
                error_rate=0.95,
                latency_p99_ms=0,
                cpu_utilisation=0.02,
                memory_utilisation=0.10,
                recent_deployment=False,
                dependencies=["mongo-db"],
                log_lines=[
                    "2026-03-27T13:55:01Z [ERROR] Lost connection to mongo-db: connection reset",
                    "2026-03-27T13:55:05Z [ERROR] Reconnect attempt 1/5 failed",
                    "2026-03-27T13:55:10Z [ERROR] Reconnect attempt 2/5 failed",
                    "2026-03-27T13:55:30Z [FATAL] Failed to reconnect to mongo-db — entering crash loop",
                    "2026-03-27T13:57:00Z [INFO]  mongo-db connection restored",
                    "2026-03-27T13:57:01Z [ERROR] Service stuck in bad state — needs restart",
                ],
                error_summary=(
                    "Inventory-service entered a crash loop after losing its MongoDB connection. "
                    "MongoDB has since recovered but inventory-service is stuck in a bad state. "
                    "A restart will allow it to reconnect successfully."
                ),
                suggested_action="RESTART inventory-service to clear the bad connection state.",
            ),
            ServiceSpec(
                name="payment-service",
                status="healthy",
                error_rate=0.01,
                latency_p99_ms=120,
                cpu_utilisation=0.30,
                memory_utilisation=0.45,
                recent_deployment=False,
                dependencies=["postgres-db"],
                log_lines=["2026-03-27T14:00:00Z [INFO]  All transactions processing normally"],
                error_summary="Payment service is healthy.",
                suggested_action=None,
            ),
            ServiceSpec(
                name="postgres-db",
                status="healthy",
                error_rate=0.0,
                latency_p99_ms=5,
                cpu_utilisation=0.25,
                memory_utilisation=0.50,
                recent_deployment=False,
                dependencies=[],
                log_lines=["2026-03-27T14:00:00Z [INFO]  Database healthy. 85 active connections."],
                error_summary="PostgreSQL is healthy.",
                suggested_action=None,
            ),
            ServiceSpec(
                name="mongo-db",
                status="healthy",
                error_rate=0.0,
                latency_p99_ms=3,
                cpu_utilisation=0.20,
                memory_utilisation=0.55,
                recent_deployment=False,
                dependencies=[],
                log_lines=[
                    "2026-03-27T13:57:00Z [INFO]  Recovered from brief network partition",
                    "2026-03-27T13:57:01Z [INFO]  All replica set members healthy",
                ],
                error_summary="MongoDB recovered from a brief network partition. Now healthy.",
                suggested_action=None,
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# HARD scenarios — cascading failures, misleading signals, multi-hop RCA
# ---------------------------------------------------------------------------

HARD_SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id="hard_001",
        difficulty="hard",
        incident_title="[P1] Site-wide degradation — multiple services failing simultaneously",
        incident_severity="P1",
        root_cause_service="auth-service",
        root_cause_type="traffic_spike",
        # Optimal path: investigate api-gateway (misleading), investigate auth-service
        # (root cause found), scale_up auth-service, resolve
        correct_action_sequence=[
            "investigate", "investigate", "scale_up", "resolve"
        ],
        time_budget_minutes=20,  # Tight SLA
        alerts=[
            ScenarioAlert(
                alert_id="ALT-020",
                service="api-gateway",
                severity="P1",
                message="api-gateway: error rate 45% — multiple upstream failures",
            ),
            ScenarioAlert(
                alert_id="ALT-021",
                service="user-service",
                severity="P1",
                message="user-service: all requests returning 401 Unauthorized",
            ),
            ScenarioAlert(
                alert_id="ALT-022",
                service="notification-service",
                severity="P2",
                message="notification-service: high error rate, emails not sending",
            ),
            ScenarioAlert(
                alert_id="ALT-023",
                service="auth-service",
                severity="P2",  # Lower severity — a misleading signal
                message="auth-service: response time elevated to 3200ms",
            ),
        ],
        services=[
            ServiceSpec(
                name="api-gateway",
                status="degraded",
                error_rate=0.45,
                latency_p99_ms=5500,
                cpu_utilisation=0.80,
                memory_utilisation=0.75,
                recent_deployment=False,
                dependencies=["user-service", "auth-service", "notification-service"],
                log_lines=[
                    "2026-03-27T16:00:01Z [ERROR] auth-service: token validation timeout (3100ms)",
                    "2026-03-27T16:00:02Z [ERROR] auth-service: token validation timeout (3400ms)",
                    "2026-03-27T16:00:03Z [WARN]  user-service: returning 401 (auth rejected)",
                    "2026-03-27T16:00:05Z [ERROR] notification-service: upstream auth failure",
                    "2026-03-27T16:00:06Z [WARN]  High request rate detected: 48,000 rpm (2x normal)",
                ],
                error_summary=(
                    "API Gateway is seeing failures across multiple upstreams — but all "
                    "failures are auth-related. auth-service is timing out causing cascading "
                    "401s everywhere. Traffic volume is 2x normal. auth-service is the bottleneck."
                ),
                suggested_action=None,  # Hard: agent must reason from data
            ),
            ServiceSpec(
                name="auth-service",
                status="degraded",
                error_rate=0.30,
                latency_p99_ms=3200,
                cpu_utilisation=0.97,  # ← the real signal: CPU saturated
                memory_utilisation=0.85,
                recent_deployment=False,
                dependencies=["redis-cache"],
                log_lines=[
                    "2026-03-27T15:55:00Z [WARN]  Token validation queue depth: 450 (normal: 20)",
                    "2026-03-27T15:58:00Z [WARN]  CPU throttling detected — validation slowing down",
                    "2026-03-27T16:00:00Z [WARN]  Queue depth: 1,200 — approaching limit",
                    "2026-03-27T16:00:05Z [ERROR] Token validation timeout — queue full",
                    "2026-03-27T16:00:06Z [INFO]  Current replicas: 2 (requested capacity: ~6)",
                ],
                error_summary=(
                    "auth-service is CPU-saturated due to a 2x traffic spike. "
                    "Token validation is backed up — queue depth is 60x normal. "
                    "Only 2 replicas running; need at least 6 to handle current load. "
                    "Scaling up replicas will immediately relieve the pressure."
                ),
                suggested_action="SCALE_UP auth-service — it is CPU-saturated due to traffic spike.",
            ),
            ServiceSpec(
                name="user-service",
                status="degraded",
                error_rate=0.88,
                latency_p99_ms=3500,
                cpu_utilisation=0.40,
                memory_utilisation=0.50,
                recent_deployment=False,
                dependencies=["auth-service"],
                log_lines=[
                    "2026-03-27T16:00:01Z [ERROR] auth-service rejected token: timed out",
                    "2026-03-27T16:00:02Z [ERROR] Returning 401 — cannot validate user session",
                ],
                error_summary=(
                    "user-service itself is healthy but auth-service is failing to validate "
                    "tokens fast enough. All 401s are a downstream consequence of auth-service "
                    "being overloaded. Fix auth-service to fix user-service."
                ),
                suggested_action=None,
            ),
            ServiceSpec(
                name="notification-service",
                status="degraded",
                error_rate=0.65,
                latency_p99_ms=4000,
                cpu_utilisation=0.35,
                memory_utilisation=0.45,
                recent_deployment=True,  # ← red herring: recent deploy, but not the cause
                dependencies=["auth-service"],
                log_lines=[
                    "2026-03-27T16:00:00Z [INFO]  Deployed v1.8.3 (minor email template update)",
                    "2026-03-27T16:00:01Z [ERROR] Cannot authenticate outbound requests: auth timeout",
                    "2026-03-27T16:00:02Z [ERROR] Email send failed: auth rejected",
                ],
                error_summary=(
                    "notification-service had a minor deployment (template update only) "
                    "but its failures are all auth-related, not deployment-related. "
                    "The deployment is a red herring — fix auth-service."
                ),
                suggested_action=None,
            ),
            ServiceSpec(
                name="redis-cache",
                status="healthy",
                error_rate=0.0,
                latency_p99_ms=1,
                cpu_utilisation=0.20,
                memory_utilisation=0.35,
                recent_deployment=False,
                dependencies=[],
                log_lines=["2026-03-27T16:00:00Z [INFO]  Cache healthy. Hit rate: 96%."],
                error_summary="Redis is healthy.",
                suggested_action=None,
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Lookup by difficulty
# ---------------------------------------------------------------------------

ALL_SCENARIOS: Dict[str, List[Scenario]] = {
    "easy": EASY_SCENARIOS,
    "medium": MEDIUM_SCENARIOS,
    "hard": HARD_SCENARIOS,
}

SCENARIO_BY_ID: Dict[str, Scenario] = {
    s.scenario_id: s
    for scenarios in ALL_SCENARIOS.values()
    for s in scenarios
}


def get_scenario(scenario_id: Optional[str], difficulty: Optional[str], rng) -> Scenario:
    """
    Select a scenario deterministically.

    Priority:
      1. If scenario_id is given, return that exact scenario.
      2. If difficulty is given, pick randomly from that tier.
      3. Otherwise pick randomly from all scenarios.
    """
    if scenario_id and scenario_id in SCENARIO_BY_ID:
        return SCENARIO_BY_ID[scenario_id]

    if difficulty and difficulty in ALL_SCENARIOS:
        pool = ALL_SCENARIOS[difficulty]
    else:
        pool = [s for scenarios in ALL_SCENARIOS.values() for s in scenarios]

    return rng.choice(pool)
