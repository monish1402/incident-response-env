"""
app.py
======
FastAPI server for the Incident Response OpenEnv environment.

Uses openenv-core's create_app() factory to mount:
  - WebSocket endpoint for persistent sessions (ws://.../ws)
  - REST endpoints: /reset, /step, /state, /health, /schema

Additional custom endpoints:
  - GET  /tasks      : List all tasks with action schemas
  - POST /grader     : Run grader for a task (returns score 0.0–1.0)
  - POST /baseline   : Run all three graders, return scores + mean

Usage
-----
    # Development
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

    # Production (in Docker)
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app
from pydantic import BaseModel

from server.environment import IncidentResponseEnv
from server.graders import GRADERS, grade_all
from server.models import IncidentAction, IncidentObservation


# ---------------------------------------------------------------------------
# Build the base OpenEnv application
# ---------------------------------------------------------------------------

app: FastAPI = create_app(
    env=IncidentResponseEnv,
    action_cls=IncidentAction,
    observation_cls=IncidentObservation,
    env_name="incident-response-env",
)

# Allow cross-origin requests (required for HF Spaces embedding)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom request / response schemas
# ---------------------------------------------------------------------------


class GraderRequest(BaseModel):
    """Request body for the /grader endpoint."""

    task_id: str
    actions: Optional[list[dict]] = None  # If None, uses default grader actions


class GraderResponse(BaseModel):
    """Response from the /grader endpoint."""

    task_id: str
    score: float
    message: str


class BaselineResponse(BaseModel):
    """Response from the /baseline endpoint."""

    scores: Dict[str, float]
    mean_score: float
    message: str


# ---------------------------------------------------------------------------
# Custom endpoints
# ---------------------------------------------------------------------------


@app.get("/tasks", tags=["openenv"], summary="List all available tasks")
def list_tasks() -> JSONResponse:
    """
    Return all available tasks with difficulty levels and action schemas.

    This endpoint is required by the OpenEnv pre-submission checklist.
    """
    env = IncidentResponseEnv()
    tasks = env.get_tasks()
    return JSONResponse(content={"tasks": tasks})


@app.post(
    "/grader",
    response_model=GraderResponse,
    tags=["openenv"],
    summary="Run the grader for a specific task",
)
def run_grader(request: GraderRequest) -> GraderResponse:
    """
    Run the deterministic grader for a given task ID.

    Accepts an optional list of actions to grade. If not provided,
    runs the default optimal action sequence (used for baseline scoring).

    Returns a score in [0.0, 1.0].
    """
    if request.task_id not in GRADERS:
        return GraderResponse(
            task_id=request.task_id,
            score=0.0,
            message=(
                f"Unknown task_id '{request.task_id}'. "
                f"Valid options: {list(GRADERS.keys())}"
            ),
        )

    # If custom actions provided, parse and grade them
    if request.actions:
        try:
            parsed = [IncidentAction(**a) for a in request.actions]
            score = GRADERS[request.task_id](custom_actions=parsed)
        except Exception as exc:
            return GraderResponse(
                task_id=request.task_id,
                score=0.0,
                message=f"Failed to parse actions: {exc}",
            )
    else:
        score = GRADERS[request.task_id]()

    return GraderResponse(
        task_id=request.task_id,
        score=score,
        message=f"Grader completed. Score: {score:.4f}",
    )


@app.post(
    "/baseline",
    response_model=BaselineResponse,
    tags=["openenv"],
    summary="Run baseline inference across all tasks",
)
def run_baseline() -> BaselineResponse:
    """
    Run the default (optimal) grader action sequence across all three tasks.

    Returns per-task scores and the mean. Used by the competition's
    automated validation pipeline to verify reproducibility.

    This endpoint is required by the OpenEnv pre-submission checklist.
    """
    from baseline.baseline import run_baseline_agent
    scores = run_baseline_agent()
    mean = round(sum(scores.values()) / len(scores), 4)

    return BaselineResponse(
        scores=scores,
        mean_score=mean,
        message=(
            f"Baseline complete. Mean score: {mean:.4f} across "
            f"{len(scores)} tasks."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the server — used by the pyproject.toml script entry point."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
