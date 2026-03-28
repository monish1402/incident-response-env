#!/usr/bin/env bash
# =============================================================================
# submit.sh — Pre-Submission Validator & Hugging Face Spaces Deployment
#
# Usage:
#   bash submit.sh YOUR_HF_USERNAME
#
# What it does:
#   1. Runs the full pytest suite (34 tests)
#   2. Runs the baseline agent on all 3 tasks
#   3. Validates openenv.yaml is parseable YAML
#   4. Boots the server and checks all required HTTP endpoints
#   5. Confirms scores are in [0.0, 1.0]
#   6. Offers to push to Hugging Face Spaces
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✅ $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; exit 1; }
info() { echo -e "  ${BLUE}ℹ  $1${NC}"; }
header() { echo -e "\n${BOLD}${YELLOW}$1${NC}"; }

HF_USERNAME="${1:-YOUR_HF_USERNAME}"
REPO_ID="${HF_USERNAME}/incident-response-env"
SERVER_PORT=17860  # Use non-standard port to avoid conflicts
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Incident Response OpenEnv — Pre-Submission     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Test suite ─────────────────────────────────────────────────────────────
header "[1/6] Running pytest suite..."
python -m pytest tests/ -q --tb=short 2>&1
pass "34 tests passed"

# ── 2. Baseline reproducibility ───────────────────────────────────────────────
header "[2/6] Running baseline agent (seed=42)..."
python -m baseline.baseline 2>&1
pass "Baseline scores produced successfully"

# ── 3. YAML validation ────────────────────────────────────────────────────────
header "[3/6] Validating openenv.yaml..."
python -c "
import yaml, sys
with open('openenv.yaml') as f:
    data = yaml.safe_load(f)
required = ['name', 'version', 'description', 'tasks', 'observation_space', 'action_space']
missing = [k for k in required if k not in data]
if missing:
    print(f'Missing keys: {missing}', file=sys.stderr)
    sys.exit(1)
print(f'  tasks: {[t[\"id\"] for t in data[\"tasks\"]]}')
"
pass "openenv.yaml is valid"

# ── 4. Boot server ────────────────────────────────────────────────────────────
header "[4/6] Starting server on port $SERVER_PORT..."
uvicorn server.app:app --host 0.0.0.0 --port "$SERVER_PORT" --log-level warning &
SERVER_PID=$!
sleep 4

check() {
    local method=$1 url=$2 body=${3:-}
    local status
    if [ -n "$body" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" \
            -X "$method" "$url" \
            -H "Content-Type: application/json" \
            -d "$body")
    else
        status=$(curl -s -o /dev/null -w "%{http_code}" -X "$method" "$url")
    fi
    if [ "$status" = "200" ]; then
        pass "$method $url → HTTP $status"
    else
        fail "$method $url → HTTP $status (expected 200)"
    fi
}

BASE="http://localhost:$SERVER_PORT"

header "[5/6] Checking required endpoints..."
check GET  "$BASE/health"
check GET  "$BASE/schema"
check GET  "$BASE/tasks"
check POST "$BASE/reset" '{"difficulty":"easy","seed":42}'
check GET  "$BASE/state"
check POST "$BASE/step" \
    '{"action_type":"investigate","target_service":"api-gateway","rationale":"Investigating the user-facing entry point first."}'
check POST "$BASE/grader" '{"task_id":"easy_incident_triage"}'
check POST "$BASE/baseline"

# ── 5. Score validation ───────────────────────────────────────────────────────
info "Verifying grader scores are in [0.0, 1.0]..."
python -c "
from server.graders import grade_all
results = grade_all()
for task_id, score in results.items():
    assert 0.0 <= score <= 1.0, f'{task_id}: {score} out of range'
    print(f'  {task_id}: {score:.4f}')
mean = sum(results.values()) / len(results)
print(f'  mean: {mean:.4f}')
"
pass "All grader scores in [0.0, 1.0]"

kill "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

# ── 6. Push to HF Spaces ──────────────────────────────────────────────────────
header "[6/6] Hugging Face Spaces deployment"
echo ""

if [ "$HF_USERNAME" = "YOUR_HF_USERNAME" ]; then
    echo -e "  ${YELLOW}⚠  No HF username provided. Skipping push.${NC}"
    echo ""
    echo "  To push, run:"
    echo "    bash submit.sh YOUR_HF_USERNAME"
else
    echo -e "  Target: ${BOLD}https://huggingface.co/spaces/${REPO_ID}${NC}"
    echo ""
    echo -e "  ${YELLOW}Remember to replace 'YOUR_HF_USERNAME' in:${NC}"
    echo "    - openenv.yaml  (author field)"
    echo "    - README.md     (all URL references)"
    echo ""
    read -r -p "  Push to HF Spaces now? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        git init -b main
        git add .
        git commit -m "feat: initial competition submission — incident-response-env v1.0.0" \
            2>/dev/null || git commit --allow-empty -m "update"
        git remote add hf "https://huggingface.co/spaces/${REPO_ID}" 2>/dev/null || \
            git remote set-url hf "https://huggingface.co/spaces/${REPO_ID}"
        git push hf main --force
        echo ""
        pass "Pushed to https://huggingface.co/spaces/${REPO_ID}"
        echo ""
        echo -e "  ${BOLD}Submit this URL to the competition:${NC}"
        echo -e "  ${GREEN}https://${HF_USERNAME}-incident-response-env.hf.space${NC}"
    else
        info "Skipped. To push manually:"
        echo ""
        echo "    git remote add hf https://huggingface.co/spaces/${REPO_ID}"
        echo "    git push hf main"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ✅  All pre-submission checks PASSED            ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
