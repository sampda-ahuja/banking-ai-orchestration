#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eval.sh — Banking AI Orchestration Evaluation Pipeline
#
# Usage (run from project root):
#   chmod +x evals/run_eval.sh
#   ./evals/run_eval.sh                      # full suite (direct graph)
#   ./evals/run_eval.sh --filter TC02        # only TC02x cases
#   ./evals/run_eval.sh --category pii_block # only a category
#   ./evals/run_eval.sh --show-all           # print per-check breakdown always
#
# No running API server needed — uses compiled_workflow directly.
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Locate project root (script lives in evals/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load environment ──────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
  export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi
if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Banking AI — Evaluation Runner  (direct compiled graph)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project  : $PROJECT_ROOT"
echo "  Dataset  : evals/eval_dataset_v2.json  (35 cases)"
echo "  Runner   : evals/run_eval.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Check venv ────────────────────────────────────────────────────────────────
VENV="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV" ]; then
  echo ""
  echo "  Virtual environment not found at .venv/"
  echo "    Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
source "$VENV"

# ── Check GROQ_API_KEY ────────────────────────────────────────────────────────
if [ -z "$GROQ_API_KEY" ]; then
  echo ""
  echo "  GROQ_API_KEY is not set — required for graph execution"
  echo "    Set it in .env or: export GROQ_API_KEY=gsk_..."
  exit 1
fi
echo ""
echo "GROQ_API_KEY found (${GROQ_API_KEY:0:8}...)"

# ── Run Python evaluation ─────────────────────────────────────────────────────
echo ""
echo " Running evaluation suite..."
echo ""

cd "$PROJECT_ROOT"
python evals/run_eval.py "$@"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results saved → evals/results/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
