#!/usr/bin/env python3
"""
evals/run_eval.py
─────────────────────────────────────────────────────────────────────────────
Standalone evaluation runner for the Banking AI Orchestration system.

Captures ALL metrics required by the assignment:
  • Per-agent latency (P50, P95, max)
  • Per-agent token usage (prompt, completion, total)
  • Intent classification accuracy
  • Field extraction accuracy
  • Compliance pass rate
  • Disclaimer presence rate
  • Response quality scoring
  • Overall pass rate by category

Usage (from project root):
    source .venv/bin/activate
    python evals/run_eval.py                     # full suite
    python evals/run_eval.py --filter TC02       # only cases whose id starts with TC02
    python evals/run_eval.py --category pii_block
    python evals/run_eval.py --show-all          # print per-check breakdown for every case
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing app modules
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.graph.workflow import compiled_workflow  # noqa: E402
from app.graph.state import WorkflowState         # noqa: E402
from app.utils.config import get_settings         # noqa: E402

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

DATASET_PATH = Path(__file__).parent / "eval_dataset_v2.json"
RESULTS_DIR  = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (strips JS-style // comments before parsing)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_js_comments(text: str) -> str:
    """Remove // line-comments so the file can be parsed as JSON."""
    return re.sub(r"//.*", "", text)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    cleaned = _strip_js_comments(raw)
    data = json.loads(cleaned)
    return data["cases"]


# ─────────────────────────────────────────────────────────────────────────────
# Workflow runner
# ─────────────────────────────────────────────────────────────────────────────

def run_case(user_input: str) -> WorkflowState:
    """Invoke the compiled LangGraph workflow and return the final state."""
    initial: WorkflowState = {"user_input": user_input, "events": [], "metrics": {}}
    return compiled_workflow.invoke(initial)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_route(state: WorkflowState) -> str:
    """Infer which terminal route was taken from the events log."""
    for event in state.get("events", []):
        if event.get("agent") == "safe_terminate":
            return "safe_terminate"
        if event.get("agent") == "escalation":
            return "escalation"
        if event.get("agent") == "compliance_fallback":
            return "fallback"
        if event.get("agent") == "jailbreak_detected":
            return "fallback"
    # If compliance ran and passed → happy path
    if "compliance_result" in state:
        return "happy_path"
    return "unknown"


class Check:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name   = name
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:
        icon = f"{GREEN}✓{RESET}" if self.passed else f"{RED}✗{RESET}"
        suffix = f"  {DIM}{self.detail}{RESET}" if self.detail else ""
        return f"  {icon} {self.name}{suffix}"


def score_case(case: dict[str, Any], state: WorkflowState) -> list[Check]:
    checks: list[Check] = []
    expected = case.get("expected", {})
    actual_route = _detect_route(state)

    # ── Route ────────────────────────────────────────────────────────────────
    exp_route = case.get("expected_route", "")
    if not exp_route and "route" in expected:
        exp_route = expected["route"]
    if exp_route:
        checks.append(Check(
            "route",
            actual_route == exp_route,
            f"expected={exp_route}  actual={actual_route}",
        ))

    # ── Guardrails ───────────────────────────────────────────────────────────
    exp_gr = expected.get("guardrails", {})
    gr = state.get("guardrail_result")
    if gr and exp_gr:
        for field, exp_val in exp_gr.items():
            actual_val = getattr(gr, field, None)
            checks.append(Check(
                f"guardrails.{field}",
                actual_val == exp_val,
                f"expected={exp_val}  actual={actual_val}",
            ))

    # ── Inquiry parser ───────────────────────────────────────────────────────
    exp_parser = expected.get("inquiry_parser", {})
    inq = state.get("inquiry_result")
    if inq and exp_parser:
        for field, exp_val in exp_parser.items():
            actual_val = getattr(inq, field, None)
            if exp_val is None:
                continue
            # Flexible string comparison (case-insensitive, partial)
            if isinstance(exp_val, str) and isinstance(actual_val, str):
                passed = exp_val.lower() in actual_val.lower()
            else:
                passed = actual_val == exp_val
            checks.append(Check(
                f"inquiry_parser.{field}",
                passed,
                f"expected={exp_val!r}  actual={actual_val!r}",
            ))

    # ── Risk policy ──────────────────────────────────────────────────────────
    exp_risk = expected.get("risk_policy", {})
    risk = state.get("risk_result")
    if risk and exp_risk:
        if "risk_score_estimate" in exp_risk:
            actual_risk = risk.risk_score_estimate
            exp_risk_val = exp_risk["risk_score_estimate"]
            checks.append(Check(
                "risk_policy.risk_score_estimate",
                actual_risk.upper() == exp_risk_val.upper(),
                f"expected={exp_risk_val}  actual={actual_risk}",
            ))

    # ── Compliance ───────────────────────────────────────────────────────────
    exp_comp = expected.get("compliance", {})
    comp = state.get("compliance_result")
    if comp and exp_comp:
        if "compliance_pass" in exp_comp:
            checks.append(Check(
                "compliance.compliance_pass",
                comp.compliance_pass == exp_comp["compliance_pass"],
                f"expected={exp_comp['compliance_pass']}  actual={comp.compliance_pass}",
            ))
        if "violations" in exp_comp and exp_comp["violations"] == []:
            checks.append(Check(
                "compliance.no_violations",
                comp.violations == [],
                f"actual_violations={comp.violations}",
            ))

    # ── Disclaimer always present on happy-path ───────────────────────────────
    if actual_route == "happy_path" and state.get("final_response"):
        disclaimer = "formal credit review and approval"
        checks.append(Check(
            "disclaimer_present",
            disclaimer.lower() in state["final_response"].lower(),
        ))

    # ── Response quality (basic heuristic scoring) ────────────────────────────
    if actual_route == "happy_path" and state.get("final_response"):
        response = state["final_response"]
        quality_checks = _score_response_quality(response)
        checks.extend(quality_checks)

    return checks


def _score_response_quality(response: str) -> list[Check]:
    """
    Heuristic response quality scoring:
    - Professional tone (no slang, no exclamation marks overuse)
    - Neutral (no guarantee language)
    - Concise (under 500 chars)
    """
    checks: list[Check] = []

    # Conciseness: response should be under 500 chars for a summary
    is_concise = len(response) <= 600
    checks.append(Check(
        "quality.concise",
        is_concise,
        f"length={len(response)} chars (target ≤600)",
    ))

    # Professional: no exclamation marks overuse
    excl_count = response.count("!")
    is_professional = excl_count <= 1
    checks.append(Check(
        "quality.professional_tone",
        is_professional,
        f"exclamation_marks={excl_count} (target ≤1)",
    ))

    # Neutral: no guarantee/promise language
    guarantee_words = ["guaranteed", "definitely", "certainly will", "promise", "assured"]
    has_guarantee = any(w in response.lower() for w in guarantee_words)
    checks.append(Check(
        "quality.neutral_tone",
        not has_guarantee,
        "no guarantee language" if not has_guarantee else "contains guarantee language",
    ))

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Metrics aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile from a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-agent metrics across all cases into the summary table
    required by the assignment:
      Agent | Model | Avg Tokens | P50 Latency | Accuracy | Compliance Pass
    """
    settings = get_settings()

    # Collect per-agent data
    agent_data: dict[str, dict[str, list[float]]] = {
        "guardrails": {"latency": [], "tokens": []},
        "inquiry_parser": {"latency": [], "tokens": []},
        "risk_policy": {"latency": [], "tokens": []},
        "compliance": {"latency": [], "tokens": []},
    }

    # Accuracy counters
    guardrail_correct = 0
    guardrail_total = 0
    intent_correct = 0
    intent_total = 0
    field_correct = 0
    field_total = 0
    compliance_pass_count = 0
    compliance_total = 0
    disclaimer_present = 0
    disclaimer_total = 0
    quality_pass = 0
    quality_total = 0

    for r in results:
        # Collect per-node metrics
        metrics = r.get("node_metrics", {})
        for agent_name, data in metrics.items():
            if agent_name in agent_data:
                agent_data[agent_name]["latency"].append(data.get("latency_ms", 0))
                agent_data[agent_name]["tokens"].append(data.get("total_tokens", 0))

        # Score accuracy from checks
        for check in r.get("checks", []):
            name = check["name"]
            passed = check["passed"]

            if name.startswith("guardrails."):
                guardrail_total += 1
                if passed:
                    guardrail_correct += 1

            elif name == "inquiry_parser.intent":
                intent_total += 1
                if passed:
                    intent_correct += 1

            elif name.startswith("inquiry_parser."):
                field_total += 1
                if passed:
                    field_correct += 1

            elif name == "compliance.compliance_pass":
                compliance_total += 1
                if passed:
                    compliance_pass_count += 1

            elif name == "disclaimer_present":
                disclaimer_total += 1
                if passed:
                    disclaimer_present += 1

            elif name.startswith("quality."):
                quality_total += 1
                if passed:
                    quality_pass += 1

    # Build summary table
    model_map = {
        "guardrails": settings.guardrail_model,
        "inquiry_parser": settings.parser_model,
        "risk_policy": "deterministic (no LLM)",
        "compliance": settings.compliance_model,
    }

    summary_table: list[dict[str, Any]] = []
    for agent_name in ["guardrails", "inquiry_parser", "risk_policy", "compliance"]:
        data = agent_data[agent_name]
        avg_tokens = sum(data["tokens"]) / len(data["tokens"]) if data["tokens"] else 0
        p50_latency = _percentile(data["latency"], 50)
        p95_latency = _percentile(data["latency"], 95)
        max_latency = max(data["latency"]) if data["latency"] else 0

        summary_table.append({
            "agent": agent_name,
            "model": model_map[agent_name],
            "avg_tokens": round(avg_tokens, 1),
            "p50_latency_ms": round(p50_latency, 1),
            "p95_latency_ms": round(p95_latency, 1),
            "max_latency_ms": round(max_latency, 1),
            "total_calls": len(data["latency"]),
        })

    # Accuracy metrics
    accuracy_metrics = {
        "guardrail_boolean_accuracy": f"{guardrail_correct}/{guardrail_total} ({_pct(guardrail_correct, guardrail_total)}%)",
        "intent_classification_accuracy": f"{intent_correct}/{intent_total} ({_pct(intent_correct, intent_total)}%)",
        "field_extraction_accuracy": f"{field_correct}/{field_total} ({_pct(field_correct, field_total)}%)",
        "compliance_pass_rate": f"{compliance_pass_count}/{compliance_total} ({_pct(compliance_pass_count, compliance_total)}%)",
        "disclaimer_presence": f"{disclaimer_present}/{disclaimer_total} ({_pct(disclaimer_present, disclaimer_total)}%)",
        "response_quality": f"{quality_pass}/{quality_total} ({_pct(quality_pass, quality_total)}%)",
    }

    # End-to-end latency
    e2e_latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    e2e_metrics = {
        "p50_ms": round(_percentile(e2e_latencies, 50), 1),
        "p95_ms": round(_percentile(e2e_latencies, 95), 1),
        "max_ms": round(max(e2e_latencies), 1) if e2e_latencies else 0,
    }

    # Total token usage
    total_tokens_all = sum(
        sum(agent_data[a]["tokens"]) for a in agent_data
    )

    return {
        "per_agent_summary": summary_table,
        "accuracy_metrics": accuracy_metrics,
        "end_to_end_latency": e2e_metrics,
        "total_tokens_consumed": total_tokens_all,
        "avg_tokens_per_request": round(total_tokens_all / len(results), 1) if results else 0,
    }


def _pct(num: int, denom: int) -> int:
    return int(num / denom * 100) if denom else 0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    filter_id: str | None = None,
    filter_category: str | None = None,
    show_all: bool = False,
) -> None:
    cases = load_dataset(DATASET_PATH)

    # Apply filters
    if filter_id:
        cases = [c for c in cases if c["id"].startswith(filter_id)]
    if filter_category:
        cases = [c for c in cases if c.get("category") == filter_category]

    if not cases:
        print(f"{YELLOW}No cases matched the filter.{RESET}")
        return

    print(f"\n{BOLD}{'━'*70}{RESET}")
    print(f"{BOLD}  Banking AI — Evaluation Runner  (direct graph invocation){RESET}")
    print(f"{BOLD}{'━'*70}{RESET}")
    print(f"  Dataset  : {DATASET_PATH.name}")
    print(f"  Cases    : {len(cases)}")
    print(f"  Source   : compiled_workflow (no HTTP)")
    print(f"{BOLD}{'━'*70}{RESET}\n")

    results: list[dict[str, Any]] = []
    passed_cases  = 0
    failed_cases  = 0

    for case in cases:
        case_id   = case["id"]
        is_hard   = case.get("hard", False)
        hard_tag  = f" {YELLOW}[HARD]{RESET}" if is_hard else ""
        input_preview = case['input'][:70] + "..." if len(case['input']) > 70 else case['input']
        print(f"{BOLD}{case_id}{RESET}{hard_tag}  {DIM}{input_preview}{RESET}")

        t0 = time.perf_counter()
        try:
            state = run_case(case["input"])
            latency_ms = (time.perf_counter() - t0) * 1000
            checks = score_case(case, state)
            case_passed = all(c.passed for c in checks)

            if case_passed:
                passed_cases += 1
                print(f"  {GREEN}PASS{RESET}  {DIM}{int(latency_ms)}ms{RESET}")
            else:
                failed_cases += 1
                print(f"  {RED}FAIL{RESET}  {DIM}{int(latency_ms)}ms{RESET}")

            if show_all or not case_passed:
                for check in checks:
                    print(repr(check))

            # Collect node-level metrics from state
            node_metrics = state.get("metrics", {})

            results.append({
                "id":        case_id,
                "category":  case.get("category"),
                "hard":      is_hard,
                "passed":    case_passed,
                "latency_ms": latency_ms,
                "node_metrics": node_metrics,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in checks
                ],
                "final_response": state.get("final_response", ""),
            })

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            failed_cases += 1
            print(f"  {RED}ERROR{RESET}  {int(latency_ms)}ms  — {exc}")
            results.append({
                "id": case_id, "category": case.get("category"),
                "hard": is_hard, "passed": False,
                "latency_ms": latency_ms, "error": str(exc),
                "checks": [], "node_metrics": {},
            })

        print()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg = aggregate_metrics(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed_cases + failed_cases
    pct   = int(passed_cases / total * 100) if total else 0
    colour = GREEN if pct >= 80 else YELLOW if pct >= 60 else RED

    print(f"{BOLD}{'━'*70}{RESET}")
    print(f"  {BOLD}Overall Results{RESET}  {colour}{passed_cases}/{total} passed ({pct}%){RESET}")
    print(f"{BOLD}{'━'*70}{RESET}")

    # Category breakdown
    cats: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("category") or "other"
        cats.setdefault(cat, {"pass": 0, "fail": 0})
        if r["passed"]:
            cats[cat]["pass"] += 1
        else:
            cats[cat]["fail"] += 1
    print(f"\n  {BOLD}Category Breakdown:{RESET}")
    for cat, counts in sorted(cats.items()):
        total_cat = counts["pass"] + counts["fail"]
        c = GREEN if counts["fail"] == 0 else RED
        print(f"    {c}{counts['pass']}/{total_cat}{RESET}  {cat}")

    # ── Per-Agent Summary Table ───────────────────────────────────────────────
    print(f"\n  {BOLD}Per-Agent Performance:{RESET}")
    print(f"  {'Agent':<16} {'Model':<28} {'Avg Tokens':<12} {'P50 (ms)':<10} {'P95 (ms)':<10}")
    print(f"  {'─'*16} {'─'*28} {'─'*12} {'─'*10} {'─'*10}")
    for row in agg["per_agent_summary"]:
        model_short = row["model"][:26] + ".." if len(row["model"]) > 28 else row["model"]
        print(f"  {row['agent']:<16} {model_short:<28} {row['avg_tokens']:<12} {row['p50_latency_ms']:<10} {row['p95_latency_ms']:<10}")

    # ── Accuracy Metrics ──────────────────────────────────────────────────────
    print(f"\n  {BOLD}Accuracy Metrics:{RESET}")
    for metric_name, value in agg["accuracy_metrics"].items():
        print(f"    {metric_name:<35} {value}")

    # ── End-to-End Latency ────────────────────────────────────────────────────
    e2e = agg["end_to_end_latency"]
    print(f"\n  {BOLD}End-to-End Latency:{RESET}")
    print(f"    P50={e2e['p50_ms']}ms  P95={e2e['p95_ms']}ms  Max={e2e['max_ms']}ms")

    # ── Token Efficiency ──────────────────────────────────────────────────────
    print(f"\n  {BOLD}Token Efficiency:{RESET}")
    print(f"    Total tokens consumed: {agg['total_tokens_consumed']}")
    print(f"    Avg tokens per request: {agg['avg_tokens_per_request']}")

    print(f"\n{BOLD}{'━'*70}{RESET}\n")

    # ── Save JSON results ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_results_{ts}.json"
    out_path.write_text(
        json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "passed": passed_cases,
            "failed": failed_cases,
            "pass_rate_pct": pct,
            "per_agent_summary": agg["per_agent_summary"],
            "accuracy_metrics": agg["accuracy_metrics"],
            "end_to_end_latency": agg["end_to_end_latency"],
            "token_efficiency": {
                "total_tokens_consumed": agg["total_tokens_consumed"],
                "avg_tokens_per_request": agg["avg_tokens_per_request"],
            },
            "category_breakdown": {
                cat: counts for cat, counts in sorted(cats.items())
            },
            "results": results,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Results saved → {out_path.relative_to(PROJECT_ROOT)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Banking AI evaluation runner — captures latency, tokens, accuracy, quality"
    )
    parser.add_argument(
        "--filter", metavar="PREFIX",
        help="Only run cases whose ID starts with this prefix (e.g. TC02)",
    )
    parser.add_argument(
        "--category", metavar="NAME",
        help="Only run cases in this category (e.g. pii_block, escalation)",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print per-check breakdown for every case, not just failures",
    )
    args = parser.parse_args()
    run_evaluation(
        filter_id=args.filter,
        filter_category=args.category,
        show_all=args.show_all,
    )
