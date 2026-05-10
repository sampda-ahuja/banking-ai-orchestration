# Banking AI Orchestration

> A production-style multi-agent AI orchestration system for banking loan inquiry workflows, built using [LangGraph](https://www.langchain.com/langgraph?utm_source=chatgpt.com), [LangChain](https://www.langchain.com/?utm_source=chatgpt.com), [Groq](https://groq.com/?utm_source=chatgpt.com), and [FastAPI](https://fastapi.tiangolo.com/?utm_source=chatgpt.com).

---

# Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Workflow Execution](#workflow-execution)
* [Core Design Principles](#core-design-principles)
* [Project Structure](#project-structure)
* [Environment Setup](#environment-setup)
* [Running the Application](#running-the-application)
* [API Usage](#api-usage)
* [Streaming Architecture](#streaming-architecture)
* [Testing](#testing)
* [Evaluation Framework](#evaluation-framework)
* [Model Selection & Tradeoffs](#model-selection--tradeoffs)
* [Deterministic Policy Enforcement](#deterministic-policy-enforcement)
* [Security Considerations](#security-considerations)
* [License](#license)

---

# Overview

This project implements a production-oriented **multi-agent orchestration system** for handling banking loan inquiries safely and reliably.

The workflow combines:

* **LLM-powered semantic understanding**
* **Deterministic policy enforcement**
* **Layered compliance validation**
* **Structured outputs**
* **Streaming observability**

The system is designed around a core principle:

> **LLMs should assist with language understanding, but deterministic systems should enforce policy and compliance.**

---

# Architecture

The orchestration layer is implemented using a **LangGraph state machine** with conditional routing and enterprise-style guardrails.

```text
START
  ↓
guardrail_node
  ↓ [conditional routing]
  ├── PII detected        → safe_terminate_node → END
  ├── Non-banking topic   → safe_terminate_node → END
  ├── Emotional distress  → escalation_node     → END
  └── All clear ↓

inquiry_parser_node
  ↓

risk_policy_node
  ↓

compliance_node
  ↓ [conditional routing]
  ├── Compliance pass     → END
  └── Compliance fail     → fallback_node → END
```

---

# Workflow Execution

## 1. Guardrail Node

**Purpose:** Early-stage safety and relevance filtering.

### Responsibilities

* PII detection using regex rules
* Banking relevance classification
* Emotional escalation detection
* Safe workflow termination when required

### Implementation Style

Hybrid architecture:

* Deterministic regex rules for sensitive data
* LLM-based semantic classification for nuanced language understanding

---

## 2. Inquiry Parser Node

**Purpose:** Structured extraction and compliant response generation.

### Responsibilities

* Loan intent classification
* Entity extraction
* Financial detail parsing
* Structured response generation

### Example Extracted Fields

* Loan type
* Income
* Employment status
* Loan amount requested
* Risk-related attributes

---

## 3. Risk Policy Node

**Purpose:** Deterministic underwriting-style risk estimation.

### Responsibilities

* Debt-to-income analysis
* Employment stability scoring
* Risk categorization

### Important Constraint

This node does **not** use any LLMs.

All policy decisions are fully deterministic and reproducible.

---

## 4. Compliance Node

**Purpose:** Final compliance and safety validation.

### Responsibilities

* ECOA checks
* UDAAP validation
* Guarantee claim detection
* Disclaimer enforcement
* Final response validation

### Failure Handling

If compliance fails:

```text
compliance_node → fallback_node → END
```

This guarantees that unsafe or non-compliant outputs never reach end users.

---

# Core Design Principles

| Principle                     | Implementation                                                  |
| ----------------------------- | --------------------------------------------------------------- |
| **LLMs for semantics only**   | LLMs classify intent, detect escalation, and generate responses |
| **Deterministic enforcement** | PII, compliance, and risk rules are regex/logic-based           |
| **Structured outputs**        | All agents return typed Pydantic v2 schemas                     |
| **Layered safety**            | Guardrails → Processing → Compliance                            |
| **Enterprise observability**  | Streaming events expose workflow execution                      |
| **Fail-safe architecture**    | Compliance fallback prevents unsafe responses                   |

---

# Project Structure

```text
banking-ai-orchestration/
├── app/
│   ├── agents/
│   │   ├── guardrails.py
│   │   ├── inquiry_parser.py
│   │   └── compliance.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   └── workflow.py
│   │
│   ├── policies/
│   │   ├── pii_rules.py
│   │   ├── compliance_rules.py
│   │   └── risk_engine.py
│   │
│   ├── prompts/
│   │   ├── guardrails_prompt.txt
│   │   ├── inquiry_prompt.txt
│   │   └── compliance_prompt.txt
│   │
│   ├── schemas/
│   │   └── models.py
│   │
│   ├── streaming/
│   │   └── event_stream.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   └── logger.py
│   │
│   └── main.py
│
├── evals/
│   ├── guardrails.csv
│   ├── parser.csv
│   ├── compliance.csv
│   └── promptfooconfig.yaml
│
├── tests/
│   ├── test_guardrails.py
│   ├── test_parser.py
│   └── test_compliance.py
│
├── .env.example
├── requirements.txt
└── README.md
```

---

# Environment Setup

## 1. Create a Virtual Environment

### Using Python venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### Using uv

```bash
uv venv
source .venv/bin/activate
```

---

## 2. Install Dependencies

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv

```bash
uv pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free API key from:

[Groq Console](https://console.groq.com/keys?utm_source=chatgpt.com)

---

# Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --port 8000
```

Application endpoints:

| Endpoint        | Description                      |
| --------------- | -------------------------------- |
| `/docs`         | Swagger API documentation        |
| `/health`       | Health check                     |
| `/loan-inquiry` | Main streaming workflow endpoint |

Base URL:

```text
http://localhost:8000
```

---

# API Usage

## Example Request

```bash
curl -N -X POST http://localhost:8000/loan-inquiry \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need a mortgage loan for $350,000. I earn $90,000 annually and work full-time."
  }'
```

---

## Example SSE Response

```text
event: agent_event
data: {
  "agent": "guardrails",
  "status": "completed",
  "output": {
    "is_banking_related": true,
    "no_pii": true,
    "needs_escalation": false
  }
}

event: agent_event
data: {
  "agent": "inquiry_parser",
  "status": "completed",
  "output": {
    "intent": "Mortgage",
    "loan_amount_requested": 350000,
    "employment_status": "full-time"
  }
}

event: agent_event
data: {
  "agent": "risk_policy",
  "status": "completed",
  "output": {
    "risk_score_estimate": "LOW"
  }
}

event: agent_event
data: {
  "agent": "compliance",
  "status": "completed",
  "output": {
    "compliance_pass": true,
    "violations": []
  }
}

event: done
data: {
  "agent": "workflow",
  "status": "done",
  "output": {
    "final_response": "...",
    "risk_score": "LOW",
    "compliance_pass": true
  }
}
```

---

# Streaming Architecture

The system streams **agent-level execution events**, not token-level outputs.

## Streaming Flow

1. `POST /loan-inquiry` triggers the SSE pipeline
2. Workflow execution runs in a background thread using `asyncio.to_thread`
3. Each node appends structured events into shared graph state
4. Events are replayed sequentially as SSE messages
5. Final workflow output is emitted through a `done` event

---

## Event Types

| Event         | Description                     |
| ------------- | ------------------------------- |
| `agent_event` | Node execution completed        |
| `done`        | Workflow successfully completed |
| `error`       | Unhandled exception occurred    |

---

## Why Agent-Level Streaming?

Token streaming was intentionally avoided because:

* Intermediate reasoning should remain hidden
* Compliance boundaries become harder to enforce
* Token ordering across agents becomes unreliable
* Agent-level observability is sufficient for monitoring

---

# Testing

All unit tests are deterministic and do not require network access or API keys.

Run all tests:

```bash
pytest tests/ -v
```

---

## Test Coverage

| File                 | Coverage                                  |
| -------------------- | ----------------------------------------- |
| `test_guardrails.py` | PII detection + schema validation         |
| `test_parser.py`     | Structured extraction + disclaimer checks |
| `test_compliance.py` | Compliance rules + fallback behavior      |

---

# Evaluation Framework

The project uses [Promptfoo](https://www.promptfoo.dev/?utm_source=chatgpt.com) for automated evaluation.

---

## Install Promptfoo

```bash
npm install -g promptfoo
```

---

## Start the API

```bash
uvicorn app.main:app --port 8000
```

---

## Run Evaluations

```bash
cd evals
npx promptfoo eval --config promptfooconfig.yaml
```

---

## View Evaluation Results

```bash
npx promptfoo view
```

---

## Evaluation Dataset Breakdown

| Category                        | Count |
| ------------------------------- | ----- |
| Normal loan inquiries           | 10    |
| Adversarial / jailbreak prompts | 5     |
| Compliance-focused cases        | 5     |
| PII / sensitive data            | 5     |
| Emotional escalation scenarios  | 5     |

---

## Metrics Tracked

* Disclaimer presence rate
* Compliance pass rate
* PII block rate
* Adversarial resilience
* Latency threshold compliance

---

# Model Selection & Tradeoffs

| Agent          | Model                     | Rationale                                   |
| -------------- | ------------------------- | ------------------------------------------- |
| Guardrails     | `llama-3.1-8b-instant`    | Fast semantic classification                |
| Inquiry Parser | `llama-3.3-70b-versatile` | Better extraction and instruction-following |
| Compliance     | `llama-3.1-8b-instant`    | Lightweight semantic compliance review      |

---

## Latency vs Accuracy Tradeoff

### Why Use a 70B Model for Parsing?

The parser handles:

* Intent classification
* Entity extraction
* Financial reasoning
* Response generation

The 70B model improves:

* Extraction reliability
* Structured output accuracy
* Nuanced language understanding

At the cost of approximately:

```text
+2–4 seconds latency
```

This tradeoff is acceptable in a banking workflow where correctness matters more than raw speed.

---

## Why Use 8B Models for Guardrails?

Guardrails and compliance tasks are primarily:

* Binary classification
* Lightweight semantic checks
* Pattern validation

Smaller Groq-hosted models provide:

* Extremely low latency
* Lower operational cost
* Sufficient accuracy for constrained tasks

---

# Deterministic Policy Enforcement

A core architectural principle of this project:

> **LLMs are never trusted to enforce policy.**

---

## Deterministic Components

| Component                 | Method                           |
| ------------------------- | -------------------------------- |
| PII detection             | Regex patterns                   |
| Risk scoring              | Deterministic logic              |
| Guarantee claim detection | Regex rules                      |
| ECOA checks               | Protected class pattern matching |
| UDAAP checks              | Deterministic pattern rules      |
| Disclaimer enforcement    | String validation                |

---

## Responsibilities Assigned to LLMs

LLMs are only used for:

* Banking relevance classification
* Emotional escalation detection
* Intent classification
* NLP extraction
* Natural language response generation
* Nuanced semantic review

