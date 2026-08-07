# AgenticNAS - Multi-Agent Systems for Neural
Architecture Search

> Master's thesis project implementing an orchestrated multi-agent system for Neural Architecture Search (NAS).

This repository is the implementation artifact for a thesis investigating whether an orchestrated agentic mesh can improve the efficiency and quality of NAS compared with more conventional or single-agent approaches. The system combines large-language-model reasoning, structured architecture representation, zero-cost performance estimation, and real model training into a single experimental workflow.

---

## Research context

Neural Architecture Search automates the design of neural networks across the core dimensions of search space, search strategy, and performance estimation. This project extends that idea by embedding LLM-based agents into the NAS loop so that candidate generation, refinement, evaluation, and termination decisions are handled by an orchestrated mesh rather than a single monolithic search procedure.

The implementation is organized around the thesis' echelons of design:

| Echelon | Component | Purpose |
|---|---|---|
| E1 | Workflow integration | Cross-agent coordination and stateful process execution in n8n |
| E2 | nn-core-taxonomy | A structured taxonomy for representing neural architectures and converting them to PyTorch models |
| E5 | Orchestrator | High-level decision making and hypothesis-driven search control |
| E6 | Model generator mesh | Candidate architecture proposal and refinement |
| E7 | Model evaluator mesh | Performance estimation and evaluation feedback |

---

## What the system does

The repository implements a closed-loop NAS pipeline with the following stages:

1. An orchestrator agent receives a task and search constraints.
2. A model-generator mesh proposes or refines candidate architectures.
3. A performance predictor estimates candidate quality using zero-cost proxies.
4. A model trainer validates promising architectures with actual training and testing.
5. A plateau detector monitors performance trends and informs the next search step.
6. Agent state and run history are persisted through the runtime and memory services.

This makes the process iterative, hypothesis-driven, and suitable for experimentation with agentic search strategies.

---

## Main components

### Orchestration and runtime
- n8n workflows coordinate the agent mesh and store workflow state.
- agent-runtime provides the FastAPI-based runtime for agent execution and graph-based orchestration.
- agent-memory stores agent state and memory in Postgres.

### Architecture generation and evaluation
- model-generator-mesh contains the generator agents responsible for architecture synthesis and refinement.
- model-evaluator-mesh contains evaluator agents that process model feedback and propose next actions.
- performance-predictor exposes zero-cost estimation endpoints for fast screening of architectures.
- model-trainer performs full training and evaluation for promising candidates.
- plateau-detector tracks performance gains over time to detect stagnation.

### Representation layer
- nn-core-taxonomy contains the neural architecture taxonomy used to serialize, validate, and convert models into PyTorch.

---

## Repository structure

```text
AutoML/
├── agents/
│   ├── model-generator-mesh/
│   ├── model-evaluator-mesh/
│   ├── orchestrator/
│   └── tools/
├── apis/
│   ├── agent-memory/
│   ├── agent-runtime/
│   ├── model-faciliator/
│   ├── model-trainer/
│   ├── performance-predictor/
│   └── plateau-detector/
├── datasets/
├── logs/
├── n8n/
├── nn-core-taxonomy/
├── notebooks/
├── volumes/
├── docker-compose.yml
└── README.md
```

---

## Architecture stack

| Component | Role |
|---|---|
| Docker Compose | Starts the full multi-service environment |
| PostgreSQL | Stores workflow, runtime, and model-related metadata |
| FastAPI services | Expose training, prediction, runtime, and memory APIs |
| Azure OpenAI | Supplies language-model capabilities to the agent mesh |
| PyTorch + nn_core | Builds and evaluates architecture specifications |
| n8n | Hosts the orchestration workflow layer |

---

## Getting started

### 1. Prepare environment variables

Copy [.env.example](.env.example) to [.env](.env) and adjust the values for your machine.

The root [.env](.env) file is ignored by git.

### 2. Start the stack

From the repository root, run:

```bash
docker compose up --build -d
```

This starts the API services, Postgres, and n8n workflow environment.

### 3. Access the system

- n8n UI: http://localhost:5678
- API services are exposed on the configured Docker ports, including the runtime, trainer, predictor, and detector services.

### 4. Configure secrets

The compose stack expects credentials and secrets for:
- Azure OpenAI endpoint and API key
- Postgres passwords for the runtime, memory, and model databases

The example secrets are located under the apis/secrets directory.

---

## Notes for running experiments

- GPU support is recommended for the training and performance-prediction services.
- The datasets directory is mounted into the prediction and training services.
- Logs are written under the logs directory for each service.
- The serve directory contains optional legacy tooling and is not required for the main orchestrated workflow.

---

## Project intent

The goal of this project is not only to build a functioning NAS agentic system, but to evaluate how structured agent collaboration can improve search behavior, reduce wasted computation, and produce stronger architectures than simpler search loops.

This repository therefore serves both as a practical implementation and as a reproducible experimental platform for the thesis work.

---

## Citation

If you use this repository in research or writing, please cite the accompanying thesis work and reference this codebase as the implementation artifact for the agentic NAS system.
