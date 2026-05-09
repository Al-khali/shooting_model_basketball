<div align="center">

# 🏀 AI Shoot

**Real-time basketball intelligence powered by computer vision, biomechanics, and agentic AI**

[![CI](https://github.com/Al-khali/shooting_model_basketball/actions/workflows/ci.yml/badge.svg)](https://github.com/Al-khali/shooting_model_basketball/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)

[Roadmap](ROADMAP.md) · [Backlog](BACKLOG.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

</div>

---

## What is AI Shoot?

AI Shoot is a **real-time sports intelligence platform** — starting with basketball as the first vertical.

It doesn't just give you a score. It **sees** your movement, **understands** your biomechanics, **reasons** like a coach, and **explains** what to fix in plain language:

> *"Your right elbow is at 52° — 17° above the optimal range. This causes leftward deviation on mid-range shots. Fix: wall drill, 10 min, elbow aligned with wrist."*

### The system pipeline

```
📹 Video Input
    ↓
👁️  PERCEPTION      → ViTPose/YOLOv11 extracts 133 keypoints per frame
    ↓
📐 BIOMECHANICS     → elbow angles, knee flex, release timing, ball arc
    ↓
🧠 VLM INTELLIGENCE → Gemini Flash / Qwen2-VL sees video + metrics together
    ↓
🤖 AGENTIC COACHING → Google ADK 2.0 agents reason, prioritize, personalize
    ↓
💬 REAL FEEDBACK    → structured analysis + actionable drill recommendation
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Pose estimation | ViTPose + MMPose | SOTA accuracy, 133 keypoints, extensible |
| Object detection | YOLOv11 | Real-time speed + precision |
| Temporal understanding | VideoMAE v2 | Understands motion sequences |
| VLM Intelligence | Gemini Flash / Qwen2-VL | Video understanding + natural language |
| Agent orchestration | Google ADK 2.0 | Production bidirectional video streaming, built-in Session Service |
| API | FastAPI async + WebSocket | Real-time, modern |
| Inference | ONNX Runtime + TensorRT | Edge + cloud deployment |
| Experiment tracking | Weights & Biases | Reproducibility |
| Data versioning | DVC | Versioned datasets |
| Python packaging | uv + pyproject.toml | 2025 standard |

---

## Project Structure

```
shoot-ai/
├── src/
│   ├── perception/     # Computer vision: pose estimation, tracking, video pipeline
│   ├── analysis/       # Biomechanics: angles, timing, shot phase detection
│   ├── agents/         # Agentic system: perceiver, analyzer, coach, planner
│   ├── vlm/            # VLM integration: Gemini, Qwen2-VL, prompt templates
│   ├── api/            # FastAPI: routes + Pydantic schemas
│   ├── data/           # Data loaders, preprocessors, dataset schemas
│   └── core/           # Config, logging, exceptions
├── tests/              # Unit + integration tests
├── scripts/            # Training, evaluation, benchmarking
├── docs/               # Documentation
├── legacy/             # Original prototype (archived, not active)
└── pyproject.toml      # Packaging + all dependencies
```

---

## Quickstart

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
# Clone
git clone https://github.com/Al-khali/shooting_model_basketball.git
cd shooting_model_basketball

# Create environment and install all dependencies
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync --extra dev

# Copy env template and add your API keys
cp .env.example .env

# Run tests
uv run pytest
```

---

## Development Status

> 🚧 **Active development — Phases 0–4 complete, Phase 5 (Auth + Docker + Edge) next**

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 — Foundations | ✅ Done | Structure, packaging, CI, contracts, governance |
| Phase 1 — Perception | ✅ Done | ViTPose integration, biomechanics pipeline, shot phase detection |
| Phase 2 — VLM Intelligence | ✅ Done | Gemini Flash, prompt engineering, evaluation framework |
| Phase 3 — Agentic System | ✅ Done | Google ADK 2.0 agents, player memory, coaching pipeline |
| Phase 4 — API + Real-time | ✅ Done | FastAPI async, WebSocket streaming, player history, health endpoint |
| Phase 5 — Security + Deploy | 🔄 In Progress | Python 3.12 ✅, Security CI (pip-audit + bandit) ✅, Auth API key, Docker, Terraform/GCP |

See [ROADMAP.md](ROADMAP.md) for the full plan and [BACKLOG.md](BACKLOG.md) for open tasks.

---

## Contributing

This project is heading toward open source. Contributions are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev workflow, commit conventions, and branch strategy.

---

## License

[MIT](LICENSE) — © 2024 Al-khali
