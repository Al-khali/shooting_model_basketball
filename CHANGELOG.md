# Changelog

Toutes les modifications notables sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
Ce projet suit le [Semantic Versioning](https://semver.org/lang/fr/).

## [Unreleased] — Phase 4: API production-ready

### Planned
- `POST /analyze` — analyse vidéo asynchrone (upload + WebSocket streaming)
- `GET /session/{id}` — état et résultats de session
- `WebSocket /analyze/stream` — streaming temps réel du pipeline
- `GET /player/{id}/history` — historique coaching joueur
- `GET /health` — endpoint de santé

---

## [0.3.0] — Phase 3: Système agentique (2026-05)

### Added
- `src/agents/state.py` — modèles `PlayerSession` + `ShotRecord` (Pydantic v2)
  - `recurring_issues` : issues vues ≥2× triées par fréquence
  - `recent_drills` : 5 derniers drills uniques pour personnalisation
- `src/agents/memory.py` — `PlayerMemoryService` : persistance JSON par joueur
  - Miroir du ADK Memory Bank (swap 1 fichier en production)
  - `build_context()`, `record_feedback()` avec déduplication via `set()`
- `src/agents/tools/` — 5 fonctions-outils ADK-compatibles
  - `extract_shot_frames` (perception), `compute_biomechanics` (analyse)
  - `generate_coaching_feedback` (VLM), `build_training_plan` (planificateur)
  - `load_player_history` / `save_coaching_result` (mémoire)
- `src/agents/orchestrator.py` — double mode d'exécution :
  - `ShotAnalysisPipeline` : pipeline synchrone standalone (CI/test friendly)
  - `create_adk_pipeline()` : factory 4 `LlmAgent` Google ADK 2.0
- `tests/unit/test_agents.py` — 40 tests (PlayerSession, Memory, 5 tools, Pipeline)
- `docs/agentic-frameworks-comparison.md` — étude comparative complète
  (LangGraph vs Google ADK 2.0 vs Anthropic vs OpenAI vs CrewAI vs AutoGen)

### Architecture decision
- **Google ADK 2.0** choisi comme framework agentique principal
  (seul framework avec streaming vidéo bidirectionnel production-ready via Gemini Live API)
- **Deferred imports** systématiques : ultralytics, mediapipe, google-adk importés
  dans le corps des fonctions — `ImportError` → stub data, pas d'exception levée

---

## [0.2.0] — Phase 2: VLM Intelligence (2026-05)

### Added
- `src/vlm/gemini_client.py` — client Gemini Flash avec retry + gestion quota
- `src/vlm/basketball_analyzer.py` — `BasketballVLMAnalyzer` : analyse vidéo → `CoachingFeedback`
- `src/vlm/prompts/basketball.py` — templates de prompts few-shot avec exemples NBA
- `src/vlm/evaluator.py` — framework d'évaluation qualité du feedback (pertinence, clarté, actionnabilité)
- `src/vlm/base.py` — interface abstraite `BaseVLMClient`
- Contrats Pydantic stables : `CoachingFeedback`, `DrillRecommendation`, `BiomechanicsReport`

### Changed
- `src/api/schemas/domain.py` — enrichissement des schémas avec champs VLM

---

## [0.1.0] — Phase 1: Perception pipeline (2026-05)

### Added
- `src/perception/pose_estimator.py` — `PoseEstimator` : ViTPose/MediaPipe, 133 keypoints/frame
- `src/perception/video_pipeline.py` — `VideoPipeline` : découpage + preprocessing vidéo
- `src/analysis/biomechanics.py` — `BiomechanicsAnalyzer` : angles articulaires, arc, timing
- `src/analysis/shot_detector.py` — `ShotPhaseDetector` : détection 4 phases (setup/jump/release/follow)
- `src/analysis/metrics.py` — métriques basketball (Q-angle, release window, wrist snap)
- Modèles Pydantic : `PoseFrame`, `PerceptionOutput`, `ShotPhase`
- 40+ tests unitaires pour perception et analyse

---

## [0.0.2] — Phase 0: Fondations (2026-05)

### Added
- Nouvelle structure de projet (`src/perception`, `src/analysis`, `src/agents`, `src/vlm`, `src/api`)
- Migration vers `uv` + `pyproject.toml` (Python 3.11+)
- Contrats d'interfaces Pydantic (`VideoInput`, `CoachingFeedback`, `BiomechanicsReport`, etc.)
- Configuration centralisée via `pydantic-settings` + `SecretStr` pour toutes les clés API
- Logging structuré avec `structlog`
- CI GitHub Actions (lint ruff + pytest + mypy)
- Fichiers de gouvernance open-source (`CONTRIBUTING`, `CODE_OF_CONDUCT`, `LICENSE`, `SECURITY`)
- Stratégie de commits Conventional Commits + `.pre-commit-config.yaml`
- `CLAUDE.md` — guide workflow pour assistants IA (PR, Gemini review, challenger findings)
- GitHub Project Kanban + issues structurées par phase

### Changed
- `requirements.txt` remplacé par `pyproject.toml`
- Architecture entièrement repensée (vision plateforme d'intelligence basketball)

### Removed
- Code prototype non fonctionnel (`app.py` script dupliqué, imports cassés)
- Stack périmée (tensorflow 2.11, sklearn 0.24, opencv 4.5)

---

## [0.0.1] — 2024 (prototype initial)

### Added
- Prototype initial : CNN custom + RandomForest pour analyse tir basket
- API FastAPI basique (non fonctionnelle)
- Utilitaires OpenCV pour chargement vidéo
