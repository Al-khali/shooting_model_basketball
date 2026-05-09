# Changelog

Toutes les modifications notables sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
Ce projet suit le [Semantic Versioning](https://semver.org/lang/fr/).

## [0.8.0] — Phase 5b: Terraform IaC + GCP infrastructure (2026-05)

### Added
- `infra/terraform/` — complete GCP infrastructure as Terraform:
  - `main.tf` — GCS remote backend, google provider, local variables
  - `project.tf` — GCP project + 7 API enablements (run, artifactregistry, secretmanager, iam, cloudbuild, sts, iamcredentials)
  - `registry.tf` — Artifact Registry docker repository
  - `secrets.tf` — Secret Manager: `gemini-api-key` + `api-keys`
  - `iam.tf` — Cloud Run SA (secretAccessor) + CI/CD SA (run.developer, artifactregistry.writer) + Workload Identity Federation pool+provider
  - `cloud_run.tf` — Cloud Run v2 service (min=0/max=2, 1CPU/2Gi, liveness + startup probes, secrets via env refs)
  - `variables.tf` — all variables including `image_tag` (SHA-based), `api_keys` as `list(string)`, `github_repository` for WIF
  - `outputs.tf` — service_url, registry_url, WIF provider, SA emails
  - `environments/dev/terraform.tfvars` — dev config (no secrets)
- `infra/scripts/bootstrap.sh` — one-time: create GCP project + billing + GCS tfstate bucket
- `infra/scripts/deploy.sh` — local alternative to CI: build → push → terraform apply
- `.gitignore` — Terraform state files excluded

### Security
- Workload Identity Federation: GitHub Actions → GCP with zero static JSON keys
- WIF locked to `var.github_repository` (prevents fork abuse)
- IAM principle of least privilege: Cloud Run SA has only secretAccessor
- Secrets passed at apply time via `TF_VAR_*` env vars, never in tfvars files
- Gemini review PR #29 — 6 findings, all accepted (import workflow, image_tag, list(string), jsonencode, depends_on IAM, parameterised WIF)

## [0.7.0] — Phase 5b: Docker containerisation (2026-05)

### Added
- `Dockerfile` — multi-stage build `linux/amd64` (Cloud Run target)
  - Stage 1 builder: uv from official image (`ghcr.io/astral-sh/uv:0.11.12`), two-layer caching (deps → project `--no-editable`)
  - Stage 2 runtime: `python:3.12-slim`, non-root `appuser`, `libgl1`/`libglib2.0-0` for OpenCV
  - `CMD` uses `exec python -m uvicorn` for direct SIGTERM reception (graceful Cloud Run shutdown)
- `docker-compose.yml` — local dev orchestration (`platform: linux/amd64`, data volume, healthcheck)
- `.dockerignore` — excludes `data/`, `models/`, `tests/`, dev tooling
- `.env.example` — comprehensive template (all settings documented, consolidated from `.env.local.example`)

### Changed
- `.gitignore` — `.env.local.example` removed; `.env.example` is the canonical template

### Notes
- MediaPipe has no `linux/arm64` wheel — `--platform linux/amd64` required on Apple Silicon (QEMU)
- uv binary copied from official image to avoid install-script QEMU issues
- Gemini review PR #28 — 3 findings, all accepted: env consistency (F1 HIGH), uv pinning (F2 MEDIUM), exec signal handling (F3 MEDIUM)

## [0.6.0] — Phase 5b: Auth middleware (2026-05)

### Added
- `src/api/middleware/auth.py` — pure-ASGI `APIKeyMiddleware`
  - `X-API-Key` header required on all endpoints when `settings.api_keys` is non-empty
  - `GET /health` + `GET /health/` always exempt (Cloud Run liveness probe)
  - `OPTIONS` requests always pass (CORS preflight)
  - WebSocket: close with code 4403 on auth failure
  - Disabled when `api_keys = []` (default — dev / CI mode)
- `src/core/config.py` — `api_keys: list[str] = []` (env var `API_KEYS` as JSON array)
- 6 new integration tests in `TestAuth` (182 tests total)

### Changed
- `src/api/main.py` — `APIKeyMiddleware` registered (last added = outermost in Starlette)
- `src/api/schemas/responses.py` — version `0.4.0` → `0.5.0`

### Security
- Gemini review PR #27 — 5 findings all accepted:
  - F1 HIGH: OPTIONS requests exempted from auth (CORS preflight)
  - F2 HIGH: version updated in schemas + test aligned
  - F3 MEDIUM: middleware order comment corrected (last added = outermost)
  - F4 MEDIUM: `/health/` trailing slash added to exempt paths
  - F5 MEDIUM: multi-key test uses protected endpoint

---



### Added
- `.github/workflows/security.yml` — workflow de sécurité dédié
  - `pip-audit` : scan CVE de toutes les dépendances Python, déclenché sur chaque push/PR + cron hebdomadaire (lundi 08:00 UTC)
  - `bandit` SAST : scan statique de `src/` — rapporte MEDIUM+, **fail CI sur HIGH**
- `[tool.bandit]` config dans `pyproject.toml` : `exclude_dirs = ["tests", "legacy"]`, seul B104 supprimé (bind `0.0.0.0` intentionnel pour Cloud Run)
- `pip-audit>=2.7.0` + `bandit[toml]>=1.7.0` + `anyio[trio]>=4.0.0` dans dev deps
- CLAUDE.md step 3b : scan sécurité local obligatoire avant toute PR touchant `src/` ou les deps

### Changed
- Upgrade Python 3.11 → **3.12** — ~20% perf gain, compatibilité complète (MediaPipe bloque 3.13)
- `pyproject.toml` : `requires-python = ">=3.12"`, `ruff target-version = py312`, `mypy python_version = 3.12`
- `.python-version` : `3.12` (nouveau fichier pour `uv`)
- `.github/workflows/ci.yml` : tous les jobs sur Python 3.12
- README.md : badge Python 3.12+, quickstart mis à jour

### Security
- Gemini review PR #26 — 3 findings :
  - **ACCEPTED** : sévérité locale alignée sur CI (`--severity-level medium`)
  - **ACCEPTED** : B101 (assert) et B603 (subprocess) retirés des skips globaux — utiliser `# nosec` au call site
  - **REJECTED** : suppression du flag `-c pyproject.toml` — prouvé nécessaire (sans lui, bandit ignore `[tool.bandit]` et flag B104)

---

## [0.4.0] — Phase 4: API production-ready (2026-05)

### Added
- `src/api/main.py` — application FastAPI avec lifespan, CORS config-driven
- `src/api/store.py` — `TaskStore` singleton in-memory (asyncio.Lock)
- `src/api/routes/analyze.py`
  - `POST /analyze` — upload multipart vidéo → 202 + task_id (background task)
  - `GET /session/{task_id}` — polling lifecycle : processing / done / error
  - `WS /analyze/stream` — streaming binaire sécurisé : client envoie les bytes vidéo, le serveur contrôle le chemin (pas de path traversal)
- `src/api/routes/health.py` — `GET /health` avec statut 3-niveaux : ok / degraded / down
- `src/api/routes/players.py` — `GET /player/{id}/history` : historique coaching + issues récurrentes
- `src/agents/orchestrator.py` — paramètre `progress_callback` sur `analyze()` pour streaming WS
- 23 nouveaux tests d'intégration (176 tests au total)

### Security
- WS `video_path` path traversal corrigé : le client envoie les bytes bruts, jamais un chemin serveur
- CORS : `allow_origins` via `settings.cors_origins` (env `CORS_ORIGINS`), `allow_credentials=False`

### Fixed
- Bug health : `overall` ignorait le statut `"down"` — corrigé avec priorité `down > degraded > ok`
- Upload OOM : lecture par chunks 1 MB au lieu de `await file.read()` complet en mémoire
- `contextlib` importé au niveau module (était différé inutilement dans `emit()`)

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
