# ROADMAP.md

Feuille de route AI Shoot — progression réelle par phase (sans dates figées).

---

## ✅ Phase 0 — Fondations

**Objectif :** passer du prototype cassé à un socle propre et exécutable.

- [x] Audit technique complet (imports cassés, scripts dupliqués, incompatibilités)
- [x] Nouvelle structure `src/` + packaging `uv` + `pyproject.toml`
- [x] Contrats d'interfaces Pydantic stables (`VideoInput`, `CoachingFeedback`, …)
- [x] CI GitHub Actions (ruff + pytest + mypy)
- [x] Gouvernance open-source (`CONTRIBUTING`, `LICENSE`, `SECURITY`, `CODE_OF_CONDUCT`)
- [x] `CLAUDE.md` — guide workflow PR + Gemini review pour assistants IA
- [x] GitHub Project Kanban + issues structurées par phase

---

## ✅ Phase 1 — Perception pipeline

**Objectif :** pipeline vision complet de la vidéo jusqu'aux métriques biomécaniques.

- [x] `PoseEstimator` — ViTPose/MediaPipe, 133 keypoints/frame
- [x] `VideoPipeline` — découpage + preprocessing vidéo
- [x] `BiomechanicsAnalyzer` — angles articulaires, arc de tir, release timing
- [x] `ShotPhaseDetector` — 4 phases : setup / jump / release / follow-through
- [x] Métriques basket (Q-angle, release window, wrist snap)
- [x] Modèles Pydantic : `PoseFrame`, `PerceptionOutput`, `ShotPhase`
- [x] 40+ tests unitaires perception + analyse

---

## ✅ Phase 2 — VLM Intelligence

**Objectif :** couche VLM qui comprend vidéo + métriques et génère un feedback coaching structuré.

- [x] `GeminiClient` — client Gemini Flash avec retry + gestion quota
- [x] `BasketballVLMAnalyzer` — analyse vidéo → `CoachingFeedback`
- [x] Prompt engineering few-shot avec exemples NBA
- [x] Framework d'évaluation qualité du feedback (pertinence, clarté, actionnabilité)
- [x] Interface abstraite `BaseVLMClient` (extensible Qwen2-VL, Claude Vision…)
- [x] Contrats Pydantic stables : `CoachingFeedback`, `DrillRecommendation`, `BiomechanicsReport`

---

## ✅ Phase 3 — Système agentique

**Objectif :** orchestration multi-agent avec mémoire de session joueur et personnalisation.

- [x] Étude comparative frameworks agentiques (LangGraph / Google ADK 2.0 / Anthropic / OpenAI)
  → **Décision : Google ADK 2.0** — seul framework avec streaming vidéo bidirectionnel production-ready
- [x] `PlayerSession` + `ShotRecord` — état mémoire Pydantic par joueur
- [x] `PlayerMemoryService` — persistance JSON, miroir ADK Memory Bank
- [x] 5 outils ADK-compatibles : perception, biomécanique, coaching, planificateur, mémoire
- [x] `ShotAnalysisPipeline` — pipeline synchrone standalone (CI/test)
- [x] `create_adk_pipeline()` — factory 4 `LlmAgent` Google ADK 2.0
- [x] 40 tests unitaires agents

---

## ✅ Phase 4 — API production-ready

**Objectif :** exposer le pipeline complet via une API async + streaming temps réel.

- [x] `POST /analyze` — upload vidéo + analyse async (background task)
- [x] `GET /session/{id}` — état et résultats de session
- [x] `WebSocket /analyze/stream` — streaming binaire sécurisé (pas de path traversal)
- [x] `GET /player/{id}/history` — historique coaching + progression
- [x] `GET /health` — liveness + readiness (ok / degraded / down)
- [x] CORS config-driven via `settings.cors_origins`
- [x] Upload chunké 1 MB (pas d'OOM sur gros fichiers)
- [x] 176 tests · ruff · mypy · Gemini review 5/5 findings résolus

---

## ✅ Phase 5a — Python 3.12 + Security CI *(livré 2026-05)*

**Objectif :** upgrade runtime + scanning sécurité en CI.

- [x] Upgrade Python 3.11 → 3.12 (~20% perf, prêt pour Python 3.13 free-threaded quand MediaPipe suit)
- [x] `pip-audit` — scan CVE deps sur chaque PR + cron hebdomadaire
- [x] `bandit` SAST — fail CI sur HIGH, report MEDIUM+ (B104 seul skip intentionnel)
- [x] CLAUDE.md : step 3b scan local obligatoire

---

## ✅ Phase 5b — Auth + Docker + GCP Deploy *(livré 2026-05)*

**Objectif :** API sécurisée + containerisée + déployée sur GCP Cloud Run.

- [x] Auth `X-API-Key` middleware (401/403, /health exempt, OPTIONS pass)
- [x] Docker multi-stage + docker-compose local (linux/amd64, uv pinned, exec signal handling)
- [x] Terraform IaC (Cloud Run v2, Artifact Registry, Secret Manager, IAM, Workload Identity)
- [x] GitHub Actions deploy workflow (Workload Identity Federation, Trivy container scan)
- [x] `scripts/smoke_test.sh` — validation locale + GCP Cloud Run (health, auth, CORS, 404)

---

## 📋 Phase 6 — Edge & Mobile *(vision moyen terme)*

**Objectif :** inférence on-device pour usage terrain sans cloud.

- [ ] Export ONNX + TensorRT des modèles de perception
- [ ] SDK mobile (iOS/Android) — capture + preprocessing local
- [ ] Mode hybride : perception on-device + VLM/agents cloud
- [ ] Latence cible : < 200ms perception, < 2s feedback complet

---

## 📋 Phase 7 — Multi-sport & Scale *(vision long terme)*

**Objectif :** étendre la plateforme au-delà du basketball.

- [ ] Abstraction sport-agnostique (keypoints génériques, phases configurables)
- [ ] Deuxième vertical sport (tennis ou football)
- [ ] Dashboard équipe + analytics agrégées
- [ ] API publique + marketplace de drills

---

## Livrables visibles sur GitHub

- `README.md` — entrée principale
- `ROADMAP.md` — cette feuille de route
- `BACKLOG.md` — priorisation opérationnelle Phase 4
- `CHANGELOG.md` — historique des phases livrées
- `CLAUDE.md` — guide workflow pour assistants IA
- `docs/agentic-frameworks-comparison.md` — étude ADK vs LangGraph vs autres
