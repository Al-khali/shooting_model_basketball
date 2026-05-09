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

## 🔄 Phase 4 — API production-ready *(en cours)*

**Objectif :** exposer le pipeline complet via une API async + streaming temps réel.

- [ ] `POST /analyze` — upload vidéo + analyse async (background task)
- [ ] `GET /session/{id}` — état et résultats de session
- [ ] `WebSocket /analyze/stream` — streaming frame-by-frame du pipeline
- [ ] `GET /player/{id}/history` — historique coaching + progression
- [ ] `GET /health` — liveness + readiness
- [ ] Auth JWT ou API key
- [ ] Docker + docker-compose pour déploiement local

---

## 📋 Phase 5 — Edge & Mobile *(planifiée)*

**Objectif :** inférence on-device pour usage terrain sans cloud.

- [ ] Export ONNX + TensorRT des modèles de perception
- [ ] SDK mobile (iOS/Android) — capture + preprocessing local
- [ ] Mode hybride : perception on-device + VLM/agents cloud
- [ ] Latence cible : < 200ms perception, < 2s feedback complet

---

## 📋 Phase 6 — Multi-sport & Scale *(vision long terme)*

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
