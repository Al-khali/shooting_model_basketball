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

## 🚀 Programme d'enrichissement v2.0 *(en cours — démarré 2026-05-10)*

**Objectif :** post-audit v1.0.0, enrichir la plateforme sur 7 tracks parallèles vers v2.0 (3-6 semaines). Voir [`BACKLOG.md`](BACKLOG.md) pour le détail des tickets et le plan complet dans `~/.claude/plans/`.

| Track | Focus | Priorité | État |
|-------|-------|----------|------|
| Track 0 | Stabilisation (Trivy, deploy, error handling, VLM resilience) | P0 | 🟡 1/3 PRs |
| Track 1 | Challenge qualif tech + ADR | P1 | 📋 |
| Track 2 | Reliability + observability (OTel, Firestore, idempotency, SLO) | P1 | 📋 |
| Track 3 | Security hardening (Cosign/SLSA, rate limit, video validation, secret rotation) | P1 | 📋 |
| Track 4 | Code quality + tests (VLM, hypothesis biomech, e2e DVC, eval golden) | P2 | 📋 |
| Track 5 | Phase 6 Edge & Mobile (ONNX, TensorRT, iOS/Android SDK, hybrid) | P2 | 📋 |
| Track 6 | Architecture (Pub/Sub async, live streaming, DDD bounded contexts) | P3 | 📋 |
| Track 7 | DX (uv cache CI, pre-commit, Renovate) | P3 | 📋 |

- [x] **T0-1** Trivy hardening — pin `@v0.36.0`, schedule scan, HIGH+CRITICAL gating, SARIF upload (PR #32 / v1.0.1)
- [x] **T0-2** Deploy job preflight — graceful skip when GCP secrets missing, diagnostic of zero-secret repo state (PR #33 / v1.0.2)
- [x] **T0-5** End-to-end local validation — 6 silent bugs uncovered + `scripts/local_e2e.sh` automation + VideoProcessor caching (PR #35 / v1.0.3)
- [x] **T0-6** Bootstrap GCP réel — projet `shoot-ai-poc` live à `https://shoot-ai-dev-chf52ondba-uc.a.run.app`, 25 ressources Terraform créées, 5 GitHub secrets configurés, `deletion_protection` environment-conditional (PR #36 / v1.0.4)
- [x] **T0-3/T0-7** Narrow exceptions (3 sites) + VLM retry/timeout avec full jitter (AWS pattern) + 7 tests unitaires retry. Rebase clean post-T0-5 (PR #34 / v1.0.5). 190 tests total.
- [x] **T0-9** YOLO `yolo11n-pose.pt` pre-fetched + baked dans l'image Docker. +50 MB image, ~500ms-2s économisés au cold-start. Premier REJECT challenger validé par Gemini (PR #37 / v1.0.6)
- [x] **T0-11** Unblock CI/CD deploy round 1-3 — `storage.admin` binding sur tfstate bucket, `cloudresourcemanager.googleapis.com` API, project ref vs var. Cascade débloquée par 2 findings Gemini + 1 fix anticipé (PR #38 / v1.0.7)
- [x] **T0-13** Unblock CI/CD deploy round 5 — `ignore_changes = [billing_account]` (placeholder dans tfvars créait drift à chaque CI apply) + `cloudbilling.googleapis.com` API. Cascade complète des 5 root causes documentée (PR #39 / v1.0.8)
- [ ] **T0-4** Dependabot alert #228 — CVE-2025-69872 DiskCache MEDIUM, pas de patch upstream, ignore-rule à documenter
- [ ] **T0-8** (follow-up T0-6) `/unknown` route retourne 401 au lieu de 404
- [ ] **T0-10** (optionnel follow-up T0-9) Centraliser le choix YOLO via `ARG`/`ENV`/`os.getenv()`
- [ ] **T0-12** (follow-up T0-11) Formaliser le `roles/editor` du SA cicd dans `iam.tf` (idéalement narrow roles : iam.serviceAccountAdmin + serviceusage.serviceUsageAdmin + resourcemanager.projectIamAdmin)

---

## 📋 Phase 6 — Edge & Mobile *(vision moyen terme — recouvre Track 5 du programme v2.0)*

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
