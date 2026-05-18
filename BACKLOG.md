# BACKLOG.md

Backlog priorisé. Focus actuel : **Programme d'enrichissement v2.0** (Track 0-7).

> Phases 0–5b complètes. Voir [CHANGELOG.md](CHANGELOG.md) pour le détail.

---

## 🚀 Programme v2.0 — enrichissement *(focus actuel — démarré 2026-05-10)*

Post-audit v1.0.0 : 7 tracks parallèles pour passer la plateforme à la prochaine vague de robustesse, observabilité et sécurité. Détail complet du plan dans `~/.claude/plans/`.

### Track 0 — Stabilisation *(P0 — 2-3 jours)*

- [x] **T0-1** Trivy hardening — pin `@v0.36.0`, schedule cron, HIGH+CRITICAL gating, SARIF upload → **livré v1.0.1 (PR #32)**
- [x] **T0-2** Deploy preflight skip — diagnostic zero-secret + preflight job → **livré v1.0.2 (PR #33)**
- [x] **T0-5** Validation locale end-to-end + 6 bugs silencieux + cache VideoProcessor → **livré v1.0.3 (PR #35)**
- [x] **T0-6** Bootstrap GCP réel — projet `shoot-ai-poc` live à `https://shoot-ai-dev-chf52ondba-uc.a.run.app`, 25 ressources, 5 GH secrets, `deletion_protection` environment-conditional → **livré v1.0.4 (PR #36)**. Workflow Deploy CI désormais débloqué.
- [x] **T0-3/T0-7** Narrow 3 `except Exception:` + VLM retry/timeout (full jitter AWS pattern) + 7 nouveaux tests. Rebase clean post-T0-5 (T0-3 et T0-5 sur fichiers disjoints) → **livré v1.0.5 (PR #34)**. 190 tests total.
- [x] **T0-9** YOLO `yolo11n-pose.pt` pre-fetched + baked. Premier REJECT challenger validé par Gemini → **livré v1.0.6 (PR #37)**
- [x] **T0-11** Unblock CI/CD deploy round 1-3 — `iam.tf` cicd_tfstate binding (roles/storage.admin), `project.tf` cloudresourcemanager API. 2 fixes anticipés par Gemini review → **livré v1.0.7 (PR #38)**
- [x] **T0-13** Unblock CI/CD deploy round 5 — `lifecycle ignore_changes = [billing_account]` + `cloudbilling.googleapis.com` API. Cascade des 5 root causes documentée → **livré v1.0.8 (PR #39)**
- [x] **T0-15** Unblock bg pipeline live — `cpu_idle = false` débloque FastAPI BackgroundTasks (était throttlée → YOLO init ~3min observé), `YOLO_CONFIG_DIR=/tmp/Ultralytics` (skip warning ultralytics fallback), `apt-get upgrade` runtime fix 2 CVE HIGH fixed-upstream Debian (libcap2 CVE-2026-4878 + libsystemd0 CVE-2026-29111). Premier vrai déploiement applicatif fonctionnel → **livré v1.0.9 (PR #40)**
- [ ] **T0-4** Triage Dependabot alert #228 — CVE-2025-69872 DiskCache (MEDIUM, transitive, no upstream patch). Décider : ignore-rule documenté, downstream pinning, ou retrait dep si possible
- [ ] **T0-8 (follow-up T0-6)** `/unknown` route → 401 au lieu de 404. Le middleware auth tourne avant le routing FastAPI. Trade-off : anti-enumeration vs UX. Options : (a) garder + documenter ; (b) auth en `Depends()` per-route ; (c) middleware qui laisse passer si la route n'existe pas
- [ ] **T0-10 (optionnel follow-up T0-9)** Centraliser `YOLO_MODEL` via `ARG`/`ENV`/`os.getenv()` quand un besoin réel apparaît (Phase 6 ONNX/mobile ou A/B variants)
- [ ] **T0-12 (follow-up T0-11)** Formaliser le `roles/editor` du SA cicd (hot-fix appliqué manuellement) dans `iam.tf`. Idéalement substitué par 3 rôles plus étroits : `iam.serviceAccountAdmin` (lire les SAs), `serviceusage.serviceUsageAdmin` (lister APIs enabled), `resourcemanager.projectIamAdmin` (read project IAM bindings). Pour le POC le `roles/editor` est OK ; à resserrer avant prod

### Track 1 — Challenge qualif tech *(P1 — 3-5 jours)*

- [x] **T1-1** Critique structurée du doc — 10 findings dont 3 HIGH étayés par `grep` sur src/ (Live API pas codée, Agent Runtime pas déployée, Memory Bank en JSON local). Gemini ack avec 3 fixes cohérence → **livré v1.0.10 (PR #41)**
- [x] **T1-2** ADR-001 — 5 décisions formelles (D1 garder ADK, D2 déférer Agent Runtime, D3 AI Gateway, D4 Memory Bank en 2 paliers, D5 doc révisé) + révision `agentic-frameworks-comparison.md` avec sections "Stack actuelle vs cible" et "Risques résiduels". Génère T3-1 (AI Gateway impl) → **livré v1.0.10**

### Track 2 — Reliability + observability *(P1 — 5-7 jours)*

- [ ] **T2-1** OpenTelemetry traces + metrics → Cloud Trace + Cloud Monitoring
- [ ] **T2-2** Structured logging avec correlation ID middleware
- [ ] **T2-3** TaskStore persistant (Firestore) — remplace l'in-memory dict (perte d'état au restart Cloud Run)
- [ ] **T2-4** Idempotency keys + circuit breaker Gemini
- [ ] **T2-5** SLO/SLI dashboard + alerts

### Track 3 — Security hardening *(P1 — 4-5 jours)*

- [x] **T3-1** AI Gateway POC — `LiteLLMClient` in-process avec failover optionnel (LITELLM_FALLBACK_MODELS env var). Premier step ADR-001 D3. Posture POC zero-budget : Gemini-only par défaut. 13 tests, 5 Gemini findings ACCEPTED → **livré v1.0.11 (PR #43)**. T3-1b à venir : déploiement LiteLLM Proxy sidecar quand volume justifie
- [ ] **T3-1b** (nouveau) Cosign image signing + SLSA provenance + SBOM (syft) — initialement T3-1, renommé car T3-1 est devenu l'AI Gateway POC d'après ADR-001 D3
- [ ] **T3-2** Rate limiting (per-API-key + per-IP) + Cloud Armor optionnel
- [ ] **T3-3** Video input validation hardening (MIME magic byte, taille, durée, codec whitelist)
- [ ] **T3-4** API key hashing at rest + Secret Manager rotation policy

### Track 4 — Code quality + tests *(P2 — 4-6 jours)*

- [ ] **T4-1** VLM client tests (0% → 80%) — mock GoogleAI, retry, timeout
- [ ] **T4-2** Property-based tests biomécanique (`hypothesis`)
- [ ] **T4-3** Integration tests e2e avec fixtures vidéo DVC
- [ ] **T4-4** Eval framework golden dataset (regression catch en CI)

### Track 5 — Phase 6 Edge & Mobile *(P2 — 1-2 semaines)*

Recouvre le backlog P6-1 → P6-4 ci-dessous.

### Track 6 — Architecture evolution *(P3 — 1 semaine)*

- [ ] **T6-1** Pipeline async via Pub/Sub pour vidéos longues (>30s)
- [ ] **T6-2** Streaming inference live (Gemini Live API enfin câblé sur du vrai vidéo)
- [ ] **T6-3** DDD bounded contexts + ADR-002

### Track 7 — DX *(P3 — 2-3 jours)*

- [ ] **T7-1** uv cache dans CI (`actions/cache@v4`) → -30-40% temps CI
- [ ] **T7-2** Pre-commit hooks (ruff, ruff-format, bandit, gitleaks)
- [ ] **T7-3** Renovate config + fermeture Dependabot PRs #6 #7

---

## ✅ Phase 5b : Auth + Docker + GCP Deploy *(terminée)*

Tous les items P5-1 → P5-5 livrés. Voir [CHANGELOG.md](CHANGELOG.md) v0.7.0 → v1.0.0.
- [x] **P5-6** Trivy container scanning (livré v0.9.0, durci v1.0.1 par T0-1)

---

## 📋 Phase 6 : Edge & Mobile *(détaillée — exécution via Track 5 du programme v2.0)*

### [P6-1] Export ONNX des modèles de perception
**Priorité :** HIGH

- Export `PoseEstimator` + `ShotPhaseDetector` en ONNX
- Optimisation TensorRT pour GPU embarqué
- Validation : même sorties que le modèle PyTorch original

### [P6-2] SDK mobile — capture + preprocessing
**Priorité :** HIGH

- iOS (Swift) + Android (Kotlin) SDK
- Capture vidéo + preprocessing local (resize, normalize)
- Envoi frames + métriques bruts vers API Cloud Run

### [P6-3] Mode hybride on-device / cloud
**Priorité :** MEDIUM

- Perception on-device (ONNX) + VLM/agents cloud
- Cible latence : < 200 ms perception, < 2 s feedback complet
- Fallback cloud si device trop lent

### [P6-4] Tests de performance & benchmarks
**Priorité :** MEDIUM

- Benchmark latence iPhone 15 / Pixel 8
- Profiling mémoire modèle ONNX
- CI automated perf regression test


---

## ✅ Terminé (Phases 0–5a)

Toutes les tâches des phases 0, 1, 2, 3, 4 et 5a sont closes.
Voir les issues GitHub fermées et [CHANGELOG.md](CHANGELOG.md) pour le détail.
