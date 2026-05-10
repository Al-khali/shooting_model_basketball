# BACKLOG.md

Backlog priorisé. Focus actuel : **Programme d'enrichissement v2.0** (Track 0-7).

> Phases 0–5b complètes. Voir [CHANGELOG.md](CHANGELOG.md) pour le détail.

---

## 🚀 Programme v2.0 — enrichissement *(focus actuel — démarré 2026-05-10)*

Post-audit v1.0.0 : 7 tracks parallèles pour passer la plateforme à la prochaine vague de robustesse, observabilité et sécurité. Détail complet du plan dans `~/.claude/plans/`.

### Track 0 — Stabilisation *(P0 — 2-3 jours)*

- [x] **T0-1** Trivy hardening — pin `@v0.36.0`, schedule cron, HIGH+CRITICAL gating, SARIF upload → **livré v1.0.1 (PR #32)**
- [x] **T0-2** Deploy preflight skip — diagnosed: zero secrets configured (`gh api .../actions/secrets` → 0). Fix: preflight job + `if: needs.preflight.outputs.ready == 'true'` on build/deploy/smoke → **livré v1.0.2 (PR #33)**. ⚠️ Bootstrap manuel encore requis (operator runs `infra/scripts/bootstrap.sh` + `terraform apply` + `gh secret set`)
- [ ] **T0-3** Resserrer 3 `except Exception:` (analyze.py:219,248 + video_pipeline.py:246) + `VLMConfig.retry_attempts/timeout_s` appliqué dans `gemini_client`
- [ ] **T0-4 (follow-up)** Triage Dependabot alert #228 — CVE-2025-69872 DiskCache (MEDIUM, transitive, no upstream patch). Décider : ignore-rule documenté, downstream pinning, ou retrait dep si possible

### Track 1 — Challenge qualif tech *(P1 — 3-5 jours)*

- [ ] **T1-1** Critique structurée du doc `docs/agentic-frameworks-comparison.md` (cost realism, lock-in réel, Live API gaps, Qwen streaming non démontré, abstraction provider absente, etc.)
- [ ] **T1-2** ADR-001 — décision finale archi agents + couche d'isolation provider (AI Gateway/LiteLLM, abstraction `MemoryService`)

### Track 2 — Reliability + observability *(P1 — 5-7 jours)*

- [ ] **T2-1** OpenTelemetry traces + metrics → Cloud Trace + Cloud Monitoring
- [ ] **T2-2** Structured logging avec correlation ID middleware
- [ ] **T2-3** TaskStore persistant (Firestore) — remplace l'in-memory dict (perte d'état au restart Cloud Run)
- [ ] **T2-4** Idempotency keys + circuit breaker Gemini
- [ ] **T2-5** SLO/SLI dashboard + alerts

### Track 3 — Security hardening *(P1 — 4-5 jours)*

- [ ] **T3-1** Cosign image signing + SLSA provenance + SBOM (syft)
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
