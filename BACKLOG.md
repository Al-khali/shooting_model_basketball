# BACKLOG.md

Backlog priorisé — Phase 5 : Edge, Auth & Déploiement.

> Phases 0–4 complètes. Voir [CHANGELOG.md](CHANGELOG.md) pour le détail de ce qui a été livré.

---

## 📋 Phase 5 : Auth + Docker + Edge

### [P5-1] Auth API key
**Priorité :** HIGH

- Header `X-API-Key` sur tous les endpoints
- `settings.api_keys: list[str]` depuis env var
- 401 si clé absente ou invalide
- Tests d'intégration auth

### [P5-2] Docker + docker-compose
**Priorité :** HIGH

- `Dockerfile` multi-stage (builder + runtime)
- `docker-compose.yml` : api + volumes data/models
- Health check Docker sur `GET /health`

### [P5-3] Export ONNX perception
**Priorité :** MEDIUM

- Export YOLOv11-pose → ONNX
- Inférence ONNX Runtime (CPU + CUDA)
- Benchmark latence vs PyTorch

### [P5-4] Rate limiting
**Priorité :** LOW

- `slowapi` ou middleware custom
- Quotas par API key : N uploads/minute

---

## ✅ Terminé (Phases 0–4)

Toutes les tâches des phases 0, 1, 2, 3 et 4 sont closes.
Voir les issues GitHub fermées et [CHANGELOG.md](CHANGELOG.md) pour le détail.
