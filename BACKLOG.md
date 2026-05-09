# BACKLOG.md

Backlog priorisé — Phase 5b : Auth + Docker + GCP Deploy.

> Phases 0–4 + Phase 5a (Python 3.12 + Security CI) complètes. Voir [CHANGELOG.md](CHANGELOG.md) pour le détail.

---

## 📋 Phase 5b : Auth + Docker + GCP Deploy

### [P5-1] Auth API key
**Priorité :** HIGH

- Header `X-API-Key` sur tous les endpoints
- `settings.api_keys: list[str]` depuis env var `API_KEYS` (comma-separated)
- 401 si clé absente, 403 si invalide
- `GET /health` exempt (Cloud Run liveness probe)
- Tests d'intégration : 401 no key, 403 invalid, 200 valid, health sans clé

### [P5-2] Docker + docker-compose
**Priorité :** HIGH

- `Dockerfile` multi-stage (builder: `uv sync --frozen --no-dev` + runtime: port 8080)
- `docker-compose.yml` : api + volumes data/models
- `.dockerignore`, `.env.local.example`
- Health check Docker sur `GET /health`
- Smoke : `docker build` + `compose up` + `curl /health`

### [P5-3] Terraform IaC (GCP)
**Priorité :** HIGH

- `infra/terraform/` : project, registry, secrets, iam, cloud_run
- Cloud Run : min=0, max=3, cpu=1, mem=2Gi, timeout=300s
- Artifact Registry + GCP Secret Manager (GEMINI_API_KEY, API_KEYS)
- `infra/scripts/bootstrap.sh` pour gcloud billing setup

### [P5-4] GitHub Actions deploy workflow
**Priorité :** MEDIUM

- `.github/workflows/deploy.yml`
- Workload Identity Federation (pas de JSON key)
- docker build → push Artifact Registry → terraform apply dev

### [P5-5] Smoke tests GCP DEV
**Priorité :** MEDIUM

- `curl /health` depuis Cloud Run URL
- Upload vidéo test → vérifier 202 + task_id
- Valider auth (401 sans clé, 200 avec clé)

### [P5-6] Trivy container scanning
**Priorité :** LOW

- Ajouter au `security.yml` une fois le Dockerfile créé
- Scan image pour CVEs OS-level + deps

---

## ✅ Terminé (Phases 0–5a)

Toutes les tâches des phases 0, 1, 2, 3, 4 et 5a sont closes.
Voir les issues GitHub fermées et [CHANGELOG.md](CHANGELOG.md) pour le détail.
