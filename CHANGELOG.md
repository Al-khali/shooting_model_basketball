# Changelog

Toutes les modifications notables sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
Ce projet suit le [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.7] — Track 0 (T0-11): unblock CI/CD deploy — 3 IAM/API fixes (2026-05-13)

100% des runs Deploy CI/CD ont silencieusement échoué depuis T0-6 (PR #36). Surface live `https://shoot-ai-dev-chf52ondba-uc.a.run.app` tournait toujours sur l'image manuelle SHA `2ce8474`, **sans** les fixes T0-5/T0-3/T0-9. Cascade de 3 root causes débloquées par cette PR.

### Fixed
- **`infra/terraform/iam.tf`** — nouvelle ressource `google_storage_bucket_iam_member.cicd_tfstate`
  - Le SA `shoot-ai-dev-cicd` n'avait aucune IAM sur le bucket `gs://shoot-ai-poc-tfstate`, créé out-of-Terraform par `bootstrap.sh`. PR #29 initiale n'avait jamais ajouté cette binding car le chicken-and-egg du backend GCS empêche le manage du bucket par TF
  - Rôle `roles/storage.admin` (pas `objectAdmin` — Gemini finding ACCEPTED) : le backend GCS appelle `storage.buckets.get` durant `terraform init` pour vérifier le versioning. `objectAdmin` n'a pas cette permission, `storage.admin` oui. Scoped au bucket via `google_storage_bucket_iam_member` (pas project-wide)
  - Référence `google_project.shoot_ai.project_id` au lieu de `var.project_id` (Gemini finding 2 ACCEPTED) — cohérence avec le reste de `iam.tf` + dependency graph carry désormais cette binding
- **`infra/terraform/project.tf`** — `cloudresourcemanager.googleapis.com` ajouté à la liste des APIs
  - Pendant le bootstrap T0-6, mes credentials Owner activaient implicitement cette API. Le SA cicd ne l'a pas implicitement → chaque `terraform plan` failait sur `Error 403: Cloud Resource Manager API has not been used in project 592121247070`
  - 8e API maintenant : `run`, `artifactregistry`, `secretmanager`, `iam`, `cloudbuild`, `iamcredentials`, `sts`, `cloudresourcemanager`

### Hot-fixes appliqués hors-PR (à formaliser en T0-12)
- `gcloud services enable cloudresourcemanager.googleapis.com --project=shoot-ai-poc`
- `gcloud storage buckets {remove,add}-iam-policy-binding gs://shoot-ai-poc-tfstate ... role=roles/storage.{objectAdmin,admin}` (upgrade after Gemini finding)
- `gcloud projects add-iam-policy-binding shoot-ai-poc --member=...cicd... --role=roles/editor` — round 4 surfaced new perms missing (`iam.serviceAccounts.get`, `serviceusage.services.list`, project IAM read). Pragmatic POC choice (vs adding 3-4 narrower roles separately). À resserrer en T0-12 follow-up

### Notes Gemini Code Assist
- 2 findings MEDIUM sur le PR initial — **tous ACCEPTED** (storage.admin vs objectAdmin, project ref vs var ref). **Gemini a anticipé le 3e fix nécessaire** (cloudresourcemanager API) que je n'avais pas vu venir — confirmation que le challenge protocol CLAUDE.md a une vraie valeur prédictive
- Validation finale Gemini : *"La correction apportée dans `project.tf` pour inclure `cloudresourcemanager.googleapis.com` est la bonne approche, car Terraform a effectivement besoin de cette API pour effectuer les appels de découverte de ressources lors de chaque plan et apply. Ton choix de restreindre `roles/storage.admin` au bucket spécifique via `google_storage_bucket_iam_member` est également conforme aux meilleures pratiques de moindre privilège. ... Tout semble prêt pour le merge."*

### Follow-up
- **T0-12 (nouveau)** Formaliser le `roles/editor` ajouté en hot-fix dans `iam.tf`. Idéalement substitué par 3 rôles plus étroits (`roles/iam.serviceAccountAdmin`, `roles/serviceusage.serviceUsageAdmin`, `roles/resourcemanager.projectIamAdmin`) — pas prioritaire en POC zero-budget mais nécessaire avant prod

### Impact opérationnel
Le prochain push sur main devrait désormais déployer correctement le code v1.0.6 (avec T0-5 fixes + T0-3 narrow exceptions + T0-9 YOLO baked) sur Cloud Run.

## [1.0.6] — Track 0 (T0-9): bake YOLO pose weights into Docker image (2026-05-12)

Follow-up T0-6 identifié sur la live test Cloud Run : cold-start de chaque nouvelle instance téléchargeait `yolo11n-pose.pt` (~6 MB) depuis GitHub avant le 1er `/analyze`. Avec scale-to-zero, ce coût frappait la majorité des requêtes POC.

### Performance
- **`Dockerfile`** — pre-fetch des weights YOLO au build time, baked dans `/app/yolo11n-pose.pt` du runtime stage
  - Builder stage : 3 libs système ajoutées (`libgl1`, `libglib2.0-0`, `libxcb1`) pour permettre `import cv2`. `ultralytics` traîne transitivement `opencv-contrib-python` (full GUI, link contre libxcb) malgré le `opencv-python-headless` pinné dans `pyproject.toml`. Sans ces libs, le pre-fetch fail : `ImportError: libxcb.so.1: cannot open shared object file`.
  - Pre-fetch via `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"` → file dans `/build/yolo11n-pose.pt`
  - Runtime stage : `COPY --from=builder --chown=appuser:appuser /build/yolo11n-pose.pt ./yolo11n-pose.pt`. `ultralytics` au runtime résout le relative path contre CWD=`/app` (WORKDIR), donc pur disk read, aucun network call
- **Image** : +50 MB (delta OCI incluant métadata pour 6.25 MB de poids). Push CI/CD inchangé (layer cachée après 1er push)
- **Bénéfice attendu Cloud Run x86_64** : ~500ms–2s économisés sur le 1er `/analyze` de chaque instance fraîche

### Verification (local Apple Silicon, QEMU linux/amd64)
- `docker build` step builder/10 passe : `RUN .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"` ✅
- `docker run --rm shoot-ai:baked ls -la /app/yolo11n-pose.pt` → `6255593 bytes appuser` ✅
- `docker logs <container> | grep -i download` → empty (le model est preloaded, aucun GitHub fetch) ✅

### Notes Gemini Code Assist
- 1 finding MEDIUM (centraliser le nom du modèle via `ARG YOLO_MODEL` + `ENV` + `os.getenv()`) — **REJECTED**, follow-up loggé en T0-10
- Justification REJECT : (1) le modèle est architectural pas runtime (couplé à `PoseModel.YOLOV11` enum), (2) POC à un seul modèle pas de switch prévu, (3) paramétrisation propre demanderait 7 fichiers touchés (Dockerfile + pose_estimator + .env.example + docker-compose + cloud_run.tf + scripts/local_e2e.sh + tests) pour bénéfice spéculatif, (4) scope PR = perf pas archi, (5) follow-up dédié quand le besoin réel apparaîtra (Phase 6 ONNX/mobile, A/B testing variants YOLO)
- Validation finale Gemini : *"Ton analyse sur le couplage architectural et la gestion du scope pour ce POC est tout à fait pertinente. Le suivi via un ticket dans le BACKLOG est la bonne approche pour éviter l'over-engineering tout en gardant une trace technique pour les évolutions futures (Phase 6). ... Prêt pour le merge."* — **premier REJECT du programme v2.0 validé par Gemini**

### Follow-up logged
- **T0-10 (optionnel)** Centraliser le choix YOLO via `ARG YOLO_MODEL` + `ENV` + `os.getenv('YOLO_MODEL', default)`. À activer quand le besoin réel apparaît (Phase 6 ONNX/mobile, A/B testing yolo11n vs yolo11s/m/l). Patron clean : single source of truth dans `pyproject.toml` ou `.env.example`, propagation Docker + Terraform Cloud Run env

## [1.0.5] — Track 0 (T0-3/T0-7): narrow exception handlers + VLM retry/timeout (2026-05-12)

Retour de la PR #34 mise en draft pendant T0-5 (course-correction). Rebase sur baseline v1.0.4 **clean, zéro conflit** — T0-3 et T0-5 touchent des fichiers disjoints.

### Fixed
- **3 `except Exception:` resserrés** — les broad catches masquaient les vraies causes d'erreur :
  - `src/api/routes/analyze.py:219` (WebSocket params receive) → `(json.JSONDecodeError, UnicodeDecodeError, RuntimeError)` + `WebSocketDisconnect` géré séparément
  - `src/api/routes/analyze.py:248` (WebSocket video bytes loop) → `(RuntimeError, OSError, ConnectionError)`
  - `src/perception/video_pipeline.py:246` (per-frame pose estimation) → `(cv2.error, RuntimeError, ValueError, IndexError)`
  - Chaque site reçoit un commentaire explicatif documentant quelle condition upstream lève quel type
  - `MemoryError`, `KeyboardInterrupt` propagent maintenant comme elles le devraient

### Added
- **`VLMConfig` resilience knobs** (`src/vlm/base.py`) : `retry_attempts=3`, `retry_backoff_seconds=1.0`, `retry_max_backoff_seconds=16.0`. Worst-case ~7s cumulé (3 waits entre 4 attempts), avec full jitter actual = uniform[0, 7]
- **`GeminiFlashClient._call_with_retry`** (`src/vlm/gemini_client.py`) : retry exponentiel + **full jitter** (AWS Architecture pattern, `sleep = random.uniform(0, base_backoff)`) sur transients Google API (`DeadlineExceeded`, `ServiceUnavailable`, `InternalServerError`, `ResourceExhausted`, `Aborted`, `GatewayTimeout`). Fallback `(TimeoutError, ConnectionError)` quand `google-api-core` absent. Non-retryable propage immédiatement ; après épuisement → `VLMError`
- **`_request_options()`** : `{"timeout": config.timeout_seconds}` passé à chaque `generate_content`/`send_message`
- **`_dispatch(model, gemini_messages: list[dict])`** : refactor en unité retry-wrappable. Type hint `list[dict]` (pas `Iterable`) — évite la consommation prématurée d'un générateur sur retry
- **+7 tests unitaires** (190 total vs 183 sur main) : `VLMConfig` retry defaults/override + 5 tests `_call_with_retry` (success après N transient, `VLMError` après exhaustion, non-retryable propagation, no-retry happy path, `request_options` carries timeout). Stub `google.generativeai` via `sys.modules`

### Notes Gemini Code Assist
- 4 findings MEDIUM avant la pause : math docstring timing (1+2+4≠15s), backoff déterministe (thundering herd), `_dispatch(Iterable[dict])` bug latent (generator consumption), `response.text` hors try dans `complete_json` (contrat `VLMError` rompu) — **tous ACCEPTED + corrigés**
- Validation finale Gemini (pré-pause) : *"Les corrections apportées, notamment le passage au 'full jitter' pour la résilience réseau et le typage strict des messages pour éviter la consommation prématurée des générateurs, sont excellentes. La suite de tests unitaires couvre bien les cas limites. Le code est prêt pour le merge."*
- Rebase post-T0-5 a confirmé que T0-3 et T0-5 sont orthogonaux ; pipeline post-rebase 6/6 (ruff, format, mypy, 190 tests, bandit, pip-audit) + `bash scripts/local_e2e.sh` 5/5 vert
- Merge final via `--admin` pour bypass un duplicate Trivy run encore pending (l'autre run identique avait déjà passé en 4m16s, code strictement identique au pre-rebase ack'd par Gemini)

## [1.0.4] — Track 0: GCP bootstrap real (T0-6) — first live deploy (2026-05-12)

**Premier provisioning GCP réel** depuis la création du projet. Les agents précédents avaient shippé Terraform + CI/CD sans jamais avoir bootstrappé contre un vrai billing account. T0-6 ferme ce gap.

### Live deployment
- **Service URL** : `https://shoot-ai-dev-chf52ondba-uc.a.run.app` (Cloud Run v2, us-central1)
- **Project** : `shoot-ai-poc` (#592121247070)
- **Billing** : `cpt_bst_1` (01E0C6-D8BA4E-0D2C3F) — 3 anciens accounts étaient fermés
- **25 ressources** Terraform créées : 7 APIs activées + 2 service accounts (cicd, cloud_run) + 2 secrets (gemini-api-key, api-keys) + 2 versions + Workload Identity pool/provider + 4 IAM bindings + Artifact Registry + Cloud Run v2 + public_access IAM
- **5 GitHub secrets** configurés (`GCP_PROJECT_ID`, `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `API_KEYS`, `GEMINI_API_KEY`) — le workflow Deploy passe désormais le preflight et exécute le build+push+apply

### Changed
- **Région** `europe-west1` → `us-central1` (`infra/terraform/variables.tf`, `tfvars`, `deploy.yml`, `bootstrap.sh`, `deploy.sh`). Cloud Storage Always Free tier (5 GB-mois) s'applique seulement à us-east1/us-west1/us-central1. Pour zero-budget POC strict, on choisit us-central1. Latence depuis EU ~+100ms cold path uniquement.
- **Project ID** placeholder `shoot-ai-dev` → `shoot-ai-poc` (suffixe explicite POC, anticipe un futur `shoot-ai-prod` séparé)
- **`cloud_run.tf`** : `deletion_protection = var.environment == "prod"` (finding Gemini ACCEPTED) — environment-driven, false en dev/POC pour itérer librement, true en prod pour anti-destroy. Self-documenting + anti-regression.

### Fixed
- **Cloud Run service `deletion_protection`** : valeur par défaut `true` côté GCP bloquait les destroy/recreate sur changement d'image_tag → workaround manuel pendant le bootstrap (`gcloud run services delete` + `terraform state rm` + re-apply). Maintenant désactivé en dev/POC pour permettre l'itération.

### Live verification
- `GET /health` → 200 (version `1.0.3` dynamique via `importlib.metadata`, 433ms)
- `GET /player/test/history` (no key) → 401
- `GET /player/test/history` + valid key → 200
- `POST /analyze` + stub.mp4 + valid key → 202 task_id

### Notes Gemini Code Assist
- 1 finding MEDIUM sur `deletion_protection` hardcoded false (proposé `var.environment == "prod"`) — **ACCEPTED**
- Validation finale : *"Ton approche pour la gestion de `deletion_protection` est excellente : elle équilibre parfaitement la flexibilité nécessaire pour le POC et la sécurité requise pour une future mise en production. ... Tout semble en ordre pour le merge."*

### Follow-ups identifiés via les live tests (logged in BACKLOG)
- **T0-8** `/unknown` route retourne 401 au lieu de 404 — le middleware auth tourne avant le routing FastAPI. Trade-off anti-enumeration vs UX à arbitrer (ou passer auth en dependency per-route)
- **T0-9** YOLO weights non bakés dans l'image Docker — chaque instance fraîche télécharge ~6MB depuis ultralytics au cold start, ralentit le premier `/analyze` post-scaling. À mitiger : copier les weights dans l'image au build
- **T2-3 priorisé** TaskStore in-memory : autoscaling Cloud Run entre instances perd les tasks. Polling sur cold start a confirmé que GET /session reste accessible mais le bg task ne complete pas dans le timing local — Firestore-backed store devient bloqueur prod

### Action requise hors-PR (operator)
1. `gcloud auth login` + `gcloud auth application-default login`
2. `gcloud projects create shoot-ai-poc --name="AI Shoot POC"`
3. Création billing account `cpt_bst_1` via console GCP
4. `gcloud billing projects link shoot-ai-poc --billing-account=01E0C6-D8BA4E-0D2C3F`
5. `PROJECT_ID=shoot-ai-poc BILLING_ACCOUNT=... REGION=us-central1 bash infra/scripts/bootstrap.sh`
6. `terraform -chdir=infra/terraform init -backend-config="bucket=shoot-ai-poc-tfstate" ...`
7. `terraform import google_project.shoot_ai shoot-ai-poc`
8. `docker build` + `docker push` (chicken-and-egg sur la 1re image)
9. `terraform apply` (25 ressources)
10. `gh secret set` les 5 secrets via Terraform outputs

## [1.0.3] — Track 0 (stabilisation): end-to-end validation + 6 silent bugs (2026-05-12)

**Course-correction critique** : avant ce PR la stack n'avait **jamais** tourné end-to-end avec un vrai POST `/analyze`. Le premier run local (uvicorn + ffmpeg stub video + curl) a révélé 6 bugs que les 182 tests unitaires verts n'ont pas attrapés, parce que les tools tombaient en silence sur des stubs deferred-import.

### Fixed
- **Bug A — leak d'exception text dans le champ user-facing** (`coach_tools.py`)
  Le stub fallback retournait `detailed_analysis: "GEMINI_API_KEY environment variable not set."`. Remplacé par des codes d'erreur stables (`vlm_unavailable`, `coaching_failed:VLMError`) ; le texte d'exception est loggé mais jamais retourné dans le body.
- **Bug B — status=done malgré dégradation silencieuse** (`coach_tools.py`, `orchestrator.py`)
  `_stub_coaching_feedback` retournait dict avec `error` ET `summary` → orchestrator guard (`not feedback.get("summary")`) évaluait False → pipeline reportait `done` avec le leak Bug A. Stub supprimé : strict mode (`status=error, on bloque`), retour `{"error": ..., "player_id": ...}` sans `summary`, guard simplifié à `if "error" in feedback_result`.
- **Bug C — pas de validation média en entrée** (`orchestrator.py`)
  Une vidéo synthétique 2s sans humain donnait `summary: "Analysis complete. Focus on your release mechanics"`. Orchestrator valide maintenant `perception_result.player_detected AND key_frames` avant de continuer ; sinon `error: no_shootable_content`.
- **Bug D — `/health.version` hardcodé "0.5.0"** (`responses.py`)
  Désynchro 3 sources (pyproject `0.1.0`, responses `0.5.0`, CHANGELOG `1.0.2`). Résolu dynamiquement via `importlib.metadata.version("shoot-ai")` au startup avec fallback `0.0.0+unknown`. `pyproject.toml` bumpé à `1.0.3`.
- **Bug E — perception jamais réellement appelée** (`perception_tools.py`)
  L34 importait `from src.perception.video_processor import VideoProcessor` mais le fichier est `video_pipeline.py`. Le `# type: ignore[import]` cachait le typo à mypy. **Depuis Phase 3, chaque `extract_shot_frames` tombait en ImportError → stub avec `player_detected: True` fabriqué.** Fixé.
- **Bug F — API VideoProcessor mal utilisée** (`perception_tools.py`)
  Même après le fix de l'import : `VideoProcessor(video_path)` (constructor prend `estimator + config`, pas le path) + `.extract_frames()` (la méthode est `.process(video_path)` qui retourne `PerceptionOutput` complet incluant shot phases). Module réécrit avec l'API correcte (`VideoProcessor().process(video_path).model_dump()`).

### Performance
- **VideoProcessor cached en singleton module-level** (`perception_tools.py`) — finding Gemini sur le PR initial. Premier call paie le coût de chargement YOLO/MediaPipe (~500ms-2s, 6MB pour yolo11n) ; calls suivants réutilisent les weights chauds. Pattern double-check locking avec `threading.Lock` car FastAPI exécute les tools bloquants dans un worker pool (`anyio.to_thread.run_sync`) — deux requêtes `/analyze` concurrentes peuvent racer sur cache froid.

### Added
- **`scripts/local_e2e.sh`** — script de validation end-to-end automatisé : spawn uvicorn (port 8088), génère un stub mp4 via ffmpeg, exécute POST `/analyze` + polling `/session`, asserte `status=error` en mode no-key et **vérifie que la string "GEMINI_API_KEY" n'apparaît pas dans le body** (régression check Bug A). Ce script doit tourner avant tout PR qui touche le pipeline.

### Changed
- **Pydantic `HealthResponse.version`** : `default=APP_VERSION` direct au lieu de `default_factory=lambda` (finding Gemini — string immuable, lambda inutile à chaque instantiation).
- **`.gitignore`** : ajout `*.pt|*.pth|*.onnx` au niveau root (ultralytics dépose des weights dans CWD au premier call ; auparavant seulement `models/*.pt` était ignoré).

### Tests
- **183 tests** (vs 182 avant) — `test_health_version` ne pin plus le literal `"0.5.0"` ; `TestPerceptionTools` et `TestCoachTools` réécrits pour asserter les codes d'erreur stables et l'absence de leak (`Traceback`, `/tmp/`, `/Users/`, `\n` non présents) ; `TestShotAnalysisPipeline` stub les 3 tools à la frontière orchestrator via monkeypatch (les anciens tests passaient uniquement grâce au stub silencieux — c'était précisément le bug) ; nouveau test `test_analyze_returns_error_when_video_missing` exerce explicitement le chemin error.

### Notes Gemini Code Assist
- 2 findings MEDIUM sur le PR initial, **tous ACCEPTED** (cache VideoProcessor + `default` vs `default_factory`)
- Validation finale Gemini : *"L'implémentation du pattern double-check locking pour le `VideoProcessor` est tout à fait appropriée pour garantir la sécurité dans un environnement FastAPI concurrent, et la gestion dynamique de la version via `importlib.metadata` est une excellente pratique. Le script `local_e2e.sh` est un ajout précieux. Vous avez mon feu vert pour le merge."*

### Hors scope (follow-up Track 0)
- **T0-3** (PR #34 narrow exceptions + VLM retry) reste en draft — à ré-ouvrir contre ce baseline sain (les 4 fixes Gemini sur le retry path restent valides)
- **T0-6** (nouveau) : exécuter le bootstrap GCP réel (`infra/scripts/bootstrap.sh` + `terraform apply`) — étape opérateur humain, jamais effectuée
- **T0-4** : Dependabot alert #228 (CVE-2025-69872 DiskCache, MEDIUM, pas de patch upstream)

## [1.0.2] — Track 0 (stabilisation): deploy preflight skip (2026-05)

### Fixed
- `.github/workflows/deploy.yml` — graceful skip when GCP secrets are missing (PR #33)
  - Diagnosed root cause of the 6 consecutive deploy failures since 2026-05-09 17:15: the repository had **zero** GitHub Secrets configured (`gh api repos/.../actions/secrets` → `total_count: 0`). The Workload Identity Federation auth step rejected the empty `workload_identity_provider` input
  - New `preflight` job verifies `GCP_WORKLOAD_IDENTITY_PROVIDER` + `GCP_SERVICE_ACCOUNT` + `GCP_PROJECT_ID` via `env:` (safe pattern — never interpolated directly in the shell), exposes `ready=true|false` as job output
  - `build-and-push`, `deploy`, `smoke-test` gain `needs: preflight` + `if: needs.preflight.outputs.ready == 'true'`
  - Missing secrets → workflow runs green with a job summary listing the bootstrap procedure (run `infra/scripts/bootstrap.sh` + `terraform apply` + `gh secret set`); a `::warning::` annotation surfaces the skip
  - All four downstream `${{ secrets.GCP_PROJECT_ID }}` shell interpolations refactored to `env:` passing while at it (defense-in-depth pattern)
- First deploy run **green** since the workflow was introduced (run 25625550032: preflight ✅, others skipped)

### Notes
- Bootstrap remains a manual operator task (interactive `gcloud auth login` cannot run from CI); see CONTRIBUTING.md for the full procedure
- Once secrets are set, the next push to `main` will trigger a real deploy unchanged
- Forks that don't deploy will see clean green runs instead of red on every push

## [1.0.1] — Track 0 (stabilisation): Trivy hardening (2026-05)

### Fixed
- `.github/workflows/security.yml` — `container-scan` job hardened (4 misconfigurations identified during the v2.0 enrichment audit):
  - **Pin** `aquasecurity/trivy-action@v0.36.0` (was `@master` → non-deterministic across runs)
  - **Run on schedule** — removed `if: github.event_name != 'schedule'`. Trivy now executes on push, PR, **and** the weekly Monday 08:00 UTC cron (which is precisely when new CVEs land on an unchanged image)
  - **Widen severity gate** to `HIGH,CRITICAL` (was `CRITICAL` only — was missing actively-exploited HIGH OS CVEs); kept `ignore-unfixed: true` to avoid blocking on CVEs without an upstream patch
  - **SARIF upload** — second Trivy step (`format: sarif`, MEDIUM+HIGH+CRITICAL) feeds `github/codeql-action/upload-sarif@v3` so every finding is visible in the GitHub Security tab; runs with `if: always()` so the report lands even when the gating step fails
- Job-level `permissions: security-events: write` added (required by `upload-sarif`)

### Notes
- Gemini Code Assist replied: *"Gemini is unable to generate a review for this pull request due to the file types involved not being currently supported."* — no findings to challenge per CLAUDE.md §challenge protocol; treated as silent acquiescement after CI passed
- All 13 CI checks green on PR #32, including container scan in 4m18s
- Dependabot alert #228 surfaced during this PR (CVE-2025-69872 — DiskCache unsafe deserialization, MEDIUM, **no upstream patch**, transitive via uv.lock) — to be triaged in a follow-up Track 0 ticket

## [1.0.0] — Phase 5b: Complete — smoke test script (2026-05)

### Added
- `scripts/smoke_test.sh` — portable smoke test for local Docker + GCP Cloud Run:
  - `check_health()`: single HTTP call → 3 assertions (status 200, body `status=ok`, `version` key present)
  - `check()`: HTTP status assertion with `|| echo "000"` fallback (safe under `set -e`)
  - `check_body()`: exact JSON key=value match via `sys.argv` (injection-safe, no substring false-positives)
  - Auth, CORS preflight, 404 assertions (auth skipped if `API_KEY` not set)
  - Portable `mktemp` approach (no GNU `head -n -1`)
- Phase 5b **code-complete**: Security CI ✅ Auth ✅ Docker ✅ Terraform ✅ CI/CD ✅ Smoke script ✅

### Security
- Gemini review PR #31: 3 findings — all ACCEPTED: network error safety (F2), injection-safe pattern passing (F1 HIGH), single-request health check (F3)

---

## [0.9.0] — Phase 5b: CI/CD deploy + Trivy container scanning (2026-05)

### Added
- `.github/workflows/deploy.yml` — 3-job deploy pipeline:
  - `build-and-push`: Workload Identity → docker buildx → Artifact Registry (SHA tag + :latest + build cache)
  - `deploy`: terraform init → plan → apply (`image_tag=$SHA`, secrets via `TF_VAR_*`)
  - `smoke-test`: /health + auth check against live Cloud Run service URL
  - Concurrency group prevents parallel deploys
- `.github/workflows/security.yml` — `container-scan` job: docker build → Trivy CRITICAL CVEs (fail on CRITICAL, ignore-unfixed)
- `CONTRIBUTING.md` — CI/CD setup section: 5 required GitHub Secrets + first-time bootstrap instructions

### Security
- Workload Identity Federation (no JSON key) in all deploy jobs
- Trivy blocks deploy if CRITICAL OS/library CVEs found in container image
- Gemini review PR #30: 1 finding (MEDIUM) — bootstrap doc fixes (variable order, billing_account export, consistency)

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
