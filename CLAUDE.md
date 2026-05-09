# CLAUDE.md — AI Shoot

> Ce fichier est lu automatiquement par Claude/Copilot à chaque session.
> Il contient les conventions, workflows et décisions clés du projet.
> Ne jamais dévier des règles définies ici sans discussion explicite.

---

## 🏀 Vision produit

**AI Shoot** est une plateforme d'intelligence sportive en temps réel — basketball comme premier vertical.

Le système **voit** le mouvement (ViTPose/YOLOv11), **comprend** la biomécanique, **raisonne** comme un coach (LangGraph agents), et **explique** en langage naturel via VLM (Gemini Flash / Qwen2-VL).

Pipeline: `Vidéo → Perception → Biomécanique → VLM → Agents → Feedback actionnable`

---

## 🛠️ Stack technique

| Couche | Technologie |
|--------|-------------|
| Packaging | `uv` + `pyproject.toml` (Python 3.11+) |
| Pose estimation | ViTPose / YOLOv11-pose |
| VLM | Gemini Flash / Qwen2-VL |
| Agents | Google ADK 2.0 (+ `ShotAnalysisPipeline` standalone) |
| API | FastAPI async |
| Validation | Pydantic v2 |
| Lint/Format | ruff |
| Tests | pytest |
| CI | GitHub Actions |

---

## 📁 Structure

```
src/perception/   → extraction pose, tracking, vidéo pipeline
src/analysis/     → biomécanique, phases tir, métriques
src/agents/       → LangGraph agents (perceiver, analyzer, coach, planner)
src/vlm/          → intégration Gemini/Qwen2-VL, prompt templates
src/api/          → FastAPI routes + Pydantic schemas
src/core/         → config (SecretStr pour les clés), logging, exceptions
tests/            → unit + integration
legacy/           → prototype archivé (ne pas toucher)
```

---

## 🔁 Workflow PR — obligatoire, ne jamais dévier

Chaque ticket suit ce cycle exact. **Ne pas sauter d'étape.**

```
1.  Checkout branche dédiée (feat/xxx ou fix/xxx)
2.  Implémenter + tester localement (uv run pytest)
3.  Lint + format propres — **obligatoire avant tout push** :
    ```
    uv run ruff check .
    uv run ruff format --check .
    ```
    Si format échoue → `uv run ruff format .` puis re-vérifier.
3b. **Security scan local avant toute PR qui touche à src/ ou deps** :
    ```
    uv run pip-audit               # CVEs packages
    uv run bandit -r src/ -c pyproject.toml --severity-level high  # SAST HIGH
    ```
    Si pip-audit remonte des vulns → évaluer mise à jour dep + noter dans PR.
    Si bandit remonte un HIGH → corriger avant de push.
4.  Commits atomiques (Conventional Commits)
5.  Push + ouvrir la PR sur GitHub
6.  Attendre la review automatique Gemini Code Assist (~2 min)
7.  Lire et challenger chaque finding (voir section ci-dessous)
8.  Appliquer les fixes légitimes, committer + push
9.  Poster UN comment de synthèse sur la PR qui tague @gemini-code-assist
    avec ACCEPTED/REJECTED par finding + preuve source (voir format ci-dessous)
10. Attendre réponse Gemini (~3 min)
11. Si Gemini nuance → retour étape 7
12. CI verte + Gemini acquiesce (ou silence ~3 min) → merger
13. Puis seulement passer au ticket suivant
14. **Après merge** : mettre à jour en un seul commit docs/ :
    - README.md  → statut des phases, tech stack, quickstart
    - CHANGELOG.md → entrée pour la phase/feature livrée
    - ROADMAP.md  → cocher la phase complétée, décrire la suivante
    - BACKLOG.md  → retirer les tickets fermés, afficher le prochain focus
    - Fermer les issues GitHub correspondantes + mettre à jour le Kanban
```

**Règles absolues :**
- Ne jamais merger sans CI verte
- Ne jamais merger sans avoir répondu à chaque finding (même les reject)
- Ne jamais commencer le ticket suivant avant que la PR soit mergée
- Si Gemini ne répond pas dans ~3 min après le comment taggué : le silence vaut acquiescement
- Les mises à jour docs (étape 14) font partie du ticket — elles ne sont pas optionnelles

---

## 🔍 Protocole — challenger les findings des reviewers

**Ne jamais appliquer un finding sans l'avoir vérifié.** Gemini peut suggérer des fixes incorrects, inutiles, ou créant de nouveaux bugs.

### Avant d'appliquer un fix, toujours vérifier :

1. **Le problème est-il réel ?** → Lire le code exact dans son contexte complet
2. **La lib se comporte-t-elle vraiment comme décrit ?** → Vérifier la doc officielle ou le code source de la lib
3. **Le fix ne crée-t-il pas de régression ?** → Tracer l'impact downstream, vérifier les tests
4. **Y a-t-il une meilleure solution ?** → Le fix du reviewer est un point de départ, pas une vérité

### Quand rejeter un finding

Rejeter explicitement si :
- La doc officielle contredit le comportement décrit
- Le problème n'existe pas dans le code réel (faux positif)
- Le fix viole une convention de ce CLAUDE.md
- Le finding porte sur du code archivé (`legacy/`) — ce code est intentionnellement mort

### Format du commentaire de synthèse (obligatoire)

```markdown
Thanks @gemini-code-assist — <résumé global 1 ligne> (fix: <SHA>)

**Finding 1 (<chemin/fichier:ligne>):** ACCEPTED/REJECTED.
<Preuve : extrait source, citation doc, snippet, lien...>

**Finding 2 (…):** ACCEPTED/REJECTED.
<Preuve…>
```

Exemples de preuves solides :
- **Accept** : "Pydantic v2 valide et sérialise `datetime` en ISO-8601 automatiquement — `str` bypasse cette validation."
- **Reject** : "Ce fichier est dans `legacy/` — code archivé intentionnellement (voir `legacy/README.md`). Les bugs sont connus et documentés, le code n'est pas exécuté."

---

## 📐 Conventions code

### Commits (Conventional Commits)
```
feat:  nouvelle fonctionnalité
fix:   correction de bug
chore: maintenance (deps, config)
docs:  documentation
test:  ajout/modification de tests
ci:    GitHub Actions
refactor: refactoring sans changement de comportement
```

### Branches
- `main` → toujours stable, mergeable
- `feat/xxx` → nouvelle fonctionnalité
- `fix/xxx` → correction bug
- PRs required pour merger dans main

### Python / Pydantic
- `StrEnum` (pas `str + Enum`) pour les enums string
- `SecretStr` pour **toutes** les clés API et secrets (jamais `str`)
- `datetime` (pas `str`) pour les timestamps dans les schemas Pydantic
- `model_post_init` pour le calcul de champs dérivés (pas `field_validator` sur champs siblings)
- Annotations `from __future__ import annotations` en tête de chaque fichier

### Sécurité
- **Jamais de secret en clair dans le code** — utiliser `SecretStr` + `.env`
- `.env` est dans `.gitignore` — utiliser `.env.example` comme template
- Accéder à la valeur d'un `SecretStr` uniquement via `.get_secret_value()` au moment de l'utilisation

---

## 🧪 Commandes de développement

```bash
# Install
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev

# Tests
uv run pytest                          # tous les tests
uv run pytest tests/unit/ -v           # tests unitaires
uv run pytest --cov=src               # avec couverture

# Lint + format
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/         # format
uv run mypy src/                       # type check

# Security
uv run pip-audit                       # CVEs packages
uv run bandit -r src/ -c pyproject.toml --severity-level high  # SAST
```

---

## 📚 Documents de référence

- Vision complète & quickstart: `README.md`
- Feuille de route phases 0-5: `ROADMAP.md`
- Tickets ouverts: `BACKLOG.md`
- Changelog: `CHANGELOG.md`
- Guide contribution: `CONTRIBUTING.md`

