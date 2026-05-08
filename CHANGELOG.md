# Changelog

Toutes les modifications notables sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
Ce projet suit le [Semantic Versioning](https://semver.org/lang/fr/).

## [Unreleased]

### Added
- Nouvelle structure de projet (`src/perception`, `src/analysis`, `src/agents`, `src/vlm`, `src/api`)
- Migration vers `uv` + `pyproject.toml` (Python 3.11+)
- Contrats d'interfaces Pydantic (`VideoInput`, `CoachingFeedback`, `BiomechanicsReport`, etc.)
- Configuration centralisée via `pydantic-settings`
- Logging structuré avec `structlog`
- CI GitHub Actions (lint + tests)
- Fichiers de gouvernance open-source (`CONTRIBUTING`, `CODE_OF_CONDUCT`, `LICENSE`, `SECURITY`)
- Stratégie de commits Conventional Commits documentée
- `.pre-commit-config.yaml` avec ruff

### Changed
- `requirements.txt` remplacé par `pyproject.toml`
- Architecture entièrement repensée (vision plateforme d'intelligence basketball)

### Removed
- Code prototype non fonctionnel (`app.py` script dupliqué, imports cassés)
- Stack périmée (tensorflow 2.11, sklearn 0.24, opencv 4.5)

## [0.0.1] - 2024 (prototype initial)

### Added
- Prototype initial : CNN custom + RandomForest pour analyse tir basket
- API FastAPI basique (non fonctionnelle)
- Utilitaires OpenCV pour chargement vidéo
