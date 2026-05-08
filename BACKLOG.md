# BACKLOG.md

Backlog priorisé pour la relance du projet.

## P0 — Bloquants de relance
- [ ] Corriger la structure Python (imports cassés `ModuleNotFoundError`, modules et packaging).
- [ ] Supprimer la duplication de logique (`app.py` et `metrics.py` actuellement redondants).
- [ ] Définir et matérialiser l'architecture cible (`src/api`, `src/ml`, `src/data`, `tests`, `scripts`).
- [ ] Rendre un flux MVP exécutable: `video -> preprocess -> inference -> feedback`.
- [ ] Corriger incompatibilités de modèles et signatures (`VideoProcessor`, `RandomForest` sans `load_weights`, imports `np` manquants).
- [ ] Ajouter un test minimum exécutable (actuellement 0 test découvert).

## P1 — Qualité & reproductibilité
- [ ] Mettre en place des tests unitaires minimaux sur utils/models.
- [ ] Ajouter CI GitHub Actions (lint + tests).
- [ ] Standardiser la gestion des dépendances/environnements.
- [ ] Ajouter des jeux de données d'exemple et conventions dataset.

## P2 — Produit & open-source readiness
- [ ] Refonte complète du README (vision, quickstart, architecture, limites).
- [ ] Ajouter `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`, `SECURITY.md`.
- [ ] Définir stratégie de commits, branches et release notes.
- [ ] Ajouter templates d'issues/PR.

## P3 — Cutting edge 2026+
- [ ] Intégrer pipeline multimodal (vidéo + audio + contexte).
- [ ] Ajouter couche agentique (analyse biomécanique, contexte, décision).
- [ ] Introduire suivi d'expériences et versioning de modèles/datasets.
- [ ] Définir protocole d'évaluation robustesse/biais/généralisation.
