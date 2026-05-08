# ROADMAP.md

Feuille de route de relance AI Shoot (sans dates figées, par étapes).

## Phase 1 — Reconception (Now)
**Objectif:** passer du prototype à un socle propre et exécutable.

- Audit technique complet du code existant (fait): imports cassés, scripts dupliqués, incompatibilités modèle/API, aucun test exécutable.
- Blueprint d'architecture cible.
- Spécification MVP (priorité joueur, hook coach).
- Plan de migration progressive.

### Architecture cible (V2)
```text
shoot-ai/
  src/
    api/
      main.py
      routes/
        health.py
        inference.py
      schemas/
        request.py
        response.py
    ml/
      pipelines/
        inference_pipeline.py
      models/
        cnn_model.py
        classical_model.py
      services/
        coaching_engine.py
    data/
      loaders/
      preprocessors/
      validators/
    core/
      settings.py
      logging.py
  tests/
    unit/
    integration/
  scripts/
    train.py
    evaluate.py
```

### Migration technique (ordre)
1. Isoler l'API FastAPI dans `src/api/` avec endpoint de santé.
2. Centraliser le pipeline d'inférence dans `src/ml/pipelines/`.
3. Déplacer les utilitaires data dans `src/data/`.
4. Remplacer les imports plats par imports package (`from src...`).
5. Supprimer la duplication `app.py`/`metrics.py` et garder des rôles clairs (`scripts/` vs `api/`).

## Phase 2 — Base production (Next)
**Objectif:** fiabiliser et rendre reproductible.

- Tests unitaires + intégration minimum.
- CI GitHub (qualité continue).
- Standard data/model versioning.
- API FastAPI modulaire et contrats stables.

## Phase 3 — Open-source readiness (Next+)
**Objectif:** ouvrir la contribution GitHub proprement.

- Documentation centrale (README, architecture, guides).
- Gouvernance projet (`CONTRIBUTING`, licence, sécurité, code of conduct).
- Stratégie de commit/branch/release.

## Phase 4 — AI multimodale agentique (Future)
**Objectif:** différenciation forte et intelligence contextuelle.

- Fusion multimodale (vision, audio, contexte).
- Orchestration multi-modèles et agents spécialisés.
- Coaching explicable orienté décision.
- Personnalisation par profil joueur/coaching.

## Livrables visibles sur GitHub
- `README.md` (entrée principale)
- `ROADMAP.md` (feuille de route)
- `BACKLOG.md` (priorisation opérationnelle)
- `CLAUDE.md` (guide assistants IA)
