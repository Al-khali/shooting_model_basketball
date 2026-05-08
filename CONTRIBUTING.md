# Contributing to AI Shoot

First off — merci de t'intéresser au projet ! 🏀

## Comment contribuer

### Signaler un bug
Ouvre une [issue](https://github.com/Al-khali/shooting_model_basketball/issues) avec :
- une description claire du problème
- les étapes pour reproduire
- l'environnement (OS, Python, GPU/CPU)

### Proposer une feature
Ouvre une issue avec le label `enhancement` avant de coder. Ça évite qu'on travaille sur la même chose.

### Soumettre du code

```bash
# 1. Fork + clone
git clone https://github.com/your-username/shooting_model_basketball.git
cd shooting_model_basketball

# 2. Setup avec uv
uv sync --extra dev

# 3. Crée une branche depuis main
git checkout -b feat/ma-feature   # ou fix/mon-bug, docs/ma-doc

# 4. Code, teste, commit
uv run pytest
git commit -m "feat: description courte"

# 5. Push + ouvre une PR
git push origin feat/ma-feature
```

## Conventions

### Messages de commit (Conventional Commits)
```
feat: nouvelle fonctionnalité
fix: correction de bug
docs: documentation uniquement
test: ajout/modification de tests
refactor: refactoring sans nouvelle feature
chore: maintenance (deps, config, CI)
perf: amélioration de performance
```

### Style de code
- **Ruff** pour le lint et le format : `uv run ruff check . && uv run ruff format .`
- Type hints partout (Pydantic pour les données, annotations Python pour les fonctions)
- Docstrings sur les classes et fonctions publiques
- Pas de conseils hardcodés — si une métrique ne peut pas être calculée, on renvoie `None`, pas un texte inventé

### Tests
- Tout nouveau module doit avoir des tests unitaires dans `tests/unit/`
- Les intégrations VLM/modèles peuvent être mockées dans les tests
- `uv run pytest` doit passer avant tout PR

### Branches
```
main          ← stable, protégée (PR obligatoire)
feat/*        ← nouvelles features
fix/*         ← corrections
docs/*        ← documentation
refactor/*    ← refactoring
chore/*       ← maintenance
```

## Questions ?
Ouvre une issue ou contacte directement via GitHub.
