# Security Policy

## Versions supportées

| Version | Support |
|---------|---------|
| 0.1.x   | ✅ Active |

## Signaler une vulnérabilité

**Ne pas ouvrir une issue publique pour une faille de sécurité.**

Contacte directement le mainteneur via GitHub (profil @Al-khali) avec :
- Description de la vulnérabilité
- Étapes pour reproduire
- Impact potentiel

Tu recevras une réponse sous 72h.

## Bonnes pratiques

- Ne jamais committer de clés API ou secrets (utiliser `.env`, voir `.env.example`)
- Les données des joueurs/vidéos ne doivent jamais être commitées dans le repo
- Les modèles entraînés sont gérés via DVC, pas Git
