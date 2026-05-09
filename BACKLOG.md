# BACKLOG.md

Backlog priorisé — Phase 4 : API production-ready.

> Phases 0–3 complètes. Voir [CHANGELOG.md](CHANGELOG.md) pour le détail de ce qui a été livré.

---

## 🔄 En cours — Phase 4 : API

### [P4-1] FastAPI production-ready (async + WebSocket)
**GitHub issue:** #19

- `POST /analyze` — upload vidéo multipart, lance le pipeline en background task
- `GET /session/{id}` — polling sur l'état + résultats
- `WebSocket /analyze/stream` — streaming temps réel (frames reçues + analyse en cours)
- `GET /health` — liveness + readiness (dépendances optionnelles)
- Gestion d'erreurs structurée (codes HTTP cohérents, messages JSON)
- Tests d'intégration FastAPI (TestClient)

### [P4-2] Vue coach — historique et progression joueur
**GitHub issue:** #20

- `GET /player/{id}/history` — liste des sessions avec métriques clés
- `GET /player/{id}/trends` — évolution des issues récurrentes + drills assignés
- Agrégation `PlayerMemoryService` → réponse structurée
- Tests unitaires sur les routes coach

---

## 📋 Backlog moyen terme — Phase 5

- Export ONNX des modèles de perception (YOLOv11-pose)
- Packaging Docker + docker-compose
- Auth API key simple (header `X-API-Key`)
- Rate limiting + quotas par joueur

---

## ✅ Terminé (Phases 0–3)

Toutes les tâches des phases 0, 1, 2 et 3 sont closes.
Voir les issues GitHub fermées et [CHANGELOG.md](CHANGELOG.md) pour le détail.
