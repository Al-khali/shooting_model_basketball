# BACKLOG.md

Backlog priorisé — Phase 6 : Edge & Mobile.

> Phases 0–5b complètes. Voir [CHANGELOG.md](CHANGELOG.md) pour le détail.

---

## ✅ Phase 5b : Auth + Docker + GCP Deploy *(terminée)*

Tous les items P5-1 → P5-5 livrés. Voir [CHANGELOG.md](CHANGELOG.md) v0.7.0 → v1.0.0.

---

## 📋 Phase 6 : Edge & Mobile *(prochaine)*

### [P6-1] Export ONNX des modèles de perception
**Priorité :** HIGH

- Export `PoseEstimator` + `ShotPhaseDetector` en ONNX
- Optimisation TensorRT pour GPU embarqué
- Validation : même sorties que le modèle PyTorch original

### [P6-2] SDK mobile — capture + preprocessing
**Priorité :** HIGH

- iOS (Swift) + Android (Kotlin) SDK
- Capture vidéo + preprocessing local (resize, normalize)
- Envoi frames + métriques bruts vers API Cloud Run

### [P6-3] Mode hybride on-device / cloud
**Priorité :** MEDIUM

- Perception on-device (ONNX) + VLM/agents cloud
- Cible latence : < 200 ms perception, < 2 s feedback complet
- Fallback cloud si device trop lent

### [P6-4] Tests de performance & benchmarks
**Priorité :** MEDIUM

- Benchmark latence iPhone 15 / Pixel 8
- Profiling mémoire modèle ONNX
- CI automated perf regression test

### [P5-6] Trivy container scanning
**Priorité :** LOW

- Ajouter au `security.yml` une fois le Dockerfile créé
- Scan image pour CVEs OS-level + deps

---

## ✅ Terminé (Phases 0–5a)

Toutes les tâches des phases 0, 1, 2, 3, 4 et 5a sont closes.
Voir les issues GitHub fermées et [CHANGELOG.md](CHANGELOG.md) pour le détail.
