# ADR-001 — Architecture agents : ADK + AI Gateway, déferrer Agent Runtime

| | |
|---|---|
| **Statut** | Accepted |
| **Date** | 2026-05-18 |
| **Décideur** | Owner projet (validé) |
| **Tickets** | T1-1 (critique), T1-2 (cette ADR), Track 2 T2-3 (TaskStore Firestore), Track 3 T3-1 (AI Gateway impl) |
| **Supersedes** | Section "Recommandation finale" du `docs/agentic-frameworks-comparison.md` (révisée en parallèle) |

---

## Context

Le projet AI Shoot a choisi **Google ADK 2.0** comme framework agent en Phase 3 (PR #23, mai 2026), avec une recommandation principale `ADK + Vertex AI Agent Engine + Gemini 2.5 Flash`. Cette décision était documentée dans `docs/agentic-frameworks-comparison.md`.

L'audit T1-1 (PR #41) a montré que **3 features-clés présentées comme actives ne sont pas implémentées** :

1. **Live API streaming** (`monitor_video_stream`, `LiveRequestQueue`) — argument central du choix ADK, mais zéro usage dans `src/agents/` (`grep -rn "LiveRequestQueue" src/` → vide).
2. **Agent Runtime / Vertex AI Agent Engine** — cible de production annoncée, mais le déploiement réel est `google_cloud_run_v2_service` (T0-6 PR #36, v1.0.4). Aucune ressource `vertex_ai_agent_engine_*` dans Terraform.
3. **Session Service + Memory Bank managed** — `src/agents/memory.py` implémente en réalité un store JSON local (`data/players/*.json`) qui « mirror the interface » de Memory Bank mais ne l'utilise pas.

Par ailleurs l'audit a identifié des gaps stratégiques :

- **Aucune stratégie de fallback Gemini** (downtime, quota dur, safety filter block) — T0-3 retry/jitter gère les transients mais pas les pannes longues
- **Lock-in non gradué** : le doc qualifie tout en bloc "Moyen-haut" alors qu'il y a 3 paliers distincts (POC JSON local → managed DB → Memory Bank) avec des coûts de migration et risques très différents
- **Couche AI Gateway absente** — pattern industry-standard pour la portabilité provider, non discuté

La stack live POC v1.0.9 (validée 2026-05-18, Cloud Run `https://shoot-ai-dev-chf52ondba-uc.a.run.app`) tourne donc avec une partie significative de la cible théorique encore à construire.

Il faut maintenant une décision qui :
- **Garde** ce qui marche (ADK SDK, Gemini Flash, Cloud Run, JSON store POC)
- **Anticipe** les risques restants (provider down, scale-out perdant la mémoire JSON, migration Memory Bank)
- **Ne précipite pas** l'adoption de features non encore justifiées par un usage réel (Agent Runtime, Live API)

---

## Decision

### D1. Conserver ADK 2.0 comme orchestrateur d'agents (status quo)

Les arguments du doc original tiennent toujours pour le **SDK**, indépendamment de l'infrastructure cible :
- Intégration Gemini native (function calling, vision)
- Workflow agents (Sequential / Parallel / Loop / LlmAgent)
- License Apache 2.0
- Cadence release bi-hebdo (v1.33.0 stable)

Ce qui change : on cesse de présenter ADK comme **lié** à Agent Runtime. Le SDK seul est portable.

### D2. Déférer Agent Runtime jusqu'à preuve d'usage Live API en production

Aujourd'hui le pipeline `POST /analyze` est **batch-async** sur vidéo uploadée. Tant que ce mode reste suffisant, Cloud Run est une cible plus simple, moins chère, moins lock-in. **Critère de bascule** : implémenter d'abord un POC réel de Live API streaming dans `src/agents/` (avec `LiveRequestQueue` + `AsyncGenerator`) ; si la valeur produit justifie le coût opérationnel, alors migrer vers Agent Runtime.

Ce critère est volontairement explicite pour éviter le "cargo cult" — adopter Agent Runtime parce que c'était dans le doc original.

### D3. Introduire une couche AI Gateway (LiteLLM Proxy)

Décorréler la décision **framework** (ADK) de la décision **provider** (Gemini → Claude → GPT). Le `BaseVLMClient` actuel (`src/vlm/base.py`) est déjà l'interface logique ; l'ajout d'un proxy runtime (LiteLLM Proxy ou Vercel AI Gateway) donne :

- **Failover automatique** si Gemini quota / down / safety block bloque une analyse
- **Cost tracking** unifié par provider et par player_id
- **Prompt caching cross-provider** (réduit le coût premium runs)
- **Rate limiting global** indépendant du provider

Implémentation : nouveau ticket **Track 3 T3-1** — déployer LiteLLM Proxy en Cloud Run sidecar (ou Helicone managed), reconfigurer `GeminiFlashClient` pour pointer vers le proxy en mode passthrough, ajouter le fallback Claude Haiku 4.5 en backup. Coût marginal du proxy : ~$0-15/mois en POC (LiteLLM Proxy est open-source).

### D4. Préparer la migration Memory Bank en deux paliers (pas un saut)

Le code actuel `PlayerMemoryService` est en palier (a) **JSON local** — pas multi-instance safe (Cloud Run autoscale perd des sessions). C'est exactement ce que pointe Track 2 T2-3.

**Palier (b) intermédiaire — Firestore-backed `PlayerMemoryService`** :
- Multi-instance safe (Cloud Run scale-out OK)
- Toujours dans le free tier GCP (Firestore 1 GiB free + 50K reads/jour)
- Lock-in faible (Firestore est utilisable depuis n'importe quel runtime)
- Interface `PlayerMemoryService` inchangée (single-file change réel cette fois)

**Palier (c) éventuel — Memory Bank** :
- Si et seulement si on adopte aussi Agent Runtime (D2)
- Pour exploiter les features long-terme (apprentissage cross-sessions managed)
- Migration non triviale (data migration + quota planning + tests) — pas "single-file change" malgré ce que dit le code actuel

Ce palier (b) Firestore est tracké par **Track 2 T2-3** dans le programme v2.0.

### D5. Documenter explicitement stack actuelle vs cible

Le doc `docs/agentic-frameworks-comparison.md` est mis à jour en parallèle (cette PR) avec :

- Section **"Stack actuelle (v1.0.9)"** : ADK SequentialAgent batch + Cloud Run + JSON local + Gemini direct
- Section **"Stack cible (selon D2-D4)"** : ADK + Cloud Run + Firestore (palier b) + AI Gateway + Gemini-as-default-with-fallback
- Section **"Stack théorique premium (différée)"** : ADK + Agent Runtime + Memory Bank + Live API streaming
- Section **"Risques résiduels et mitigations"** : provider failure, autoscale memory loss, cost overrun, Live API quota

Ces 4 sections rendent la trajectoire d'adoption lisible à un opérateur extérieur.

---

## Consequences

### Positives

- **Réduction de l'incohérence doc / code** : la critique T1-1 disparaît avec la révision du doc qualif (T1-2 inclut cette révision)
- **Risque provider isolé** : un seul PR T3-1 (AI Gateway) débloque le failover Gemini → Claude/GPT, sans toucher au code agent
- **Trajectoire d'adoption claire** : chaque palier (POC JSON → Firestore → Memory Bank) est testable indépendamment, avec un go/no-go explicite
- **Pas de "cargo cult"** : Agent Runtime et Live API ne sont adoptés que si un besoin documenté les justifie, pas parce que c'était dans le doc original

### Négatives

- **Effort projet** : 3 nouveaux tickets (T3-1 AI Gateway, T2-3 Firestore PlayerMemoryService renforcée, révision doc qualif) — déjà sur le backlog programme v2.0 mais maintenant **bloquants** plutôt qu'optionnels
- **Coût additionnel marginal** : ~$0-15/mois pour AI Gateway sidecar (négligeable mais non zéro)
- **Complexité opérationnelle légère** : un proxy à monitorer en plus, configuration provider keys dans 2 endroits si on garde aussi un access direct

### Neutres

- **Pas de changement immédiat sur la stack live** — la décision est de pivoter la **trajectoire**, pas de tout réécrire. La v1.0.9 reste valide jusqu'à ce que les paliers suivants soient livrés

### Mitigations des nouvelles dépendances

- **AI Gateway (LiteLLM Proxy)** : open-source MIT-licensed. Tournable en self-hosted ou en backup remplaçable par n'importe quel autre proxy compatible OpenAI-API (Helicone, OpenRouter)
- **Firestore palier (b)** : restreint au free tier au POC. Migration ultérieure vers Cloud SQL / Spanner reste possible (Firestore expose une API standard documentée)
- **Tests d'intégration** : doivent couvrir le path `AI Gateway → provider down → failover successful → status=done with model_used="claude-haiku-4-5"` — recouvre Track 4 T4-1 (VLM tests)

---

## Status

**Accepted** — appliqué dans cette PR :

- ADR-001 écrit dans `docs/adr/001-agentic-architecture.md` (ce fichier)
- `docs/agentic-frameworks-comparison.md` révisé : section "Recommandation finale" mise à jour pour pointer vers cette ADR + nouvelles sections **"Stack actuelle vs cible"** et **"Risques résiduels et mitigations"**

Suivi opérationnel :
- **Track 3 T3-1** (à ouvrir) : POC AI Gateway LiteLLM Proxy en sidecar Cloud Run
- **Track 2 T2-3** (déjà au backlog) : Firestore PlayerMemoryService — bumper en P1 après T1-2
- **Track 0 T0-14** : sync pyproject version au bump CHANGELOG (le `/health` retourne encore 1.0.3 alors que CHANGELOG est à 1.0.10)
- **Track 6 dédiée** (déjà au roadmap) : si la valeur produit Live API streaming est confirmée, alors démarrer ADR-002 sur Agent Runtime

---

## Références

- [T1-1 PR #41](https://github.com/Al-khali/shooting_model_basketball/pull/41) — Critique structurée du doc qualif (10 findings)
- [T0-15 PR #40](https://github.com/Al-khali/shooting_model_basketball/pull/40) — Cloud Run cpu_idle false (a révélé l'incompatibilité bg task / autoscale)
- [Programme v2.0](../../.claude/plans/) — plan complet (7 tracks)
- [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start) — AI Gateway open-source recommandé
- [Vercel AI Gateway](https://vercel.com/docs/ai-gateway) — alternative managed
- [Google ADK 2.0](https://google.github.io/adk-docs/) — documentation officielle (status quo D1)
- [Firestore free tier](https://cloud.google.com/firestore/docs/quotas#free_quota) — base du palier (b) Memory Bank intermédiaire
