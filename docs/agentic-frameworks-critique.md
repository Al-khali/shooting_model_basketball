# Agentic Frameworks Comparison — Critique technique

> **Statut** : critique structurée du doc `agentic-frameworks-comparison.md` (735 lignes, mai 2026), produite dans le cadre du programme d'enrichissement v2.0 (Track 1 T1-1).
>
> **Méthode** : pour chaque finding, confronter une affirmation du doc avec **(a) le code réel du projet shippé** et **(b) la documentation officielle des frameworks cités**. Aucune critique sans preuve. Inspiré du protocole de challenge de findings Gemini Code Assist défini dans `CLAUDE.md` (§challenger les findings des reviewers).
>
> **Verdict global** : le doc est bien sourcé pour les comparaisons de coûts et de licences, et le choix d'ADK est **défensable**. Mais plusieurs claims présentent une feature (Live API, Session Service, Memory Bank) comme déjà active dans la stack alors que **rien dans le code shippé ne les utilise** — la recommandation est en réalité plus modeste que ne le suggère le doc. Cette critique vise à faire converger doc et réalité, et à expliciter les risques résiduels.

---

## Findings (par sévérité)

### 🔴 Finding 1 — Le streaming vidéo Live API n'est PAS implémenté (HIGH)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | "`monitor_video_stream()` avec `LiveRequestQueue` + `AsyncGenerator` est **exactement** ce que l'analyse basketball temps réel nécessite — aucun autre framework ne propose ça en production" (lignes 244-256, 665) | Aucune occurrence dans `src/` |
| **Preuve** | `grep -rn "LiveRequestQueue\|run_live\|monitor_video_stream\|AsyncGenerator" src/` → **vide** | `src/agents/orchestrator.py:create_adk_pipeline()` construit 4 `LlmAgent` standard avec `tools=[adk_tool(extract_shot_frames), ...]`. Aucun `BaseAgent` avec `_run_async_impl`, aucune `LiveRequestQueue` |
| **Impact** | Le doc présente Live API comme **le** différenciateur d'ADK pour AI Shoot. C'est l'argument principal de la recommandation (cité 3 fois). Mais en pratique le projet utilise `POST /analyze` → background task → pipeline synchrone batch sur vidéo uploadée. **Le streaming temps réel reste un objectif Phase 6, pas une feature livrée.** |
| **Recommandation** | Soit (a) implémenter un POC `LiveRequestQueue` pour justifier le claim, soit (b) reformuler la recommandation : ADK choisi pour son **future-proofing** Live API, pas pour une utilisation actuelle |

---

### 🔴 Finding 2 — "Agent Runtime serverless" pas déployé — c'est Cloud Run standard (HIGH)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | "**Agent Runtime** = zéro gestion d'infra — serverless, auto-scaling" (ligne 290) — l'infrastructure de production cible est Agent Runtime (Vertex AI Agent Engine) | `infra/terraform/cloud_run.tf` : `google_cloud_run_v2_service`. Pas de `google_vertex_ai_agent_engine_*`. Pas de mention Agent Runtime dans Terraform |
| **Preuve** | `grep -rn "agent_engine\|AgentEngine\|vertex_ai_agent" infra/` → **vide**. Le projet déploie sur Cloud Run v2 (POC zero-budget choix T0-6, voir CHANGELOG v1.0.4) | `src/agents/orchestrator.py:43` (docstring) : *"In production this maps to an ADK SequentialAgent with Agent Runtime's Session Service and Memory Bank."* — c'est un commentaire intentionnel ("**maps to**"), pas une assertion d'exécution |
| **Impact** | Les chiffres de coût `$10-35/mois` (ligne 311, table p.616) supposent Agent Runtime. En réalité on paie : Cloud Run requests + CPU-seconds + Gemini API + Artifact Registry. Pour le POC c'est OK et **encore moins cher**, mais le doc raconte une stack qui n'est pas la stack |
| **Recommandation** | Ajouter une section "stack actuelle (Cloud Run) vs stack cible (Agent Runtime)" — pratiques opérationnelles et coûts diffèrent (Agent Runtime facture aussi du compute storage par session active, Cloud Run non) |

---

### 🔴 Finding 3 — "Session Service / Memory Bank intégrés" — implémentés en JSON local (HIGH)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | "**Session Service** intégré : mémoire per-joueur sans Redis/Postgres custom" + "**Memory Bank** : historique d'amélioration joueur cross-sessions" (lignes 291-292) | `src/agents/memory.py:1-5`: *"This module implements the POC in-memory + JSON file store that **mirrors the interface** of Google ADK Memory Bank."* + `DEFAULT_STORE_DIR = Path("data/players")` |
| **Preuve** | `ls data/players/` → fichiers JSON par joueur. Aucun appel à `vertexai.agent_engines` ou équivalent | `PlayerMemoryService.build_context(player_id)` lit/écrit du JSON. Le commentaire dit *"Swapping to ADK Agent Runtime's Memory Bank... is a single-file change"* — c'est la stratégie, pas l'implémentation |
| **Impact** | (a) Le lock-in présenté comme "Moyen-haut" parce que "Session Service + Memory Bank créent une adhérence GCP" (ligne 316) est **factuel pour la cible**, **mais nul pour le code shippé** — le code est portable trivialement vers n'importe quel KV store. (b) Cette mémoire JSON locale ne survit pas à un redémarrage Cloud Run multi-instance (autoscaling) — c'est exactement le problème surfaceé dans Track 2 T2-3 |
| **Recommandation** | Clarifier dans le doc : POC = JSON local (= aucun lock-in, **mais pas multi-instance safe**) → Phase X = Firestore/Cloud SQL (= lock-in marginal sur GCP managed DB) → Phase Y = Memory Bank (= lock-in maximum sur Agent Runtime). Les 3 paliers ont des trade-offs très différents qui méritent d'être nommés |

---

### 🟠 Finding 4 — Lock-in sous-évalué OU sur-évalué selon le palier (MEDIUM)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | "Vendor lock-in : **Moyen-haut**" (ligne 316) — formulation unique pour ADK | Le lock-in dépend du *palier d'adoption* : (a) ADK + LiteLLM + state custom = lock-in **faible** ; (b) ADK + Session Service in-VPC = lock-in **moyen** ; (c) ADK + Agent Runtime + Memory Bank = lock-in **haut** |
| **Preuve** | Ligne 638 : *"Plus difficile à quitter : ADK (Session/Memory services créent une adhérence GCP)"* — contradit la qualification "moyen-haut" en disant que c'est "le plus difficile à quitter" | Le projet shippé est en palier (a) — lock-in factuel **faible** sur du code (juste ADK SDK), pas de cloud lock-in (Cloud Run + Cloud Storage sont remplaçables par EKS + S3 trivialement) |
| **Impact** | Décision archi opérée sur un lock-in "moyen-haut" alors que la pratique actuelle est faible. Le risque émergera **quand** on activera Agent Runtime — pas avant |
| **Recommandation** | Tableau 3-paliers explicite + identifier le palier visé pour chaque environnement (dev / staging / prod) |

---

### 🟠 Finding 5 — Aucune discussion de fallback Gemini (downtime/quota/safety) (MEDIUM)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | Le doc compare les **forces** de Gemini Flash (vidéo native, prix bas) mais ne mentionne pas le risque opérationnel | Pour un produit live, **un seul provider VLM = single point of failure**. Vertex AI a eu plusieurs incidents documentés en 2024-2026 (incidents.cloud.google.com archives) |
| **Preuve** | Code T0-3 livré (PR #34, v1.0.5) : `src/vlm/gemini_client.py` a retry exponentiel + jitter sur transient errors, mais **pas de fallback provider** — quota dur exhausted = pipeline DOWN | Aucune section "Disaster recovery" ou "Provider availability" dans le doc qualif |
| **Impact** | En cas de panne Vertex AI > retry budget : tous les `/analyze` retournent `coaching_failed:VLMError`. Pas de dégradation gracieuse, pas de basculement Claude/GPT-4V |
| **Recommandation** | Section dédiée "Stratégie multi-provider + AI Gateway" — Vercel AI Gateway, LiteLLM Proxy, ou OpenRouter permettent de basculer en quelques secondes. Coût d'intégration : 1 PR. Coût d'absence : panne complète si Gemini est down |

---

### 🟠 Finding 6 — Qwen2-VL "supporté via LiteLLM" — non démontré en streaming (MEDIUM)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | Ligne 561 : *"Intégration Qwen2-VL : ✅ Via LiteLLM ou BaseAgent custom"* — présenté comme path alternatif évident | Aucun exemple shippé. Le `pyproject.toml` n'inclut ni `litellm` ni client Qwen. `src/vlm/base.py:BaseVLMClient` est l'abstraction mais seul `GeminiFlashClient` la concrétise |
| **Preuve** | `grep -rn "qwen\|litellm" src/ pyproject.toml` → **vide** | LiteLLM **supporte Qwen2-VL en mode standard** (input image), mais **pas en mode streaming vidéo bidirectionnel comme Live API**. Le doc ne fait pas la distinction |
| **Impact** | Si Gemini Live API devient prohibitive, le path "basculer Qwen2-VL" qui est censé être trivial demanderait en réalité de **réécrire toute la couche streaming**. Ce n'est plus "via LiteLLM" — c'est un projet à part |
| **Recommandation** | Reformuler : "Qwen2-VL supporté pour les modes async/batch (frames statiques), pas pour streaming live. Le streaming live est Gemini-only à ce jour" |

---

### 🟠 Finding 7 — Comparaison de coûts apples-to-oranges (MEDIUM)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | Table p.616 : `ADK + Gemini Flash $10-30` vs `LangGraph + LangSmith $206` — facteur **7-20×** | ADK ligne = self-hosted (Cloud Run, free tier Storage). LangGraph ligne inclut LangSmith Plus + Prod deployment **always-on** ($155/mois) |
| **Preuve** | Le doc ligne 145-148 explicite : *"Prod deployment uptime $0.0036/min = ~$155/mois en always-on"* — c'est de l'**uptime billing**. LangGraph est **MIT-licensed et self-hostable** (Pregel runtime) | Comparable équitable serait : ADK self-hosted vs LangGraph self-hosted (~$5-15 modèle, ~$10-30 Cloud Run/EKS) — facteur réel **2-3×** au lieu de 7-20× |
| **Impact** | La recommandation ADK est sur-vendue économiquement. C'est toujours moins cher, mais l'avantage est moins flagrant qu'annoncé |
| **Recommandation** | Ajouter une ligne "LangGraph self-hosted (Pregel runtime, no LangSmith)" dans la table — coût ~$15-45/mois (modèle + infra). Le delta vrai vs ADK ressort propre |

---

### 🟡 Finding 8 — Pas de couche AI Gateway / proxy provider (LOW)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | Aucune mention | Pattern industry-standard 2025-2026 : interposer une couche **AI Gateway** (Vercel AI Gateway, Helicone, LiteLLM Proxy, OpenRouter) entre app et provider |
| **Preuve** | Bénéfices documentés : (a) failover transparent provider→provider ; (b) cost tracking unifié ; (c) prompt caching cross-provider ; (d) audit/observability ; (e) rate limiting global | Le doc traite **provider-agnostic** comme une property du framework (LangGraph, OpenAI SDK) — mais ignore que le **runtime API gateway** offre la même portabilité au-dessus de n'importe quel framework |
| **Impact** | On peut très bien faire ADK + AI Gateway et obtenir simultanément (a) la qualité ADK + (b) la portabilité provider d'un OpenAI SDK. Le doc traite ces deux dimensions comme exclusives |
| **Recommandation** | Section "Couche d'isolation provider" — recommander un AI Gateway en plus du framework choisi |

---

### 🟡 Finding 9 — Le Memory Bank "interface mirror" est plus fragile qu'annoncé (LOW)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | "Swapping to ADK Agent Runtime's Memory Bank... is a single-file change" (`src/agents/memory.py:30`) | C'est techniquement vrai pour l'interface (`build_context`, `record_feedback`), mais : (a) la sérialisation JSON locale est **lossless** par construction ; Memory Bank impose ses propres types et limites de taille. (b) Les rate limits Memory Bank (Agent Runtime quota) sont différents d'un disk write. (c) Le coût bascule de `$0` à un usage facturé |
| **Preuve** | Doc Vertex AI Memory Bank cite quotas + pricing | Le "single-file change" est l'**interface code**. Le **comportement opérationnel** demande tests, observabilité, migration de data |
| **Impact** | Faible aujourd'hui (POC), mais à anticiper avant prod : la migration Memory Bank n'est pas "single-file" en pratique |
| **Recommandation** | Documenter explicitement les étapes de migration : (1) test compatibility, (2) data migration JSON→Memory Bank, (3) feature flag rollout, (4) monitoring tests |

---

### 🟡 Finding 10 — Bench production pas reproductible (LOW)

| | Doc | Réalité shippée |
|---|---|---|
| **Claim** | Coût `$10-30/mois pour 1000 analyses` est précis sans détailler les hypothèses | Hypothèses non explicitées : taille vidéo (durée, résolution), nombre de frames sample, taille des prompts, hits cache Gemini, latence acceptable |
| **Preuve** | Pour `testsrc 2s 640x480 @ 30fps` (utilisé en T0-15 live test) : ~60 frames × 1 seul appel Gemini (un summary par shot) → ~5K tokens output × $2.50/M = `~$0.0125 par analyse`. Sur 1K analyses ≈ `$12.50`. La table tient pour ce profil. Pour une analyse "1080p 10s match complet 60 fps sample 1/3" → différent par 10-30× |
| **Impact** | Le coût final dépend de la définition de "1 analyse". Un produit qui scale = profils mixtes (entraînement court vs match) |
| **Recommandation** | Tableau coûts en 3 profils : `(a) shot court ≤5s low-res, (b) shot HD 1080p, (c) match analysis 10min sample 1/30`. Coûts varient sur 1-2 ordres de magnitude entre les 3 |

---

## Résumé exécutif

| # | Finding | Sévérité | Code shippé contredit le doc ? |
|---|---------|----------|-------------------------------|
| 1 | Live API streaming pas implémenté | 🔴 HIGH | Oui (`grep` vide sur LiveRequestQueue) |
| 2 | Agent Runtime pas déployé (Cloud Run à la place) | 🔴 HIGH | Oui (`google_cloud_run_v2_service` dans Terraform) |
| 3 | Session/Memory Bank = JSON local, pas managed | 🔴 HIGH | Oui (`PlayerMemoryService` lit `data/players/*.json`) |
| 4 | Lock-in qualification "moyen-haut" trompeuse | 🟠 MEDIUM | Indirect (palier de lock-in non explicité) |
| 5 | Aucune stratégie de fallback provider | 🟠 MEDIUM | N/A (gap dans le doc, pas dans le code) |
| 6 | Qwen2-VL "via LiteLLM" pas démontré en streaming | 🟠 MEDIUM | Oui (`grep` vide sur litellm/qwen) |
| 7 | LangGraph $206 vs ADK $30 — comparaison apples-to-oranges | 🟠 MEDIUM | N/A (gap dans la méthodologie) |
| 8 | AI Gateway / proxy provider absent | 🟡 LOW | N/A (gap) |
| 9 | Memory Bank "single-file change" sous-évalué | 🟡 LOW | Partiel (interface OK, opérations à anticiper) |
| 10 | Bench coût pas reproductible | 🟡 LOW | N/A (gap méthodologique) |

**3 findings HIGH** où le doc présente comme actif ce qui est encore un commentaire docstring ("maps to..."). Les **3 fixes immédiats** pour aligner doc et réalité sont :

1. **Préciser dans le doc** que la stack actuelle est *(Cloud Run + JSON local + ADK SequentialAgent batch)*, et que la stack cible est *(Agent Runtime + Memory Bank + Live API streaming)* — c'est la différence entre un POC fonctionnel et la "platform 10× better" promise.
2. **Ajouter une section "Risques résiduels"** (à intégrer dans la révision T1-2) qui liste les paliers de lock-in, la stratégie multi-provider, et les conditions de bascule entre paliers.
3. **Tracker un ADR-001** (T1-2) qui fige la décision archi : *garder ADK pour le SDK + l'écosystème, mais introduire un AI Gateway pour découpler la décision provider, et différer l'adoption d'Agent Runtime jusqu'à preuve d'usage du streaming Live API*.

---

## Verdict

**Le choix ADK n'est pas remis en cause.** Pour les bonnes raisons (intégration Gemini native, future-proofing Live API, license Apache 2.0, courbe d'apprentissage acceptable), c'est un choix défensable. La critique porte sur la **présentation** : le doc vend une stack premium qui n'est pas déployée, et minimise des risques opérationnels réels (fallback provider, lock-in graduel, comparaison de coûts).

La révision T1-2 (ADR-001) doit explicitement formaliser :
- **Garder ADK** comme orchestrateur d'agents
- **Différer Agent Runtime** jusqu'à preuve d'usage Live API en prod
- **Introduire un AI Gateway** (LiteLLM Proxy ou Vercel AI Gateway) pour la couche provider
- **Préparer la migration Memory Bank** avec un palier Firestore intermédiaire (multi-instance safe)
- **Documenter explicitement** la stack actuelle vs cible

Cette critique a été produite suivant le protocole de challenge défini dans `CLAUDE.md` (challenger toute affirmation avec preuve par le code ou la doc officielle, jamais sur foi).
