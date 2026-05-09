# Agentic AI Frameworks — Étude Comparative 2026

> **Contexte :** Ce document compare les écosystèmes agentics majeurs pour le projet **AI Shoot** — plateforme d'analyse vidéo basketball en temps réel. Il couvre LangChain/LangGraph, Google ADK 2.0, Anthropic, OpenAI, et les frameworks indépendants.
>
> **Auteur :** Analyse produite via recherche approfondie — sources citées en fin de document.
> **Date :** Mai 2026

---

## Table des matières

1. [Résumé exécutif](#résumé-exécutif)
2. [LangChain / LangGraph](#1-langchain--langgraph)
3. [Google ADK 2.0](#2-google-adk-20)
4. [Anthropic](#3-anthropic)
5. [OpenAI](#4-openai)
6. [Frameworks indépendants](#5-frameworks-indépendants-crewai-autogen-smolagents)
7. [ADK vs LangGraph — Head-to-Head](#adk-vs-langgraph--head-to-head)
8. [Comparaison des coûts](#comparaison-des-coûts--1000-analyses-mois)
9. [Matrice de sélection](#matrice-de-sélection)
10. [Recommandation finale pour AI Shoot](#recommandation-finale-pour-ai-shoot)
11. [Sources](#sources)

---

## Résumé exécutif

**Recommandation principale : Google ADK + Vertex AI Agent Engine + Gemini 2.5 Flash**

LangGraph reste un excellent choix secondaire si la flexibilité model-agnostic est prioritaire. Voici pourquoi ADK gagne pour AI Shoot :

| Critère | ADK | LangGraph |
|---------|-----|-----------|
| Streaming vidéo temps réel | ✅ Natif (Gemini Live API) | ⚠️ À construire |
| Mémoire session par joueur | ✅ Built-in (Session Service) | ⚠️ DIY (Redis/Postgres) |
| Coût 1000 analyses/mois | **~$10–35** | **~$206** |
| Stabilité | v1.33.0 stable | v1.2.0a7 **alpha** |
| Lock-in | Moyen-haut (GCP) | Moyen (LangSmith) |

---

## 1. LangChain / LangGraph

### État actuel (2025-2026)

LangGraph est un **runtime d'orchestration bas-niveau** pour agents stateful multi-acteurs, construit sur le modèle distribué Pregel/Apache Beam. Version actuelle : **v1.2.0a7 (alpha)**.

La famille produit LangChain s'est divisée :
- **LangGraph** — runtime d'orchestration (durable execution, streaming, HITL)
- **LangChain** — intégrations et composants (modèles, outils, chaînes)
- **LangSmith** — observabilité + plateforme de déploiement hébergé (ex-LangGraph Platform)
- **Deep Agents** — harness haut-niveau sur LangGraph

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      LangGraph Ecosystem                     │
│                                                              │
│  LangSmith Cloud (Déploiement + Observabilité)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  LangSmith Studio (visual debugger / prototypage)      │  │
│  │  LangSmith Fleet (no-code agent builder)               │  │
│  │  LangSmith Deployment (LangGraph server, 30+ API ends) │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                  │
│  LangGraph Core (Python/JS)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ StateGraph   │  │ Prebuilt     │  │  Checkpointers   │   │
│  │ (DAG nodes,  │  │ react_agent, │  │  SQLite,         │   │
│  │  edges,      │  │ supervisor,  │  │  Postgres,       │   │
│  │  routing     │  │ swarm        │  │  Redis           │   │
│  │  conditionnel│  │              │  │  in-memory       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         │                                                    │
│  LangChain Integrations (100+ modèles)                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ ChatGoogleGenerativeAI, ChatAnthropic, ChatOpenAI... │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Patterns multi-agents

- **Supervisor** : LLM central décide quel sous-agent appeler. Visibilité complète sur l'état.
- **Swarm** : Agents se passent la main peer-to-peer selon spécialisation. Pas d'orchestrateur central.
- **DAG custom** : N'importe quelle topologie de graphe est exprimable.

### Streaming (7 modes)

LangGraph a le système de streaming le plus sophistiqué de tout l'écosystème :

```python
for chunk in graph.stream(
    {"topic": "shot_analysis"},
    stream_mode=["updates", "messages", "custom"],
    version="v2",  # format StreamPart unifié
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]   # token streaming LLM
    elif chunk["type"] == "updates":
        pass  # changements d'état des nœuds
    elif chunk["type"] == "custom":
        pass  # données de progression arbitraires
```

Modes disponibles : `values`, `updates`, `messages`, `custom`, `checkpoints`, `tasks`, `debug`

### Pricing LangSmith

| Tier | Coût | Inclus |
|------|------|--------|
| Developer | Gratuit | 5k traces/mois, 1 siège |
| Plus | $39/siège/mois | 10k traces + 1 déploiement dev |
| Runs de déploiement | $0.005/run | Par invocation (prod) |
| Dev deployment uptime | $0.0007/min | ~$30/mois en always-on |
| Prod deployment uptime | $0.0036/min | ~$155/mois en always-on |
| Enterprise | Sur devis | VPC/hybrid, SSO, SLA |

> Programme startup disponible pour entreprises VC-backed.

### Forces

- Observabilité inégalée via LangSmith (visualisation des traces, chemins d'exécution)
- Topologie multi-agents la plus flexible — n'importe quel graphe exprimable
- 7 modes de streaming — idéal pour UI progressive et mises à jour temps réel
- Model-agnostic — même code fonctionne avec Gemini, Claude, GPT-4, Qwen2-VL
- Grand écosystème (100k+ téléchargements PyPI/jour), communauté mature (Klarna, Replit, Elastic)
- Human-in-the-loop natif (interrupt, resume, édition d'état)

### Faiblesses

- **v1.2.0a7 encore en alpha** — breaking changes attendus
- Courbe d'apprentissage élevée sur l'abstraction de graphe
- Pas de pipeline vidéo/multimodal natif — à construire from scratch
- LangSmith Deployment cher pour always-on : ~$155+/mois prod
- LangSmith Live API Gemini non supporté (streaming vidéo bidirectionnel impossible)

### Coût estimé : 1000 analyses/mois

| Poste | Estimation |
|-------|-----------|
| Modèle (Gemini 2.5 Flash via LangChain) | ~$5.80 |
| LangSmith Plus (1 siège) | $39 |
| Prod deployment always-on | $155 |
| Runs (1000 × $0.005) | $5 |
| **Total** | **~$205/mois** |

**Vendor lock-in : Moyen** — LangGraph MIT-licensed (self-hostable), LangSmith propriétaire.

---

## 2. Google ADK 2.0

### État actuel

ADK est le **framework open-source (Apache 2.0) de Google** pour construire des agents de production, sorti en 2025. Version stable actuelle : **v1.33.0** (8 mai 2026).

Disponible en Python, TypeScript, Go et Java. Conçu pour le déploiement enterprise via **Agent Runtime (ex-Vertex AI Agent Engine)**.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Google ADK Ecosystem                           │
│                                                                  │
│  Google Cloud Agent Platform                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Agent Runtime (fully managed, serverless)               │   │
│  │  ├── Session Service (état per-joueur/per-session)       │   │
│  │  ├── Memory Bank (mémoire persistante long-terme)        │   │
│  │  ├── Code Execution Sandbox (AgentEngineSandbox)         │   │
│  │  ├── Cloud Trace + Monitoring (observabilité incluse)    │   │
│  │  └── IAM + Secret Manager (sécurité)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ADK Core Framework                                              │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ Agent Types   │  │ Workflow Agents  │  │  Live API        │   │
│  │ LlmAgent      │  │ Sequential      │  │  Toolkit         │   │
│  │ BaseAgent     │  │ Parallel        │  │  (vidéo/audio    │   │
│  │ (logique      │  │ Loop            │  │   bidirectionnel, │   │
│  │  custom)      │  │                 │  │   WebSocket)      │   │
│  └───────────────┘  └─────────────────┘  └──────────────────┘   │
│         │                                                        │
│  Outils (built-in + extensible)                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ google_search, BigQuery, Spanner, GCS, Pub/Sub           │   │
│  │ MCP Toolset, OpenAPI specs, Code Executor, A2A protocol  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│  Modèles                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Gemini 2.5 Flash/Pro/Flash Lite (natif, première classe)  │   │
│  │ Anthropic Claude (via extensions), LiteLLM (100+ models)  │   │
│  │ LangGraph agents (via adaptateur LangGraphAgent)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

Protocole A2A (communication cross-framework)
┌───────────────────────────────────────────────────────────────┐
│ JSON-RPC 2.0 over HTTP(S) │ Agent Cards │ SSE Streaming       │
│ ADK ↔ LangGraph ↔ BeeAI ↔ tout agent conforme               │
└───────────────────────────────────────────────────────────────┘
```

### Patterns multi-agents (3 types)

**1. Délégation par LLM (sous-agents)**
```python
coordinator = LlmAgent(
    name="ShotOrchestrator",
    model="gemini-2.5-flash",
    sub_agents=[shot_detector, pose_analyzer, feedback_agent]
    # Le LLM décide lui-même quel sous-agent appeler
)
```

**2. Workflow déterministe**
```python
# Séquentiel : extraction → analyse → classification
pipeline = SequentialAgent(sub_agents=[extractor, analyzer, classifier])

# Parallèle : analyse bras + jambes simultanément
parallel = ParallelAgent(sub_agents=[arm_agent, leg_agent])

# Boucle : raffiner jusqu'au seuil de confiance atteint
loop = LoopAgent(sub_agents=[refinement_agent], max_iterations=5)
```

**3. BaseAgent custom (logique non-LLM)**
```python
class VideoFrameAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        # Python pur : OpenCV + YOLOv8, aucun appel LLM
        yield Event(...)
```

### Streaming — Gemini Live API Toolkit

ADK v0.5.0+ intègre le **Gemini Live API Toolkit** — système de streaming vidéo/audio bidirectionnel en production. **C'est ce qui différencie ADK pour l'analyse sportive temps réel :**

```python
async def analyze_shot_stream(
    input_stream: LiveRequestQueue,  # flux vidéo live de la caméra
) -> AsyncGenerator[str, None]:
    """Analyse les tirs basket frame par frame en temps réel."""
    while True:
        frame = await input_stream.get()
        if frame.blob and frame.blob.mime_type == "image/jpeg":
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=[image_part, text_part]
            )
            yield response  # résultats streamés vers l'agent
        await asyncio.sleep(0.5)
```

Fonctionnalités :
- WebSocket bidirectionnel faible latence (`LiveRequestQueue` + `run_live()`)
- Traitement texte, audio **et vidéo** en simultané
- Détection d'interruption vocale
- Streaming tools avec `AsyncGenerator` (monitore un flux vidéo en continu)

### Protocole A2A

ADK intègre nativement le **protocole A2A** (Linux Foundation, Apache 2.0) :
- Communication cross-framework : ADK ↔ LangGraph ↔ BeeAI
- **Agent Cards** pour la découverte de capacités
- SSE streaming pour tâches longues
- Les agents collaborent sans exposer leur état interne

```bash
pip install "google-adk[a2a]"
```

### Tarifs Gemini (Vertex AI)

| Modèle | Usage AI Shoot | Input/M tokens | Output/M tokens |
|--------|---------------|---------------|----------------|
| Gemini 2.5 Flash | Agent d'analyse principal | $0.30 | $2.50 |
| Gemini 2.5 Flash Lite | Pré-screening, détection pose | $0.10 | $0.40 |
| Gemini 2.5 Flash (Live API) | Streaming temps réel | $3.00 (vidéo) | $2.00 |
| Gemini 2.5 Pro | Feedback coaching premium | $1.25 | $10.00 |

### Forces

- **Meilleure intégration Gemini** — Live API, streaming vidéo, function calling natif
- **Gemini Live API Toolkit** est exactement ce que l'analyse basketball temps réel nécessite
- **Agent Runtime = zéro gestion d'infra** — serverless, auto-scaling
- **Session Service** intégré : mémoire per-joueur sans Redis/Postgres custom
- **Memory Bank** : historique d'amélioration joueur cross-sessions
- **OpenTelemetry intégré** — pas d'abonnement supplémentaire pour l'observabilité
- Apache 2.0 — plus permissif pour produits commerciaux
- v1.33.0 stable, cadence bi-hebdomadaire

### Faiblesses

- **Dépendance Google Cloud** pour les fonctionnalités production (Agent Runtime, Memory Bank)
- Écosystème moins mature que LangGraph (communauté plus petite)
- Streaming tools requis Gemini Live API — Qwen2-VL non supporté nativement dans le chemin streaming
- Developer tooling (ADK Web UI) encore tôt-stade
- Install plus lourd — `google-adk` tire 30+ packages Google Cloud SDK

### Coût estimé : 1000 analyses/mois

| Poste | Estimation |
|-------|-----------|
| Gemini 2.5 Flash (standard, 5 frames/analyse) | ~$2–5 |
| Agent Runtime compute | ~$5–20 |
| **Total (standard)** | **~$10–30/mois** |
| Avec Live API (streaming temps réel) | **~$15–35/mois** |

> **7–20× moins cher que LangGraph + plateforme comparable.**

**Vendor lock-in : Moyen-haut** — ADK est open-source et tourne en local/Docker, mais Agent Runtime, Session Service, Memory Bank créent une adhérence GCP. Le protocole A2A offre une sortie de secours.

---

## 3. Anthropic

### État actuel

Anthropic **ne propose pas de framework agent ni de runtime hébergé**. L'écosystème comprend :
- **Claude API** (Messages API) — tool use, vision, extended thinking
- **Model Context Protocol (MCP)** — standard ouvert pour la connectivité outils/contexte
- **Prompt caching** — réduction des coûts sur contexte répété
- **Batch API** — 50% de réduction pour traitement asynchrone

### Modèles Claude (2025-2026)

| Modèle | Input/MTok | Output/MTok | Contexte | Usage |
|--------|-----------|------------|---------|-------|
| Claude Opus 4.7 | $5 | $25 | 1M tokens | Raisonnement complexe, codage agentique |
| Claude Sonnet 4.6 | $3 | $15 | 1M tokens | Équilibre vitesse + intelligence |
| Claude Haiku 4.5 | $1 | $5 | 200K tokens | Volume élevé, faible latence |

Prompt caching : **10% du prix d'entrée** sur les hits cache — très bénéfique pour les profils joueurs répétés.

### MCP — Ce que ça permet

MCP est le **USB-C de l'IA** — standard ouvert (JSON-RPC 2.0) pour connecter Claude (et tout client conforme) à des sources de données et outils externes :

- Claude → base de données vidéo basket → récupère l'historique des tirs
- Claude → outils OpenCV custom → extraction de frames
- **Compatible ADK, OpenAI Agents SDK, et Claude simultanément** (standard convergent)

### Pattern Claude comme orchestrateur

```python
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-opus-4-7",
    tools=[frame_extractor_tool, pose_analyzer_tool, shot_classifier_tool],
    messages=[{"role": "user", "content": "Analyse ce tir basket"}]
)
# Claude retourne des blocs tool_use → vous exécutez → retour tool_result
# Claude décide ensuite du prochain appel outil (boucle agentique)
```

### Architecture pour AI Shoot

```
┌────────────────────────────────────────────┐
│          Backend Python (BYO)              │
│  ┌──────────────────────────────────────┐  │
│  │  Boucle agentique (hand-rolled)      │  │
│  │  ┌────────────────────────────────┐  │  │
│  │  │  Claude Opus 4.7 (orchestrateur│  │  │
│  │  │  - Décide quel outil appeler   │  │  │
│  │  │  - Interprète l'analyse vidéo  │  │  │
│  │  │  - Génère le texte coaching    │  │  │
│  │  └────────────────────────────────┘  │  │
│  │         │                            │  │
│  │  ┌──────┴────┐  ┌────────────────┐   │  │
│  │  │ OpenCV    │  │  MCP Server    │   │  │
│  │  │ Tool      │  │  (outils       │   │  │
│  │  │           │  │   customs)     │   │  │
│  │  └───────────┘  └────────────────┘   │  │
│  └──────────────────────────────────────┘  │
│  BYO : Redis/Postgres (état de session)    │
│  BYO : Déploiement (Cloud Run / EC2 / ...) │
└────────────────────────────────────────────┘
```

### Forces

- **Meilleure qualité de raisonnement** pour la génération de feedback coaching complexe
- **Prompt caching** réduit fortement les coûts sur contextes répétés (profils joueurs)
- **1M tokens de contexte** — historique complet de match dans un seul appel
- **Tool use strict** — sorties structurées garanties par schéma
- MCP converge comme standard universel (tous les frameworks le supportent)
- Pas de coût de plateforme — pay-per-token uniquement

### Faiblesses

- **Pas de runtime agent hébergé** — boucle agentique à construire et déployer soi-même
- **Pas de streaming vidéo** — Claude traite des frames statiques, pas de flux live
- **Pas d'orchestration multi-agents native** — à construire en Python
- **Coût le plus élevé** pour les modèles premium ($25/MTok output pour Opus 4.7)
- Compréhension vidéo limitée aux frames (pas de vidéo native comme Gemini)

### Coût estimé : 1000 analyses/mois

| Modèle | Coût modèle seul | + Infra self-hosted | Total |
|--------|-----------------|--------------------|----|
| Claude Haiku 4.5 | ~$7 | $30–100 | ~$37–107 |
| Claude Sonnet 4.6 | ~$21 | $30–100 | ~$51–121 |

**Vendor lock-in : Faible** — API simple à swapper, MCP est ouvert.

---

## 4. OpenAI

### Stack produit (2025)

OpenAI propose une **stack à trois niveaux** pour les agents :
1. **Responses API** — bas niveau, contrôle total
2. **OpenAI Agents SDK** (v0.14.0+) — orchestration multi-agents haut niveau
3. **Realtime API** — agents bas-latence audio/vidéo via WebSocket

### Architecture OpenAI Agents SDK

```
┌────────────────────────────────────────────────────────────┐
│              OpenAI Agents SDK                             │
│                                                            │
│  ┌──────────────────┐  ┌───────────────────────────────┐  │
│  │  Agent           │  │  Realtime Agents               │  │
│  │  (instructions + │  │  gpt-realtime-2 over WebSocket│  │
│  │   tools +        │  │  - Voice input/output          │  │
│  │   guardrails +   │  │  - Semantic VAD                │  │
│  │   handoffs)      │  │  - Détection interruption      │  │
│  └──────────────────┘  └───────────────────────────────┘  │
│         │                                                  │
│  ┌──────┴────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  Runner       │  │  Sessions  │  │  Tracing         │  │
│  │  (boucle      │  │  (mémoire  │  │  (dashboard      │  │
│  │   agent)      │  │   cross-   │  │   OpenAI)        │  │
│  │               │  │   turns)   │  │                  │  │
│  └───────────────┘  └────────────┘  └──────────────────┘  │
│         │                                                  │
│  Provider-Agnostic (LiteLLM → 100+ modèles)                │
│  ┌────────────────────────────────────────────────────┐    │
│  │ GPT-4o, GPT-4o-mini │ Claude │ Gemini │ ...        │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

### Tarifs (2025-2026)

| Modèle | Input/MTok | Output/MTok | Usage AI Shoot |
|--------|-----------|------------|---------------|
| GPT-4o | $2.50 | $10 | Analyse complexe |
| GPT-4o-mini | $0.15 | $0.60 | Screening volume |
| gpt-realtime-2 | Plus élevé | Plus élevé | Coaching vocal |

### Forces

- **Sorties structurées excellentes** — JSON schema strict
- **Realtime API** pour fonctionnalités de coaching vocal
- **Sessions** gèrent la mémoire conversationnelle automatiquement
- SDK bien poli — le moins d'abstractions, le plus facile à apprendre
- Provider-agnostic — peut utiliser Gemini Flash ou Claude via LiteLLM
- **Tracing intégré** dashboard OpenAI

### Faiblesses

- **Pas de runtime agent hébergé** — BYO infra comme Anthropic
- **Traitement vidéo limité** — frames uniquement, pas de streaming vidéo natif
- GPT-4o **10–17× plus cher** que Gemini 2.5 Flash pour des tâches comparables
- Realtime API WebSocket server-side uniquement (pas de WebRTC browser en Python SDK)
- Pas d'équivalent au Gemini Live API pour l'analyse vidéo temps réel

### Coût estimé : 1000 analyses/mois

| Configuration | Coût modèle | + Infra | Total |
|--------------|------------|---------|-------|
| GPT-4o | ~$37.50 | $30–100 | ~$70–140 |
| GPT-4o-mini | ~$2.25 | $30–100 | ~$32–100 |

**Vendor lock-in : Faible-Moyen** — SDK open-source, provider-agnostic.

---

## 5. Frameworks indépendants (CrewAI, AutoGen, Smolagents)

### CrewAI

**Architecture** : Flows (machine à états event-driven) + Crews (équipes d'agents autonomes)

```
Flow (scaffolding) → Délégation Crew → Collaboration agents → Suite Flow
```

**Pour AI Shoot :**
```python
# Flow pour le cycle de vie d'un match
@listen(shot_detected)
def analyze_shot(shot):
    crew = ShotAnalysisCrew()
    return crew.kickoff(inputs={"shot": shot})
```

**Forces** : prototypage rapide, métaphore de rôles intuitive, 100K+ développeurs certifiés.
**Faiblesses** : streaming faible, pas de hosting managé, moins optimisé pour vidéo/multimodal.
**Coût** : $0 framework + coûts modèles + infra self-hosted (~$32–55/mois total).

---

### AutoGen 2.0 (Microsoft)

**Architecture** : 3 tiers
1. **Core** — runtime event-driven distribué, multi-agents
2. **AgentChat** — agents conversationnels Python haut niveau
3. **Extensions** — MCP, OpenAI Assistant, Docker code execution, gRPC distribué

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

agent = AssistantAgent(
    "shot_analyzer",
    OpenAIChatCompletionClient(model="gpt-4o")
)
```

**Forces** : distribué natif (gRPC), MCP natif, sandbox Docker pour exécution de code.
**Faiblesses** : doc fragmentée, migration v0.2→2.0 difficile, pas de streaming vidéo.
**Coût** : $0 framework + modèles + infra.

---

### Smolagents (Hugging Face)

**Architecture** : Minimaliste — ~1000 lignes de code core.

```python
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[], model=InferenceClientModel())
agent.run("Analyse la technique de tir depuis ces frames")
```

**Feature unique — CodeAgent** : écrit du code Python pour résoudre les tâches (plutôt que des appels JSON). Puissant pour l'analytique sportive ad-hoc.

**Forces** : prototype ultra-rapide, intégration HuggingFace Hub (Qwen2-VL), pas de frais.
**Faiblesses** : pas production-grade, pas de state management ni de mémoire, streaming limité.
**Coût** : ~$21–55/mois total.

---

## ADK vs LangGraph — Head-to-Head

| Dimension | Google ADK 2.0 | LangGraph |
|-----------|---------------|-----------|
| **Streaming vidéo temps réel** | ✅ Natif (Gemini Live API Toolkit) | ⚠️ BYO (7 modes texte ; vidéo à construire) |
| **Intégration Gemini Flash** | ✅ Natif (Live API, function calling) | ✅ Bon (via `langchain-google-genai`, sans Live API) |
| **Intégration Qwen2-VL** | ✅ Via LiteLLM ou BaseAgent custom | ✅ Via `langchain-community` |
| **Orchestration multi-agents** | ✅ Sequential/Parallel/Loop/LLM-délégation | ✅ Supervisor/Swarm/DAG libre |
| **Mémoire session per-joueur** | ✅ Session Service intégré (Agent Runtime) | ✅ Manuel (Postgres/Redis checkpointer) |
| **Mémoire long-terme joueur** | ✅ Memory Bank (Agent Runtime) | ⚠️ DIY (vector DB, fetch manuel) |
| **Déploiement hébergé** | ✅ Agent Runtime (serverless, auto-scaling) | ✅ LangSmith (~$155/mois prod) |
| **Observabilité** | ✅ Cloud Trace + OTel (inclus) | ✅ LangSmith (abonnement requis) |
| **Human-in-the-loop** | ✅ Tool Confirmation flow | ✅ interrupt/resume intégré |
| **Model-agnostic** | ✅ (LiteLLM, Anthropic, LangGraph adapters) | ✅ (100+ intégrations) |
| **Interop cross-framework** | ✅ Protocole A2A natif | ⚠️ Limité (pas d'A2A natif) |
| **Taille de communauté** | Moyenne (croissance rapide) | Grande (Klarna, Uber, JPMorgan) |
| **Coût 1K analyses/mois** | **~$10–35 total** | **~$206 total** |
| **Friction démarrage** | Moyenne (setup GCP requis) | Faible (tourne en local) |
| **Licence** | Apache 2.0 | MIT |
| **Maturité production** | **v1.33.0 (stable, bi-hebdo)** | v1.2.0a7 **(alpha !)** |

---

## Architecture recommandée pour AI Shoot

```
┌────────────────────────────────────────────────────────────────────┐
│              Architecture AI Shoot — Stack Google ADK              │
│                                                                    │
│  App mobile/web ──WebSocket──→ Cloud Run API Gateway               │
│                                              │                     │
│  ┌───────────────────────────────────────────▼──────────────────┐  │
│  │  ADK Multi-Agent System (Agent Runtime / Cloud Run)          │  │
│  │                                                              │  │
│  │  ShotOrchestrator (LlmAgent, gemini-2.5-flash)              │  │
│  │  ├── VideoStreamAgent (BaseAgent, OpenCV + Gemini Live)     │  │
│  │  │   └── monitor_video_stream() → AsyncGenerator            │  │
│  │  ├── PoseAnalyzer (LlmAgent, gemini-2.5-flash)              │  │
│  │  ├── ShotClassifier (LlmAgent, gemini-2.5-flash-lite)       │  │
│  │  └── FeedbackAgent (LlmAgent, gemini-2.5-pro premium)       │  │
│  │                                                              │  │
│  │  Session Service → état per-joueur (analyse en cours)       │  │
│  │  Memory Bank → historique d'amélioration cross-sessions      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐  │
│  │  Couche stockage                                             │  │
│  │  GCS (vidéo brute) | BigQuery (analytics) | Firestore (users)│  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Observabilité : Cloud Trace + Cloud Monitoring (ADK OTel natif)   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Comparaison des coûts : 1000 analyses/mois

| Framework | Coût modèle | Coût plateforme | **Total/mois** | Notes |
|-----------|------------|----------------|----------------|-------|
| **ADK + Gemini 2.5 Flash** | ~$2–5 | ~$5–25 (Agent Runtime) | **~$10–30** | ⭐ Meilleur rapport |
| **ADK + Gemini Live API** | ~$4–10 | ~$5–25 | **~$15–35** | Streaming temps réel |
| **LangGraph + LangSmith** | ~$6 | ~$200 (prod+siège) | **~$206** | Meilleure traçabilité |
| **OpenAI Agents + GPT-4o** | ~$38 | ~$30–100 (self-host) | **~$70–140** | Coût modèle élevé |
| **OpenAI Agents + GPT-4o-mini** | ~$2 | ~$30–100 | **~$32–100** | Modèle moins cher |
| **Anthropic + Claude Haiku 4.5** | ~$7 | ~$30–100 (self-host) | **~$37–107** | Pas de streaming vidéo |
| **Anthropic + Claude Sonnet 4.6** | ~$21 | ~$30–100 | **~$51–121** | Meilleur raisonnement |
| **CrewAI + Gemini Flash** | ~$2–5 | ~$30–50 (self-host) | **~$32–55** | Session DIY |
| **Smolagents + HF Qwen2-VL** | ~$1–5 | ~$20–50 | **~$21–55** | Qualité prototype |

---

## Complexité de migration

| Migration | Complexité | Ce qui change |
|-----------|-----------|--------------|
| ADK → LangGraph | **Moyenne** | Réécrire hiérarchie agents en StateGraph ; perdre Session Service ; gagner LangSmith |
| LangGraph → ADK | **Moyenne** | Réécrire nœuds StateGraph en agents ADK ; gagner Session Service ; perdre LangSmith Studio |
| L'un ou l'autre → Claude-natif | **Faible** | Drop framework, boucle agentique manuelle ; perdre state management |
| L'un ou l'autre → OpenAI SDK | **Faible-Moyenne** | Abstractions similaires ; perdre streaming vidéo ; gagner Realtime API |
| ADK ↔ LangGraph | **Faible** (avec A2A) | A2A permet aux deux de coexister et communiquer |

**Plus portable** : OpenAI Agents SDK (provider-agnostic, primitives propres, MCP-natif)
**Plus difficile à quitter** : ADK (Session/Memory services créent une adhérence GCP)

---

## Matrice de sélection

| Priorité | Meilleur choix | Raisonnement |
|----------|--------------|-------------|
| Coût minimum à l'échelle | ADK + Gemini Flash | Vidéo native, compute serverless |
| Streaming vidéo temps réel | ADK + Live API | Seul framework avec streaming vidéo bidirectionnel |
| Meilleure qualité de raisonnement | Anthropic Claude Opus 4.7 | Mais pas de vidéo, pas de runtime |
| Topologie la plus flexible | LangGraph | Modèle DAG : n'importe quelle architecture |
| Meilleure observabilité | LangGraph + LangSmith | UI de tracing supérieure |
| Prototype le plus rapide | Smolagents / CrewAI | Moins de concepts |
| Meilleur pour Qwen2-VL | LangGraph ou OpenAI SDK | Via LiteLLM |
| Moins de vendor lock-in | OpenAI Agents SDK | Provider-agnostic, MCP-natif |
| Meilleur pricing startup | ADK + Gemini Flash Lite | 10–20× moins cher que concurrents |
| Production-ready aujourd'hui | **ADK (v1.33.0)** | LangGraph encore alpha (v1.2.0a7) |

---

## Recommandation finale pour AI Shoot

### 🥇 Recommandation principale : Google ADK + Vertex AI Stack

**Pourquoi :**

1. `monitor_video_stream()` avec `LiveRequestQueue` + `AsyncGenerator` est **exactement** ce que l'analyse basketball temps réel nécessite — aucun autre framework ne propose ça en production
2. **Gemini 2.5 Flash** comprend la vidéo nativement — pas besoin d'extraire et encoder des frames manuellement
3. **Agent Runtime** donne l'état session per-joueur et la mémoire long-terme (tracking de progression) **sans boilerplate Redis/Postgres**
4. À **$10–35/mois** pour 1000 analyses, c'est startup-friendly
5. **Licence Apache 2.0** plus propre pour produit commercial
6. **v1.33.0 stable** avec cadence bi-hebdomadaire — LangGraph toujours en alpha

```python
from google.adk.agents import LlmAgent, BaseAgent, ParallelAgent, SequentialAgent

# Détection et pose en parallèle
detection_pipeline = ParallelAgent(sub_agents=[
    shot_detector,   # Gemini 2.5 Flash Lite (rapide, économique)
    pose_analyzer,   # Gemini 2.5 Flash (précis)
])

# Pipeline complet : détecter → classifier → feedback
shot_analysis_agent = SequentialAgent(sub_agents=[
    detection_pipeline,
    shot_classifier,
    FeedbackAgent(model="gemini-2.5-flash"),
])
```

### 🥈 Recommandation secondaire : LangGraph

Choisir LangGraph si :
- Expérimenter **Gemini Flash et Qwen2-VL en parallèle** (swap trivial)
- Besoin de **LangSmith pour l'observabilité** (meilleur UX de debugging)
- Éviter toute adhérence GCP
- L'équipe connaît déjà LangGraph

⚠️ **Attention :** Pinner la version à cause du statut alpha (v1.2.0a7).

### 🥉 Stratégie hybride (le meilleur des deux)

Puisqu'**ADK intègre déjà `langgraph>=0.2.60`** dans ses extensions :

1. **Pipeline vidéo dans ADK** (Live API, Session Service)
2. **Logique coaching complexe dans LangGraph** (pattern supervisor pour multi-coach)
3. **Communication via A2A** (agent ADK ↔ agent LangGraph)
4. **LangSmith pour l'observabilité** des deux (framework-agnostic)

Cette architecture scale de startup à enterprise avec migration minimale.

---

## Sources

| Affirmation | Source |
|-------------|--------|
| LangGraph v1.2.0a7 | `langchain-ai/langgraph:libs/langgraph/pyproject.toml` |
| LangGraph streaming 7 modes + format v2 | `docs.langchain.com/oss/python/langgraph/streaming` |
| LangSmith pricing ($39/siège, $0.005/run, $0.0036/min) | `langchain.com/pricing-langsmith` |
| ADK v1.33.0 | `google/adk-python:CHANGELOG.md` (2026-05-08) |
| ADK multi-agents : Sequential/Parallel/Loop | `google.github.io/adk-docs/agents/multi-agents/` |
| ADK Live API Toolkit + code streaming vidéo | `google.github.io/adk-docs/streaming/streaming-tools/` |
| ADK dépendances complètes | `google/adk-python:pyproject.toml` |
| Protocole A2A (JSON-RPC 2.0, Agent Cards, SSE) | `google-a2a/A2A:README.md` |
| Gemini 2.5 Flash pricing ($0.30/$2.50/M) | `cloud.google.com/vertex-ai/generative-ai/pricing` |
| Gemini 2.5 Flash Live API ($3/M vidéo, $2/M out) | `cloud.google.com/vertex-ai/generative-ai/pricing` |
| Claude Opus 4.7 ($5/$25), Sonnet 4.6 ($3/$15), Haiku 4.5 ($1/$5) | `docs.anthropic.com/en/docs/about-claude/pricing` |
| OpenAI Agents SDK (sandbox, sessions, realtime) | `openai/openai-agents-python:README.md` |
| OpenAI Realtime API (gpt-realtime-2, WebSocket) | `openai.github.io/openai-agents-python/realtime/quickstart/` |
| GPT-4o pricing ($2.50/$10/M) | `openrouter.ai/openai/gpt-4o` |
| GPT-4o-mini pricing ($0.15/$0.60/M) | `openrouter.ai/openai/gpt-4o-mini` |
| CrewAI (Flows + Crews) | `docs.crewai.com/introduction` |
| AutoGen 2.0 (Core/AgentChat/Extensions) | `microsoft.github.io/autogen/stable/` |
| Smolagents (CodeAgent, ~1K LOC) | `huggingface.co/docs/smolagents/en/index` |
| MCP comme standard universel | `modelcontextprotocol.io/introduction` |
| ADK Agent Runtime (serverless, session/memory) | `google.github.io/adk-docs/deploy/agent-runtime/` |
