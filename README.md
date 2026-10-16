# NEXUS

Assistant IA multi-agents — mobile + web + cloud.

## Vision
Un assistant qui comprend, planifie, recherche, code, teste, corrige et déploie.
Accessible depuis un téléphone comme depuis un ordinateur.

## Stack
- **Web** : Next.js 14 + TypeScript + Tailwind
- **Mobile** : React Native + Expo
- **API** : FastAPI + Python 3.12
- **Agents** : LangGraph
- **DB** : PostgreSQL 16 + pgvector
- **Cache** : Redis
- **Storage** : S3 / Cloudflare R2

## Structure
- `apps/web`     — Interface web
- `apps/mobile`  — Application mobile
- `apps/api`     — Backend FastAPI
- `agents/`      — Logique multi-agents
- `packages/`    — Code partagé
- `infra/`       — Docker, K8s, Terraform

## Auteur
Aboubacar Boubafofana — Abidjan
