# AutoVibe Gym — Live Status

**Last updated:** 2026-05-28
**Phase:** First experiments running on server — ablation data being collected

---

## Current Sprint Goal
Collect ablation results across all 3 datasets × 3 experiment types (baseline / multishot / gym) and produce comparison table for presentation.

---

## Status by Component

### Core Gym (`gym/`)
| File | Status | Notes |
|------|--------|-------|
| `env.py` | ✅ Done | GymEnv, EnvState, JSON Action handling, Observation + CellHistory recording, submit gate, sandbox_timeout param |
| `executor.py` | ✅ Done | subprocess + pickle tempfiles + hard timeout (ADR-002) |
| `checklist.py` | ✅ Done | 8 checks, keyword-based, implicit hints |
| `protocol.py` | ✅ Done | Explicit JSON Action / Observation contract |
| `workspace.py` | ✅ Done | Persistent visible namespace without hidden test split |
| `cell_history.py` | ✅ Done | Notebook-like cells with code, stdout/stderr, hints, coverage, and submit result |
| `datasets.py` | ✅ Done | Loader for CSV mode and fixed split dataset dirs |
| `llm.py` | ✅ Done | OpenAI-compatible LLMClient adapter |
| `agent.py` | ✅ Done | JSON actions, OpenAI-compatible provider adapter, budget tracking |
| `__init__.py` | ✅ Done | |

### Experiments (`experiments/`)
| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | ✅ Done | MLflow logging, `--mode cloud/local`, dataset-dir/CSV loader, sandbox_timeout |
| `run_baseline.py` | ✅ Done | Single-shot, no env, MLflow logging |
| `compare.py` | ✅ Done | Aggregates MLflow runs into comparison table |
| `run_multishot.py` | ✅ Done | N-shot iteration, execution feedback only, no checklist hints |

### Datasets (`datasets/`)
| Dataset | Status | Notes |
|---------|--------|-------|
| `scripts/prepare_datasets.py` | ✅ Done | Downloads + splits Wine Quality, Bank Marketing, Heart Disease |
| Fixed split format | ✅ Done | `train.csv` / `val.csv` / `test.csv` / `meta.json` contract supported |
| Wine Quality splits | ❌ TODO | Run `python scripts/prepare_datasets.py --dataset wine_quality` |
| Bank Marketing splits | ❌ TODO | |
| Heart Disease splits | ❌ TODO | |

### Infrastructure
| Item | Status | Notes |
|------|--------|-------|
| `requirements.txt` | ✅ Done | openai, mlflow, xgboost, lightgbm, python-dotenv |
| `CLAUDE.md` | ✅ Done | Points to STATUS, PROJECT, GIT_WORKFLOW |
| `AGENTS.md` | ✅ Done | Codex workflow points to status, project, and Git workflow docs |
| `docs/GIT_WORKFLOW.md` | ✅ Done | Team Git/PR workflow and AI-agent collaboration rules |
| `docs/PROJECT.md` | ✅ Done | Stack updated to reflect ADR-001..005 and Action/Observation protocol |
| `docs/ARCHITECTURE_DECISIONS.md` | ✅ Done | ADR-001..009 |
| `scripts/start_vllm.sh` | ✅ Done | vLLM launcher for H200, auto-detects AWQ |
| `Dockerfile` | ✅ Done | Based on booml-backend:latest; entrypoint python -m |
| `docker-compose.yml` | ✅ Done | MLflow service with named volume |
| Unit smoke tests | ✅ Done | Protocol/workspace/submit plus checklist, executor, env, and CellHistory coverage |
| `.env.example` | ✅ Done | LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MLFLOW_TRACKING_URI |
| Server deployment | ✅ Done | Docker on booml@10.8.52.11; MLflow on :8002 |
| API access | ✅ Done | http://llm.letovo.site:8809/openai — gemma-4-26b, deepseek-v4-flash |

---

## Experiment Results (preliminary, deepseek-v4-flash)

| Dataset | Baseline | Gym | Notes |
|---------|----------|-----|-------|
| wine_quality | null (timeout) | 0.216 | Gym achieves result; checklist 100% |
| bank_marketing | null (encode error) | null (pipeline mismatch) | Both fail; gym encodes train but not test |
| heart_disease | **0.910** | 0.887 | Comparable; checklist 100% |

Multishot (no checklist) running — results pending.
Gemma-4-26b gym comparison running — previous local result: wine_quality 0.649.

## Blocked / Needs Decision

- **bank_marketing pipeline bug**: gym encodes train with get_dummies but doesn't apply same transform to test at submit. Consider injecting a preprocessing hint or using pipelines in the system prompt.
- **deepseek wine_quality score low (0.216)**: 5 errors in 15 steps — model got stuck. Need to investigate logs or re-run.

---

## Next Actions (приоритет)

1. [ ] Дождаться результатов multishot + gemma gym экспериментов
2. [ ] Собрать финальную таблицу: `python -m experiments.compare`
3. [ ] Исправить pipeline баг для bank_marketing (preprocessing в обучении и на submit)
4. [ ] Подготовить слайды с таблицей сравнения baseline / multishot / gym
5. [ ] Смержить текущую ветку в main

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-28 | Added notebook-like CellHistory, Gym feedback context, MLflow cell_history artifact, and tests |
| 2026-05-27 | Resolved PR #3 conflicts with ADR-001..005 implementation on main |
| 2026-05-27 | Implemented base Action/Observation protocol, Workspace, dataset split loader, and smoke tests |
| 2026-05-27 | ADR-001..005: agent→openai SDK, executor→subprocess, MLflow, datasets script, vLLM startup |
| 2026-05-27 | Added docs/ARCHITECTURE_DECISIONS.md (ADR-001..007) |
| 2026-05-27 | Codex PR #1: hardened GIT_WORKFLOW.md, added AGENTS.md |
| 2026-05-27 | Added docs/GIT_WORKFLOW.md, .gitignore, .env.example; pushed to GitHub |
| 2026-05-27 | Initial scaffolding: gym/, executor, checklist, agent, run_gym.py, docs |
