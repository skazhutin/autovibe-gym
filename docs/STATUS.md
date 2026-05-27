# AutoVibe Gym — Live Status

**Last updated:** 2026-05-27
**Phase:** Core implementation complete, first end-to-end run needed

---

## Current Sprint Goal
First end-to-end run: `python scripts/prepare_datasets.py` → `python -m experiments.run_gym` → MLflow summary printed.

---

## Status by Component

### Core Gym (`gym/`)
| File | Status | Notes |
|------|--------|-------|
| `env.py` | ✅ Done | GymEnv, EnvState, StepResult, submit gate, sandbox_timeout param |
| `executor.py` | ✅ Done | subprocess + pickle tempfiles + hard timeout (ADR-002) |
| `checklist.py` | ✅ Done | 8 checks, keyword-based, implicit hints |
| `agent.py` | ✅ Done | openai SDK, configurable LLM_BASE_URL/LLM_MODEL (ADR-001) |
| `__init__.py` | ✅ Done | |

### Experiments (`experiments/`)
| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | ✅ Done | MLflow logging, --mode cloud/local, sandbox_timeout |
| `run_baseline.py` | ✅ Done | Single-shot, no env, MLflow logging |
| `compare.py` | ✅ Done | Aggregates MLflow runs into comparison table |
| `run_multishot.py` | ❌ TODO | Multi-shot, same token budget, no checklist |

### Datasets (`datasets/`)
| Dataset | Status | Notes |
|---------|--------|-------|
| `scripts/prepare_datasets.py` | ✅ Done | Downloads + splits Wine Quality, Bank Marketing, Heart Disease |
| Wine Quality splits | ❌ TODO | Run `python scripts/prepare_datasets.py --dataset wine_quality` |
| Bank Marketing splits | ❌ TODO | |
| Heart Disease splits | ❌ TODO | |

### Infrastructure
| Item | Status | Notes |
|------|--------|-------|
| `requirements.txt` | ✅ Done | openai, mlflow, xgboost, lightgbm, python-dotenv |
| `CLAUDE.md` | ✅ Done | Points to STATUS, PROJECT, GIT_WORKFLOW |
| `AGENTS.md` | ✅ Done | Codex instructions (added by Codex PR #1) |
| `docs/PROJECT.md` | ✅ Done | Stack updated to reflect ADR-001..005 |
| `docs/GIT_WORKFLOW.md` | ✅ Done | Rewritten by Codex PR #1 — much improved |
| `docs/ARCHITECTURE_DECISIONS.md` | ✅ Done | ADR-001..007 |
| `scripts/start_vllm.sh` | ✅ Done | vLLM launcher for H200, auto-detects AWQ |
| `.env.example` | ✅ Done | LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MLFLOW_TRACKING_URI |
| First end-to-end test run | ❌ TODO | Need vLLM server running + dataset prepared |
| MLflow server running | ❌ TODO | `mlflow server --host 0.0.0.0 --port 5000` on H200 |

---

## Blocked / Needs Decision

- **vLLM server**: нужен запущенный сервер на H200 для первого прогона. До этого можно тестировать с облачным API (OpenAI/Anthropic proxy).
- **Sandbox security**: subprocess изолирует процесс, но не ОС. Docker — вариант для финального деплоя если нужна полная изоляция.
- **run_multishot.py**: последний недостающий режим сравнения. Нужен для честного сравнения "gym vs. multi-shot с тем же токен-бюджетом".

---

## Next Actions (приоритет)

1. [ ] Запустить `python scripts/prepare_datasets.py` — скачать и нарезать датасеты
2. [ ] Поднять vLLM на H200: `./scripts/start_vllm.sh Qwen/Qwen2.5-Coder-7B-Instruct`
3. [ ] Поднять MLflow: `mlflow server --host 0.0.0.0 --port 5000`
4. [ ] Заполнить `.env` (скопировать `.env.example`, прописать IP сервера)
5. [ ] Первый тестовый прогон: `python -m experiments.run_gym --dataset-dir datasets/wine_quality --mode local`
6. [ ] Прогнать baseline: `python -m experiments.run_baseline --dataset-dir datasets/wine_quality --mode local`
7. [ ] Сравнить: `python -m experiments.compare`
8. [ ] Написать `experiments/run_multishot.py`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-27 | Codex PR #1: hardened GIT_WORKFLOW.md, added AGENTS.md |
| 2026-05-27 | ADR-001..005: agent→openai SDK, executor→subprocess, MLflow, datasets script, vLLM startup |
| 2026-05-27 | Added docs/ARCHITECTURE_DECISIONS.md (ADR-001..007) |
| 2026-05-27 | Added docs/GIT_WORKFLOW.md, .gitignore, .env.example; pushed to GitHub |
| 2026-05-27 | Initial scaffolding: gym/, executor, checklist, agent, run_gym.py, docs |
