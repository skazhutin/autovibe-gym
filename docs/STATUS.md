# AutoVibe Gym — Live Status

**Last updated:** 2026-05-27  
**Phase:** Scaffolding complete, first run not yet tested

---

## Current Sprint Goal
Get a first end-to-end run working: dataset → GymEnv → GymAgent → summary printed.

---

## Status by Component

### Core Gym (`gym/`)
| File | Status | Notes |
|------|--------|-------|
| `env.py` | ✅ Done | GymEnv, EnvState, StepResult, submit gate |
| `executor.py` | ✅ Done | Isolated namespace exec, strips private keys |
| `checklist.py` | ✅ Done | 8 checks, keyword-based, implicit hints |
| `agent.py` | ✅ Done | Anthropic API, code parser, budget tracking |
| `__init__.py` | ✅ Done | |

### Experiments (`experiments/`)
| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | ✅ Done | CLI entry point, auto metric detection |
| `run_baseline.py` | ❌ TODO | Single-shot, no env |
| `run_multishot.py` | ❌ TODO | Multi-shot, same token budget, no checklist |
| `compare.py` | ❌ TODO | Aggregate results into comparison table |

### Datasets (`datasets/`)
| Dataset | Status | Notes |
|---------|--------|-------|
| Wine Quality | ❌ TODO | Download from UCI / Kaggle |
| Bank Marketing | ❌ TODO | |
| Heart Disease | ❌ TODO | |

### Infrastructure
| Item | Status | Notes |
|------|--------|-------|
| `requirements.txt` | ✅ Done | |
| `CLAUDE.md` | ✅ Done | |
| `docs/PROJECT.md` | ✅ Done | |
| First end-to-end test run | ❌ TODO | Need ANTHROPIC_API_KEY + a dataset |
| `.env.example` template | ✅ Done | |
| `.env` / secrets setup | ❌ TODO | Каждый копирует `.env.example` → `.env` |

---

## Blocked / Needs Decision

- **API key**: нужен `ANTHROPIC_API_KEY` для запуска агента. Пока нет — можно заглушить `agent.py` и тестировать `env.py` + `checklist.py` напрямую.
- **Sandbox security**: сейчас `exec()` в том же процессе. Для финального деплоя стоит рассмотреть `subprocess` или Docker. Пока оставляем как есть.
- **Датасеты**: нужно выбрать 2-3 финальных датасета и зафиксировать сплиты (seed=42 уже задан).

---

## Next Actions (приоритет)

1. [ ] Скачать 1 датасет (Wine Quality) → `datasets/wine_quality.csv`
2. [ ] Запустить `python -m experiments.run_gym --dataset datasets/wine_quality.csv --target quality` и проверить первый цикл
3. [ ] Написать `experiments/run_baseline.py` (single-shot без env)
4. [ ] Сравнить результаты gym vs. baseline на одном датасете
5. [ ] Расширить checklist по итогам первых прогонов

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-27 | Initial scaffolding: gym/, executor, checklist, agent, run_gym.py, docs |
| 2026-05-27 | Added docs/GIT_WORKFLOW.md, .gitignore, .env.example; pushed to GitHub |
