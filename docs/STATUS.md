# AutoVibe Gym — Live Status

**Last updated:** 2026-05-29 (all experiments complete, docs final, merged to main)
**Phase:** COMPLETE. All 4 modes × 2 datasets. All 47 tests passing. Pushed to main.

---

## Current Sprint Goal
Complete all TZ deliverables: 4 modes × 2 datasets, Protocol spec, Reproducibility guide, Experiment report.

---

## Status by Component

### Core Gym (`gym/`)
| File | Status | Notes |
|------|--------|-------|
| `env.py` | ✅ Done | GymEnv, EnvState, JSON Action handling, Observation + CellHistory recording, submit gate, sandbox_timeout param; pre-flight model validation (`_raw_validation_hint`); **[P3]** model.predict wrapped in try/except for schema errors |
| `executor.py` | ✅ Done | **[P1 fixed]** subprocess + pickle tempfiles + hard timeout + `cwd=tmpdir` (no test.csv leak) |
| `checklist.py` | ✅ Done | **[P2 fixed]** 8 checks; `model_selection` now includes current code, not only history |
| `protocol.py` | ✅ Done | Explicit JSON Action / Observation contract |
| `workspace.py` | ✅ Done | Persistent visible namespace without hidden test split |
| `cell_history.py` | ✅ Done | Notebook-like cells with code, stdout/stderr, hints, coverage, and submit result |
| `datasets.py` | ✅ Done | Loader for CSV mode and fixed split dataset dirs |
| `llm.py` | ✅ Done | OpenAI-compatible + LiteLLMClient (provider/model routing) |
| `agent.py` | ✅ Done | **[P2 fixed]** Pipeline prescription removed from SYSTEM_PROMPT; forced-submit scans all workspace vars with predict() |
| `__init__.py` | ✅ Done | |

### Experiments (`experiments/`)
| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | ✅ Done | MLflow logging, `--mode cloud/local`, dataset-dir/CSV loader, sandbox_timeout |
| `run_baseline.py` | ✅ Done | Single-shot, no env, MLflow logging |
| `compare.py` | ✅ Done | Aggregates MLflow runs into comparison table |
| `run_multishot.py` | ✅ Done | N-shot iteration, execution feedback only, no checklist hints |
| `run_fixed.py` | ✅ Done | Fixed transitions: 5 mandatory stages (EDA→Prep→FE→ModelSel→HPT), per-stage budget, checklist feedback |

### Datasets (`datasets/`)
| Dataset | Status | Notes |
|---------|--------|-------|
| `scripts/prepare_datasets.py` | ✅ Done | Discovers `datasets/*/config.json`, prepares splits into `prepared/` with `meta.json` |
| Fixed split format | ✅ Done | `train.csv` / `val.csv` / `test.csv` / `meta.json` contract supported (incl. `prepared/` subdir) |
| Example dataset configs | ✅ Done | `dry_bean`, `student_dropout`, `room_occupancy`, `naticusdroid` (**[DATA fixed]** deduplicate before split), `phiusiil_phishing` (**[DATA fixed]** drops FILENAME/URL/Domain/TLD/Title) |
| `prepare_datasets.py` deduplication | ✅ Done | `"deduplicate": true` in config removes duplicates before splitting |

### Infrastructure
| Item | Status | Notes |
|------|--------|-------|
| `requirements.txt` | ✅ Done | openai, mlflow, xgboost, lightgbm, python-dotenv |
| `CLAUDE.md` | ✅ Done | Points to STATUS, PROJECT, GIT_WORKFLOW |
| `AGENTS.md` | ✅ Done | Codex workflow points to status, project, and Git workflow docs |
| `docs/GIT_WORKFLOW.md` | ✅ Done | Team Git/PR workflow and AI-agent collaboration rules |
| `docs/PROJECT.md` | ✅ Done | Stack updated to reflect ADR-001..005 and Action/Observation protocol |
| `docs/ARCHITECTURE_DECISIONS.md` | ✅ Done | ADR-001..010 |
| `scripts/start_vllm.sh` | ✅ Done | vLLM launcher for H200, auto-detects AWQ |
| `Dockerfile` | ✅ Done | Based on booml-backend:latest; entrypoint python -m |
| `docker-compose.yml` | ✅ Done | MLflow service with named volume |
| Unit smoke tests | ✅ Done | Protocol/workspace/submit plus checklist, executor, env, and CellHistory coverage |
| `.env.example` | ✅ Done | LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MLFLOW_TRACKING_URI |
| Server deployment | ✅ Done | Docker on booml@10.8.52.11; MLflow on :8002 |
| API access | ✅ Done | http://llm.letovo.site:8809/openai — gemma-4-26b, deepseek-v4-flash |

---

## Experiment Results ✅ COMPLETE

### student_dropout (deepseek-v4-flash, f1_macro)

| Mode | test_metric | steps/attempts | errors | elapsed | input tokens |
|------|-------------|----------------|--------|---------|--------------|
| single-shot (baseline) | **0.743** | 1 shot | 0 | 13s | 2 120 |
| repeated single-shot | 0.740 | 5 attempts | 6 | 71s | 11 284 |
| flexible transitions (gym) | 0.735 | 12/15 steps | 4 | 199s | 136 115 |
| fixed transitions | 0.699 | 22 steps | 9 | 707s | — |

### room_occupancy (deepseek-v4-flash, f1_macro, temporal split)

| Mode | test_metric | steps/attempts | errors | elapsed | input tokens |
|------|-------------|----------------|--------|---------|--------------|
| single-shot (baseline) | **1.000** | 1 shot | 0 | 30s | 1 126 |
| repeated single-shot | 1.000 | 10 attempts | 12 | 315s | 12 600 |
| flexible transitions (gym) | null† | 15/15 steps | 6 | 143s | 186 822 |
| fixed transitions | null† | 22 steps | 6 | 207s | 335 131 |

†Agent extracted temporal features (hour/minute/day/month) without Pipeline wrapper → forced submit failed.

**Key findings:**
- More interaction ≠ better score. Baseline wins on score AND cost for student_dropout.
- Gym achieves checklist_coverage=1.0 (all 8 DS stages covered) — diagnostic value baseline can't provide.
- Room_occupancy: gym/fixed surface Pipeline-encapsulation failure mode (test_metric=null with explanation).
- Environment handles both stratified_random and temporal splits without code changes — extensibility confirmed.

## Deliverables Status

| Артефакт (ТЗ) | Файл | Статус |
|---|---|---|
| Prototype | `gym/`, `experiments/` | ✅ |
| Protocol spec | `docs/PROTOCOL.md` | ✅ |
| Experiment report | `docs/EXPERIMENT_REPORT.md` | ✅ |
| Trajectory logs | MLflow `cell_history.md`, `stage_log.json` | ✅ |
| Reproducibility guide | `docs/REPRODUCIBILITY.md` | ✅ |
| Demo task | student_dropout + room_occupancy | ✅ |

## Next Actions

1. [x] Завершить эксперименты room_occupancy — все 4 режима запущены и завершены
2. [x] Обновить EXPERIMENT_REPORT.md с результатами
3. [x] Смержить в main — commit 983ec54 pushed to origin/main 2026-05-29

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

| 2026-05-28 | Added dataset-centric config-driven pipeline scaffold for example datasets (student_dropout, room_occupancy, naticusdroid, phiusiil_phishing, dry_bean) with legacy compatibility and tests |
| 2026-05-28 | Fixed dataset `meta.json` generation (JSON-serializable distributions), added `pytest` to `requirements.txt` |
| 2026-05-28 | [P1] executor.py: added `cwd=tmpdir` to subprocess.run — closes test.csv filesystem leak (25% grading weight) |
| 2026-05-28 | [P1] agent.py: _try_forced_submit scans all workspace vars with predict() — closes null test_metric on budget exhaustion |
| 2026-05-28 | [P2] checklist.py: _check_model_selection includes current code, not only prior history (off-by-one) |
| 2026-05-28 | [P2] agent.py: removed Pipeline prescription from SYSTEM_PROMPT — was a confound for gym vs baseline comparison |
| 2026-05-28 | [DATA] naticusdroid config: `deduplicate: true` — 74.5% duplicate rows removed before splitting |
| 2026-05-28 | [DATA] phiusiil_phishing config: drops FILENAME/URL/Domain/TLD/Title non-numeric identifier columns |
| 2026-05-28 | prepare_datasets.py: added `deduplicate` preparation step support |
| 2026-05-28 | experiments/run_fixed.py: Fixed transitions mode — 5 stages, per-stage budget, stage_log artifact |
| 2026-05-28 | run_multishot.py: rewritten as true repeated single-shot (independent attempts, no traceback) |
| 2026-05-28 | run_baseline.py: migrated from openai.OpenAI to _default_client(); added elapsed_seconds |
| 2026-05-28 | gym/env.py: submit() now catches label-type mismatch and attempts coercion before failing |
| 2026-05-28 | gym/env.py: added pre-flight model validation (_raw_validation_hint) to catch schema errors before submit |
| 2026-05-28 | docs/PROTOCOL.md: full protocol spec (4 modes, feedback taxonomy, privacy boundary) |
| 2026-05-28 | docs/REPRODUCIBILITY.md: reproducibility guide + how to add a new dataset |
| 2026-05-28 | docs/EXPERIMENT_REPORT.md: mode comparison, cost-quality, failure taxonomy, TZ criteria |
| 2026-05-29 | [P3] gym/env.py: model.predict(X_test) wrapped in try/except — catches schema/missing-column errors |
| 2026-05-29 | room_occupancy experiments complete — all 4 modes run, findings documented |
| 2026-05-29 | .gitignore: added *.arff, *.xlsx, *.txt exclusions for raw dataset binary files |
