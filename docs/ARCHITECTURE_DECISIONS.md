# Architecture Decision Records (ADR)

Ключевые решения по архитектуре, стеку и особенностям проекта.
Каждое ADR: контекст → варианты → решение → последствия.

---

## ADR-001: Provider-agnostic LLM agent

**Статус:** ✅ Принято  
**Дата:** 2026-05-27

### Контекст
Изначально `agent.py` использует `anthropic` SDK напрямую. Цель проекта — сравнивать
разные LLM в одной среде. У нас есть H200 с локальными моделями и потенциально доступ
к облачным API. Жёсткая привязка к одному провайдеру блокирует основной эксперимент.

### Варианты
| Вариант | Плюсы | Минусы |
|---------|-------|--------|
| Отдельный класс на каждый провайдер | Максимальный контроль | Дублирование кода |
| LiteLLM как прокси | Единый интерфейс, 100+ моделей | Лишняя зависимость, абстракция |
| `openai` SDK + `base_url` | Один клиент, OpenAI-совместимо | Нужен OpenAI-совместимый endpoint |

### Решение
**Использовать `openai` SDK с конфигурируемым `base_url`.**

Все целевые модели доступны через OpenAI-совместимый API:
- `vLLM` (локальные модели) → `http://localhost:8000/v1`
- OpenAI GPT → `https://api.openai.com/v1`
- Anthropic (через прокси или `litellm proxy`) → `http://localhost:4000/v1`

```python
# Конфигурация через .env / CLI флаг
client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.getenv("LLM_API_KEY", "local"))
```

### Последствия
- `agent.py` рефакторится на `openai` SDK
- `requirements.txt`: убрать `anthropic`, добавить `openai`
- Добавить в `.env.example`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`

---

## ADR-002: Изолированный sandbox через subprocess

**Статус:** ✅ Принято  
**Дата:** 2026-05-27

### Контекст
Текущий `executor.py` использует `exec()` в том же процессе. LLM-код запускается в
production-сервере с GPU. Риски: бесконечный цикл, утечка памяти, доступ к файловой
системе сервера, `import os; os.system(...)`.

### Варианты
| Вариант | Изоляция | Сложность |
|---------|----------|-----------|
| `exec()` в процессе (текущее) | Нет | Минимальная |
| `subprocess` с timeout | Процесс | Низкая |
| Docker container | Полная | Средняя |
| `RestrictedPython` | Частичная | Средняя |

### Решение
**subprocess с timeout + `resource` limits (Linux).**

Docker — идеально, но добавляет overhead к каждому шагу и усложняет передачу namespace.
`subprocess` + timeout + убийство процесса при превышении — достаточная защита для
нашего контекста (доверенная команда, не публичный сервис).

```python
result = subprocess.run(
    ["python", "-c", code_with_context],
    capture_output=True, text=True,
    timeout=30,  # шаг не может длиться > 30 сек
)
```

Передача namespace: сериализуем входные данные в tempfile, код читает их и пишет
результаты обратно.

### Последствия
- `executor.py` полностью переписывается
- Namespace передаётся через pickle/JSON в tempfile
- Прирост latency ~0.1-0.3с на шаг — приемлемо
- Добавить `--sandbox-timeout` флаг в CLI

---

## ADR-003: MLflow для трекинга экспериментов

**Статус:** ✅ Принято  
**Дата:** 2026-05-27

### Контекст
Будем запускать десятки экспериментов: 3 модели × 3 датасета × 4 режима (gym/single/multi/free)
= 36+ прогонов минимум. Текущий вывод — `print(json.dumps(summary))` в терминал.
Нужно сравнивать прогоны, строить таблицы, видеть тренды.

### Варианты
| Вариант | Плюсы | Минусы |
|---------|-------|--------|
| JSON файлы + pandas | Просто, без зависимостей | Нет UI, ручная агрегация |
| MLflow (локальный) | UI, API, сравнение runs, без облака | Ещё одна зависимость |
| Weights & Biases | Красивый UI, облако | Нужен интернет, аккаунт |
| TensorBoard | Известен всем | Не заточен под tabular ML |

### Решение
**MLflow локально.**

```bash
mlflow server --host 0.0.0.0 --port 5000  # запустить на сервере
# Доступ: http://server-ip:5000
```

Каждый прогон логирует: `test_metric`, `checklist_coverage`, `steps_used`,
`error_count`, `input_tokens`, `output_tokens`, `model`, `dataset`, `mode`.

### Последствия
- Добавить `mlflow` в `requirements.txt`
- В `run_gym.py` и других experiment-скриптах: `mlflow.start_run()` + `mlflow.log_metrics()`
- `MLFLOW_TRACKING_URI=http://localhost:5000` в `.env.example`
- Один `compare.py` просто делает `mlflow.search_runs()` и выводит таблицу

---

## ADR-004: Модели для экспериментов (H200, 141GB VRAM)

**Статус:** ✅ Принято  
**Дата:** 2026-05-27

### Контекст
H200 SXM5 = 141GB HBM3e VRAM, 1TB RAM, единственный GPU.
Задача — сравнить несколько LLM разного размера в одинаковых условиях.

### Что влезает

| Модель | VRAM (bf16) | VRAM (AWQ 4-bit) | Рекомендуется |
|--------|-------------|-------------------|---------------|
| Qwen2.5-Coder-7B-Instruct | ~14GB | ~4GB | ✅ Small baseline |
| Llama-3.1-8B-Instruct | ~16GB | ~5GB | ✅ Small alt |
| DeepSeek-R1-Distill-7B | ~14GB | ~4GB | ✅ Reasoning small |
| Qwen2.5-Coder-32B-Instruct | ~64GB | ~20GB | ✅ Medium |
| Qwen2.5-72B-Instruct | ~144GB ❌ | ~40GB ✅ | ✅ Large (AWQ) |
| Llama-3.3-70B-Instruct | ~140GB ≈ | ~40GB ✅ | ✅ Large (AWQ) |
| Llama-3.1-405B | ~810GB ❌ | ~200GB ❌ | ✗ Не влезает |

### Решение — матрица экспериментов

**Tier S (Small):** `Qwen2.5-Coder-7B-Instruct`  
**Tier M (Medium):** `Qwen2.5-Coder-32B-Instruct`  
**Tier L (Large):** `Qwen2.5-72B-Instruct-AWQ`

Qwen2.5-Coder выбраны потому что они code-focused и хорошо знают sklearn/pandas.
DeepSeek-R1-Distill можно добавить как интересный эксперимент (reasoning model на DS).

### Обслуживание через vLLM

```bash
# Запуск (один раз на модель, затем гоним эксперименты)
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768

# Смена модели — остановить, запустить другую на том же порту
```

### Параллельность

- vLLM обрабатывает concurrent requests к одной модели: можно запускать несколько агентов
  параллельно на одном датасете (разные random seeds / разные режимы).
- 7B модель: можно запустить 5-8 одновременных сессий
- 70B модель: 1-2 одновременных сессии

### Последствия
- `scripts/start_vllm.sh` — скрипт запуска сервера под каждую модель
- Добавить `vllm` в `requirements.txt` (серверная зависимость, можно в отдельный файл)
- `LLM_BASE_URL=http://localhost:8000/v1` для локальных моделей

---

## ADR-005: Бюджет шагов и токенов

**Статус:** ✅ Принято  
**Дата:** 2026-05-27

### Контекст
Облачные LLM стоят денег → осторожные лимиты (15 шагов, 4096 токенов).
Локальный vLLM → inference бесплатный → можно быть щедрее.

### Решение

| Параметр | Облако | Локальный vLLM |
|----------|--------|----------------|
| `max_steps` | 15 | 30 |
| `max_tokens` (per response) | 4096 | 8192 |
| Timeout per step | — | 30 сек |

Добавить `--mode cloud/local` флаг в CLI, который автоматически выставляет дефолты.

### Последствия
- `run_gym.py` принимает `--mode {cloud,local}` с разными дефолтами
- Важно для честного сравнения: при сравнении облако vs локальная использовать
  одинаковые `max_steps` и считать total_tokens, не время

---

## ADR-006: Checklist — эвристики vs LLM-judge

**Статус:** ⏳ Отложено (v2)  
**Дата:** 2026-05-27

### Контекст
Текущий `checklist.py` использует keyword/regex matching.
Плюс: быстро, без дополнительных API-вызовов.
Минус: хрупко — LLM может сделать правильные вещи с нестандартными именами функций.

### Варианты
- **Regex (текущее):** быстро, детерминированно, легко отлаживать
- **LLM-judge:** отдельный вызов малой модели (7B) оценивает код — надёжнее, но
  добавляет задержку и токены к каждому шагу

### Решение (временное)
**Остаться на regex для MVP.** После первых экспериментов — посмотреть сколько
false negatives у чеклиста на реальных прогонах. Если много — добавить LLM-judge
как опциональный режим (`--judge-model`).

---

## ADR-007: Датасеты — фиксированные сплиты

**Статус:** ✅ Принято  
**Дата:** 2026-05-27

### Контекст
Для честного сравнения нужно чтобы все модели и все режимы работали на одних и тех же
train/val/test сплитах. Сейчас `run_gym.py` генерирует сплиты из CSV каждый раз
(random_state=42, повторяемо, но зависит от версии pandas).

### Решение
**Один раз создать сплиты и сохранить как отдельные CSV.**

```
datasets/
  wine_quality/
    train.csv
    val.csv
    test.csv
    meta.json  ← target_col, metric, источник, seed
```

Скрипт `scripts/prepare_datasets.py` генерирует эти файлы.
После этого эксперименты читают готовые сплиты — гарантированная воспроизводимость.

### Последствия
- `scripts/prepare_datasets.py` — написать
- `run_gym.py` принимает `--dataset-dir datasets/wine_quality/` вместо одного CSV
- `datasets/*/train.csv` — в `.gitignore` (большие файлы), но `meta.json` — в git

---

## ADR-008: Явный Action / Observation протокол

**Статус:** ✅ Принято и базово реализовано
**Дата:** 2026-05-27

### Контекст
Изначально агент возвращал plain Python code, а `SUBMIT` распознавался как магическая
строка. Это удобно для MVP, но плохо масштабируется: нельзя надёжно различить
намерение агента, сложно логировать действия, трудно добавлять новые action-типы
и легко получить неоднозначный parsing.

### Решение
Ввести явный JSON-контракт:

```json
{"type": "code", "code": "print(train_df.shape)"}
```

```json
{"type": "submit", "model_var": "best_model"}
```

Среда возвращает структурированный `Observation`:

- `stdout`
- `stderr`
- `hints`
- `checklist_coverage`
- `budget_remaining`
- `done`
- `submitted`
- `test_metric`

Workspace создаётся и сбрасывается внутри `GymEnv`, а не снаружи в experiment runner.
В workspace доступны `train_df`, `val_df`, `target_col`, `pd`, `np`; `test_df` туда
не попадает.

### Последствия
- `gym/protocol.py` содержит `Action`, `Observation`, parser и legacy fallback.
- `gym/workspace.py` отвечает за видимый namespace агента.
- `GymEnv.step()` принимает `Action`, dict или JSON string.
- `GymAgent` просит LLM возвращать JSON action, больше не зависит от `SUBMIT`
  и использует OpenAI-compatible `LLMClient`.
- `run_gym.py` не инжектит namespace вручную.
- Добавлены базовые unit tests на протокол и workspace lifecycle.

---

## Итоговый стек (обновлённый)

| Слой | Технология | Статус |
|------|-----------|--------|
| Model serving | vLLM | ✅ ADR-004 |
| LLM client | `openai` SDK + `base_url` | ✅ ADR-001 |
| Agent protocol | JSON Action / Observation | ✅ ADR-008 |
| Sandbox | subprocess + timeout | ✅ ADR-002 |
| Experiment tracking | MLflow (local) | ✅ ADR-003 |
| Data | pandas, numpy | Без изменений |
| ML | scikit-learn, xgboost, lightgbm | Добавить xgb/lgbm |
| Metrics | sklearn.metrics | Без изменений |
| Env config | python-dotenv | Добавить |
| Python | 3.11+ | Без изменений |

### requirements.txt (целевой)
```
# LLM
openai>=1.40.0
python-dotenv>=1.0.0

# Data & ML
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Experiment tracking
mlflow>=2.14.0

# Server-side only (на H200, не обязательно у всех)
# vllm>=0.5.0
```

---

## ADR-009: Notebook-like CellHistory поверх Workspace

**Статус:** ✅ Принято и базово реализовано

**Дата:** 2026-05-28

### Контекст
LLM работает удобнее, когда среда похожа на notebook: каждый ход является новой
ячейкой, предыдущие переменные сохраняются, а вывод и ошибки видны рядом с кодом.
Один только `Workspace` хранит runtime-переменные, но не дает явной notebook-like
истории действий.

### Решение
**Добавить `CellHistory` как отдельный слой поверх `Workspace`.**

- `Workspace` остается источником живых Python-переменных между шагами.
- `CellHistory` хранит ячейки: code, stdout, stderr, hints, coverage, budget,
  submit result.
- `GymAgent` добавляет компактный recent notebook context в feedback для LLM.
- `run_gym.py` логирует `cell_history.md` как MLflow artifact.

### Последствия
- LLM может работать инкрементально, как в notebook, без переписывания всего кода
  с нуля.
- История сессии становится удобной для анализа неудачных запусков.
- Контекст ограничивается последними ячейками, чтобы не раздувать токены.

---

## ADR-010: Raw-validation model readiness diagnostics

**Статус:** ✅ Принято и базово реализовано

**Дата:** 2026-05-28

### Контекст
LLM часто делает preprocessing вручную на train, например `pd.get_dummies`, а
затем сохраняет только estimator. На hidden submit среда вызывает
`model.predict(raw_X_test)`, и такой estimator падает, потому что test не
преобразован тем же способом. Давать LLM доступ к `test.csv` нельзя.

### Решение
**Проверять submit-кандидатов на raw validation features до hidden-test submit.**

- После code action, если в workspace есть `best_model` или `model`, Gym вызывает
  `model.predict(raw_X_val_sample)`.
- Если predict падает, observation получает `[MODEL CHECK]` feedback.
- Submit по такой модели не закрывает среду, чтобы агент мог исправить pipeline.
- Если модель проходит validation check, но падает уже на hidden test, Gym
  закрывает submit безопасным generic-сообщением без деталей hidden values.

### Последствия
- LLM получает ранний сигнал, что preprocessing должен быть внутри submitted
  model/pipeline.
- Hidden test остается скрытым.
- Ошибка `train encoded, test raw` ловится до финального submit в большинстве
  практических случаев.

---

## Статус реализации (2026-05-29)

Все ADR реализованы. Стек запущен в production на booml@10.8.52.11.

| ADR | Задача | Статус |
|-----|--------|--------|
| ADR-001 | agent.py на openai SDK + LiteLLM routing | ✅ Done |
| ADR-002 | subprocess + pickle tempfiles + `cwd=tmpdir` | ✅ Done |
| ADR-003 | MLflow в run_gym/baseline/multishot/fixed + compare.py | ✅ Done |
| ADR-004 | vLLM на H200; deepseek-v4-flash через cloud endpoint | ✅ Done |
| ADR-005 | `--mode cloud/local` в CLI | ✅ Done |
| ADR-006 | Checklist regex (v1); LLM-judge отложен на v2 | ⏳ v2 |
| ADR-007 | `scripts/prepare_datasets.py` + 5 датасетов | ✅ Done |
| ADR-008 | JSON Action / Observation протокол | ✅ Done |
| ADR-009 | CellHistory поверх Workspace | ✅ Done |
| ADR-010 | Pre-flight validation + `[MODEL CHECK]` hints | ✅ Done |
| ADR-011 | Real Jupyter notebook + ipykernel backend | ✅ Done |
| ADR-012 | Persistent interaction + mandatory clean replay | ✅ Done |
| ADR-013 | Environment-controlled validate before submit | ✅ Done |
| ADR-014 | Feedback channel separation | ✅ Done |
| ADR-015 | Generic selective hidden-checklist policy | ✅ Done |

---

## ADR-011: Real Jupyter notebook as the iterative environment

**Status:** Accepted
**Date:** 2026-05-29

### Context
The previous runtime behaved like a notebook but stored a custom cell history
and executed code through a subprocess namespace. That was useful for early
experiments, but it could not faithfully model notebook editing, rich outputs,
dirty interactive state, or clean replay.

### Decision
Use a real nbformat v4 `.ipynb` file and a persistent `ipykernel` session for
iterative Gym episodes. `NotebookDocument` owns the notebook document and
`JupyterKernelSession` executes authored code cells through the Jupyter message
protocol. Legacy `code` actions are mapped to "add code cell and execute".

### Consequences
Episodes now save `solution.ipynb`, `final_notebook.ipynb`,
`final_notebook.py`, `notebook_events.json`, `feedback_trace.json`,
`validation_trajectory.json`, and `episode_summary.json`. Rich outputs stay in
the notebook, while agent feedback receives compact text.

---

## ADR-012: Persistent interaction plus mandatory clean replay

**Status:** Accepted
**Date:** 2026-05-29

### Context
Interactive notebooks can contain stale variables from deleted or edited cells.
Validation based only on the current kernel can accept a solution that cannot be
reproduced.

### Decision
`restart_and_run_all` terminates the current kernel, starts a new kernel,
injects the allowed bootstrap context, executes the current notebook top to
bottom, saves real outputs, and records a clean-run id. Editing or running cells
after a clean run invalidates validation until another clean run succeeds.

### Consequences
`validate` and `submit` are blocked unless the current notebook revision matches
the latest successful clean run.

---

## ADR-013: Environment-controlled validate before submit

**Status:** Accepted
**Date:** 2026-05-29

### Context
Candidate readiness checks after every cell are expensive and can produce side
effects. Hidden test evaluation must remain one-shot and private.

### Decision
The agent must call `validate` explicitly. The environment extracts the named
model from the clean-run kernel, checks `predict`, evaluates raw validation
features host-side, stores a `CandidateRecord`, and returns only the validation
metric. `submit` is allowed only for the validated clean-run candidate.

### Consequences
`[MODEL CHECK]` is contract feedback during validate/submit readiness checks,
not an automatic post-cell side effect.

---

## ADR-014: Feedback channel separation

**Status:** Accepted
**Date:** 2026-05-29

### Context
Runtime errors, contract violations, checklist hints, and private evaluator
state have different visibility rules.

### Decision
Use explicit feedback items with `runtime`, `contract`, `checklist`, and
`terminal` channels. Runtime and contract feedback can be visible to all
iterative modes. Checklist feedback is visible only in `gym_with_checklist`.
Terminal/private evaluation details are never sent to the agent.

### Consequences
Agent-facing messages no longer contain private checklist coverage or hidden
test metrics.

---

## ADR-015: Generic selective hidden-checklist policy

**Status:** Accepted
**Date:** 2026-05-29

### Context
Dataset-specific hints can leak facts that the LLM should discover by analysis.
Mandatory tuning/feature-engineering requirements can make small-budget agents
worse.

### Decision
Use a generic checklist covering task understanding, schema review, target
distribution, missing values, categorical/cardinality audit, duplicates,
suspicious columns, target exclusion, reproducibility, validation, and
raw-input readiness. Emit at most one generic hint per eligible execution, with
cooldown and suppression on runtime or contract blockers.

### Consequences
Coverage is private. Hints ask the agent to check whether issues exist; they do
not reveal dataset properties or prescribe specific sklearn components.

---

## ADR-016: Hidden test score privacy

**Status:** Accepted
**Date:** 2026-05-29

### Context
Returning the hidden score to the LLM after submit contaminates interaction
history and weakens experimental validity.

### Decision
Successful submit returns only `[SUBMITTED] Final candidate accepted. Episode
finished.` to agent-facing context. Hidden score is stored only in private
summary and MLflow metrics. Hidden-test failures remain generic and omit hidden
rows, labels, categories, and score.

### Consequences
Tests assert the hidden score is absent from feedback traces and notebook
outputs while still present in private summaries.

---

## ADR-017: Iterative no-checklist as the fair control

**Status:** Accepted
**Date:** 2026-05-29

### Context
The previous multishot runner was not the right control for checklist feedback
because it did not use the same notebook environment.

### Decision
Define `EpisodeMode` and run both `iterative_no_checklist` and
`gym_with_checklist` through the same Jupyter backend, action protocol, budget,
runtime feedback, and contract feedback. The only behavioral difference is
checklist feedback visibility.

### Consequences
`experiments.run_multishot` is logged as `repeated_single_shot`; the fair
checklist ablation is `experiments.run_gym --episode-mode iterative_no_checklist`.

---

## ADR-018: Security boundary for the local Jupyter backend

**Status:** Accepted
**Date:** 2026-05-29

### Context
A real local Jupyter kernel gives notebook fidelity but not full isolation for
untrusted code.

### Decision
For this PR, use `LocalJupyterKernelBackend`, sanitize common secret
environment variables, and physically keep hidden evaluator artifacts outside
the episode workspace. Define `KernelExecutionBackend` and a future
`ContainerJupyterKernelBackend` placeholder for container isolation.

### Consequences
Hidden test isolation is enforced now. Full filesystem/network/CPU/RAM sandbox
for notebook kernels remains future work and is documented as a limitation.
>>>>>>> eba926e (feat(gym): add real jupyter notebook environment)
