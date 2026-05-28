# AutoVibe Gym — Project Specification

## 1. Суть кейса

Цель — построить **среду (gym)**, в которой языковая модель (LLM) итеративно решает задачи машинного обучения, и **доказать**, что такая среда повышает качество решения по сравнению с single-shot запросом.

Ключевая гипотеза: LLM уже умеет решать DS-задачи, но ей нужна структурированная обратная связь, чтобы довести решение до максимума. Среда эту обратную связь обеспечивает.

---

## 2. Цели

### Основная
Создать воспроизводимую среду (Gym), в которой LLM-агент итеративно улучшает ML-решение, получая неявные подсказки через чеклист DS-пайплайна.

### Исследовательские
- Сравнить метрику на тесте: Gym vs. single-shot vs. multi-shot без среды vs. free-agent
- Измерить покрытие чеклиста и количество критических ошибок по типам LLM
- Проверить: улучшается ли слабая LLM в среде до уровня более сильной без среды?

---

## 3. Технические требования (ТЗ)

### 3.1 Среда (GymEnv)

| Требование | Описание |
|---|---|
| Изоляция кода | LLM-код исполняется в изолированном namespace; тест-выборка недоступна |
| Итеративность | Цикл: JSON action → execution → observation → JSON action, до N шагов или `submit` action |
| Submit gate | `env.submit(model)` вызывается ровно один раз; возвращает финальную метрику и закрывает сессию |
| Бюджет | Агент знает сколько шагов осталось; при исчерпании — авто-сабмит лучшей модели из namespace |
| Логирование | Каждый шаг пишется в историю: код, stdout, stderr, хинты, покрытие чеклиста |

### 3.2 Чеклист

8 обязательных пунктов DS-пайплайна. Проверки **неявные** — LLM получает хинт-намёк, не прямую инструкцию.

| Пункт | Триггер проверки |
|---|---|
| `eda` | `.describe()`, `.info()`, `.head()`, `.shape` |
| `missing_values` | `isnull`, `fillna`, `dropna`, `impute` |
| `duplicates` | `drop_duplicates`, `duplicated()` |
| `target_leak` | Явное отделение X от y (`.drop(target_col)`) |
| `train_val_split` | Использование `val_df` или `train_test_split` |
| `feature_engineering` | Логарифм, дамми, скейлер, polynomial и др. |
| `model_selection` | ≥2 разных классов моделей в истории сессии |
| `hyperparameter_tuning` | `GridSearchCV`, `n_estimators=`, `Optuna` и др. |

### 3.3 Агент

- Один LLM-агент (не мульти-агент на старте, можно расширить)
- Общается с API (Anthropic / OpenAI / local)
- Получает от LLM явный JSON action:
  - `{"type": "code", "code": "..."}`
  - `{"type": "submit", "model_var": "best_model"}`
- Получает observation: stdout + stderr + pending hints + checklist coverage + оставшийся бюджет

### 3.4 Датасеты

**Критерии выбора:**
- Не слишком популярный (не Titanic, не Iris — LLM знает ответ наизусть)
- Не слишком непопулярный (должен встречаться в интернете хотя бы в нескольких ноутбуках)
- Бинарная классификация или регрессия, табличные данные
- Размер: 1k–50k строк, 5–30 признаков

**Кандидаты:**
- Wine Quality (Kaggle, умеренно популярный)
- Breast Cancer Wisconsin (sklearn built-in, но не Titanic-уровень)
- Bank Marketing (UCI)
- Heart Disease Cleveland (UCI)

### 3.5 Метрики сравнения

| Метрика | Описание |
|---|---|
| `test_metric` | Финальный score на тест-выборке (f1_weighted / neg_rmse) |
| `checklist_coverage` | Доля закрытых пунктов чеклиста (0.0 – 1.0) |
| `error_count` | Количество шагов с непустым stderr |
| `steps_used` | Шагов потрачено до submit |
| `input_tokens` / `output_tokens` | Бюджет токенов на сессию |

---

## 4. Архитектура

```
┌───────────────────────────────────────────────────────────┐
│                       GymAgent                            │
│  OpenAI-compatible LLMClient ─► Action JSON ─► parser     │
│      ▲                                      │              │
│      └──────── Observation feedback ◄──────┘              │
└───────────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────┐
│                        GymEnv                             │
│                                                           │
│  EnvState                                                 │
│  ├── train / val / test (test hidden from workspace)       │
│  ├── Workspace(namespace: train_df, val_df, target_col)    │
│  ├── CellHistory(notebook-like cells with outputs)         │
│  ├── history: List[Observation]                            │
│  └── step counter / submitted flag                         │
│                                                           │
│  CodeExecutor ──► subprocess(code, workspace.namespace)    │
│  Checklist    ──► evaluate() → implicit hints             │
│  submit       ──► metric_fn(hidden test) → final score     │
└───────────────────────────────────────────────────────────┘
```

### Файловая структура

```
autovibe-gym/
├── AGENTS.md                  ← инструкции для Codex
├── CLAUDE.md                  ← инструкции для Claude
├── README.md
├── requirements.txt
│
├── gym/
│   ├── __init__.py
│   ├── env.py                 ← GymEnv, EnvState, Observation history
│   ├── executor.py            ← subprocess CodeExecutor с timeout
│   ├── checklist.py           ← Checklist, CheckItem, HINTS
│   ├── protocol.py            ← Action / Observation JSON-контракт
│   ├── workspace.py           ← видимый namespace агента без test_df
│   ├── cell_history.py        ← notebook-like история ячеек и outputs
│   ├── datasets.py            ← DatasetSplits, meta.json loader, metric resolver
│   ├── llm.py                 ← OpenAI-compatible LLMClient adapter
│   └── agent.py               ← GymAgent JSON action loop
│
├── experiments/
│   ├── run_gym.py             ← Gym режим (CLI)
│   ├── run_baseline.py        ← Single-shot (без среды)
│   ├── run_multishot.py       ← Multi-shot (без чеклиста)   [TODO]
│   └── compare.py             ← Сводная таблица по запускам
│
├── datasets/
│   ├── *.csv                  ← legacy single-file mode
│   └── <dataset_name>/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── meta.json          ← target_col, metric, source, seed
│
├── docs/
│   ├── GIT_WORKFLOW.md        ← правила Git, веток, PR и AI-агентов
│   ├── PROJECT.md             ← этот файл
│   └── STATUS.md              ← живой статус проекта
│
└── memory/                    ← Claude memory
    └── project_context.md
```

---

## 5. Стек

| Слой | Технология | Причина |
|---|---|---|
| Agent protocol | JSON `Action` / structured `Observation` | Явный контракт между LLM и средой |
| LLM client | `LLMClient` + `openai` SDK + `LLM_BASE_URL` | Работает с vLLM (H200), OpenAI, любым OpenAI-совместимым прокси |
| Model serving | `vLLM` (на сервере H200) | OpenAI-compatible API, лучший throughput |
| Sandbox | `subprocess` + timeout + pickle | Изолированный процесс, сервер-safe |
| Experiment tracking | `MLflow` (local) | Без облака, visual UI, сравнение runs |
| Data | `pandas`, `numpy` | Стандарт DS |
| ML | `scikit-learn`, `xgboost`, `lightgbm` | LLM хорошо знает эти библиотеки |
| Metrics | `sklearn.metrics` | |
| Config | `python-dotenv` | `.env` файл для API keys и URL |
| CLI | `argparse` | Без фреймворков |
| Python | 3.11+ | Type hints, `X \| Y` union syntax |

---

## 6. Эксперименты (план)

| Режим | Файл | Статус |
|---|---|---|
| **Gym** — итеративно с чеклистом | `experiments/run_gym.py` | Готово |
| **Single-shot** — один запрос | `experiments/run_baseline.py` | Готово |
| **Multi-shot** — несколько запросов, тот же бюджет токенов, без чеклиста | `experiments/run_multishot.py` | Готово |
| **Free-agent** — делает что хочет, без структуры | (расширение gym без checklist) | TODO |
| **Compare** — сводная таблица всех режимов | `experiments/compare.py` | Готово |

### Ожидаемый результат
Gym > Multi-shot > Single-shot по `test_metric`.  
Gym > Free-agent на малых/средних LLM.

---

## 7. Ограничения и допущения

- Тест-выборка недоступна агенту до `submit()` — нет утечки данных
- Один `submit()` на сессию — имитирует реальный деплой
- Code actions ведут себя как notebook cells: код добавляется шагами, а переменные живут в Workspace между шагами
- Проверки чеклиста — эвристические (regex + keyword match), не семантические
- Sandbox не изолирован на уровне ОС (нет Docker) — запускать только доверенные агенты
- Датасеты — только табличные, CSV

---

## 8. Команда и таймлайн

- Команда: ~4 человека
- Дедлайн: ~2 недели от 27.05.2026 (т.е. ~10.06.2026)
- Формат сдачи: презентация + демо запуска
