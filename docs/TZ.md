# ТЗ: AutoML Gym Challenge

> Источники: устная постановка задачи (27.05.2026, приоритет) + официальный brief (PDF)

---

## Суть и цель

Спроектировать среду, которая **честно оценивает LLM-агентов**, решающих табличные ML-задачи через AutoML-процесс.

**Цель — не максимальный score на одном датасете, а проверяемая среда оценки.**

Ваш результат должен показать: может ли LLM-агент улучшать AutoML-решение **безопасно, воспроизводимо и измеримо**.

Главный вопрос: **может ли LLM-агент улучшать AutoML-решение через контролируемое взаимодействие?**

---

## Почему нужна такая среда

Финальная метрика не показывает, где агент ошибся: в планировании, EDA, feature engineering, model selection, validation или submission.

Нужна среда, где агент взаимодействует с задачей по правилам, получает разрешённый feedback и сдаёт воспроизводимый артефакт.

Оценивается не только **«что получилось»**, но и **«как агент улучшал решение»**.

---

## Что должна уметь среда

| Компонент | Описание |
|---|---|
| **Task interface** | train / validation / private final split; metric; task metadata |
| **Action protocol** | разные режимы: без feedback, с жёсткими переходами, с гибкими переходами |
| **Candidate control** | регистрация, validation, выбор/фиксация модели перед private test |
| **Safe execution** | sandbox, reproducibility, no leakage, raw-input submission |
| **Trajectory audit** | логи действий, failures, budget, selected candidate, final outcome |
| **Extensibility** | новые табличные задачи без переписывания всей среды |

**Сильное решение делает ошибку агента наблюдаемой, а private test — защищённым.**

---

## Режимы взаимодействия (все 4 нужно реализовать и сравнить)

| Режим | Что разрешено агенту | Что проверяет |
|---|---|---|
| **Single-shot** | Один полный ответ с моделью; без environment feedback | Сила первичной генерации |
| **Repeated single-shot** | Несколько независимых попыток; между ними только scalar validation metric | Отделяет iteration от перебора |
| **Fixed transitions** | Feedback доступен, но порядок этапов задан заранее | Полезна ли структурированная траектория |
| **Flexible transitions** | Feedback доступен, агент сам выбирает следующий шаг | Полезна ли автономная доработка |

> **Важно:** Single-shot и repeated single-shot **не получают** stage feedback, validator hints, reward или traceback.
> Feedback выдаётся только в интерактивных режимах: fixed и flexible.

---

## Жизненный цикл кандидата

```
Train → Validate → Choose → Replay → Final
```

| Шаг | Описание |
|---|---|
| **Train** | Агент строит candidate-модель |
| **Validate** | Среда даёт разрешённый score на validation |
| **Choose** | Агент явно фиксирует модель ИЛИ среда берёт best validation |
| **Replay** | Кандидат воспроизводится с raw held-out rows **без ручной доработки** |
| **Final** | Private test — **один раз**, результат необратим |

**Правило выбора:** в режимах с feedback агент должен иметь возможность явно выбрать валидированный кандидат.

**Fallback:** если выбор невалиден или отсутствует, среда использует лучший кандидат по validation metric.

> **Private final score не должен быть инструментом выбора модели.**

---

## Чеклист (для интерактивных режимов)

Набор этапов DS-пайплайна, которые среда проверяет **неявно**. Фидбэк — намёки, не прямые инструкции.

| Этап | Пример намёка (не инструкции!) |
|---|---|
| EDA | «Вы исследовали структуру данных перед моделированием?» |
| Пропущенные значения | «Некоторые столбцы могут содержать пропуски — стоит на это обратить внимание» |
| Дубликаты | «Дублирующиеся строки могут искажать обучение» |
| Утечка таргета | «Убедитесь, что ни одна фича не является прямым производным таргета» |
| Разделение train/val | «Полезно оценивать модель на отложенной выборке в процессе разработки» |
| Feature engineering | «Сырые признаки не всегда оптимальны — пробовали преобразования?» |
| Выбор модели | «Сравнивали ли вы несколько типов моделей?» |
| Гиперпараметры | «Дефолтные параметры редко оптимальны» |

- ❌ Плохо: «Тебе нужно вызвать `drop_duplicates()`»
- ✅ Хорошо: «Возможно, стоит проверить наличие дублирующихся строк»

---

## Датасеты

- Не слишком игрушечные (не Titanic, не Iris)
- Не слишком редкие (LLM должна иметь базовое понимание предметной области)
- Фиксированные сплиты train / val / test (воспроизводимость)
- Минимум несколько датасетов разного типа для проверки extensibility

---

## Эксперименты

Эксперименты должны **доказывать свойства среды**, а не только качество одной модели.

| Эксперимент | Зачем нужен | Ожидаемый вывод |
|---|---|---|
| **Mode comparison** | single-shot vs repeated vs fixed vs flexible | Помогает ли interaction |
| **Validation-final gap** | validation score vs private final score | Есть ли overfitting к feedback |
| **Cost-quality** | score на токен / вызов / секунду | Стоит ли interaction своих затрат |
| **Robustness** | leakage-seeking, sloppy, non-reproducible agents | Не ломается ли среда |
| **Extensibility** | новая табличная задача | Не привязана ли среда к одному датасету |
| **Failure taxonomy** | план, данные, модель, submission, reproducibility | Даёт ли среда диагностику |

---

## Что нужно сдать

| Артефакт | Описание |
|---|---|
| **Prototype** | Запускаемая среда или сервис для LLM AutoML agents |
| **Protocol spec** | Описание режимов, доступного feedback и privacy boundary |
| **Experiment report** | Сравнение режимов, cost-quality и failure modes |
| **Trajectory logs** | Действия агента, validation events, candidate choice, final outcome |
| **Reproducibility guide** | Как повторить запуск и добавить новую задачу |
| **Demo task** | Пример на одной или нескольких табличных задачах |

> **Отчёт должен объяснять не только победителей, но и причины провалов.**

---

## Критерии оценки

| Критерий | Вес |
|---|---|
| Privacy & isolation | **25%** |
| Interaction protocol | **20%** |
| Candidate/replay correctness | **15%** |
| Diagnostics & feedback | **15%** |
| Experiments & statistics | **15%** |
| Extensibility & reproducibility | **10%** |

**Низкий балл:** leaderboard без диагностики процесса и без защиты final evaluation.

**Высокий балл:** протоколы, логи и эксперименты показывают, почему агент улучшился или провалился.

---

## Статус реализации (актуально на 2026-05-29)

| Компонент | Статус | Примечание |
|---|---|---|
| `NotebookGymEnv` — среда на реальном Jupyter kernel | ✅ | основная среда |
| `GymEnv` (legacy) — subprocess sandbox | ✅ | retained for compatibility |
| `GymAgent` — LLM-агент с JSON-действиями | ✅ | |
| `Checklist` — 8 пунктов с generic hints | ✅ | режим gym_with_checklist |
| `CodeExecutor` — Docker/subprocess sandbox | ✅ | legacy path |
| `ContainerJupyterKernelBackend` — Docker kernel | ✅ | `AUTOVIBE_KERNEL_BACKEND=docker` |
| `run_baseline.py` — single-shot | ✅ | |
| `run_multishot.py` — repeated single-shot | ✅ | N независимых попыток, только scalar val metric |
| `run_fixed.py` — fixed transitions | ✅ | legacy GymEnv runtime |
| `run_gym.py` — flexible transitions | ✅ | NotebookGymEnv |
| `run_matrix.py` — batch experiment runner | ✅ | N датасетов × M режимов |
| `iterative_no_checklist` — fair control | ✅ | тот же backend, без hints |
| Candidate lifecycle: Train→Validate→Replay→Final | ✅ | `restart_and_run_all` + `validate` + `submit` |
| Trajectory artifacts (notebook, feedback_trace, etc.) | ✅ | MLflow artifacts |
| MLflow трекинг | ✅ | все режимы |
| Privacy: test.csv скрыт, score не виден агенту | ✅ | `private_dir` изолирован |
| Датасеты (5 штук с fixed splits) | ✅ | |
| `scripts/prepare_datasets.py` | ✅ | |
| `docs/REPRODUCIBILITY.md` | ✅ | |
| CI (GitHub Actions) + Docker integration tests | ✅ | `tests.yml` + `docker-sandbox.yml` |

## Открытые вопросы (архитектурные решения)

### Fallback auto-submit — намеренный строгий контракт

ТЗ (строка 74): *«среда использует лучший кандидат по validation metric»*.

В `NotebookGymEnv` fallback (`_try_forced_submit`) работает только если был успешный `validate`. Это **намеренно строже**: без clean replay нет гарантии воспроизводимости на raw test rows.

**Решение:** поведение задокументировано в `docs/PROTOCOL.md`. Для legacy `GymEnv` fallback по-прежнему работает через `workspace.first_existing`.

### Эксперименты на legacy runtime

Результаты в `EXPERIMENT_REPORT.md` получены на legacy `GymEnv` (subprocess). Notebook-era эксперименты запускаются через `run_matrix.py` на H200.

### Data

`naticusdroid` — 74.5% дублей исправлено (`"deduplicate": true` в config). `dry_bean` требует `openpyxl` (есть в `requirements.txt`, нужна пересборка Docker-образа).

---

## Дедлайн

2 недели от постановки задачи (27.05.2026).
