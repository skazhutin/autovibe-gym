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

## Итоговый стек (обновлённый)

| Слой | Технология | Статус |
|------|-----------|--------|
| Model serving | vLLM | ✅ ADR-004 |
| LLM client | `openai` SDK + `base_url` | ✅ ADR-001 |
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

## Что требует реализации (приоритет)

| Приоритет | ADR | Задача | Затронутые файлы |
|-----------|-----|--------|-----------------|
| 🔴 High | ADR-001 | Рефактор agent.py на openai SDK | `gym/agent.py`, `requirements.txt` |
| 🔴 High | ADR-002 | Subprocess sandbox | `gym/executor.py` |
| 🔴 High | ADR-007 | Скрипт подготовки датасетов | `scripts/prepare_datasets.py` |
| 🟡 Medium | ADR-003 | MLflow в run_gym.py + compare.py | `experiments/` |
| 🟡 Medium | ADR-005 | `--mode cloud/local` в CLI | `experiments/run_gym.py` |
| 🟢 Low | ADR-004 | vLLM startup скрипт | `scripts/start_vllm.sh` |
| 🟢 Low | ADR-006 | LLM-judge (v2) | `gym/checklist.py` |
