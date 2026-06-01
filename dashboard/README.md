# AutoVibe Gym — Local Dashboard

Локальная веб-панель для запуска LLM-агентов на AutoVibe Gym: выбрать модель +
тип прогона + датасет, настроить бюджет, запустить, наблюдать в реальном времени,
читать ноутбук/траекторию решения, видеть скор и ошибки, управлять датасетами и
моделями, сравнивать прогоны.

Это **отдельный** от `gym/` модуль. Бэкенд переиспользует тот же `.venv` проекта
(в нём уже есть FastAPI/uvicorn/MLflow/pandas) и запускает раннеры `experiments.*`,
читая их результаты из MLflow.

```
dashboard/
├── server/            FastAPI backend
│   ├── app/           роутеры (health, settings, datasets, models, runs) + сервисы
│   ├── requirements.txt
│   └── run.sh         запуск через .venv проекта
└── web/               Vite + React + TypeScript фронтенд
    └── src/           токены, UI-примитивы, экраны, API-клиент
```

## Запуск

**Бэкенд** (из корня репозитория):

```bash
dashboard/server/run.sh                 # → http://127.0.0.1:8000  (docs: /docs)
# или: .venv/bin/python -m uvicorn dashboard.server.app.main:app --reload --port 8000
```

**Фронтенд:**

```bash
cd dashboard/web
npm install
npm run dev                             # → http://localhost:5173  (проксирует /api → :8000)
```

## Архитектура

- Прогоны = процессы `experiments.run_gym / run_baseline / run_multishot`, логируются
  в MLflow (`mlflow.db` + `mlruns/` в корне репо). Бэкенд читает их оттуда и из
  артефактов эпизода (`solution.ipynb`, `validation_trajectory.json`, `episode_summary.json`).
- 4 режима UI → 3 раннера: single→baseline, repeated→multishot,
  iterative→`run_gym --episode-mode iterative_no_checklist`, gym→`run_gym --episode-mode gym_with_checklist`.
- Датасеты читаются из `datasets/<name>/prepared/meta.json`; загрузка кладёт CSV туда же.
- Реестр моделей хранится в `dashboard/server/data/models.json` (у gym своего реестра нет).

Дизайн-референс: `~/Desktop/design` (hi-fi прототип + токены). Воссоздаём в этом стеке.
```
