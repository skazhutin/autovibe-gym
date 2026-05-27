# Git Workflow — AutoVibe Gym

Нас трое, у каждого свой AI-агент (Claude Code, Codex, и др.) на своей машине.
Агенты генерируют код быстро — без договорённостей получим конфликты и сломанный `main`.

---

## Золотые правила

1. **`main` — всегда рабочий.** Прямой пуш в `main` запрещён.
2. **Каждый работает в своей ветке.** Никаких общих feature-веток без договорённости.
3. **Пушим ветку после каждой сессии.** Даже незаконченный код — чтобы другие видели прогресс.
4. **Перед мёржем — `rebase` от `main`.** Не `merge`, чтобы история была линейной.
5. **Датасеты не коммитим.** Только скрипт загрузки или ссылка.

---

## Быстрый старт (для нового участника)

```bash
# 1. Клонировать
git clone https://github.com/skazhutin/autovibe-gym.git
cd autovibe-gym

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Создать свою ветку
git checkout -b dev/<твоё-имя>-<что-делаешь>
# Примеры:
#   dev/klim-gym-env
#   dev/nikita-checklist
#   dev/masha-experiments

# 4. Работать, коммитить, пушить
git add <файлы>
git commit -m "feat: описание что сделал"
git push -u origin dev/<твоя-ветка>
```

---

## Именование веток

```
dev/<имя>-<фича>
```

| Пример | Смысл |
|--------|-------|
| `dev/klim-gym-env` | Klim пишет GymEnv |
| `dev/nikita-checklist` | Nikita пишет чеклист |
| `dev/masha-experiments` | Masha пишет эксперименты |

Не называть ветки `fix`, `update`, `test` — непонятно кто и что.

---

## Стиль коммитов (Conventional Commits)

```
<тип>: короткое описание (до 72 символов)
```

| Тип | Когда использовать |
|-----|-------------------|
| `feat:` | новая функциональность |
| `fix:` | исправление бага |
| `chore:` | инфра, зависимости, конфиги |
| `docs:` | документация |
| `test:` | тесты |
| `refactor:` | рефактор без изменения поведения |
| `exp:` | эксперимент / исследование |

**Примеры:**
```
feat: add Checklist with 8 DS pipeline checks
fix: executor strips private namespace keys
docs: add GIT_WORKFLOW guide
exp: run gym vs baseline on wine_quality dataset
chore: add .gitignore, requirements.txt
```

---

## Синхронизация с main (делать регулярно)

```bash
# Забрать последние изменения из main в свою ветку
git fetch origin
git rebase origin/main

# Если конфликты — решить, потом:
git rebase --continue

# Запушить (после rebase нужен --force-with-lease)
git push --force-with-lease
```

---

## Влить свою ветку в main

Два варианта — договоритесь какой используете:

### Вариант A: GitHub Pull Request (рекомендуется)
```bash
git push origin dev/<твоя-ветка>
# Открыть PR на GitHub, попросить кого-то из команды посмотреть
```

### Вариант B: Локальный merge (если некогда)
```bash
git checkout main
git pull origin main
git merge --no-ff dev/<твоя-ветка>
git push origin main
```

---

## Работа с AI-агентами (Claude Code / Codex / др.)

AI-агенты пишут много кода за один раз — это хорошо, но создаёт риски:

**Правила:**
- Агент работает только в вашей ветке, никогда не пушит напрямую в `main`
- Перед большим изменением — `git stash` или промежуточный коммит
- После сессии агента — обязательно просмотреть `git diff` перед пушем
- Директория `.claude/`, `.codex/`, `memory/` — в `.gitignore`, не коммитить

**Что делать если агент накосячил:**
```bash
# Отменить все незакоммиченные изменения
git restore .

# Откатить последний коммит (сохранив файлы)
git reset --soft HEAD~1
```

---

## Структура веток на сейчас

```
main
├── dev/klim-gym-env        ← gym/, executor, env
├── dev/...-checklist       ← checklist, hints
└── dev/...-experiments     ← experiments/, datasets/
```

---

## Что НЕ коммитить

| Что | Почему |
|-----|--------|
| `datasets/*.csv` | Большие файлы, у каждого локально |
| `.env` | Секреты (API ключи) |
| `memory/`, `.claude/` | Локальные конфиги AI-агентов |
| `__pycache__/`, `*.pyc` | Артефакты Python |
| `runs/`, `outputs/` | Результаты экспериментов (логи отдельно) |

---

## FAQ

**Q: Агент закоммитил `.env` случайно — что делать?**
```bash
git rm --cached .env
echo ".env" >> .gitignore
git commit -m "chore: remove .env from tracking"
# Сменить API ключ — он уже скомпрометирован
```

**Q: Хочу посмотреть что делает другой участник**
```bash
git fetch origin
git checkout dev/<его-ветка>
```

**Q: Конфликт при rebase**
Открыть файл, найти `<<<<<<`, решить вручную, `git add <файл>`, `git rebase --continue`.
