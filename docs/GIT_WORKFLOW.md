# AutoVibe Gym - Git Workflow

Этот документ - единый источник правил по Git, веткам, pull request'ам и
AI-generated коду. Обновляйте его в том же PR, где меняется командный процесс.

## Жесткие правила

1. `main` всегда должен быть рабочим и пригодным для демо.
2. Прямой push в `main` запрещен. Все изменения идут через pull request.
3. Одна ветка - одна понятная задача или один эксперимент.
4. Общие feature-ветки запрещены без явной договоренности команды.
5. Перед готовностью PR ветка должна быть rebased на свежий `origin/main`.
6. В конце каждой рабочей сессии ветку нужно push'ить, даже если PR остается draft.
7. Секреты, локальные датасеты, кеши и outputs экспериментов не коммитятся.
8. В конце каждого PR-цикла обновляется `docs/STATUS.md`.
9. Если меняется командный Git/PR-процесс, обновляется этот файл.

## Что Читать Перед Работой

Перед кодом или планированием PR читать:

- `docs/PROJECT.md` - цели, архитектура, ограничения;
- `docs/STATUS.md` - текущий статус, блокеры, ближайшие действия;
- `docs/GIT_WORKFLOW.md` - правила веток, коммитов, PR и review.

Для AI-агентов `AGENTS.md` и `CLAUDE.md` могут добавлять tool-specific
инструкции, но Git workflow определяется этим файлом.

## Имена Веток

Основной формат:

```bash
dev/<owner>/<topic>
```

Примеры:

```bash
dev/klim/gym-env
dev/nikita/checklist-hints
dev/masha/baseline-runner
```

Ветки, созданные Codex, могут использовать:

```bash
codex/<topic>
```

Одноразовые исследовательские ветки могут использовать:

```bash
exp/<owner>/<topic>
```

Правила:

- lowercase;
- слова разделяются hyphen'ами;
- название должно объяснять владельца и смысл работы;
- не использовать `fix`, `update`, `test`, `final`, `new` как полное имя ветки.

## Стандартный Рабочий Цикл

### 1. Начать С Актуальной Базы

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c dev/<owner>/<topic>
```

Если задача уже идет в существующей ветке:

```bash
git switch dev/<owner>/<topic>
git fetch origin
git rebase origin/main
```

Не начинать крупные изменения от устаревшей ветки.

### 2. Работать Маленькими Шагами

- Держать изменения сфокусированными.
- Делать несколько небольших коммитов вместо одного смешанного.
- Не вносить unrelated formatting churn.
- Не переписывать файлы, которые активно меняет другой участник, без согласования.
- Перед коммитом запускать минимальную релевантную проверку.

### 3. Проверить Свой Diff

Перед коммитом:

```bash
git status --short
git diff --check
git diff
```

Проверить:

- случайные секреты, датасеты, кеши, notebook outputs;
- нерелевантные переписывания файлов;
- debug prints и временный код;
- изменение поведения без обновления тестов или документации.

### 4. Закоммитить И Запушить

```bash
git add <files>
git commit -m "type(scope): short description"
git push -u origin dev/<owner>/<topic>
```

Если remote branch уже существует:

```bash
git push
```

Если работа больше одного маленького коммита, открывайте draft PR рано.

## Commit Messages

Используем Conventional Commits:

```text
<type>(optional-scope): short description
```

Типы:

| Type | Когда использовать |
|------|--------------------|
| `feat` | Новая функциональность |
| `fix` | Исправление бага |
| `docs` | Документация |
| `test` | Тесты |
| `refactor` | Внутренние изменения без изменения поведения |
| `chore` | Зависимости, конфиги, обслуживание |
| `exp` | Исследования и экспериментальные запуски |

Примеры:

```text
feat(env): add submit gate
fix(executor): hide private namespace keys
docs: update git workflow
test(checklist): cover model selection detection
exp(wine): compare gym and baseline runs
```

Правила:

- commit subject по умолчанию пишем на английском;
- первая строка до 72 символов;
- один коммит - одно логическое изменение;
- не смешивать docs, refactor и behavior changes без необходимости.

## Синхронизация С Main

Регулярно rebase'иться от `main`, особенно перед review:

```bash
git fetch origin
git rebase origin/main
```

Если появились конфликты:

1. Открыть конфликтующие файлы и решить конфликт вручную.
2. Сохранить обе стороны, если это два разных реальных изменения.
3. Запустить релевантную проверку после разрешения конфликта.
4. Продолжить rebase:

```bash
git add <resolved-files>
git rebase --continue
```

После успешного rebase обновлять только свою remote branch:

```bash
git push --force-with-lease
```

Никогда не force-push'ить `main`. Никогда не force-push'ить чужую ветку.

Если rebase стал непонятным:

```bash
git rebase --abort
```

После этого лучше согласовать решение с командой.

## Pull Requests

PR обязателен для попадания в `main`.

Рекомендуемый flow:

1. Push ветки.
2. Draft PR как можно раньше.
3. Дальше push маленьких сфокусированных коммитов.
4. Rebase на `origin/main`.
5. Ready for review только после проверок и обновления docs.
6. Merge только после review approval.

Описание PR должно включать:

- context: какую проблему решаем;
- changes: что поменялось в файлах или поведении;
- verification: какие команды запущены и с каким результатом;
- risks: что не проверено, заблокировано или требует внимания.

Checklist перед Ready for review:

- ветка rebased на свежий `origin/main`;
- `docs/STATUS.md` обновлен для этого PR-цикла;
- `docs/GIT_WORKFLOW.md` обновлен, если менялся процесс;
- секреты, датасеты, кеши и outputs не попали в diff;
- релевантные тесты или smoke checks запущены;
- title PR соответствует стилю commit messages.

Предпочтительный merge method: squash merge с чистым Conventional Commit title.
Локальный merge в `main` как shortcut запрещен.

После merge:

```bash
git switch main
git pull --ff-only origin main
git branch -d dev/<owner>/<topic>
git push origin --delete dev/<owner>/<topic>
```

## Минимальная Проверка

Пока полноценного test suite нет, использовать минимальную проверку для
измененной области.

Для Python-only изменений:

```bash
python3 -m compileall gym experiments tests
```

Для изменений `GymEnv` или checklist добавить или запустить smoke script, который
исполняет среду без доступа к hidden test split.

Для agent/API изменений явно писать, запускалась ли команда с реальным API key
или проверка была только статической.

Для docs-only изменений runtime test не нужен, но `git diff --check` должен
проходить.

## Правила Для AI-Агентов

AI-агенты быстро генерируют большие diff'ы, поэтому правила строже:

- работать только в выделенной task branch;
- смотреть `git status --short` перед изменениями;
- не перетирать изменения пользователя или другого участника;
- не запускать destructive Git commands без явного человеческого approval;
- в конце PR-цикла обновлять `docs/STATUS.md`;
- обновлять `docs/GIT_WORKFLOW.md`, если меняется сам процесс;
- показывать или кратко описывать final diff перед merge;
- держать изменения в рамках поставленной задачи.

Для Codex проектные инструкции лежат в `AGENTS.md`.
Для Claude Code tool-specific инструкции лежат в `CLAUDE.md`.

## Shared Files

Эти файлы чаще всего конфликтуют и требуют аккуратности:

| File | Правило |
|------|---------|
| `docs/STATUS.md` | Сверять актуальный статус вручную; changelog entries не удалять |
| `docs/PROJECT.md` | Менять только при изменении scope или архитектуры |
| `docs/GIT_WORKFLOW.md` | Менять только при изменении командного процесса |
| `gym/env.py` | Согласовывать, потому что это core runtime contract |
| `gym/checklist.py` | Хинты должны оставаться implicit, без прямых инструкций LLM |
| `gym/agent.py` | Не hardcode'ить secrets, provider-specific local paths |

## Что Не Коммитить

| Path / Pattern | Почему |
|----------------|--------|
| `.env`, `.env.*`, `*.key` | Secrets |
| `datasets/*.csv`, `datasets/*.parquet`, `datasets/*.xlsx` | Local or large data |
| `runs/`, `outputs/`, `results/`, `*.log`, `*.jsonl` | Generated experiment output |
| `memory/`, `.claude/`, `.codex/`, `.cursor/` | Local AI-agent state |
| `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/` | Generated Python/Jupyter files |
| `.vscode/`, `.idea/`, `.DS_Store` | Local editor/OS files |

Если датасет или output должен быть воспроизводимым, коммитить download script,
metadata или короткий README вместо generated file.

## Recovery And Safety

Предпочитать точечные команды:

```bash
git restore --staged <file>
git restore -- <file>
git reset --soft HEAD~1
```

Не использовать широкие команды вроде `git restore .`, пока diff не просмотрен и
нет намерения удалить все локальные изменения.

Если секрет попал в commit:

1. Сразу rotate secret.
2. Убрать файл из index:

```bash
git rm --cached .env
git commit -m "chore: stop tracking local env file"
```

3. Если секрет уже был pushed, согласовать history cleanup с командой. Обычный
   follow-up commit не делает секрет безопасным.

## Как Вести Этот Файл

Обновлять документ при изменении:

- naming веток;
- PR или merge rules;
- обязательных verification commands;
- ignored file policy;
- dataset или experiment artifact policy;
- правил работы AI-агентов.

Если реальная практика расходится с этим файлом, нужно либо исправить практику,
либо обновить файл до merge следующего PR.
