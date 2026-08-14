# AutoVibe Gym Paper V1 — Research PR Policy

Этот документ определяет, когда создавать, переводить в Ready и сливать
исследовательские pull requests. Общие Git-правила и обязательная identity
определены в `docs/GIT_WORKFLOW.md`; при конфликте применяется более строгая
граница.

## 1. Зачем исследованию отдельная PR-policy

История Git должна позволять внешнему reviewer'у восстановить:

- когда были сформулированы гипотезы и analysis rules;
- какие решения были приняты до просмотра confirmatory outcomes;
- какая версия кода создала каждый run;
- где заканчивается pilot и начинается confirmatory evidence;
- какие failures, исключения и изменения произошли после freeze;
- как из immutable manifests были получены числа статьи.

Один PR соответствует одному научному или инфраструктурному утверждению. Нельзя
смешивать protocol, budget mechanics, runner orchestration, результаты и paper
text в одном большом diff.

## 2. Авторство и публичная идентичность

Все commits:

```text
Author:    JapanDino <klim.i.rumyantsev@gmail.com>
Committer: JapanDino <klim.i.rumyantsev@gmail.com>
```

Все PR создаются только GitHub-аккаунтом `JapanDino`. PR, commit или merge от
другого пользователя не считается допустимым research artifact и должен быть
остановлен до публикации.

Перед commit/push/PR выполняются identity-команды из `docs/GIT_WORKFLOW.md`, а
результат проверки кратко указывается в handoff/PR verification.

## 3. Когда PR еще нельзя создавать

Не открывать даже Draft PR, если выполняется хотя бы одно условие:

- branch основан не на проверенном свежем `origin/main` или содержит unrelated
  ancestry/commits;
- task diff смешан с чужими или незавершенными изменениями;
- scope нельзя описать одним предложением;
- в diff могут находиться secrets, private data, generated run outputs или
  большие trajectories;
- неизвестно, является работа pilot, confirmatory infrastructure или analysis;
- substantive research decision скрыто принята вместо explicit TODO с owner и
  deadline;
- нет минимальной offline-проверки измененной области;
- PR основан на просмотренных confirmatory results, но меняет незадокументированно
  hypothesis, endpoint, exclusion или analysis method.

Для PR 1 использована изолированная ветка `codex/paper-v1-protocol`, созданная от
проверенного `origin/main` commit `1504cc0`. Перед commit и перед публикацией base
проверяется повторно; если `origin/main` изменился, применяются обычные
rebase/verification gates.

## 4. Когда открывать Draft PR

Draft PR создается только после отдельного разрешения владельца и когда:

- base и remote проверены;
- branch содержит хотя бы один цельный reviewable commit;
- diff ограничен одной PR-задачей;
- PR title/body уже объясняют research purpose и evidence boundary;
- выполнены минимальные checks или честно перечислены blockers;
- `docs/STATUS.md` отражает состояние `In Progress`/`Blocked`;
- нет секретов и запрещенных артефактов;
- PR author подтвержден как `JapanDino`.

Draft нужно открывать рано только тогда, когда он уже полезен для review. Пустой
Draft без проверяемого diff не является прогрессом исследования.

## 5. Когда переводить PR в Ready for review

Ready допустим, когда:

- все acceptance criteria PR выполнены;
- branch rebased на свежий `origin/main`;
- обязательные unit/offline integration checks прошли;
- полный suite запущен либо невозможность/внешние failures точно описаны;
- `git diff --check` прошел;
- protocol/status/docs синхронизированы;
- reviewer может объяснить, что изменилось и что не изменилось;
- unresolved decisions перечислены с owner/deadline и не замаскированы defaults;
- evidence не содержит cherry-picking или смешения pilot/confirmatory runs;
- PR body содержит точные команды и результаты проверок;
- commit и PR identities повторно проверены.

Наличие открытого freeze-blocking human decision не всегда блокирует Ready для
инфраструктурного PR, но обязательно блокирует protocol freeze и confirmatory run.

## 6. Когда merge допустим

Merge выполняется только после отдельного разрешения владельца и когда:

- PR имеет требуемый human review/approval;
- required CI завершился с допустимым результатом;
- нет unresolved review threads;
- base снова проверен непосредственно перед merge;
- итоговый diff не изменился после последней содержательной проверки либо новые
  изменения повторно проверены;
- `docs/STATUS.md` содержит итог цикла и точные проверки;
- merge title соответствует Conventional Commits;
- merge выполняется аккаунтом `JapanDino` либо подтвержденным maintainer merge,
  не изменяющим авторство commits/PR;
- для research freeze/run соблюдены специальные gates ниже.

Предпочтительный метод — squash merge с одним чистым Conventional Commit title,
если сохранение нескольких commits не нужно для научного provenance. Если внутри
PR есть отдельные preregistration и implementation checkpoints, разрешен rebase
merge только после явного решения владельца.

## 7. Обязательная последовательность Paper V1

| Этап | PR / artifact | Когда начинать | Критерий завершения | Что запрещено до завершения |
|---|---|---|---|---|
| Phase 0 | audit внутри PR 1 | до изменения mechanics | кодовые факты и test gaps зафиксированы | выдавать pilot за доказательство |
| PR 1 | `docs(research): define paper v1 protocol` | сейчас, до causal refactor | protocol/гипотезы/analysis/failure/selection rules однозначны и validated | global-budget refactor и confirmatory runs |
| PR 2 | `feat(research): add auditable run manifests` | только после merge PR 1 | stable condition ID, unique run ID, hashes, ledgers, atomic manifest, failure class | менять causal behavior режимов |
| PR 3 | `feat(research): enforce fair global episode budgets` | после merge PR 2 | A/B/C используют общий budget, no-autofit validator и one-shot hidden evaluation | availability pilot на старой mechanics |
| PR 4 | `feat(research): add confirmatory experiment planner` | после merge PR 3 | precomputed matrix, blocked order, resume, duplicate/replacement checks | confirmatory execution |
| PR 5a | `feat(research): preregister primary analysis pipeline` | после стабильного manifest schema, до freeze | FANU, completeness, paired analysis и synthetic fixtures работают без реальных outcomes | смотреть confirmatory outcomes |
| Pilot | immutable pilot artifacts | после PR 3–5a, до freeze | endpoints/usage/budget/isolation проверены, changes result-blind | смешивать pilot с confirmatory data |
| Amendment | `docs(research): freeze paper v1 configuration` | после pilot и human decisions | exact models/datasets/budget/reference values/config hashes приняты | confirmatory runs |
| Freeze | tag `paper-v1-experiment-freeze` | после merge всех pre-run PR | clean commit/tag, matrix hash, protocol/config hashes и acceptance gate | менять код во время серии без change note/new tag |
| Runs | confirmatory series | только с freeze tag | все conditions завершены либо заранее разрешенно censored/replaced | править hypotheses или cherry-pick runs |
| PR 5b | `analysis(research): build paper v1 results` | после immutable run set | все tables/figures пересоздаются из manifests | ручной перенос чисел |
| PR 6 | `docs(paper): add paper v1 reproducibility package` | после review PR 5b | paper, limitations, provenance, disclosure, release package | преувеличивать выводы |

PR 5 разделен на preregistered analysis (`5a`) и population of results (`5b`).
Это позволяет зафиксировать primary analysis до просмотра результатов, но не
коммитить сгенерированные итоговые числа до завершения immutable серии.

## 8. Pilot, freeze и confirmatory gates

### Availability/budget pilot разрешен, когда

- PR 3–5a merged;
- runs явно имеют отдельный pilot experiment ID;
- exact provider/model IDs записаны;
- usage ledger и failure classifier работают;
- Docker isolation проверен;
- допустимый API/cost limit утвержден;
- pilot не используется для выбора красивого arm effect.

### Freeze разрешен, когда

- все `freeze_blocking` TODO protocol имеют `status: resolved`;
- protocol/config/dataset/prompt hashes записаны;
- полный список conditions с order seed сгенерирован и сохранен;
- primary analysis pipeline проверен на synthetic fixtures;
- worktree clean, tests и research acceptance gate пройдены;
- создан отдельный owner-approved freeze commit и tag.

### Во время confirmatory серии

- нельзя merge'ить изменения в freeze branch/tag;
- нельзя менять prompts, budgets, models, datasets, exclusions или analysis;
- нельзя удалять/перезаписывать attempts;
- infrastructure fix требует остановки, change note, нового commit/tag и явного
  решения о повторе затронутых conditions;
- наблюдение промежуточных aggregate outcomes ограничивается completeness и
  infrastructure health, без анализа arm effects.

## 9. Структура research PR body

```markdown
## Research context

Research question, hypothesis, invariant, or validity risk addressed.

- Protocol version/hash:
- Related hypothesis/invariant:
- Evidence class: protocol / pilot infrastructure / confirmatory infrastructure / analysis / paper

## Changes

- ...

## Scientific rationale

Why this change is required for fairness, auditability, or reproducibility.

## Invariants preserved

- Hidden test remains unavailable to agent context.
- Pilot and confirmatory artifacts remain separate.
- No result-dependent protocol change.
- No evaluator assistance unless explicitly common and recorded.

## Acceptance criteria

- [ ] ...

## Verification

| Check | Command | Result |
|---|---|---|
| Focused tests | `...` | `...` |
| Full suite | `...` | `...` |
| Diff check | `git diff --check` | `...` |

## Known failures and limitations

- ...

## Human decisions required

| Decision | Owner | Deadline | Freeze-blocking |
|---|---|---|---|

## What this PR does not prove

- ...

## Reproducibility and provenance

- Base commit:
- Head commit:
- Protocol/config hashes:
- Dataset/artifact references:
- Commit identity check:
- PR author check:
- Secret/private-data review:

## Out of scope

- ...
```

Нельзя писать `all tests pass`, если полный suite не прошел. Нужно указывать
точные counts, команды, warnings, skipped и failures с границей отношения к diff.

## 10. Commit policy внутри PR

- 1 commit = 1 логическое изменение.
- Commit subject на английском, Conventional Commits, до 72 символов.
- Не смешивать method/protocol change и generated results.
- Не переписывать опубликованный research history без явной причины и review.
- Не использовать `--no-verify`, подмену author flags или environment identity.
- Перед каждым commit показывать scoped file list и проверять staged diff.
- После commit проверять author/committer через `git show -s --format=fuller`.

Рекомендуемые commits для PR 1:

```text
docs(research): define paper v1 protocol and audit
test(research): validate protocol invariants
docs(workflow): codify research pull request gates
docs(status): record paper v1 protocol cycle
```

## 11. Post-merge и релиз

После merge:

- обновить local `main` через fast-forward;
- записать merged PR/commit в `docs/STATUS.md`;
- не удалять branch/tag, пока provenance не проверен;
- перед новым PR заново прочитать project/status/workflow/policy;
- для freeze/release создать immutable tag только после отдельного owner approval;
- GitHub/Zenodo release должен ссылаться на exact tag, protocol/config hashes и
  checksum внешних artifacts.
