# AutoVibe Gym — Claude Instructions

## On every session start
1. Read `docs/STATUS.md` to understand current state
2. Read `docs/PROJECT.md` if touching architecture or checklist logic
3. Check `memory/project_context.md` for high-level context

## On every session end (or after meaningful progress)
Update `docs/STATUS.md`:
- Move completed items to Done
- Add new In Progress / Blocked items
- Update "Last updated" date at the top
- Add a one-line entry to the Changelog section

## Project root
`C:\Users\klimi\APPS_projects\autovibe-gym`

## Key files
| File | Purpose |
|------|---------|
| `gym/env.py` | Core GymEnv class — action loop, submit gate |
| `gym/executor.py` | Isolated Python sandbox for LLM code |
| `gym/checklist.py` | 8-item DS checklist + implicit hints |
| `gym/agent.py` | LLM agent via Anthropic API |
| `experiments/run_gym.py` | CLI entry point |
| `docs/PROJECT.md` | TZ, goals, architecture, stack |
| `docs/STATUS.md` | Live project status — update this regularly |

## Coding conventions
- Python 3.11+, type hints everywhere
- No external state beyond what's in `EnvState`
- Checklist hints must be implicit (nudge, not direct instruction)
- Test data is NEVER accessible inside the executor namespace

## Team context
~4 people, 2-week deadline, final presentation. Stack decisions should stay simple and auditable.
