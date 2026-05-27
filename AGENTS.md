# AutoVibe Gym - Codex Instructions

## At The Start Of Each Work Cycle
1. Read `docs/PROJECT.md` for the project goals, architecture, and constraints.
2. Read `docs/STATUS.md` for current progress, blockers, and next actions.
3. Read `docs/GIT_WORKFLOW.md` for branch, commit, PR, and review rules.
4. Use these files as the source of truth before changing code or planning a PR.

## At The End Of Each PR Cycle
Update `docs/STATUS.md` to reflect the actual state:
- update `Last updated`;
- move completed work into the relevant status sections;
- add new TODO, In Progress, or Blocked items when needed;
- add a short changelog entry for the cycle.

Also update `docs/GIT_WORKFLOW.md` whenever the Git, PR, review, or AI-agent
collaboration process changes.

## Project Notes
- Keep the implementation simple and auditable.
- Preserve the hidden test split and submit gate behavior.
- Checklist hints should stay implicit nudges, not direct instructions.
