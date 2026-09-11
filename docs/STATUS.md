# AutoVibe Gym - Live Status

**Last updated:** 2026-09-11 (Paper V2 M1a-M9 local stack prepared as a transfer checkpoint; no PR, freeze, or experiment activation)
**Phase:** Paper V1 manuscript review; isolated Paper V2 reliability development.

---

## Current Sprint Goal

Review and merge the PR 6 working manuscript and its reproducibility package
without overstating the negative H1 result, the inconclusive checklist increment,
or the unvalidated checklist detector. External submission remains out of scope.

## Paper V2 Reliability Work

| Item | Status | Notes |
|---|---|---|
| M1a deterministic incumbent core | Implemented locally; uncommitted | Isolated branch `JD/paper-v2-m1a` is based on verified `origin/main` commit `1d94774`. Validated candidate records are immutable; the host registry uses the existing `higher`/`lower` metric-direction convention, deterministic earlier-incumbent tie handling, and separate current/incumbent pointers. Notebook mutation invalidates only current-clean-run eligibility while preserving historical records, model objects, and artifacts. Submission and hidden evaluation still accept only the current clean-run candidate. |
| M1b durable candidate bundle | Implemented locally; uncommitted | Each validated candidate is atomically published to a no-overwrite private `candidate-bundle-v1` directory containing canonical metadata, the serialized model, an immutable notebook snapshot, and `SHA256SUMS`. Strict schema/JSON/path/file checks, checksum verification before model loading, deterministic registration-index reconstruction, ignored unpublished temp directories, and fail-closed corruption/write failures are covered. Restore recreates historical registry/incumbent state but deliberately leaves `current=None`, so it cannot bypass clean-run eligibility or trigger hidden evaluation. Public events expose only the schema, candidate ID, and hashes; bundle paths and detailed failures remain private. This is registry recovery, not full kernel/episode resume. |
| M1a+M1b verification | Passing | Bundle/registry focused suite: `20 passed, 1 skipped`; direct notebook regression before the final write-failure case: `36 passed, 1 skipped`, followed by `2 passed` for the latest corruption/write-failure targets; adjacent experiment/protocol/submission/runner/artifact suite: `92 passed`; full offline suite: `416 passed, 1 skipped, 2 deselected`; one existing Jupyter path deprecation warning. Compileall and `git diff --check` pass apart from line-ending notices. No API, paid endpoint, Docker integration, hidden-test research run, staging, commit, push, or PR occurred. |
| M2a protected budget partition | Implemented locally; uncommitted; not activated | `EpisodeBudget` now has an opt-in, one-way exploration-to-finalization partition backed by a separately versionable `FinalizationReservePolicy`. Exploration stops at `global - reserve` without marking the global budget exhausted; finalization is limited to its own token, LLM-call, code, tool, and wall-time pools and cannot borrow unused exploration capacity. With no reserve, the Paper V1 policy payload, snapshot shape, events, limits, and exception behavior remain unchanged. No reserve values, CLI flags, or stopping thresholds were chosen or enabled. |
| M2a verification | Passing | Focused reserve suite: `19 passed`; adjacent budget/experiment/runner/freeze suite: `50 passed`; final offline suite: `427 passed, 1 skipped, 2 deselected`; one existing Jupyter path deprecation warning. No API, model endpoint, Docker integration, pilot, experiment, staging, commit, push, or PR occurred. |
| M2b protected producing-revision finalization | Implemented locally; uncommitted; not activated | When and only when research mode receives an explicit reserve policy, `submit` and host finalization enter the same host-owned terminal controller. It selects the frozen incumbent regardless of the requested variable, verifies its complete M1b bundle, executes the immutable notebook snapshot in a restarted kernel under finalization code limits, repeats raw-row/serialization/validation checks, rejects metric drift, atomically stores and reloads a private `protected-finalization-v1` replay artifact, and only then spends the final tool unit on the one-shot hidden gate. Missing/corrupt incumbents, replay/preflight/drift/storage failures, reserve exhaustion, and hidden failure are terminal with no code repair, live-kernel/autofit fallback, new candidate, resource borrowing, or second hidden attempt. Public receipts expose IDs and hashes only. |
| M2b verification | Passing | Dedicated artifact store: `4 passed, 2 skipped`; all ten M2b controller cases passed across focused and full runs (the two store skips are Windows symlink capabilities). Full offline suite: `441 passed, 3 skipped, 2 deselected` in 641.39s; one existing Jupyter path deprecation warning. Compileall and `git diff --check` pass apart from line-ending notices. Static search confirms that no runner/CLI constructs `FinalizationReservePolicy`; tests are the only callers selecting fixture reserve values. No API, model endpoint, Docker integration, pilot, experiment, staging, commit, push, or PR occurred. |
| M3 deterministic context compression | Implemented locally; uncommitted; not activated | An explicit, versioned `ContextCompressionPolicy` can replace the accumulated transcript with canonical UTF-8 JSON containing the public task contract, tested public hypotheses, current incumbent, active public blockers, deterministic remaining-resource counters, allowed actions, notebook state, and finalization contract. Schema validation is strict and rejects unknown/private fields and non-finite values. Truncation is deterministic and bounded: it limits recent messages and state lists, drops lower-priority history in a fixed order, preserves required core fields, and fails closed if the core cannot fit. Public receipts contain policy, sizes, reduction, counts, and the pack hash; hashes of raw/source messages remain private. The existing default path and legacy environment-variable compactor are unchanged. No policy values, runner/CLI wiring, or experimental activation were chosen. |
| M3 verification | Passing | Focused context-compression suite: `11 passed`; the M3 notebook public-pack target passed; the final full offline suite passed `453 passed, 3 skipped, 2 deselected` in 474.03s with one existing Jupyter path deprecation warning. Compileall and `git diff --check` pass apart from line-ending notices. Static search confirms that production runners/CLI do not construct or activate `ContextCompressionPolicy`. No API, model endpoint, Docker integration, pilot, experiment, staging, commit, push, or PR occurred. |
| VQ1 adaptive-validation accounting and cap | Implemented locally; uncommitted; not activated | A versioned, explicitly injected `ValidationQueryPolicy` adds a separate pre-query cap, an optional non-borrowable protected-finalization query reserve, and one frozen numeric feedback precision. Repeated, failed, cached/reused, same-revision, `validate`, auto-validation/readiness, `check_candidate`, `quick_validate`, tuning-preflight/trials/final-check, Cleanlab, submit preflight, and protected replay accesses share one source-classified ledger. Under the policy, feedback-validation labels are removed from `val_df`, the workspace CSV, dataset cards, and public profiles; host validation owns labels and charges before each information access. Scores are normalized before both incumbent selection and public feedback. Summaries report realized queries and hidden-minus-feedback-validation gap; M3 exposes only aggregate remaining query resources, not the source ledger. No limits, precision, runner/CLI wiring, or arm configuration were chosen. |
| VQ1 verification | Passing with deployment caveat | Budget suite: `29 passed`; strict context suite: `12 passed`; focused loophole/kernel suite: `16 passed`; combined budget/context/notebook suite: `93 passed, 1 skipped`; full offline suite before M5: `469 passed, 3 skipped, 2 deselected` in 545.96s with one existing Jupyter path deprecation warning. Compileall and `git diff --check` pass apart from line-ending notices. No production runner or CLI constructs `ValidationQueryPolicy`; the inactive M5 repeated arm now honors label isolation, query charging, and metric normalization if a policy is explicitly injected. Local Jupyter is not full OS isolation, so confirmatory enforcement still requires the same isolated Docker backend and frozen VQ1 policy in both arms. No API, Docker run, model endpoint, pilot, experiment, staging, commit, push, or PR occurred. |
| M5 symmetric terminal selection | Implemented locally; uncommitted; not activated | `symmetric-terminal-v1` is now the single host-owned selector/finalizer used by protected `NotebookGymEnv` finalization and by the reserve-gated repeated-single-shot path. Both arms select through `CandidateRegistry`, verify the immutable M1b bundle, clean-replay the exact producing revision in a fresh execution state, run the same isolated submission validator, reject validation-metric drift under the same direction/tolerance contract, atomically store/reload the same protected artifact schema, and open the same one-shot `HiddenEvaluationGate`. The repeated adapter stores every admitted attempt as an immutable one-cell notebook bundle, so later failures cannot erase an earlier incumbent and a live `best_model` cannot bypass replay. VQ1 label isolation/query charging/precision are also honored when injected. Existing no-reserve behavior remains on the Paper V1 path. |
| M5 verification | Passing with protocol gate | Common-controller and repeated-adapter suite: `8 passed`; reserve-gated runner/adjacent suite: `41 passed`; common-controller plus full `NotebookGymEnv` regression: `57 passed, 1 skipped`; final full offline suite: `479 passed, 5 skipped` in 586.27s with one existing Jupyter path deprecation warning. Compileall and `git diff --check` pass apart from line-ending notices. An injected reserve now requires explicit `--research-v2-metric-direction` and `--research-v2-score-tolerance`; neither has a default for the active V2 path. Static search confirms no production code constructs `FinalizationReservePolicy` or `ValidationQueryPolicy`, so no runner activates M2/VQ1/M5 automatically. Confirmatory arm symmetry still requires frozen reserve/query/tolerance values, the same Docker execution/prediction configuration, preregistration, and immutable protocol binding. No API, Docker run, model endpoint, pilot, experiment, staging, commit, push, or PR occurred. |
| M4 frozen stopping controller | Implemented locally; uncommitted; not activated | `stopping-policy-v1` is a strict, externally supplied, hash-bound contract with no threshold defaults. A deterministic first-trigger-wins controller covers the preregisterable reasons: watched exploration resources reaching the reserve boundary, exploration-pool exhaustion, a frozen number of eligible non-improving validations, an agent finalization request with a valid incumbent, and an unrecoverable safety/contract failure. Both `NotebookGymEnv`/`GymAgent` and repeated-single-shot apply the same reason/state machine. Finalize decisions enter the common M5 protected replay; unrecoverable decisions terminate without hidden evaluation. Boundary inspection is read-only and occurs before a new LLM call, so exploration cannot borrow the reserve. |
| M4 verification | Passing with protocol gate | Strict policy/controller/budget/runner suite: `15 passed`; Gym pre-call integration plus policy suite: `23 passed`; three end-to-end notebook stopping paths passed; repeated-arm observers and the existing reserve-gated M5 runner passed. Final full offline suite: `499 passed, 5 skipped` in 687.16s with one existing Jupyter path deprecation warning. Compileall, `git diff --check`, and static activation search pass apart from line-ending notices. Production code constructs no `FrozenStoppingPolicy`, `FinalizationReservePolicy`, or `ValidationQueryPolicy`. The JSON path is accepted only by the two M4-capable runners and fails closed unless a reserve was already explicitly injected. No patience value, watched-resource set, reserve/query limit, metric contract, or arm configuration was selected or activated. |
| M6 result-blind protocol admission | Implemented locally; uncommitted; not frozen or activated | `paper-v2-protocol-bundle-v1` binds the exact M2/M3/VQ1/M4/M5 policies, terminal schema versions, one-hidden-gate rule, Docker execution/prediction isolation, immutable A/B/C runner/adapter mappings, execution commit, and hashes for selection evidence, protocol/hypotheses/analysis/failure documents, datasets/models/prompts/decoding, and matrix/randomization artifacts. Strict JSON and Python admission reject duplicate/unknown fields, non-standard numbers, post-outcome selection, malformed hashes, version/backend/arm drift, and terminal reserves that cannot execute the minimum M5 sequence. Canonicalization normalizes semantically equal integer/float spellings and stopping-resource order before hashing. The module emits only a public hash receipt; it neither writes a freeze artifact nor imports into any runner. |
| M6 verification | Passing with external-artifact gate | Focused parser/schema/adversarial suite: `21 passed`; adjacent protocol/stopping/budget/runner/freeze/planner suite: `99 passed`; final full offline suite: `520 passed, 5 skipped` in 718.52s with one existing Jupyter path deprecation warning. Compileall, JSON Schema validation, `git diff --check`, and static non-wiring/non-construction searches pass apart from line-ending notices. The tests use synthetic values only. No real protocol document, policy selection, artifact manifest, independent review, preregistration, Docker digest verification, freeze record, runner activation, API, Docker run, experiment, commit, push, or PR occurred. |
| M7 semantic preregistration contract | Implemented locally; uncommitted; draft only | `paper-v2-preregistration-v1` turns the Paper V2 questions into an exact result-blind hierarchy: P1 tests B-minus-A reliability non-inferiority; P2 is a P1-gated B-minus-A superiority test on unconditional FANU; S1 tests the C-minus-B checklist effect two-sided. The authoritative analysis population retains every terminal agent outcome, pairs by dataset/model/replicate, weights dataset-model strata equally, keeps successful-only scores descriptive, and preserves the one-hidden-evaluation rule. Failure semantics retain agent-invalid outcomes at dummy utility, censor and same-condition-replace infrastructure attempts, and stop rather than delete on protocol violations. System claims remain separate from a D0-D3 excluded-development nested ablation, whose checkpoint/reserve/compression contrasts support fixed-order incremental claims only and cannot be pooled with A/B/C. |
| M7 verification | Passing with twelve explicit freeze blockers | The local artifact is `status=draft`, records `confirmatory_outcomes_visible=false`, and leaves margin, alpha, power, replicate count, resampling/randomization counts, seed, multiplicity, dataset/model scope, FANU references, power analysis, and independent statistical review explicitly unresolved with no values. Strict Python/JSON Schema admission rejects hypothesis/estimand/failure/claim drift, pseudo-resolution without rationale and evidence, duplicate/unknown fields, visible outcomes, invalid decision ranges, and `freeze_candidate` while any blocker remains. A future resolved object produces canonical preregistration/component hashes and must match the M6 selection/study references before admission. Focused suite: `20 passed`; adjacent M4-M7 research suite: `86 passed`; full offline suite: `540 passed, 5 skipped` in 813.13s with one existing Jupyter path deprecation warning. Tests use synthetic resolved fixtures only. No real decision value, artifact approval, runner activation, API, Docker run, pilot, experiment, freeze, stage, commit, push, or PR occurred. |
| M8 result-blind planning power simulation | Implemented locally; uncommitted; no real scenario or result | `paper-v2-power-scenario-v1` requires explicit M7 hypothesis/analysis/failure hashes, alpha-split gatekeeping, family-wise/primary/secondary alpha, non-inferiority margin, target power, simulation draws/seed, per-stratum replicates, paired validity assumptions, and successful-FANU means/deviations/correlation. It simulates equal-weight paired strata, applies a one-sided P1 lower bound, gates the one-sided P2 FANU bound on P1, tests S1 with a two-sided bound, and reports P1, gated P2, S1, joint confirmatory power, Monte Carlo error/intervals, runtime versions, and immutable scenario/result hashes. Strict result validation reconciles counts, probabilities, uncertainty, targets, embedded assumptions, and hashes before atomic no-overwrite output. Every output states that this is planning-only normal-bound approximation, not confirmatory evidence or the final resampling/randomization analysis. |
| M8 verification | Passing; M7 power decision remains unresolved | Focused scenario/simulation/result/schema/adversarial suite: `18 passed`; adjacent M6-M8/planner/analysis/freeze suite: `105 passed`; full offline suite: `558 passed, 5 skipped` in 648.85s with one existing Jupyter path deprecation warning. Extreme synthetic fixtures exercise full/zero P1 gate behavior, gated P2 and S1; tampered and internally inconsistent result objects fail closed even when rehashed. The module is absent from experiment runners and loads no MLflow, manifests, hidden scores, Paper V1 results, or Paper V2 outcomes. Only test fixtures contain numerical assumptions. No real scenario, sensitivity grid, power-result artifact, M7 decision resolution, API, Docker run, pilot, experiment, freeze, stage, commit, push, or PR occurred. |
| M9 excluded-development and assumption provenance | Implemented locally; uncommitted; metadata gate only | `paper-v2-excluded-development-manifest-v1` and `paper-v2-confirmatory-dataset-scope-v1` reject overlap by immutable dataset identity, exact content, or source lineage. `paper-v2-assumption-evidence-v1` binds every M8 assumption separately per required scenario to the exact scenario hash, assumption pointer, value hash, declared use, admissible source artifact, and—only for pilot-derived nuisance parameters—the exact excluded-development manifest. Historical V1 and excluded pilots may inform nuisance parameters but cannot set the non-inferiority margin, alpha/power design, or provide evidence that corrected V2 works. The robust sample-size rule must pass all predeclared scenarios; favorable-row selection is forbidden. No real manifest, scope, evidence package, scenario grid, or receipt was created. |
| M9 verification | Passing with one disclosed non-reproduced pre-existing VQ1 transient | Focused schema/parser/provenance/disjointness/adversarial suite: `28 passed`; adjacent M6-M9/planner/analysis/freeze suite: `211 passed`; final repeated full offline suite: `586 passed, 5 skipped` in 817.01s with one existing Jupyter path deprecation warning. The first full run completed `585 passed, 5 skipped, 1 failed`: the existing protected-finalization validation-query test observed one rather than two charges. It then passed alone, in its three-test VQ1 group, and in the complete rerun; cause remains unresolved and no unrelated VQ1 code was changed. JSON Schema tests, compile/format/diff checks, and static non-wiring/non-artifact searches pass apart from existing line-ending notices. M9 remains disconnected from runners and does not resolve any M7 freeze blocker. No API, Docker run, pilot, experiment, freeze, stage, commit, push, or PR occurred. |
| Transfer checkpoint verification | Ready to commit and push; not PR-ready | On 2026-09-11 the branch remained exactly based on fresh `origin/main` commit `1d94774`, with 13 modified and 31 new source/test/document files, no staged or prior local commits, no forbidden dataset/output/cache paths, no file over 1 MiB, and no credential-shaped match in the bounded changed-file scan (`gitleaks` was unavailable). A fresh full offline run completed `585 passed, 5 skipped, 1 failed` in 1051.96s: existing `test_hidden_submit_failure_is_generic` did not make its third failure terminal. The unchanged target immediately passed three isolated reruns in 44.06s, 44.37s, and 42.27s. Cause is unresolved; no runtime code was altered to hide the non-reproduced failure. This is a resumable checkpoint only; the mixed M1-M9 stack must be split into reviewable claim boundaries before any PR. |

This working tree stacks M1b, M2a, M2b, M3, VQ1, M4, M5, M6, M7, M8, and M9 locally on the still-uncommitted M1a
branch. It must be split into reviewable claim boundaries after explicit commit
authorization; it is not yet a published or PR-ready lineage. The M2 mechanism
is implemented, but the scientific milestone is not protocol-ready: reserve
values and stopping thresholds must be selected on excluded development tasks,
reviewed, preregistered, and frozen before activation. M3 is likewise only a
mechanism: its numeric limits and experimental comparison have not been selected
or frozen. VQ1, M4, M5, M6, M7, M8, and M9 are likewise inactive mechanisms/contracts: query limits, feedback
precision, reserve values, objective direction/tolerance, Docker enforcement,
and arm configuration remain unfrozen. Full kernel/episode resume, broader
task/model coverage, frozen M4 policy values, and later ablations remain separate
Paper V2 claims and review gates.

## Paper V1 Research Protocol

| Item | Status | Notes |
|---|---|---|
| Phase 0 code audit | Done | Audited freshly fetched `origin/main` commit `1504cc0`; mapped five modes, per-call token semantics, host autofit, step/tool counters, artifact paths, hidden-score privacy, hidden retry loop, and current tests |
| `research/protocol-v1` scaffold | Done | Machine-readable protocol, hypotheses, analysis plan, failure policy, selection gates, validator, and offline tests |
| Protocol freeze | Amendment v2 merged and tagged | PR #71 fast-forward merged at `60295bd`; annotated tag `paper-v1-experiment-freeze-v2` binds plan `793d85...310f`, the fresh v2 series root, and unchanged effective scientific conditions. Original PR #69/tag and the stopped pre-provider v1 root remain immutable |
| Confirmatory execution | Complete | V2 reconciliation is `terminal=120`, `pending=0`, `running=0`, `replacement_required=0`. Collection used execution commit `b4e3da2`, the frozen datasets/models/prompts/arms/randomization, and Docker image `sha256:b8c3...7466d` |
| Primary-analysis binding repair | Merged as PR #72 | The real frozen plan/reference pair validates and local/GitHub tests passed. The repair was merged before a successful analysis. GitHub rebase created `4ab8804` with correct JapanDino author/account/email but committer display name `J D`; the owner instructed work to continue without force-rewriting published `main`, and the deviation is recorded in the results audit |
| PR 5b confirmatory results | Merged as PR #73 | Fast-forward merged at `d1ec571` with exact JapanDino author/committer identity. The preregistered analysis covers all 120 outcomes with result hash `22ff7ac1...c58a`; deterministic CSV/SVG/provenance artifacts are under `research/publication/paper_v1`. Final full suite: `385 passed, 2 skipped`; GitHub tests and current-head review passed |
| PR 6 manuscript package | Ready for review | A TMLR-style working manuscript, checked bibliography, limitations/disclosures, reproducibility guide, and SHA-256 manifest are under `paper/`. The builder derives numeric Results, validates all 13 artifacts, and detects tampering across LF/CRLF checkouts. Current focused suite: `23 passed`; latest Linux CI before the final assertion/disclosure-only fix: `392 passed, 2 deselected`. The manuscript records incomplete resource, validation-gap, recovery, clean-replay, and stratified-failure reporting; no submission, release, raw trajectories, private registry, or paid API request |
| Confirmatory launcher | Merged as PR #70 | Fast-forward merged at `e07b76c` after local `383 passed`, GitHub PR tests, resolved review, and post-merge CI. It verifies the tagged contract, execution worktree, datasets, private model settings, Docker digest, and manifest/lifecycle history before every condition |
| Candidate prediction isolation | Implemented and frozen | Confirmatory Docker runs explicitly bind agent execution and candidate prediction to Docker. Readiness and hidden evaluation run in a separate ephemeral, network-none, read-only/cap-dropped evaluator; its result crosses to the host only as a validated JSON scalar vector, never executable pickle output |
| Auditable run artifacts (PR 2) | Merged | PR 2 was rebased onto merged PR 1, reverified (`289 passed, 2 skipped` locally plus required GitHub `Python tests` success), and squash-merged as `67069a3` |
| Causal runner behavior changes (PR 3) | Merged as PR #65 | Research-mode A/B/C runners share one pre-call-enforced episode budget, common no-autofit submission validation, and at most one hidden evaluation. Default product behavior remains compatible. After review fixes, `307 passed, 2 skipped` locally and both GitHub checks passed. Fast-forward merge preserved all three commits with author/committer `JapanDino`; `main` advanced to `385a09f` |
| Confirmatory experiment planner (PR 4) | Merged as PR #66 | Deterministic immutable A/B/C matrix expansion, portable blocked randomization, plan/config hashes, concurrent-safe no-overwrite writes, manifest-based resume, duplicate/condition-drift/category rejection, and same-condition infrastructure replacement queue. Review-fix full suite: `338 passed, 2 skipped`; both GitHub checks passed. Fast-forward merge preserved JapanDino authorship and advanced `main` to `1b12017` |
| Preregistered primary analysis (PR 5a) | Merged as PR #67 | Synthetic-only FANU, fail-closed completeness, paired B-A/C-B analysis, equal-stratum bootstrap/permutation, exact McNemar, 95% Wilson rate intervals, all-outcome/successful-only score summaries, ledger reconciliation, Holm correction, raw pairs/stratum summaries, content hashes, schemas, and concurrent-safe no-overwrite output. Final focused suite: `66 passed`; full: `351 passed, 2 skipped`; GitHub `Python tests` passed. Fast-forward merge preserved four JapanDino commits and advanced `main` to `b1cd5f5` |
| Research PR policy | Done | `docs/RESEARCH_PR_POLICY.md` defines authorization, identity, Draft/Ready/Merge gates, PR 1–6 timing, pilot/freeze/confirmatory boundaries, PR body, commit evidence, and post-merge rules |
| Commit/PR identity | Enforced by policy/config | Commits use `JapanDino <klim.i.rumyantsev@gmail.com>`; PR author must be GitHub login `JapanDino`; identity is checked independently before commit and before PR |

---

## Status by Component

### Core Gym (`gym/`)

| File | Status | Notes |
|------|--------|-------|
| `notebook.py` | Done | nbformat v4 document editing, stable cell ids, revisions, outputs, Python export |
| `jupyter_kernel.py` | Done | persistent local `ipykernel`; Docker kernel backend with loopback-only ZMQ ports and workspace path translation |
| `notebook_env.py` | V1 done; V2 M1/M2/M3/VQ1/M4/M5 implemented locally | Real notebook action loop plus opt-in protected incumbent replay/finalization through the common M5 controller, strict public context packing, host-owned validation-query enforcement/label isolation, and an explicitly injected M4 stopping controller; default and no-policy behavior remain on the V1 path |
| `feedback.py` | Done | runtime/contract/checklist/terminal feedback items and generic hidden checklist policy |
| `candidates.py` | V1 done; V2 M1a+M1b implemented locally | Immutable validated records, explicit registration order, deterministic current/incumbent registry, and historical reconstruction contract are implemented on isolated uncommitted branch `JD/paper-v2-m1a` |
| `candidate_store.py` | V2 M1b implemented locally | Atomic no-overwrite private bundles, canonical metadata, model/notebook snapshots, SHA-256 manifests, strict verified loading, and deterministic fail-closed reconstruction |
| `finalization.py` | V2 M2b implemented locally | Atomic no-overwrite producing-revision artifacts, canonical receipts, source/replay hashes, strict verified loading, symlink/path rejection, and public hash-only receipts |
| `terminal_contract.py` | V2 M5 implemented locally | Common host-owned incumbent selection, bundle verification, exact-revision replay boundary, isolated preflight, metric-drift rejection, protected artifact reload, and one-shot hidden gate shared by both arm adapters |
| `context_compression.py` | V2 M3/VQ1 implemented locally | Explicit versioned policy, strict public schema, canonical byte-stable serialization, deterministic bounded truncation, required-state preservation, fail-closed fitting, public/private receipt separation, and aggregate-only validation-query resources |
| `research/stopping.py` | V2 M4 implemented locally | Strict external policy loading, canonical policy hash, deterministic first-trigger-wins reasons, eligible-validation patience accounting, reserve-boundary matching, and public audit snapshots; no policy thresholds or watched resources are selected in production code |
| `research/protocol_v2.py` | V2 M6 implemented locally; admission only | Strict result-blind bundle parsing, normalized canonical hash, component receipts, real policy-object validation, M5 minimum-resource checks, Docker/network/terminal gates, immutable A/B/C adapter mapping, and hashes binding the future study and experiment artifacts; no runner imports or activates it |
| `research/schemas/paper_v2_protocol_bundle.schema.json` | V2 M6 implemented locally | Draft 2020-12 structural schema synchronized with the Python admission contract; Python remains authoritative for cross-field resource invariants |
| `research/preregistration_v2.py` | V2 M7 implemented locally; draft admission only | Strict semantic validation for the P1/P2/S1 hierarchy, paired equal-stratum estimands, failure handling, D0-D3 non-pooled nested ablation, claim limits, twelve typed freeze decisions, canonical receipts, and M6 hash-reference reconciliation; no runner imports or activates it |
| `research/protocols/paper_v2/preregistration.draft.json` | V2 M7 result-blind draft | Machine-readable preregistration source of truth with all numerical and review-dependent decisions explicitly unresolved; changing it to `freeze_candidate` currently fails closed |
| `research/schemas/paper_v2_preregistration.schema.json` | V2 M7 implemented locally | Draft 2020-12 schema synchronized with the semantic Python admission, including conditional rejection of unresolved freeze candidates |
| `research/power_v2.py` | V2 M8 implemented locally; planning only | Strict scenario/preregistration binding, paired A/B/C validity and FANU simulation, alpha-split P1→P2 gate plus S1, Monte Carlo uncertainty, semantic result reconciliation, runtime provenance, CLI, and atomic idempotent no-overwrite output; no runner imports or activates it |
| `research/schemas/paper_v2_power_scenario.schema.json` | V2 M8 implemented locally | Draft 2020-12 structural schema for explicit result-blind planning scenarios; Python additionally enforces cross-field alpha allocation and unique strata |
| `research/protocols/paper_v2/POWER_ANALYSIS.md` | V2 M8 method note | Declares required assumptions, normal-bound approximation, multiplicity boundary, no favorable-row selection, and restrictions on any later use of historical V1 evidence |
| `research/evidence_v2.py` | V2 M9 implemented locally; metadata admission only | Strict result-blind excluded-development, confirmatory-scope, and per-scenario assumption-provenance admission; exact identity/content/lineage disjointness, value/source/manifest binding, complete-grid coverage, robust-all-scenarios selection, canonical receipts, and no runner wiring |
| `research/schemas/paper_v2_*evidence*.schema.json` and dataset-scope schemas | V2 M9 implemented locally | Three Draft 2020-12 schemas synchronize the excluded-development manifest, confirmatory dataset scope, and assumption evidence structures; Python additionally enforces semantic disjointness, exact scenario/value coverage, and source/use admissibility |
| `research/protocols/paper_v2/EVIDENCE_PROVENANCE.md` | V2 M9 method note | Defines dataset separation, per-value provenance, admissible source/use classes, the historical-V1 boundary, and the all-scenarios sample-size rule; explicitly records that no real M9 artifact exists |
| `experiments/repeated_terminal.py` | V2 M5 implemented locally | Immutable one-cell attempt admission, common registry selection, fresh CodeExecutor replay, and adapter into `symmetric-terminal-v1`; activated only when an explicit finalization reserve is injected |
| `modes.py` | Done | `gym_with_checklist` and `iterative_no_checklist` share the same backend |
| `protocol.py` | Done | canonical action enum, required stage enum, canonical `thoughts`, and `think`; legacy `code` action remains compatible |
| `agent.py` | V1 done; V2 M3/M4 implemented locally | Minimal prompt requires `type`/`stage`; thoughts mode requires `thoughts` and initial `think`/`planning`; deterministic compression is used only when an explicit policy is injected; M4 boundary/latched decisions are applied before another LLM call; default and legacy paths are unchanged |
| `llm.py` | Done | OpenAI-compatible, Google/Gemini, and LiteLLM client selection |
| `env.py` | Legacy maintained | old subprocess/Docker environment retained for compatibility tests; rejects `think`/`planning`/`thoughts` because thoughts mode is disabled |
| `executor.py` | Legacy/baseline | Docker/subprocess executor retained for non-notebook baselines |

### Experiments (`experiments/`)

| File | Status | Notes |
|------|--------|-------|
| `run_gym.py` | Done; inactive M4 wiring | Uses `NotebookGymEnv`, logs notebook/process/private metrics and artifacts to MLflow; accepts a strict external M4 JSON only when a protected reserve is already injected |
| `run_baseline.py` | Done | single-shot control preserved; prompts require raw-DataFrame pipelines; missing score is not logged as zero |
| `run_multishot.py` | Done; inactive M4/M5 wiring | Logged as `repeated_single_shot`; prompts require raw-DataFrame pipelines; an injected M4 policy uses the same incumbent/no-improvement/boundary reasons and common M5 terminal path; not the fair checklist control |
| `run_fixed.py` | Done | fixed-transition control preserved; failed submit is not logged as real score 0.0 |
| Paper V1 recorder integration | Done | all four `run_*` entrypoints accept opt-in research artifact flags and link MLflow to experiment/condition/run IDs without changing default execution |
| `run.py` | Done | common single-dataset entrypoint; `--mode all` expands to five separate product runs with shared `batch_id`; `--modes ...` runs a selected batch of up to five modes |
| `compare.py` | Done | handles missing metrics without zero substitution |

### Privacy and Security

| Item | Status | Notes |
|------|--------|-------|
| Hidden test files | Done | not copied into episode workspace; no `test_df` in kernel |
| Hidden score feedback | Done | submit response hides score; score only in private summary/MLflow |
| Local Jupyter sandbox | Limited | real notebook functionality and sanitized env, but not full OS isolation |
| Docker kernel backend | Done | CI builds `Dockerfile.sandbox` and runs Docker integration smoke when Docker is available |
| Agent-visible artifacts | Done | Public workspace artifacts exclude hidden score, private checklist coverage, submit failure type, candidate pickle paths, replayed model paths, and private failure details |
| Private evaluator artifacts | Done | Private summaries, trajectories, versioned candidate bundles, and protected replay artifacts are stored outside the kernel-visible workspace; only ID/hash receipts without private paths enter public events |
| M3 context receipts | Implemented locally | The agent-visible pack is assembled from an allow-listed public schema. Public receipts exclude raw/source-message hashes; those hashes are recorded only in the private receipt persisted outside the agent-visible workspace. |
| VQ1 feedback-validation boundary | Implemented locally; inactive | With an explicit query policy, validation labels remain host-only and are absent from `val_df`, workspace files, dataset cards, public profiles, and aggregate M3 state. Query source counts remain in host summaries/ledgers. Local Jupyter is still not an OS security boundary; Docker isolation is required before scientific activation. |

### Tests

| Area | Status |
|------|--------|
| Existing legacy env/executor/agent tests | Passing |
| Jupyter kernel tests | Passing |
| Notebook editing tests | Passing |
| Clean run / validate / submit tests | Passing |
| Checklist privacy/fairness tests | Passing |
| Hidden-test privacy tests | Passing |
| Candidate persistence / corruption / resume tests | Passing |
| Protected replay / reserve / hidden-gate tests | Passing |
| Deterministic context-pack / privacy / truncation tests | Passing |
| Validation-query cap / bypass / precision / label-isolation tests | Passing |
| Docker kernel integration | Runs in GitHub Actions after sandbox image build |
| Step-budget semantics | Passing |

---

## Current Verification

Paper V2 M9 excluded-development and assumption-provenance cycle (2026-09-01):

- added strict metadata contracts for an excluded-development task manifest,
  a result-blind confirmatory dataset scope, and a reviewed assumption-evidence
  package; none of these contracts selects data, values, or policy;
- development/confirmatory separation is checked independently by dataset
  identity hash, exact content hash, and source-lineage hash, so renaming or
  resplitting the same source cannot establish independence;
- every assumption record is scoped to one scenario and binds the exact
  scenario hash, canonical assumption pointer, exact value hash, use class,
  source class, immutable source artifact, and conditional development-manifest
  reference;
- exact coverage is fail-closed across the complete supplied scenario set:
  missing, extra, duplicate, moved, scenario-drifted, or value-drifted records
  are rejected;
- the non-inferiority margin accepts only an independent methodological
  justification or external primary source; alpha/power choices have the same
  independence boundary; V1 and excluded-development observations may source
  only explicitly labelled nuisance assumptions;
- pilot-derived nuisance records must bind the exact excluded-development
  manifest, while non-pilot records are forbidden from claiming that manifest;
- the scenario-grid plan and full-grid review are hash-bound, favorable-row
  selection is false, and the only admitted sample-size decision rule is the
  minimum replicate count meeting target in all required scenarios;
- three Draft 2020-12 JSON Schemas mirror the structures; Python remains
  authoritative for disjointness, exact coverage, and source/use constraints;
- `python -m pytest tests/test_evidence_v2.py -q` -> `28 passed`;
  adjacent M6-M9/planner/analysis/freeze suite -> `211 passed`;
- the first full suite completed `585 passed, 5 skipped, 1 failed` in 765.02s:
  one existing VQ1 protected-finalization test observed one rather than two
  validation-query charges; it immediately passed alone (`1 passed`), with its
  related VQ1 group (`3 passed, 54 deselected`), and in the complete repeated
  suite; no VQ1 implementation was changed because the cause did not reproduce;
- final repeated `python -m pytest -q` -> `586 passed, 5 skipped` in 817.01s
  with one existing Jupyter path deprecation warning;
- no real development manifest, confirmatory scope, scenario grid, assumption
  package, receipt, decision resolution, API, model endpoint, Docker run, pilot,
  experiment, freeze, stage, commit, push, PR, or external registration occurred.

Paper V2 M8 result-blind power-planning cycle (2026-09-01):

- added `paper-v2-power-scenario-v1`, which is rejected unless its hypothesis,
  analysis-plan, and failure-policy hashes match the M7 preregistration;
- every scenario must explicitly provide family-wise alpha, its allocation to
  the P1/P2 sequence and S1, non-inferiority margin, target power, simulation
  draws/seed, stratum replicate counts, paired validity assumptions, and
  successful-FANU distribution/correlation assumptions; no scientific defaults
  or real scenario file were added;
- P1 uses the equal-stratum paired B-minus-A validity risk difference and passes
  when its one-sided normal lower bound exceeds the negative margin;
- P2 uses unconditional B-minus-A FANU, is tested one-sided, and counts only
  when P1 passes; S1 uses a two-sided C-minus-B FANU bound;
- the primary-sequence and S1 alpha allocations must both be positive and their
  sum cannot exceed the family-wise alpha; this M8 version models the M7
  `alpha_split_with_gatekeeping` candidate only, without selecting it for freeze;
- results report P1, gated P2, S1, joint confirmatory power, descriptive P2
  success conditional on P1, Monte Carlo standard errors and intervals, target
  attainment, exact assumptions, runtime versions, and scenario/result hashes;
- result admission first detects hash tampering, then reconstructs and verifies
  the embedded scenario and reconciles success counts, draws, power, uncertainty,
  intervals, targets, gates, claim limits, and runtime fields;
- planning results use atomic no-overwrite publication and identical reruns are
  idempotent; a different valid result cannot replace an existing artifact;
- each result explicitly states that normal-bound simulation is planning-only,
  assumption-dependent, and not the final paired resampling/randomization
  analysis or evidence that an arm works;
- the method note forbids selecting only a favorable sensitivity row and limits
  any later Paper V1 use to labelled external planning evidence for nuisance
  quantities, never the non-inferiority margin or evidence for corrected V2;
- `python -m pytest tests/test_power_v2.py -q` -> `18 passed`;
  adjacent M6-M8/planner/analysis/freeze suite -> `105 passed`;
- `python -m pytest -q` -> `558 passed, 5 skipped` in 648.85s with one existing
  Jupyter path deprecation warning;
- no real scenario, sensitivity grid, result artifact, M7 decision resolution,
  API, model endpoint, Docker run, pilot, experiment, freeze, stage, commit,
  push, PR, or external preregistration occurred.

Paper V2 M7 semantic-preregistration cycle (2026-09-01):

- added `paper-v2-preregistration-v1` as a semantic layer above the M6 hash
  contract; M6 can no longer be treated as sufficient evidence that hashed study
  documents contain complete, admissible hypotheses and analysis rules;
- froze the design shape, not its numerical values: P1 is B-minus-A reliability
  non-inferiority, P2 is P1-gated B-minus-A unconditional-FANU superiority, and
  S1 is a two-sided C-minus-B checklist comparison;
- the estimand contract uses all terminal agent outcomes, paired
  dataset/model/replicate blocks, equal dataset-model-stratum weighting, one
  hidden evaluation, and explicitly selection-biased descriptive-only
  successful-submission summaries;
- agent-invalid outcomes remain invalid at dummy FANU, infrastructure attempts
  are censored and replaced under the same condition, protocol violations stop
  the series without deleting outcomes, and complete-case primary analysis is
  forbidden;
- causal mechanism claims require a separate excluded-development D0-D3 nested
  ablation; its incremental checkpoint, protected-reserve, and compression
  contrasts cannot be pooled with confirmatory A/B/C or described as
  order-independent component effects;
- twelve exact freeze decisions remain unresolved: non-inferiority margin,
  family-wise alpha, target power, replicates, resampling/randomization counts,
  analysis seed, secondary multiplicity, reviewed dataset/model scope, FANU
  references, power analysis, and independent statistical review;
- unresolved decisions must contain no value, rationale, or evidence hash;
  resolved synthetic fixtures require typed values plus non-empty rationale and
  SHA-256 evidence, while `freeze_candidate` fails if any decision remains open;
- canonical receipts expose the preregistration and component hashes; a future
  resolved candidate must match both the M6 preregistration reference and all
  M6 study-contract hashes before it can be admitted;
- `python -m pytest tests/test_preregistration_v2.py -q` -> `20 passed`;
  adjacent M4-M7 research suite -> `86 passed`;
- `python -m pytest -q` -> `540 passed, 5 skipped` in 813.13s with one existing
  Jupyter path deprecation warning;
- the local artifact remains a result-blind draft with no real values;
  synthetic test values are not policy selections, and no runner imports M7;
- no API, model endpoint, Docker run, pilot, experiment, freeze, stage, commit,
  push, PR, or external preregistration occurred.

Paper V2 M6 result-blind-protocol-admission cycle (2026-09-01):

- added `paper-v2-protocol-bundle-v1` as a strict admission contract for a
  future owner-approved Paper V2 freeze; it does not write or activate one;
- the bundle binds execution commit plus SHA-256 references for result-blind
  selection evidence, protocol/hypotheses/analysis/failure documents,
  dataset/model/prompt/decoding manifests, and matrix/randomization artifacts;
- all M2/M3/VQ1/M4 policies are parsed through their real validated policy
  objects; M5 terminal/candidate/finalization versions, metric direction and
  tolerance, one hidden gate, Docker image digest, network-none prediction, and
  exact A/B/C runner/adapter/context mappings are admitted together;
- cross-policy admission requires at least the known M5 minimum of one replay
  code execution, three terminal tool calls, and one protected validation query,
  while leaving positive exploration time/query capacity;
- duplicate/unknown/non-string fields, non-standard JSON constants, booleans as
  numbers, malformed hashes, visible confirmatory outcomes, non-excluded policy
  selection, version/backend/network/arm drift, and incompatible reserves fail
  closed before a receipt is produced;
- canonical hashing is built from normalized policy objects, eliminating hash
  drift from integer-versus-float spellings and stopping-resource input order;
- a Draft 2020-12 JSON Schema mirrors the structural contract; Python admission
  remains authoritative for cross-field/resource invariants;
- the public receipt exposes only bundle/component hashes and immutable IDs;
  no private paths, policy activation, freeze mutation, or outcome access occurs;
- `python -m pytest tests/test_protocol_v2.py -q` -> `21 passed`; adjacent
  protocol/stopping/budget/runner/freeze/planner suite -> `99 passed`;
- `python -m pytest -q` -> `520 passed, 5 skipped` in 718.52s with one existing
  Jupyter path deprecation warning; compileall, JSON Schema validation,
  `git diff --check`, and static non-wiring checks pass apart from line-ending
  notices;
- no real policy values or artifact hashes were selected, no runner imports M6,
  and no API, model endpoint, Docker run, pilot, experiment, stage, commit, push,
  or PR occurred.

Paper V2 M4 frozen-stopping-controller cycle (2026-09-01):

- added strict `stopping-policy-v1` loading from a regular JSON file with exact
  fields, duplicate/unknown/non-standard JSON rejection, canonical ordering,
  stable SHA-256 policy identity, and no implementation-selected thresholds;
- implemented a deterministic first-trigger-wins controller for reserve
  boundary, exploration exhaustion, eligible-validation no-improvement,
  incumbent-backed agent finalization, and unrecoverable contract failure;
- added read-only `EpisodeBudget.exploration_boundary_resources()` so the
  stopping layer can detect exact token/call/code/tool/time/query boundaries
  without changing phase or consuming/borrowing resources;
- `NotebookGymEnv` records candidate improvement against the deterministic M1
  incumbent, exposes only public stopping receipts, and maps finalize decisions
  to the shared M5 protected replay; terminate decisions produce no hidden
  evaluation and do not enter the reserve;
- `GymAgent` checks for a boundary before each new model call and after every
  observation, and records step/call/budget exhaustion before its existing
  fallback; a focused test proves a boundary can submit without any LLM call;
- repeated-single-shot observes the same incumbent/tie semantics, resource
  boundaries, exhaustion reasons, and bundle failures, then uses the same M5
  terminal controller for finalize decisions;
- runner binding accepts `--research-v2-stopping-policy` only for the two
  M4-capable arms, requires an already-injected protected reserve, and includes
  both canonical policy content and hash in the budget-policy manifest input;
- `tests/test_stopping.py` -> `15 passed`; `tests/test_agent.py` plus policy
  suite -> `23 passed`; three end-to-end notebook M4 paths passed; repeated M4
  observers plus the existing reserve-gated M5 runner passed;
- `python -m pytest -q` -> `499 passed, 5 skipped` in 687.16s with one existing
  Jupyter path deprecation warning; compileall and `git diff --check` pass apart
  from line-ending notices;
- static search confirms no production construction of `FrozenStoppingPolicy`,
  `FinalizationReservePolicy`, or `ValidationQueryPolicy`; policy thresholds,
  reserve/query values, objective tolerance, and arm mapping remain excluded-
  development and preregistration gates;
- no API, model endpoint, Docker run, pilot, experiment, stage, commit, push, or
  PR occurred.

Paper V2 M5 symmetric-terminal-selection cycle (2026-08-30):

- added versioned `symmetric-terminal-v1` as the only protected terminal
  controller for both the iterative notebook adapter and the repeated-single-shot
  adapter;
- the common sequence is fixed as incumbent selection, immutable bundle
  verification, exact producing-revision replay in a fresh state, isolated
  raw-row/serialization preflight, normalized validation-score match, atomic
  protected artifact write/reload, and one `HiddenEvaluationGate` attempt;
- refactored the existing M2b `NotebookGymEnv` path onto that controller while
  retaining reserve charging, replay-cell evidence, private diagnostics,
  public redaction, terminal failure statuses, and no-repair/no-fallback behavior;
- added `RepeatedSingleShotTerminalAdapter`: every validation-ready attempt is
  stored through the same `CandidateRegistry` and `CandidateBundleStore` as an
  immutable one-cell notebook revision; terminal replay uses a fresh executor
  namespace and never falls back to the live host `best_model` object;
- the real `run_multishot` integration is gated by an already-injected
  `FinalizationReservePolicy`; without a reserve it retains the Paper V1 path;
  production code still does not construct a reserve;
- an active V2 reserve requires explicit metric direction and score tolerance;
  the new CLI fields have no active-path defaults and a missing value fails
  before the episode;
- if a `ValidationQueryPolicy` is also injected, repeated-single-shot removes
  labels from the agent dataset card, prompt, namespace, and replay namespace,
  charges attempt and protected-finalization validation accesses, and uses the
  same normalized score for feedback and selection;
- common controller plus adapter -> `8 passed`; reserve-gated runner and adjacent
  regressions -> `41 passed`; common controller plus full notebook regression ->
  `57 passed, 1 skipped`;
- `python -m pytest -q` -> `479 passed, 5 skipped` in 586.27s with one existing
  Jupyter path deprecation warning; compileall and `git diff --check` pass apart
  from line-ending notices;
- static search confirms no production construction of
  `FinalizationReservePolicy` or `ValidationQueryPolicy`; confirmatory use still
  requires frozen policy values, the same Docker backend/prediction isolation,
  preregistration, and immutable protocol binding;
- no API, model endpoint, Docker run, pilot, experiment, stage, commit, push, or
  PR occurred.

Paper V2 VQ1 adaptive-validation-control cycle (2026-08-30):

- added `validation-query-v1` as an explicitly injected policy with required
  global query cap, protected-finalization query reserve, and public numeric
  precision; no defaults, CLI arguments, runner construction, or experimental
  values were added;
- extended `EpisodeBudget` rather than adding an unrelated counter; failed,
  repeated, cached/reused, and same-revision queries are independently charged,
  source classified, event logged, capped before evaluation, and unable to
  borrow across the exploration/finalization boundary;
- centralized candidate readiness accounting covers explicit/auto validation,
  automatic model feedback, candidate checks, quick validation, submit
  preflight, and protected replay; tuning preflight, every tuning trial, final
  tuned-model check, and enabled Cleanlab diagnostics are also charged;
- when and only when the policy is active, `val_df` and the workspace validation
  CSV contain features without the target; public cards/profiles use that same
  label-free frame, while the host retains labels for charged evaluation;
- feedback-derived metrics are normalized with the frozen precision before both
  candidate selection and public display; Cleanlab confidence follows the same
  precision contract;
- host summaries expose the policy, realized count/source ledger, selected
  feedback-validation metric, and `hidden_minus_feedback_validation_metric`;
  the strict M3 pack exposes only aggregate query limits/use/remainders and
  rejects an injected source ledger or inconsistent counters;
- `python -m pytest tests/test_research_budget.py -q` -> `29 passed`;
- focused cap/bypass/label-isolation/kernel set -> `16 passed, 78 deselected`;
- `python -m pytest tests/test_context_compression.py -q` -> `12 passed`;
- combined budget/context/notebook suite -> `93 passed, 1 skipped` in 444.53s
  before the final aggregate-context test; that final context suite passed
  separately and is included in the full result;
- `python -m compileall -q gym research experiments tests`, `git diff --check`,
  and `python -m pytest -m "not integration" -q` completed successfully; the
  full suite result is `469 passed, 3 skipped, 2 deselected` in 545.96s with one
  existing Jupyter path deprecation warning;
- no production runner or CLI constructs the policy; the inactive, reserve-gated
  M5 repeated runner now enforces it when explicitly injected;
- confirmatory enforcement still requires the isolated Docker backend, frozen
  limits/precision, symmetric M5 activation, external preregistration, and an
  immutable protocol freeze;
- no API, model endpoint, Docker run, pilot, experiment, stage, commit, push, or
  PR occurred.

Paper V2 M3 deterministic-context cycle (2026-08-30):

- added a versioned, explicitly injected `ContextCompressionPolicy`; no default
  limits, CLI flag, runner wiring, or automatic activation were added;
- the new public pack retains the task contract, public tested hypotheses,
  incumbent identity and validation state, active public blockers, deterministic
  remaining-resource counters, allowed actions, notebook state, and the fixed
  finalization contract;
- strict schema validation rejects unknown/private fields, malformed state, and
  non-finite numbers before serialization;
- canonical UTF-8 JSON is byte-stable under mapping-order changes; bounded
  history and state truncation use a fixed priority order, preserve the required
  core, and fail closed when the core itself cannot fit;
- raw errors, private diagnostics, hidden values, bundle paths, and private
  checklist data are absent from the public pack; raw/source-message hashes are
  written only to the private receipt;
- later success for an action clears its active blocker, while failed or checked
  public attempts remain represented as tested hypotheses;
- `python -m pytest tests/test_context_compression.py -q` -> `11 passed`, one
  existing Jupyter path deprecation warning;
- the dedicated M3 `NotebookGymEnv` public-pack/privacy test passed after the
  final hypothesis-history and blocker-clearing behavior was added;
- `python -m compileall -q gym research experiments tests`, `git diff --check`,
  and `python -m pytest -m "not integration" -q` completed successfully; the
  full suite result is `453 passed, 3 skipped, 2 deselected` in 474.03s with one
  existing Jupyter path deprecation warning;
- no API, model endpoint, Docker integration, pilot, experiment, stage, commit,
  push, or PR occurred.

Paper V2 M2b protected-finalization cycle (2026-08-30):

- added the opt-in terminal controller without adding CLI flags or default
  reserve values; static search confirms only tests instantiate
  `FinalizationReservePolicy`;
- explicit agent submit and host-forced finalize use the same three finalization
  tool units and the same producing-revision path, including after the
  exploration cutoff has fired;
- the host selects the historical incumbent, verifies its complete bundle, and
  executes its exact snapshot in a restarted kernel without modifying either
  the snapshot or the agent's later notebook revision;
- replayed raw-row prediction, serialization, and validation metric are checked
  before any artifact or hidden evaluation; validation drift fails closed;
- the replayed model is atomically written, checksum-verified, and reloaded from
  `protected-finalization-v1` before the one-shot hidden gate opens;
- no-incumbent, bundle corruption, replay failure, preflight failure, metric
  drift, storage failure, reserve exhaustion, and hidden failure are terminal;
  none invokes code repair, live-kernel/autofit fallback, a new candidate,
  exploration-resource borrowing, or a second hidden attempt;
- `python -m pytest tests/test_finalization.py -q` -> `4 passed, 2 skipped`;
  the two skips are unavailable Windows symlink capabilities;
- all ten M2b controller cases passed across focused and full runs, including
  the real exploration-cutoff-to-reserve transition;
- `python -m pytest -m "not integration" -q` -> `441 passed, 3 skipped, 2
  deselected` in 641.39s, with one existing Jupyter path deprecation warning;
- `python -m compileall -q gym research experiments tests` and `git diff
  --check` passed apart from line-ending notices;
- no API, model endpoint, Docker integration, pilot, experiment, stage, commit,
  push, or PR occurred.

Paper V2 M2a protected-budget cycle (2026-08-30):

- added an opt-in `FinalizationReservePolicy` without modifying the frozen
  `EpisodeBudgetPolicy` fields or default serialization;
- exploration limits are computed as global limits minus the protected reserve;
  crossing that boundary raises a distinct non-terminal exploration stop while
  preserving the finalization pool;
- the explicit phase transition is one-way and idempotent, rejects pending LLM
  reservations, records its trigger/start usage, and limits finalization to its
  own realized token/call/execution/tool/time deltas;
- no unused exploration capacity can be borrowed by finalization, including
  when the phase begins early;
- reserve-aware policy payloads are available for future manifests, but no CLI
  activation or numeric reserve default exists;
- `python -m pytest tests/test_research_budget.py -q` -> `19 passed`;
- adjacent budget/experiment/runner/freeze suite -> `50 passed`, one existing
  Jupyter path deprecation warning;
- `python -m pytest -m "not integration" -q` -> `427 passed, 1 skipped, 2
  deselected`, one existing Jupyter path deprecation warning;
- no API, model endpoint, pilot, experiment, commit, push, or PR occurred.

Paper V2 M1a+M1b isolated development cycle (2026-08-30):

- fetched and verified `origin/main` at `1d94774`, then created a clean dedicated
  worktree on `JD/paper-v2-m1a`; the dirty Paper V1 manuscript worktree was not
  modified;
- implemented M1b as a dependent local layer: atomic `candidate-bundle-v1`
  storage, canonical private metadata, model/notebook snapshots, SHA-256
  verification, deterministic historical registry restoration, public receipt
  redaction, and pre-hidden-evaluation fail-closed corruption handling;
- `python -m pytest tests/test_candidate_store.py tests/test_candidates.py -q`
  -> `20 passed, 1 skipped`, one existing Jupyter path deprecation warning;
- direct notebook regression before the final storage-write case -> `36 passed,
  1 skipped`; the latest corruption/write-failure targets then passed `2 passed`,
  and the final full offline suite covers both;
- adjacent experiment, submission, runner, artifact, and protocol suite ->
  `92 passed`, one existing Jupyter path deprecation warning;
- `python -m pytest -m "not integration" -q` -> `416 passed, 1 skipped, 2
  deselected`, one existing Jupyter path deprecation warning;
- `python -m compileall -q gym tests/test_candidate_store.py
  tests/test_candidates.py tests/test_notebook_env.py` -> passed;
- `git diff --check` -> passed with line-ending notices only;
- no API, paid endpoint, Docker integration, experiment, stage, commit, push, or
  pull request was performed.

Paper V1 protocol cycle (2026-08-12/13):

- `python -m research.validate_protocol research/protocols/protocol_v1.yaml` ->
  valid structure, 120 provisional confirmatory runs, 7 open freeze blockers,
  status `draft_unfrozen`.
- `python -m pytest tests/test_research_protocol.py -q` -> `3 passed`.
- Focused offline suite (episode modes, agent, protocol, experiments) ->
  `66 passed`, one Jupyter path deprecation warning.
- Initial full `python -m pytest -q` -> `267 passed, 2 failed`, one warning; both
  failures were Docker integrations caused by the missing local
  `autovibe-gym-sandbox:latest` image.
- `docker build -f Dockerfile.sandbox -t autovibe-gym-sandbox:latest .` -> passed
  on the second attempt after one external PyPI read timeout.
- The two previously failing Docker integrations were rerun explicitly ->
  `2 passed`, one warning. Together these completed runs cover all 269 collected
  tests; a later monolithic repeat exceeded the local wrapper timeout and is not
  represented as a passing run.
- No API experiments or hidden-test research runs occurred in this protocol
  cycle; the protocol is under review in Draft PR 1.

Paper V1 auditable-manifest cycle (2026-08-13):

- `python -m pytest tests/test_research_run_artifacts.py -q` -> `19 passed`.
- `python -m pytest tests/test_llm.py tests/test_research_run_artifacts.py -q`
  -> `40 passed`, one existing Jupyter path deprecation warning.
- The offline smoke creates an atomic manifest and append-only ledgers, links
  condition/run IDs, redacts secret-shaped values/private notebook fields, and
  classifies unsuccessful outcomes without any API or hidden-test run.
- Provider retry hooks were verified to emit each attempted request and to be
  fail-open for observability, preserving the existing provider return/retry
  behavior.
- Focused experiments/LLM/protocol/research suite -> `70 passed`, one existing
  Jupyter path deprecation warning.
- Full `python -m pytest -q` -> `291 passed`, one existing Jupyter path
  deprecation warning.
- No expensive API experiment or confirmatory run was executed; generated run
  artifacts were confined to pytest temporary directories.
- PR 1 and PR 2 were merged in order on 2026-08-14. PR 2 was rebased onto the
  merged PR 1, passed `289 passed, 2 skipped` locally and the required GitHub
  `Python tests` check, then squash-merged as `67069a3`.

Paper V1 post-collection analysis-binding repair (2026-08-16):

- Freeze-v2 reconciliation completed with `terminal=120`, `pending=0`,
  `running=0`, and `replacement_required=0` before analysis was invoked.
- The first analysis invocation failed closed before emitting statistics because
  the validator expected an impossible reverse binding from the pre-plan FANU
  reference file to the later-created plan.
- `validate_references` now verifies the exact frozen generator payload and its
  one-way `analysis_reference_hash` binding; the frozen plan/reference pair
  validates with four dataset references.
- `python -m pytest tests/test_research_analysis.py
  tests/test_research_protocol.py -q` -> `18 passed`.
- `python -m pytest -q` -> `384 passed`, one existing Jupyter path deprecation
  warning.
- `git diff --check` -> passed. No API request, condition rerun, confirmatory
  statistic, or external publication occurred in this repair cycle.

Paper V1 fair-budget cycle (local PR 3 preparation, 2026-08-14):

- Added an opt-in research `EpisodeBudget` with protocol defaults of 64,000
  total reported tokens, 4,096 output tokens per call, 12 logical LLM calls,
  20 code executions, 20 host-tool calls, and 1,800 seconds. Limits are checked
  before the affected provider, execution, or tool call and emit audit events.
- Research-mode repeated single-shot, iterative no-checklist, and gym-with-
  checklist now use the same budget object. Reported reasoning tokens are
  included in total usage; cached-input usage is recorded separately while the
  provider's raw input total remains the budget input count.
- Added one common no-repair submission validator for serializability, raw-row
  prediction, output length, and non-null predictions. Research mode disables
  host-side autofit and host finalization repair/replay; the agent must leave an
  already-fitted candidate that passed clean replay and validation.
- Hidden evaluation is limited to one attempt per terminal candidate outcome.
  A hidden-evaluation failure is terminal and provides no repair feedback to the
  agent. Product-mode retry/autofit compatibility remains unchanged outside the
  opt-in research path.
- Post-outcome LLM summaries are disabled for research runs so they cannot
  consume decision-adjacent budget after the outcome.
- `pytest -q tests/test_research_budget.py tests/test_research_submission.py tests/test_research_run_artifacts.py tests/test_research_protocol.py`
  -> `34 passed`, one existing Jupyter path deprecation warning.
- `pytest -q tests/test_llm.py tests/test_experiments.py tests/test_agent.py tests/test_env.py tests/test_env_protocol.py tests/test_episode_modes.py tests/test_run_summary.py`
  -> `89 passed`, one existing Jupyter path deprecation warning.
- `pytest -q tests/test_notebook_env.py` -> `33 passed, 1 skipped`, one existing
  Jupyter path deprecation warning; the skipped case is the Docker integration
  because Docker was not available to that test process.
- Review follow-up normalizes OpenAI-compatible usage so reasoning tokens remain
  a separately reported subset of completion tokens rather than being charged
  twice. A regression test fixes the expected visible-output/reasoning split.
- A pre-call or pre-execution budget stop in the single-shot control is now a
  recorded `budget_exhausted` terminal outcome and reaches manifest
  finalization without provider access or hidden evaluation.
- Focused review regression -> `77 passed`, one existing Jupyter path
  deprecation warning.
- Full `pytest -q` -> `307 passed, 2 skipped`, one existing Jupyter path
  deprecation warning. `git diff --check` passed.
- Required GitHub `Python tests` passed on PR #65. The additional Docker sandbox
  integration job failed twice on a kernel-readiness timeout, alternating
  between its two integration cases; the production code and focused local
  suites did not reproduce a deterministic failure. CI handshake hardening is
  intentionally not mixed into this research PR without separate approval.
- No API calls, pilot episodes, or hidden-test research runs occurred. PR 3 was
  rebased onto merged `origin/main`, reverified, and published as PR #65.

Paper V1 confirmatory-planner cycle (PR 4, 2026-08-15):

- PR 3 review feedback was answered and its only unresolved thread was resolved
  after commit `385a09f`; required `Python tests` and the Docker sandbox
  integration both passed on the final head.
- PR 3 was fast-forward merged as PR #65 so its existing author/committer
  identities remained `JapanDino <klim.i.rumyantsev@gmail.com>`; merged `main`
  is exactly `385a09f`.
- Added a deterministic confirmatory planner that expands exact dataset/model/
  arm/replicate inputs into unique condition IDs and a canonical immutable plan
  hash. The protocol-sized fixture precomputes 120 conditions in 40 blocks.
- Within every dataset/model/replicate block, A/B/C order uses a portable
  SHA-256-seeded permutation. Input list order cannot change the plan/hash.
- Identical plan creation is an idempotent no-op; a different plan cannot
  overwrite an existing file. The planner rejects unresolved placeholders,
  provisional protocol matrices, pilot budgets, and config/protocol drift.
- Resume reconciliation reads append-only manifests, rejects unplanned or
  duplicate outcomes and broken replacement chains, leaves agent failures as
  terminal outcomes, and queues only declared infrastructure/provider failures
  under the same condition with `rerun_of` and a reason.
- Added JSON schemas for exact plan input and expanded plan output plus CLI/docs
  for offline `build` and `status`. The CLI does not launch episodes.
- Automated review identified and the branch fixed three fail-closed gaps:
  budget status now must equal `frozen`, completed attempts require a declared
  failure category, and concurrent writers publish via an atomic hard link so a
  different preregistered plan cannot be overwritten.
- Focused planner/manifest/protocol/budget/submission suite -> `66 passed`, one
  existing Jupyter path deprecation warning.
- Full `python -m pytest -q` -> `338 passed, 2 skipped`, one existing Jupyter
  path deprecation warning. `git diff --check` passed.
- No exact confirmatory config, real plan, API call, pilot episode, hidden-test
  evaluation, or confirmatory outcome was created or inspected.

Paper V1 preregistered-analysis cycle (PR 5a, 2026-08-15):

- Added a synthetic-only primary pipeline that binds frozen FANU references to
  the immutable plan ID/hash and refuses missing datasets, invalid metric
  directions, or zero denominators.
- Agent failures remain at dummy performance; infrastructure attempts require a
  same-condition terminal replacement and cannot silently enter the primary
  population. Successful outcomes require `valid_submit=true` and exactly one
  hidden evaluation.
- The pipeline emits a machine-readable incomplete-run error before arm effects,
  then pairs H1 `B-A` and H2 `C-B` by dataset/model/replicate when complete.
- Estimates weight dataset-model strata equally. The implementation includes
  paired stratified percentile bootstrap intervals, equal-stratum sign-flip
  permutation tests, exact McNemar valid-submit tests, and Holm correction.
- Raw pairs and per-dataset/model summaries remain in the content-hashed result;
  concurrent writers cannot overwrite a different output, while identical
  reruns are idempotent.
- Automated review identified two P1 provenance gaps: protocol analysis settings
  and FANU reference values were not cryptographically bound to the plan. The
  planner now carries both canonical hashes; analysis recomputes, rejects drift,
  and propagates them into the result. Synthetic tampering regressions cover
  changed seeds/resampling and changed directions/dummy/reference values.
- Follow-up review correctly identified that arm-specific valid-submission
  confidence intervals were required but not frozen. The protocol and pipeline
  now specify and emit 95% Wilson score intervals; paired rate differences still
  use exact McNemar. A separate identity comment cited a nonexistent commit SHA;
  both actual PR commits independently show JapanDino as author and committer.
- Final review added valid fail-closed checks for definitive `git_dirty=false`,
  recomputation of usage/execution totals from disk ledgers, and frozen
  per-arm/dataset/model hidden-score summaries over all agent outcomes and the
  explicitly selection-biased successful-only subset. Two synthetic regressions
  cover dirty provenance and edited manifest totals. A second identity comment
  again cited a SHA absent from both the local object database and GitHub API;
  the actual PR range remains entirely JapanDino-authored/committed.
- Added versioned frozen-reference/result schemas, CLI/docs, and synthetic
  fixtures. Review-fix focused planner/analysis/protocol/artifact suite:
  `66 passed`.
- Full `python -m pytest -q`: `351 passed, 2 skipped`, one existing Jupyter path
  deprecation warning. `git diff --check` passed.
- No API call, pilot, exact confirmatory plan, real manifest, hidden-test score,
  confirmatory result, table, or claim was created or inspected.
- PR #67 passed the final GitHub `Python tests` job in 5m02s. All actionable
  review threads were answered and resolved; repeated identity comments citing
  repository-absent SHAs were closed with local-object, GitHub-API, and exact
  PR-range evidence. The PR was fast-forward merged at `b1cd5f5`, preserving all
  four commits with JapanDino as both author and committer.

Paper V1 protocol-freeze cycle (2026-08-15):

- The authenticated internal OpenAI-compatible endpoint advertised exact IDs
  `deepseek-v4-flash` and `gemma-4-26b`; it did not expose immutable weight
  revisions, so the protocol uses the honest `service-snapshot-2026-08-15`
  label and records request-level provenance. No API secret is committed.
- Result-blind pilots exercised A/B/C for both models on non-confirmatory
  `example_room_occupancy`. The 64k and 128k token limits were rejected because
  iterative runs stopped on pre-call reservation. At 256k, both model-specific
  B checks reached the fixed 12-call ceiling (108283 and 134878 accounted
  tokens) without token-reservation stop. Pilot scores were not inspected for
  selection and pilot outputs cannot enter confirmatory analysis.
- A pilot candidate containing a dynamic `FunctionTransformer` reproduced a
  native Windows access violation during host-side readiness prediction. The
  validator and one-shot hidden gate now execute prediction in an isolated
  worker with timeout/crash containment. Confirmatory Docker mode selects a
  separate ephemeral network-none Docker evaluator rather than a host child.
  Two interrupted infrastructure attempts remain non-overwritten; the
  replacement pilot finalized normally with all ledgers.
- Frozen inputs remove four privileged ground-truth concentrations from Air
  Quality, group diabetes splits by patient and remove both IDs, and remove
  post-contact `duration` from Bank Marketing. Diabetes group overlap is zero.
  Dataset cards record official UCI URLs/DOIs/licenses, local/raw hashes,
  contamination, privacy, and task-framing limitations.
- The fixed HistGradientBoosting reference beats the dummy endpoint on all four
  hidden snapshots. Final immutable hashes are manifest
  `91e4a98523e336929057bb0a4b5b27b54aa2a7b9f2ec0f0057a37989b562d7ac`
  and reference
  `e867776aa650b7445a9fb17d16686e176dca8b14cc0c1460426708fe162e0570`.
- `python -m research.validate_protocol research/protocols/protocol_v1.yaml` ->
  valid 120-run structure, zero open freeze blockers, status `frozen`.
- Focused freeze/protocol/submission/notebook suite -> `48 passed`, one existing
  Jupyter path warning, in 343.49 seconds.
- Full `python -m pytest -q` -> `361 passed`, one existing Jupyter path
  deprecation warning, in 425.44 seconds. `git diff --check` passed.
- No real confirmatory plan was generated; no confirmatory episode, arm effect,
  p-value, confidence interval, or scientific conclusion was inspected.

Last local run:

```bash
python -m pytest
```

Result after deterministic action observability: `236 passed` (one sklearn
`InconsistentVersionWarning` from the Docker notebook pickle smoke).
Current deterministic action observability cycle:

- Canonical action schema now uses `type`, required `stage`, canonical
  `thoughts`, and non-mutating `think`.
- `python -m pytest` -> `236 passed`, one existing sklearn warning.
- Dashboard TypeScript build + Vite production build passed via bundled Node
  runtime (`tsc -b`, `vite build`).
- Browser smoke on `http://127.0.0.1:8010/runs/live_codex_stage`: Run Detail
  header showed `Этап = Анализ валидации`; Trajectory rendered `think` and
  type/stage badges; Thoughts tab rendered scratchpad entries; desktop and
  mobile viewport console checks had no warnings/errors.
- Follow-up launcher check after a real dashboard run: selected model records
  now explicitly set `LLM_PROVIDER` (`openai`, `google`, or `litellm`) so a
  `.env` Gemini default cannot route OpenAI-compatible dashboard models through
  the Google SDK; `python -m pytest tests/test_dashboard.py -q` -> `17 passed`.
Current model-registry config cycle:

- `.env.example` and local `.env` no longer contain LLM model settings or LLM
  API keys; model names, provider, endpoint URL, and per-model API key live in
  the shared model registry.
- Added `python -m experiments.models` for CLI model registry management.
- `experiments.run`, `run_matrix`, and the `run_*` scripts require `--model` and
  resolve it as a model id/name from the registry.
- Gemini models no longer require or send Base URL; LiteLLM receives the
  per-model key through the registry-backed runtime path.
- `python -m pytest tests/test_llm.py tests/test_dashboard.py tests/test_experiments.py -q` -> `64 passed`.
- Dashboard TypeScript build (`tsc -b`) and Vite production build passed.
Current dashboard multi-select cycle:

- `python -m pytest tests/test_dashboard.py tests/test_experiments.py` -> `39 passed`
- dashboard TypeScript build + Vite production build via bundled Node runtime
- Browser smoke on `/new`: `All modes` is absent and run types are selected as
  separate product modes.
Current dashboard responsive-hardening cycle:

- Backend API healthy on `http://127.0.0.1:8000/api/health`; Vite dev server
  running on `http://127.0.0.1:5174/new`.
- dashboard TypeScript build + Vite production build via bundled Node runtime.
- CSS hardened for compact top navigation, responsive New Run stacking, card/grid
  min-width behavior, wrapped filter/setting rows, dataset actions, compare picks,
  preview values, and chart bars so narrow and desktop breakpoints avoid clipped
  controls and horizontal layout drift.
Current dashboard five-mode selection cycle:

- New Run includes Iterative again, allows selecting up to 5 run types, removes
  the recommendation badge, and labels the interactive modes as Flexible gym and
  Fixed gym.
- Shared product-mode metadata, common `experiments.run --mode all`, dashboard
  batch launch validation, frontend types, and tests now use the same five-mode
  product set.
Current dashboard environment-badge cycle:

- New Run run-type cards now show only two badges: `Среда` for Iterative,
  Flexible gym, and Fixed gym; `Без среды` for Single-shot and
  Repeated single-shot.
- dashboard TypeScript build + Vite production build via bundled Node runtime.
Current dashboard budget-tooltip cycle:

- Removed the inline `local — длиннее, cloud — экономнее` hint from the New Run
  budget preset field.
- Added Problems-style `?` tooltips to New Run budget parameter labels.
- dashboard TypeScript build + Vite production build via bundled Node runtime.
Current common-runner failure-propagation cycle:

- `experiments.run` now exits with the first failed child return code after
  printing the batch summary, even when `--stop-on-failure` is not set.
- `python -m pytest tests/test_experiments.py` -> `27 passed`.
Current dashboard run-detail tabs cycle:

- Dashboard tab menus no longer have a 1px vertical overflow: the divider line is drawn inside the tab strip and tab buttons no longer use a negative bottom margin.
Current dashboard/product-mode label cycle:

- Dashboard mode labels now display `Flexible gym` wherever the gym product mode
  was previously shortened to `Flexible`.
- Fixed-transition product mode display text is shortened to `Fixed gym` in the
  dashboard and shared experiment matrix labels.
- `dashboard/web`: TypeScript build + Vite production build passed.
- `.venv/bin/python -m pytest tests/test_experiments.py` -> `27 passed`.
Current dashboard sidebar cleanup cycle:

- Removed the bottom sidebar team/local-mode block from the dashboard shell.
- Moved the sidebar collapse toggle down to the bottom edge of the simplified
  sidebar.
Current dashboard logo cycle:

- Replaced the old H-like sidebar logo mark with a diagonal dumbbell mark
  matching the yellow-square fitness reference while avoiding letter shapes.
- `dashboard/web`: TypeScript build + Vite production build passed.
- Browser smoke captured `.sidebar-logo`; only existing dev warnings/favicon 404
  appeared in console.
Additional checks:

- `python -m experiments.run --dataset-dir datasets/demo/prepared --mode all --model fake-model --dry-run`
- `python -m experiments.run_all_modes_matrix --datasets datasets/demo/prepared --models fake-model --dry-run`

The Docker-backed notebook integration test ran locally in this Windows
workspace and passed. GitHub Actions remains the source of truth for the Linux
sandbox image build.

---

## H200 Recon Findings (2026-06-01)

First full run of all modes × both cloud models (`deepseek-v4-flash`, `gemma-4-26b`)
on the H200 server. Cloud API (`llm.letovo.site`) exposes only these two models —
no Qwen yet (curator's Qwen3-32B suggestion needs local vLLM serving).

Fixed in this PR:

- **OpenBLAS / OpenMP thread exhaustion.** The subprocess sandbox env and the
  local Jupyter kernel env set no thread caps. On the many-core server this aborts
  model training with `OpenBLAS: Memory allocation failed` (subprocess backend) and
  xgboost ctypes `DataIter` crashes (notebook kernel). Now capped via
  `thread_limit_env()` in `executor.py`, `_minimal_kernel_env()`, and the Docker
  kernel `--env` block (overridable with `AUTOVIBE_SANDBOX_THREADS`).
- **`run_fixed.py` CLI parity.** It rejected `--max-steps` (used by every other
  runner and the matrix), crashing the fixed-transition runs with exit 2. Added.
- **scikit-learn replay skew.** Candidate pickles were written in the sandbox image
  (sklearn 1.7.2) and read on the host venv (1.8.0) → `InconsistentVersionWarning`.
  Pinned `scikit-learn==1.7.2` so the image and host resolve identically.

Gym `test_metric=null` — root-caused and fixed (next PR, branch
`dev/claude/gym-good-score`):

- **Cause 1 — action parsing.** `gemma-4-26b` wraps its JSON in chat-template
  tool-call tokens (`{...}<tool_call|>`, `<|tool_call>call:{...}`). The strict
  parser required the text to end in `}`, fell through to the legacy code
  fallback, and dumped the raw action text into a notebook cell → SyntaxError
  loop, so the agent never reached validate/submit. Fixed with robust extraction
  (strip wrapper tokens, balanced-brace JSON scan, `strict=False`).
- **Cause 2 — brittle clean run.** restart_and_run_all re-runs the whole messy
  notebook; missing libs (`seaborn`), slow GridSearchCV (per-cell timeout), and
  cross-cell `NameError`s from deleted cells made every clean run fail. Mitigated
  by installing matplotlib/seaborn, raising the per-cell timeout to 120s, and
  steering the prompt to finalize early with a small search.
- **Cause 3 — strict finalize + label skew.** Forced submit needed a prior
  validate; agents that built a good model but mismanaged the protocol got null.
  Added host-side `NotebookGymEnv.finalize()` (live-kernel fallback before any
  kernel-wiping restart) and label dtype coercion in validate/submit.
- **Verified:** a dirty-kernel, label-encoded model now finalizes to
  `final_test_metric=0.7247` f1_macro on `student_dropout` (vs baseline 0.739).

Single-shot / repeated multishot failures — addressed in this PR (branch
`dev/claude/oneshot-multishot-fix`):

- **Raw hidden-test input mismatch.** Baseline/multishot prompts now require a
  fitted scikit-learn `Pipeline` / `ColumnTransformer` assigned to `model`, so
  `model.predict(raw_df)` works on raw validation and hidden-test rows instead
  of relying on notebook-side preprocessing state.
- **Label dtype mismatch.** The label-coercion scoring helper is now shared by
  `NotebookGymEnv`, `run_baseline.py`, and `run_multishot.py`, so label-encoded
  integer predictions can be scored against string/categorical targets.
- **Joblib worker crashes in sandbox.** The subprocess/Docker executor forces
  sequential joblib/loky execution (`JOBLIB_MULTIPROCESSING=0`,
  `LOKY_MAX_CPU_COUNT=1`), preventing `n_jobs=-1` searches from failing when
  process spawning is restricted.
- **Pending verification:** rerun the H200 matrix to confirm DeepSeek
  baseline/multishot no longer end as `submit_failed`.

Still open (need server-side iteration):

- **Sandbox image name.** Default is `autovibe-gym-sandbox:latest`; the server only
  had `autovibe-gym:latest` built, and rootless Docker can't pull from docker.io.
  Server setup task: build the image under the expected tag.

## Web Dashboard (`dashboard/`, branch `dev/claude/web-dashboard`)

Local control panel, separate from `gym/`. Reuses the project `.venv`.

- **Backend** (`dashboard/server`, FastAPI): reads runs from MLflow and parses
  episode artifacts into notebook/trajectory/checklist/errors/logs (per-item
  checklist closure reconstructed by replaying public notebook events through the
  gym's own `NotebookChecklist`); datasets CRUD + CSV upload over `datasets/`;
  models JSON registry seeded from `.env` with OpenAI-compatible health probe;
  run launcher spawning `run_baseline/run_multishot/run_gym` subprocesses (shared
  MLflow store), live status + process-log tail + stop.
- **Frontend** (`dashboard/web`, Vite+React+TS): all 8 screens built to the
  T-Bank design tokens — Dashboard, New Run, Runs, Run Detail (5 tabs), Compare,
  Datasets (+detail/upload/delete), Models, Settings. Light/dark theme + accent.
- **Execution modes:** each run picks where the gym executes — **local** (on the
  machine running the backend; only the LLM call is remote — works off-VPN) or
  **server (SSH)** (gym + kernels run on the GPU server, results synced back via
  rsync; configured in Settings). The site can also be served entirely from the
  server (single-app mode) when a port is reachable.
- **Models:** registry seeded with the team's gemma/deepseek on the shared LLM
  server; any OpenAI-compatible endpoint (e.g. Cerebras/Groq/Gemini) can be added.
  Header pill shows LLM server reachability.
- **Live updates:** runs launch into a known `data/runs/<id>/workspace` dir; the
  gym already flushes public artifacts after every step, so while a run is in
  progress the dashboard reads that dir directly — step counter, checklist
  coverage, notebook cells, trajectory and logs all advance live (detail polls
  every 2.5s, running runs enriched via `episode_progress`). MLflow is used for
  finished runs.
- **Verified:** `npm run build` clean; FastAPI serves over HTTP; Vite dev proxies
  `/api`; real MLflow runs/datasets render; launcher builds correct commands;
  simulated mid-run workspace confirms live step/checklist/notebook/logs reads.
- **Hardened after merge:** Windows/default Python detection now resolves the
  repo `.venv\Scripts\python.exe` (with `sys.executable` fallback), single-shot
  and repeated single-shot launches use the correct planned step counts, MLflow
  mode/progress/status mapping handles `baseline_single_shot` and repeated
  attempts, placeholder zero scores from failed submits are hidden, checklist
  detail/list coverage both fall back to authoritative MLflow coverage when
  episode artifacts are absent, and the run detail donut uses that authoritative
  coverage value. The responsive shell now switches to compact top navigation on
  mobile so the dashboard has no page-level horizontal overflow.
- **Dataset Center expansion:** `/datasets` now manages raw and prepared data
  with search/filter/sort, staged file and URL uploads, safe archive extraction,
  table preview, raw-table splitting, prepared-file mapping, rich
  `dataset_config.json`, compatible `prepared/meta.json`, editable sources and
  agent notes, and backward-compatible display for old prepared datasets.
- **Dataset Center polish:** dataset cards render as a one-column list, the UI is
  localized to Russian while preserving common ML terms (`target`, `seed`,
  `raw`, `train/val/test`, `Target column`), dataset suite/group metadata is
  removed from dashboard/project flows, empty sources display `-`, and example
  configs now carry the repository creation timestamp plus UCI source metadata.
- **Dataset Center PR #34 hardening:** URL downloads now reject localhost/private
  targets and unsafe redirects, archive extraction budgets count decompressed
  gzip bytes, failed create-from-config attempts clean their temporary dataset
  root, root-format legacy datasets keep root `meta.json` edits, JSONL raw
  uploads are covered by tests, `/datasets` remains compatible with the new
  `/problems` navigation, and CI dataset preparation uses the current
  `prepare_datasets.py` CLI.
- **Verified after hardening:** backend API smoke confirms `/api/health`,
  `/api/runs`, and `/api/runs/{id}/checklist` agree on `11/12` and `0.88` for a
  legacy MLflow run without episode events; browser smoke on desktop/mobile
  confirms `11/12`, `88%`, no stale `92%`, no console errors, and no page-level
  horizontal overflow.
- **Visual fix:** active trajectory rows now render the "agent is executing"
  spinner inside the same marker column as step icons; the connector from the
  previous step ends at the spinner's top center.
- **Run:** `dashboard/server/run.sh` (API :8000) + `cd dashboard/web && npm i && npm run dev` (:5173).

## Blocked / Needs Decision

- The original freeze and launcher are merged, but execution is blocked until
  the result-blind v2 amendment passes regression/review, is merged, and receives
  annotated tag `paper-v1-experiment-freeze-v2`.
- The second annotator and adjudicator are nominees only. Checklist annotation
  cannot start until each gives written consent; this does not block the model
  run matrix.
- No frozen data, trajectory bundle, GitHub Release, Zenodo record, arXiv post,
  or journal submission is externally published. Those remain separate actions
  after the study/manuscript and explicit owner approval.
- Local Docker CLI is available in this Windows workspace as of 2026-06-02; the
  Docker-backed notebook integration test passed locally. GitHub Actions still
  verifies the Linux sandbox image path.
- Existing `GymEnv` remains for compatibility, but new iterative experiments
  should use `NotebookGymEnv`.
- Repository owner still needs to confirm the `main` ruleset requires the
  `Python tests` status check before merge if connector/API access cannot verify
  rulesets.

---

## Next Actions

1. [x] Completed full local regression of the frozen protocol, exact plan,
       dataset cards, pilot evidence, and crash-containment changes; human PR
       review remains a merge gate.
2. [x] PR 1 and PR 2 were reviewed through their required gates and merged in
       order; PR 2 passed local and required GitHub verification after rebase.
3. [x] Rebased, reviewed, reverified, and fast-forward merged PR #65: one fair
       global budget across A/B/C, research-mode no-autofit validation, and one
       hidden evaluation per terminal agent outcome. Merged `main` is `385a09f`.
4. [x] Reviewed and fast-forward merged PR #66: complete
       condition matrix/hash, blocked randomization, resume/idempotency,
       duplicate rejection, replacement queue, schemas, CLI, and acceptance checks.
5. [x] Published, reviewed, and fast-forward merged PR #67: preregistered FANU,
       completeness, paired analysis, valid-rate intervals, score sensitivities,
       ledger reconciliation, and synthetic fixtures before any outcome inspection.
6. [x] Completed a labeled result-blind availability/budget pilot after PR 5a;
       selected 256k/12 calls and kept all pilot outputs outside confirmatory data.
7. [x] Merged freeze PR #69 and created/verified annotated tag
       `paper-v1-experiment-freeze`; the plan/config/protocol hashes reconcile
       and every condition binds execution commit `b4e3da2`.
8. [x] Reviewed and fast-forward merged fail-closed launcher PR #70 at
       `e07b76c`; PR and post-merge GitHub tests passed.
9. [x] Merged and tagged result-blind freeze amendment v2, then executed its
       immutable 120-condition plan outcome-blind to complete reconciliation.
10. [x] Reviewed and merged the pre-analysis reference-binding implementation
        repair as PR #72, then ran the unchanged preregistered analysis over all
        120 reconciled terminal outcomes.
11. [x] Reviewed and fast-forward merged PR #73 with deterministic tables,
        figures, audit, and provenance at `d1ec571`.
12. [ ] Review and merge PR 6: generated working manuscript, primary-source
        bibliography, reproducibility guide, disclosures, and artifact manifest.
        Do not submit or publicly release the package in this cycle.
13. [ ] Keep the earlier H200 notebook-era matrix rerun and experiment-report
       refresh as product/pilot validation, clearly separated from Paper V1.
14. [x] Existing product PRs were merged to `main`; TZ/PROTOCOL/EXPERIMENT_REPORT
        were synchronized before this Paper V1 cycle.
15. [ ] Review and split the isolated Paper V2 M1a, M1b, M2a, M2b, M3, VQ1, M4, M5, M6, M7, M8, and M9 local
        stack into one-claim boundaries. Stage/commit/push/PR each require their
        own authorization. Do not activate M2/M3/VQ1/M4 or mark them protocol-ready
        until excluded-development policy selection, review, preregistration,
        and freeze are complete. VQ1, M4, M5, M6, M7, M8, and M9 must remain separate claim boundaries;
        do not make arm-symmetry claims until both arms use the frozen policies
        and the same Docker-enforced validation and terminal configuration.
16. [ ] Before any real M8/M9 artifact is admitted, independently review and
        hash the excluded-development tasks, confirmatory dataset scope,
        assumption sources, and full scenario-grid plan. Then execute every
        predeclared planning scenario and select replicates by the all-scenarios
        rule; keep all M7 decisions unresolved until substantive and statistical
        reviewers approve the actual evidence.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-11 | Prepared the complete inactive M1-M9 Paper V2 stack as a remote transfer checkpoint. Fresh `origin/main`, JapanDino Git/GitHub identity, the 44-file inventory, forbidden-path/size/credential-shaped scans, and diff cleanliness were checked before publication; `gitleaks` was unavailable. A fresh full offline suite completed `585 passed, 5 skipped, 1 failed` in 1051.96s because existing `test_hidden_submit_failure_is_generic` did not terminate on its third hidden-submit failure; the unchanged test then passed three isolated reruns. No research run, real M9 artifact, protocol freeze, PR, merge, or release was authorized; the checkpoint remains a mixed stack that must be split before review. |
| 2026-09-01 | Implemented inactive Paper V2 M9 excluded-development and assumption-provenance admission. Development and confirmatory metadata now fail closed on identity, content, or source-lineage overlap. Every M8 value is independently bound per scenario to its scenario/value hashes, declared use, admissible source artifact, and exact excluded manifest when pilot-derived. V1/pilot evidence is limited to nuisance quantities; it cannot set the non-inferiority margin or establish corrected-V2 effects. The complete scenario set and review are hash-bound, favorable-row selection is forbidden, and sample size must meet target in all required scenarios. Three synchronized schemas, a method note, and synthetic adversarial tests were added. Verification: focused `28 passed`, adjacent `211 passed`, final repeated full suite `586 passed, 5 skipped` in 817.01s. The first full run had one non-reproduced existing VQ1 failure (`585 passed, 5 skipped, 1 failed`), which passed alone, in its group, and in the complete rerun; no unrelated VQ1 code changed. No real manifest, scope, scenario grid, evidence package, API, Docker run, pilot, experiment, freeze, stage, commit, push, or PR. |
| 2026-09-01 | Implemented inactive Paper V2 M8 result-blind planning power simulation. Explicit scenarios bind M7 study hashes and declare alpha split/gatekeeping, margin, target, draws/seed, replicates, paired validity, and successful-FANU assumptions with no defaults. The simulator reports P1, gated P2, S1 and joint power plus Monte Carlo uncertainty. Strict result validation detects tampering and reconciles embedded assumptions/counts/power/intervals/targets before atomic no-overwrite output; runtime versions and planning-only claim limits are recorded. A synchronized Draft 2020-12 scenario schema and method note prohibit outcome access, favorable-row selection, and treating the normal-bound approximation as confirmatory evidence. Verification: focused `18 passed`, adjacent `105 passed`, full offline `558 passed, 5 skipped` in 648.85s. No real scenario, power result, decision resolution, API, Docker run, pilot, experiment, freeze, stage, commit, push, or PR. |
| 2026-09-01 | Implemented inactive Paper V2 M7 semantic preregistration admission. `paper-v2-preregistration-v1` fixes the P1 reliability-noninferiority, gated P2 unconditional-FANU superiority, and two-sided S1 checklist hierarchy; all-outcome paired equal-stratum estimands; failure handling; claim limits; and a separate non-pooled D0-D3 nested mechanism ablation. Twelve typed scientific/review decisions remain explicitly unresolved in the result-blind draft, so `freeze_candidate` fails closed. Canonical preregistration/component receipts must match future M6 references. A synchronized Draft 2020-12 schema and adversarial tests reject design drift, pseudo-resolution, visible outcomes, invalid values, duplicate/unknown fields, and unresolved freeze attempts. Verification: focused `20 passed`, adjacent `86 passed`, full offline `540 passed, 5 skipped` in 813.13s; no real values, API, Docker run, pilot, experiment, freeze, stage, commit, push, or PR. |
| 2026-09-01 | Implemented inactive Paper V2 M6 result-blind protocol admission. `paper-v2-protocol-bundle-v1` strictly binds the validated M2/M3/VQ1/M4 policies, M5 terminal/artifact versions and minimum reserve, one hidden gate, Docker/network isolation, exact A/B/C runner mappings, execution commit, and hashes for selection evidence, study documents, inputs, prompts, decoding, matrix, and randomization. Canonicalization removes integer/float and stopping-order hash drift. A synchronized Draft 2020-12 schema and adversarial tests reject post-outcome selection, malformed evidence, unknown/duplicate fields, backend/version/arm drift, and incompatible reserves. It remains disconnected from runners and cannot create a freeze. Verification: focused `21 passed`, adjacent `99 passed`, full offline `520 passed, 5 skipped` in 718.52s; compileall/schema/diff/static non-wiring checks pass. No real values, artifacts, API, Docker run, experiment, stage, commit, push, or PR. |
| 2026-09-01 | Implemented inactive Paper V2 M4 frozen stopping controller. `stopping-policy-v1` has strict external JSON loading, canonical hashing, no threshold defaults, and deterministic first-trigger-wins handling for reserve boundary, exploration exhaustion, eligible no-improvement, incumbent-backed agent finalization, and unrecoverable failure. Gym checks the boundary before another LLM call; repeated-single-shot uses the same incumbent and boundary observers; finalize decisions enter common M5 replay while terminate decisions never open the hidden gate. The policy is manifest-bound and cannot load without an already-injected reserve. Verification: policy/runner `15 passed`, agent+policy `23 passed`, three notebook end-to-end paths passed, repeated observer/M5 regression passed, full offline `499 passed, 5 skipped` in 687.16s; compileall/diff/static activation checks pass. No policy values, API, Docker run, experiment, stage, commit, push, or PR. |
| 2026-08-30 | Implemented inactive Paper V2 M5 symmetric terminal selection. `symmetric-terminal-v1` now owns incumbent selection, immutable bundle verification, exact producing-revision replay, the common isolated preflight, normalized score-drift check, protected artifact reload, and one hidden gate for both `NotebookGymEnv` and the reserve-gated repeated-single-shot adapter. Repeated attempts are persisted as immutable one-cell bundles and cannot fall back to live `best_model`; injected VQ1 removes labels and charges/normalizes validation consistently. Active V2 use requires explicit metric direction/tolerance, while production still constructs neither reserve nor query policy. Verification: common/adapter `8 passed`, adjacent `41 passed`, controller+notebook `57 passed, 1 skipped`, full offline `479 passed, 5 skipped` in 586.27s; compileall/diff/static activation checks pass. No policy values, API, Docker run, experiment, stage, commit, push, or PR. |
| 2026-08-30 | Implemented inactive Paper V2 VQ1 adaptive-validation control as a separate local mechanism. `ValidationQueryPolicy` now provides a pre-query global cap, non-borrowable terminal reserve, source ledger, and frozen numeric feedback precision with no defaults. All current validation-information paths are charged, repeated/cached/revision reuse cannot bypass the cap, active-policy workspaces/kernels receive validation features without labels, selection uses the same normalized score disclosed to the agent, summaries report realized queries and hidden-minus-feedback gap, and M3 retains only aggregate query resources. Local Jupyter is not full OS isolation; frozen symmetric M5 activation and Docker enforcement remain gates. Verification: budget `29 passed`, context `12 passed`, focused loophole/kernel `16 passed`, full offline `469 passed, 3 skipped, 2 deselected` in 545.96s; compileall/diff/static activation checks pass. No policy values, runner/CLI activation, API, Docker run, experiment, stage, commit, push, or PR. |
| 2026-08-30 | Implemented opt-in Paper V2 M3 deterministic context compression on the local M1/M2 stack. A strict versioned public context pack now preserves the task contract, tested public hypotheses, incumbent, active blockers, deterministic remaining resources, allowed actions, notebook state, and finalization contract. Canonical byte-stable JSON, fixed-priority bounded truncation, core-preserving fail-closed behavior, explicit public/private receipts, legacy/default compatibility, and private/hidden-field rejection are tested. No runner/CLI activates the policy and no numeric limits were selected. Verification: focused context suite `11 passed`, dedicated notebook pack/privacy target passed, and full offline `453 passed, 3 skipped, 2 deselected` in 474.03s; compileall and diff checks pass. No API, Docker, experiment, stage, commit, push, or PR. |
| 2026-08-30 | Implemented opt-in Paper V2 M2b protected producing-revision finalization on top of local M1a/M1b/M2a. The host now selects and verifies the incumbent, clean-replays its frozen snapshot under the separate reserve, repeats raw-row/serialization/metric preflight, atomically stores and reloads a hash-bound replay artifact, and then permits one hidden evaluation. Every pre-hidden and hidden failure is terminal with no repair, fallback, replacement candidate, borrowing, or retry. No runner/CLI activates the reserve and no numeric policy was selected. Verification: dedicated artifact store `4 passed, 2 skipped`, all ten M2b controller cases passed, and full offline `441 passed, 3 skipped, 2 deselected`; compileall and diff checks pass. No API, Docker, experiment, stage, commit, push, or PR. |
| 2026-08-30 | Started Paper V2 reliability implementation from verified `origin/main` commit `1d94774` in isolated branch `JD/paper-v2-m1a`. M1a makes validated records immutable, selects incumbents deterministically with the existing `higher`/`lower` convention, preserves historical candidates across notebook mutations, and keeps submission/hidden evaluation restricted to the current clean-run candidate. Added focused registry/lifecycle/boundary tests. Verification: `43 passed, 1 skipped` focused, `95 passed` adjacent, `402 passed, 2 deselected` full offline suite, and `git diff --check` with line-ending notices only. No API, experiment, staging, commit, push, or PR. M1b persistence/resume remains required. |
| 2026-08-16 | Prepared PR 6 from merged PR #73 (`d1ec571`): added a TMLR-style manuscript, checked bibliography, limitations/disclosures, reproducibility guide, and SHA-256 manifest. Review hardening validates all outputs, binds renderer/hash code, clarifies runtimes/budget, and records omitted preregistered resource and descriptive reports. Windows/Linux drift was fixed with LF-normalized hashing, artifact/output assertions, and CRLF regressions; current Linux CI passed `392 passed, 2 deselected`, and focused tests pass `23 passed` before the final assertion/disclosure-only follow-up. No submission, public release, raw-run commit, private registry access, or paid API request. |
| 2026-08-16 | Fast-forward merged deterministic Paper V1 results PR #73 at `d1ec571` after exact-identity commits, local `385 passed, 2 skipped`, successful GitHub tests, no unresolved threads, and a current-head review with no major issue. The package preserves the content-hashed analysis, deterministic CSV/SVG outputs, resource accounting including reasoning tokens, stable hidden-summary columns, provenance, and the documented PR #72 metadata deviation. |
| 2026-08-16 | Merged reference-binding repair PR #72 after local `384 passed`, GitHub tests, and review resolution, then ran the unchanged preregistered analysis once to fresh `analysis-v2/primary-analysis.json`. All 120 outcomes passed completeness. Result hash is `22ff7ac1...c58a`: H1 `B-A=-0.731124`, 95% CI `[-0.848360,-0.608697]`, Holm p=`0.000020`; H2 `C-B=-0.048985`, 95% CI `[-0.211189,0.107149]`, Holm p=`0.570312`; valid rates A/B/C are `75.0%/15.0%/7.5%`. Added deterministic, hash-validating CSV/SVG/provenance generation for PR 5b; focused suite `21 passed`, full suite `385 passed, 2 skipped`, one existing Jupyter warning. GitHub rebase rewrote PR #72 committer display name to `J D` despite correct account/email and exact source-commit identity; owner instructed continuation without a force rewrite, and the deviation is retained in the audit. |
| 2026-08-16 | Completed the immutable freeze-v2 confirmatory series with reconciliation `terminal=120`, `pending=0`, `running=0`, `replacement_required=0`. The first preregistered analysis invocation failed closed before emitting statistics because `analysis.py` required the pre-plan frozen FANU reference file to contain later-created plan identity and recomputed its hash from a payload inconsistent with the freeze generator. Prepared an auditable implementation repair that validates the actual frozen payload and preserves every scientific and statistical invariant; focused tests pass (`18 passed`), the full suite passes (`384 passed`), and no confirmatory result was inspected before the repair. |
| 2026-08-15 | Fast-forward merged fail-closed launcher PR #70 at `e07b76c` with both commits authored/committed by JapanDino; PR and post-merge GitHub tests passed. Its first real invocation stopped in `RunRecorder.create` before manifest creation, LLM client construction, provider request, hidden evaluation, or outcome because the plan hashed wall-clock `1800` as an integer while frozen argparse materialized `1800.0`. Prepared result-blind amendment v2: effective limits and all scientific factors remain unchanged; budget hash becomes `ae9a65...45f2`, regenerated plan `plan_793d85d310427d848a09302c` contains 120 conditions with hash `793d85...310f`, and the original tag/logs remain immutable. Review hardening binds a fresh v2 series root and rejects v1-root reuse. Focused suites: `47 passed` and `17 passed`; final full suite: `384 passed`. |
| 2026-08-15 | Verified the merged PR #69 freeze at `1d3e0bf` and annotated tag `paper-v1-experiment-freeze`. Added a fail-closed confirmatory launcher that validates the complete tagged contract, detached execution commit, clean worktree, dataset hashes, exact internal model settings, Docker image ID, and append-only `started`/`finished` lifecycle provenance before scheduling the next frozen condition. Built and no-network-smoked image `sha256:b8c3...7466d`; real dry-run selected sequence 0 without API access or output creation. Focused suite: `42 passed`; full suite: `383 passed`. No confirmatory outcome or external publication was created. |
| 2026-08-15 | Closed the three actionable PR #69 review gaps: Arm A now explicitly couples Docker execution to Docker candidate prediction; the evaluator returns a strict typed JSON scalar vector instead of host-unpickled result objects; and the exact result-blind confirmatory plan now exists before the freeze tag. Also bound API temperature 0.4 to actual requests and decoding hashes, normalized prompt-template hashing across datasets, added explicit A/B/C manifest identity, and reduced the default provider retry limit to the preregistered three. Execution commit `b4e3da2` passed `372 passed`; plan `plan_1c2ad2a922165e0d27471ae7` contains 120 pending conditions with hash `1c2ad2...b593`. No confirmatory outcome or external publication was created. |
| 2026-08-15 | Prepared the Paper V1 protocol freeze after nine completed result-blind pilot episodes. Froze authenticated internal `deepseek-v4-flash`/`gemma-4-26b`, 256k tokens with 12 calls and 0-ruble/no-paid-fallback policy, four leakage-reviewed dataset snapshots, FANU dummy/reference values, storage choice, annotator nominees, and TMLR-style manuscript format. Added dataset cards, endpoint/pilot snapshots, immutable manifest/reference hashes, and a no-overwrite freeze generator. Pilot-discovered native candidate prediction crashes are now contained; confirmatory prediction runs in an ephemeral network-none Docker evaluator with no host secrets. Focused suite `48 passed`; full suite `361 passed`; Docker prediction smoke passed. No confirmatory plan/outcome or external publication was created. |
| 2026-08-15 | Fast-forward merged Paper V1 PR 5a (#67) at `b1cd5f5` after final GitHub `Python tests` success and resolution of all actionable review threads. Preserved four commits with author/committer `JapanDino`. The frozen analysis now fails closed on incomplete/dirty/drifted inputs, binds protocol/reference hashes, reconciles disk ledgers, and preregisters FANU H1/H2, Wilson valid-rate intervals, McNemar/Holm, and all-outcome/successful-only summaries. No API, pilot, real manifest, confirmatory outcome, or scientific claim was created or inspected. |
| 2026-08-15 | Fast-forward merged Paper V1 PR 4 (#66) at `1b12017` after all review threads and GitHub checks passed, preserving author/committer `JapanDino`. Published PR 5a as #67: fail-closed completeness, frozen FANU references, paired H1/H2, equal-stratum bootstrap/permutation, exact McNemar, 95% Wilson rate intervals, all-outcome/successful-only score summaries, ledger reconciliation, Holm, raw/stratum outputs, schemas, and immutable content-hashed publication. Review fixes cryptographically bind protocol and reference payloads to the plan. Synthetic-only focused suite after fixes: `66 passed`; full suite: `351 passed, 2 skipped`. No API/pilot/confirmatory outcome was created or inspected. |
| 2026-08-15 | Fast-forward merged Paper V1 PR 3 (#65) at `385a09f`, preserving author/committer `JapanDino`; all final GitHub checks passed. Published PR 4 as #66: deterministic immutable condition planning, 120-condition acceptance fixture, SHA-256 blocked A/B/C ordering, concurrent-safe no-overwrite plan creation, exact frozen-budget/category gates, manifest resume/duplicate checks, and infrastructure-only replacement queue. Review-fix focused suite: `66 passed`; full suite: `338 passed, 2 skipped`. No plan with real identities and no API/pilot/confirmatory run was created. |
| 2026-08-14 | Merged Paper V1 PR 1 (`6bb75d1`) and PR 2 (`67069a3`) in order after required checks. Published PR 3 as #65: pre-call global episode budgets and audit events, common no-autofit submission validation, one-shot hidden evaluation without repair feedback, reasoning/cached-token observability, and research-only suppression of post-outcome LLM summaries. Review follow-up removed reasoning-token double counting and ensured budget exhaustion still finalizes the single-shot manifest. Full offline suite: `307 passed, 2 skipped`; required GitHub `Python tests` passed; the separate Docker integration job remains flaky at kernel readiness. No API/pilot/confirmatory run occurred. |
| 2026-08-12 | Codified mandatory authorship and publication rules: every commit must use `JapanDino <klim.i.rumyantsev@gmail.com>`, every PR must be authored by GitHub login `JapanDino`, and stage/commit/push/PR/merge require separate explicit authorization. Added `docs/RESEARCH_PR_POLICY.md` with Draft/Ready/Merge timing, PR 1–6 sequencing, preregistered PR 5a before freeze, pilot/confirmatory gates, PR-body evidence requirements, and post-merge provenance rules. |
| 2026-08-12 | Prepared the unfrozen Paper V1 research package from freshly fetched `origin/main` commit `1504cc0`: Phase 0 evidence-backed code audit, canonical machine-readable protocol, hypotheses, analysis plan, failure/retry policy, dataset/model selection gates, offline validator, and focused tests. Confirmed current per-call token semantics, host autofit in single/repeated controls, separate step/tool accounting, private hidden score, three-attempt hidden failure loop, and 8-vs-12 checklist documentation drift. No runner behavior, API experiment, commit, push, or PR was performed. |
| 2026-06-08 | Gym now beats single-shot (experiment-validated, gemma-4-26b): (1) validation-improvement nudge — after `validate`, NotebookGymEnv reports best-so-far + remaining budget and pushes the agent to beat its own baseline (honest, val-split only, feedback modes only); flips student_dropout from −0.004 (gym lost) to +0.013 over single-shot. (1b) unknown-cell-id guard — targeting a non-existent cell is now a recoverable blocker instead of a KeyError that crashed the whole episode, restoring valid-submit rate to 1.0. Measured-but-rejected: per-class-recall diagnostic and candidate-list nudge both hurt (extra feedback verbosity inflates the trajectory → more clean-run failures) |
| 2026-06-06 | Tightened run-summary UX: the prompt remains English-only, normalization no longer chops already-short section bodies mid-sentence, and the Run Detail «Саммари решения» card now renders parsed summary sections (plus inline code) as a structured two-column report instead of raw markdown paragraphs |
| 2026-06-06 | Post-run self-summary: once the model solved the task (reached a final submit for gym/fixed — even if the hidden test rejected it — or produced a usable candidate for single/repeated) one extra best-effort LLM call asks the model to summarize its own solution; saved as `run_summary.json` (`gym/run_summary.py`), served via `GET /runs/{id}/summary` + `hasSummary` on `get_run`, and rendered as a standalone report card (accent side-stripe, neutral surface — deliberately not styled like a step thought) above a «Ход рассуждений по шагам» section on the «Мысли» tab, which now shows for any run with a summary even when thoughts mode is off (old/unsolved runs without either stay hidden). Privacy preserved: the summarizer only sees the conversation it already had, never the hidden test score |
| 2026-06-06 | Dashboard launcher now passes explicit `LLM_PROVIDER` from the selected model provider, preventing OpenAI-compatible models from inheriting a Gemini `.env` default |
| 2026-06-06 | LLM model configuration moved out of `.env` into the shared model registry with CLI management, required `--model` registry resolution, provider-specific Gemini/LiteLLM runtime env, and no LLM keys in `.env` |
| 2026-06-06 | Hardened post-submit run-summary generation: stricter four-section retrospective prompt in English, lower token cap, and normalization that strips reasoning/draft/checklist scaffolding both when generating new summaries and when reading already-saved `run_summary.json` artifacts |
| 2026-06-06 | Post-submit self-summary is now explicitly retrospective and grounded in the actual submitted solution code (`final_notebook.py` / generated baseline code), while the dashboard thoughts toggle/launcher scope is limited to Iterative and Gym rather than Fixed |
| 2026-06-06 | Deterministic action observability: agent JSON actions now use canonical `type`, required ordered `stage`, canonical `thoughts` in thoughts mode, and non-mutating `think`; NotebookGymEnv/GymEnv enforce the contract, artifacts/API/dashboard expose current stage and thoughts, docs/tests updated |
| 2026-06-06 | Added LLM-agent ML toolbox to requirements.txt + pyproject.toml: catboost, seaborn, plotly, optuna, shap, imbalanced-learn, category_encoders, statsmodels, tabulate — all missing from the venv, all commonly reached for by agents; seaborn was already listed in requirements.txt but never installed (PR #53) |
| 2026-06-06 | Dashboard runs table: added `.truncate` CSS (max-width 220px, ellipsis) on model and dataset columns to eliminate horizontal micro-scroll caused by long model names after chevron removal (PR #50) |
| 2026-06-06 | Dashboard runs table: removed the chevron `›` column at row end — rows are visibly clickable without it (PR #49) |
| 2026-06-05 | Found "sweet-spot" weak model for gym-vs-single-shot gap experiments: `llama-3.1-8b-instant` on Groq (val≈0.661 single-shot); added to models.json (gitignored). OpenRouter free limit (50 req/day) inadequate for gym runs. Groq TPM cap (~6000) handled by PR #47 max-tokens clamp |
| 2026-06-05 | Failed runs now expose all available info across every mode: repeated single-shot writes a full multi-attempt episode (every attempt's code + error visible in Notebook/Trajectory/Errors/Logs, not just the best), the run record carries `failReason`/`finalStatus` from the runner's `null_reason`/`final_status`, and the detail page always shows the fail banner with the status label |
| 2026-06-05 | Dashboard launcher clamps `--max-tokens` to the selected model's `maxTokens` cap, so providers with tight per-minute token limits (e.g. Groq free ~6000 TPM) don't 413 when the New Run form leaves the default high |
| 2026-06-05 | Dashboard tabs visual fix: removed the 1px vertical overflow in tab menus by moving the divider into the tab strip and dropping the negative tab margin |
| 2026-06-04 | Dashboard logo refresh: replaced the H-like yellow-square mark with a diagonal dumbbell AutoVibe mark |
| 2026-06-04 | Dashboard sidebar cleanup: removed the bottom `Команда / локальный режим` block and its unused styles |
| 2026-06-04 | Normalized product-mode display labels: dashboard short labels now show `Flexible gym`, and fixed transitions display as `Fixed gym` in dashboard and shared matrix metadata |
| 2026-06-02 | Dashboard trajectory visual fix: aligned the live "агент выполняет шаг…" spinner with the timeline marker column and adjusted the connector so the grey line meets the spinner's top center |
| 2026-06-02 | Agent thoughts/scratchpad: initial visible-thoughts support for gym/iterative runs with `--enable-thoughts`, persistent `scratchpad.json`, context reinjection, dashboard New Run toggle, and a «Мысли» tab. |
| 2026-06-03 | Propagated child runner failures from `experiments.run`: batch summaries still print, but the wrapper exits non-zero when any selected product mode fails |
| 2026-06-03 | Cleaned up New Run budget controls: removed the budget-preset subhint and added Problems-style tooltip hints to budget parameter fields |
| 2026-06-03 | Added two-state environment badges to New Run mode cards: `Среда` for the three environment-backed modes and `Без среды` for the two non-environment modes |
| 2026-06-03 | Restored Iterative as a selectable product run type, raised dashboard/common-run batch selection to 5 modes, renamed Gym to Flexible gym and Fixed transitions to Fixed transitions gym, and removed the New Run recommendation badge |
| 2026-06-03 | Hardened dashboard responsive layout across core routes: mobile hides the desktop sidebar toggle, New Run stacks earlier, grids/cards/settings/filters/dataset actions can shrink or wrap safely, compare labels and preview values wrap, and chart bars no longer force overflow |
| 2026-06-03 | Replaced the dashboard `All modes` launch card with multi-select run types; multi-run launches use `batch` + `--modes ...` while preserving the CLI `--mode all` path |
| 2026-06-03 | Added first-class `all` run orchestration: shared product-mode metadata, `experiments.run --mode all`, matrix batch metadata, compare columns/sort for `requested_mode`/`batch_id`/`mode_label`, dashboard `All modes` and `Fixed transitions` launch options, and responsive New Run grid fix |
| 2026-06-03 | PR #34 Dataset Center hardening: fixed CI dataset preparation, restored finite upload limits, blocked localhost/private URL downloads and unsafe redirects, enforced gzip decompressed-size limits, made dataset creation atomic, preserved legacy root `meta.json` edits, added JSONL/SSRF/cleanup regressions, and kept `/datasets` route compatibility after the `/problems` UI rename |
| 2026-06-02 | Dataset Center polish: one-column dataset cards, Russian UI with common ML terms preserved, dataset suite/group metadata removed from project flows, example configs now include repository-created timestamp and UCI sources, and empty sources display `-` |
| 2026-06-02 | Dataset Center full workflow: backend staged uploads/URL downloads/safe archive extraction/table preview/create-from-config/config editing, React Dataset Center search/filter/sort, full creation wizard, seven-tab detail page, docs and backend tests |
| 2026-06-02 | Single-shot/repeated now show code + checklist coverage in the dashboard: the legacy runners emit a synthesized episode (solution.ipynb, notebook_events, feedback_trace, summary) into `--workspace-dir` and log `checklist_coverage` measured from the generated code, so Notebook/Trajectory/Checklist tabs populate for these modes too |
| 2026-06-02 | Single-shot/repeated now produce a score locally: raised legacy executor timeout 60→300s, prompts require a fitted predict-ready model, and the runner auto-fits an unfitted submitted model before scoring (gpt-oss-120b left it unfitted). Verified single_shot=0.931 f1 on example_dry_bean |
| 2026-06-02 | Dashboard local-execution fix: single-shot/repeated use the legacy executor which defaulted to `docker` from .env → "no candidate" on a Mac without Docker. Local launches now force the in-process `subprocess` executor + local kernel (env `AUTOVIBE_DASHBOARD_EXECUTOR`) |
| 2026-06-02 | Dashboard visual polish: fixed sidebar (position:fixed), dumbbell logo replacing the «A» mark, cleaner gear/trash icons, and a rebuilt trajectory timeline — per-step icon badges by step type (add/edit/delete/restart/run/validate/submit) with opaque fills and a clean connector line |
| 2026-06-02 | Dashboard checklist consistency: tab count uses the recorded `checklist_coverage` (single source of truth, matches the run banner) and exactly that many items render green; aligned the live-banner count to the same formula |
| 2026-06-02 | Dashboard execution modes: per-run selector «на сервере (SSH) / на компьютере» on New Run (overrides the global default); local mode runs gym on the machine and calls the remote LLM (works off-VPN) |
| 2026-06-02 | Dashboard remote execution: run the gym on the GPU server over SSH while the site stays local (`services/remote_exec.py`: ssh/rsync launch + artifact sync + run-summary parse; key auth or optional expect password); configurable in Settings with a connectivity probe |
| 2026-06-02 | Dashboard single-app server mode: FastAPI serves the built SPA (one process) so the whole dashboard can run on the server; added `serve.sh` and deploy docs |
| 2026-06-02 | Dashboard live updates: launches write to a known workspace dir and the backend reads in-flight artifacts, so step/checklist/notebook/trajectory/logs advance during a run (2.5s polling); models registry seeded with team gemma/deepseek; header pill switched to LLM "Сервер онлайн/офлайн" |
| 2026-06-02 | Dashboard polish: fixed Windows Python discovery for run launches, stabilized single-shot/repeated launch progress, reconciled checklist list/detail coverage for legacy MLflow runs, hid placeholder failed-submit scores, pinned `scikit-learn==1.7.2` in `pyproject.toml`, added dashboard regression tests, and fixed mobile layout overflow |
| 2026-06-01 | Dashboard run fixes: dedup the MLflow twin of a live launch by run-name; reconcile orphaned 'running' metas (server reload mid-run) against MLflow; live per-second duration on client; cap launch threads (OMP/BLAS/MKL + AUTOVIBE_SANDBOX_THREADS, sequential joblib) to stop CPU/fan spikes; `run.sh` reload off by default (gym `.py` artifact writes were restarting uvicorn mid-run) |
| 2026-06-01 | Added local web dashboard (`dashboard/`): FastAPI backend (MLflow runs, episode-artifact parsing, datasets/models CRUD, subprocess run launcher) + Vite/React/TS frontend with all 8 screens on the T-Bank design system |
| 2026-06-01 | Fixed single-shot/repeated multishot H200 failure modes: raw-input pipeline prompts, shared label-coercion scoring, sequential joblib in the subprocess/Docker executor |
| 2026-06-01 | Fixed gym `test_metric=null`: robust action parsing (tool-call tokens), host-side `finalize()` live-kernel fallback, label-encoding coercion in validate/submit, viz libs + per-cell timeout + prompt steering; verified 0.7247 on student_dropout |
| 2026-06-01 | H200 recon: capped BLAS/OMP threads in sandbox+kernel, added `run_fixed --max-steps`, pinned scikit-learn==1.7.2; documented open gym-submit issue |
| 2026-05-29 | Hardened notebook privacy artifacts, Docker kernel path/port handling, step-budget blocking, and deterministic CI sandbox image build |
| 2026-05-29 | Implemented ContainerJupyterKernelBackend: Docker sandbox with internal network, read-only rootfs, and dropped capabilities |
| 2026-05-29 | Rebasing Jupyter branch on updated `origin/main` and preserving LiteLLM/Groq provider support |
| 2026-05-29 | Added real Jupyter `.ipynb` + persistent kernel backend with notebook actions |
| 2026-05-29 | Added clean `restart_and_run_all`, host-controlled `validate`, candidate registry, and submit gate |
| 2026-05-29 | Split feedback channels and replaced dataset-specific checklist hints with generic selective hints |
| 2026-05-29 | Hid hidden test metric from agent-facing messages, feedback traces, and notebook outputs |
| 2026-05-29 | Added `iterative_no_checklist` as fair Jupyter control and renamed multishot logging to `repeated_single_shot` |
| 2026-05-29 | Added Jupyter dependencies and tests for kernel, notebook edits, clean replay, validation, privacy, and fairness |
