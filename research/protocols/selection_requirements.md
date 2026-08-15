# Dataset and model selection requirements

Selections are frozen only after result-blind eligibility and availability
checks. Pilot scores may calibrate feasibility and budget but may not be used to
pick whichever dataset/model combination makes an arm look best.

## Dataset eligibility

Each confirmatory dataset must have:

- a legally redistributable source or reproducible download procedure;
- a versioned dataset card with source URL/citation, license, retrieval date,
  row/feature counts, target definition, task type, and known risks;
- immutable raw/prepared hashes and a documented preparation script;
- a fixed train/validation/hidden-test split ID and seed;
- no hidden rows, labels, score, or source path visible to agent code;
- a metric direction and frozen dummy/reference pipelines with scores;
- enough difficulty to avoid saturation while remaining solvable within the
  shared budget;
- no target leakage, post-outcome fields, duplicate split contamination, or
  identifier proxy left unexplained;
- a contamination note describing how likely public training data contains the
  dataset or common solution notebooks.

Current provisional four-task matrix:

| Slot | Task | Status before freeze |
|---|---|---|
| 1 | `air_quality` regression | source/card/hash/split/reference values required |
| 2 | `diabetes` binary classification | exact dataset identity and all frozen metadata required |
| 3 | `student_dropout` multiclass | source/card/hash/split/reference values required |
| 4 | `bank_marketing` or `phishing` | owner decision D02 required |

`digits` is infrastructure smoke only. It cannot support the main causal claim.

The repository currently contains example task configurations, not a frozen
confirmatory dataset registry. Matching an example name to a protocol slot must
be explicit; similar names are not assumed to identify the same source/version.

## Model/provider eligibility

Each of the two model slots must record:

- provider and endpoint class;
- exact provider model ID and version/revision when available;
- context window and maximum output-token limit;
- decoding parameters, system/template behavior, and tool/JSON mode;
- usage metadata support for raw input, cached input, output, and reasoning tokens,
  with documented zero/unknown semantics;
- stable authentication/configuration without secrets in manifests;
- request ID, finish reason, retryable error, and latency observability where the
  provider exposes them;
- availability under a short predeclared pilot without systematic 413/429/capacity
  failure;
- compatibility with the same task prompt and arm protocol;
- price snapshot and timestamp for cost reporting.

The final pair should include one stronger and one compact/mid-capability model.
The choice is based on availability and predeclared capability tier, not on which
model yields a preferred arm effect in pilot results.

## Availability pilot

Before freeze, run only 6–12 explicitly labeled pilot episodes sufficient to:

- verify endpoint/model IDs and usage fields;
- estimate context/output and monetary budget feasibility;
- exercise every A/B/C path on a small non-confirmatory task;
- validate retry/error classification;
- confirm Docker isolation and artifact completeness.

Pilot outcomes are stored under a separate experiment ID and never pooled with
confirmatory results. Budget changes after the pilot are allowed only before
confirmatory outcomes are inspected and must update protocol/config hashes.

## Freeze checklist

- [x] D01: two exact models/providers/endpoints/decoding configs approved.
- [x] D02: fourth dataset approved.
- [x] D03: global budget approved from result-blind 64k/128k/256k pilots.
- [x] D05: immutable large-artifact storage approved.
- [x] D08: zero-ruble client hard stop and no paid fallback approved.
- [x] D09: dataset cards, hashes, splits, metric directions, dummy/reference
      pipelines, and scores approved.
- [x] Every selected endpoint returns usable input/output usage metadata; cached
      and reasoning zeros are conservatively interpreted as none-or-not-exposed.
- [x] Every dataset has a leakage/license/privacy card. Air Quality remains
      research-only under the conservative reading, and no raw data is committed.
- [ ] Confirmatory matrix is generated and hashed before outcomes are inspected.
