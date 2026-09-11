# Paper V2 preregistration draft

This directory contains a result-blind, non-frozen study-design draft for the
next AutoVibe experiment. The machine-readable source of truth is
`preregistration.draft.json`; `research.preregistration_v2` validates its
semantics and produces hashes that a future M6 protocol bundle must bind.

## Confirmatory hierarchy

1. **P1 — reliability non-inferiority.** Compare B minus A on the paired valid
   terminal-outcome risk difference. B is admitted as non-inferior only when
   the lower confidence bound is above the negative, independently justified
   margin.
2. **P2 — gated utility superiority.** Only after P1 passes, compare B minus A
   on unconditional failure-adjusted normalized utility (FANU). Agent-caused
   invalid outcomes remain in the population at dummy utility.
3. **S1 — checklist effect.** Compare C minus B on FANU with a two-sided test.
   Its multiplicity treatment is freeze-blocking and remains unresolved.

The analysis unit is a paired `dataset × model × replicate` block. Dataset-model
strata receive equal aggregate weight. Successful-only hidden score is
descriptive because conditioning on success can create selection bias.

## Separate mechanism study

The A/B/C comparison can establish system-level differences but cannot identify
which reliability mechanism caused them. A separate excluded-development study
therefore uses a nested sequence:

- D0: legacy iterative reference;
- D1: add immutable incumbent preservation;
- D2: add protected finalization reserve;
- D3: add deterministic context compression.

Each contrast supports only an incremental effect in that fixed order. It does
not estimate order-independent component main effects, and its observations
must never be pooled with the confirmatory A/B/C series.

## Freeze blockers

No numerical value is selected in this draft. Freeze remains blocked until all
of the following are resolved from excluded-development evidence and documented
human review:

- non-inferiority margin;
- family-wise alpha and secondary multiplicity strategy;
- target power and replicates per stratum;
- resampling/randomization counts and analysis seed;
- reviewed dataset/model scope and FANU references;
- immutable statistical power analysis;
- independent statistical review.

Changing `status` to `freeze_candidate` while any item is unresolved is rejected.
Loading or validating this draft does not activate runners, run Docker, inspect
outcomes, create a freeze, or authorize publication.
