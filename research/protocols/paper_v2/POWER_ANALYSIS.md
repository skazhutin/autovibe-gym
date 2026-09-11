# Paper V2 planning power analysis

`research.power_v2` is a result-blind planning tool for the P1, gated P2, and
S1 hierarchy defined by `paper-v2-preregistration-v1`. It does not load run
manifests, historical result tables, MLflow, hidden scores, or experiment
outputs. A scenario is rejected unless its hypothesis, analysis-plan, and
failure-policy hashes match the preregistration draft.

## What a reviewed scenario must declare

Every scenario explicitly supplies:

- family-wise alpha and its allocation between the P1→P2 sequence and S1;
- the P1 non-inferiority margin and target power;
- simulation draws, deterministic seed, and RNG contract;
- the number of replicates in every dataset-model stratum;
- paired validity assumptions through `A`, `B | A`, and `C | B` probabilities;
- successful-outcome FANU means, standard deviations, and shared correlation
  for A/B/C.

There are no code defaults for these scientific values. A real scenario file
must not be created until the assumptions have a documented source and review.
Synthetic values exist only in the test suite.

## Decision approximation

For each simulated experiment, the tool computes an equal-weight average of
paired stratum effects and its design-based standard error:

- P1 passes when the one-sided lower normal bound for the B-minus-A validity
  difference is above the negative non-inferiority margin;
- P2 passes only when P1 passes and the one-sided lower normal bound for
  B-minus-A unconditional FANU is above zero;
- S1 passes when the two-sided normal interval for C-minus-B unconditional FANU
  excludes zero.

The P1/P2 sequence shares one alpha allocation; S1 receives a separate alpha,
and their sum may not exceed the family-wise alpha. Reported power includes P1,
gated P2, S1, and the probability that gated P2 and S1 both pass.

## Claim boundary

This normal-bound simulation is a sample-size and sensitivity approximation.
It is not the final paired resampling/randomization analysis, and its output is
not evidence that an arm works. Power is conditional on unverified assumptions;
therefore the final review must inspect sensitivity across a defensible scenario
grid rather than publish the most favorable row.

Historical Paper V1 observations may later be used only as explicitly labelled
external planning evidence for nuisance quantities such as variability and
pairing. They must not determine the non-inferiority margin, be pooled with
Paper V2 outcomes, or be presented as evidence for the corrected architecture.

The CLI requires all paths explicitly:

```powershell
python -m research.power_v2 `
  --scenario <reviewed-scenario.json> `
  --preregistration research/protocols/paper_v2/preregistration.draft.json `
  --output <new-no-overwrite-result.json>
```

No reviewed real scenario or result artifact exists yet. Running synthetic
tests does not resolve the M7 power, sample-size, alpha, margin, or review gates.
