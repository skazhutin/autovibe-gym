# Hypotheses and claim rules

## Research questions

Primary: under a matched global token and execution budget, does stateful
execution feedback improve failure-adjusted tabular-ML performance relative to
best-of-N independent single-shot attempts?

Secondary: does checklist feedback add value beyond runtime, contract, and
validation feedback in the same iterative environment?

## H1 — feedback reliability (confirmatory)

- Comparison: arm B (`iterative_no_checklist`) minus arm A
  (`repeated_single_shot`).
- Primary estimand: mean paired difference in FANU across precomputed
  `dataset × model × replicate` blocks, with equal weight per dataset-model
  stratum.
- Null: the paired FANU difference is zero.
- Alternative: the paired FANU difference is non-zero. Direction and magnitude
  are reported; a positive result is not assumed.
- Supporting outcomes: paired valid-submission status, resource use, error
  recovery, and validation-to-test gap.

An agent-caused invalid outcome remains in the primary analysis with
`S_agent = S_dummy`. Infrastructure-censored runs are replaced under the failure
policy and are not silently scored as agent failures.

## H2 — checklist effect (confirmatory key secondary)

- Comparison: arm C (`gym_with_checklist`) minus arm B
  (`iterative_no_checklist`).
- FANU estimand and blocking are identical to H1.
- Process outcomes: valid submission, clean replay, error recovery, and the
  validated 12-item checklist detector.
- The two FANU hypothesis tests (H1 and H2) use Holm correction at family-wise
  alpha 0.05.

Checklist coverage is a process proxy unless human validation establishes
acceptable per-item detector performance. Higher coverage alone is not evidence
of better hidden-test quality.

## H3 — task dependence (confirmatory heterogeneity reporting)

Aggregate estimates must be accompanied by results for every dataset and model.
The expected pattern is not treated as a required result: simple or saturated
tasks may favor single-shot on the cost-quality frontier, while mixed types,
missing values, imbalance, unusual metrics, and pipeline failures may make
iteration more useful.

The study is not powered to confirm arbitrary interaction effects. Dataset/model
interactions and mixed-effects models are exploratory unless a later amendment is
frozen before confirmatory outcomes are inspected.

## H4 — fixed workflow (exploratory)

Arm D (`fixed_transitions`) tests whether fixed transitions help weaker models or constrain
stronger ones. It is outside the confirmatory A/B/C family and cannot be used to
rewrite H1/H2 after results are known.

## Claim rules

The paper may say an arm improved an outcome only when it reports:

- the matched estimand and arm definitions;
- the number of planned, completed, agent-failed, and infrastructure-censored
  conditions;
- effect size and 95% confidence interval;
- the predeclared paired test where applicable;
- total tokens, calls, executions, tools, time, and cost;
- dataset-level results and material failure modes.

Existing presentation/JSON results are pilot evidence only. They cannot be pooled
with confirmatory runs or cited as proof that Gym is better.
