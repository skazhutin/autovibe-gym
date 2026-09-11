# Paper V2 excluded-development and assumption provenance

`research.evidence_v2` is the M9 metadata gate between result-blind study
design and any future excluded-development pilot or planning power grid. It
does not select datasets, inspect outcomes, choose numerical assumptions, run
power simulations, freeze the preregistration, or activate an experiment.

No real excluded-development manifest, confirmatory dataset scope, assumption
package, scenario grid, or M9 receipt exists in this repository. Test fixtures
are synthetic and have no policy status.

## Dataset separation

The excluded-development manifest records, for each task, immutable hashes of
the dataset identity, exact content, source lineage, split, target, metric, and
dataset card. The independently reviewed confirmatory scope records identity,
content, lineage, and dataset-card hashes before confirmatory outcomes are
visible.

Admission rejects any overlap in dataset identity, exact dataset content, or
source lineage. Renaming a dataset, changing only a split, or constructing a
new task from the same source lineage therefore does not make it independent.
Model families may be shared when the later model-scope review permits it, but
development observations may not enter the confirmatory series.

## Provenance unit

Every M8 assumption is registered separately for every required scenario. A
record binds:

- scenario ID and exact scenario hash;
- canonical assumption pointer and exact value hash;
- declared scientific use class;
- admissible source class and immutable source-artifact hash;
- the excluded-development manifest hash when, and only when, the source is an
  excluded-development pilot.

This prevents one citation from silently covering different values in a
sensitivity grid. Admission requires exact coverage: a missing, additional,
duplicated, moved, or value-drifted assumption fails closed.

## Source restrictions

| Use | Admissible source |
|---|---|
| Non-inferiority acceptability threshold | Independent methodological justification or external primary source |
| Alpha, multiplicity, target power | Independent methodological justification or external primary source |
| Simulation draws, seed, RNG contract | Deterministic design choice or independent methodological justification |
| Candidate replicate count | Predeclared power-grid design |
| Validity/FANU nuisance parameters | Excluded-development pilot, historical Paper V1, or external primary source |

Paper V1 and excluded-development outcomes must not set the non-inferiority
margin or serve as evidence that the corrected Paper V2 architecture works.
They may inform only explicitly labelled nuisance assumptions such as rates,
variability, and pairing. A pilot-sourced record must bind the exact excluded
manifest; all other source classes must leave that field null.

## Scenario-grid decision rule

The grid plan and its complete-grid review are hash-bound before results are
used. The only admitted sample-size rule is the minimum candidate replicate
count that meets the target in **all** required scenarios. Selecting a
favorable row is explicitly forbidden. The required scenario ID/hash set in
the evidence package must equal the supplied verified scenario set exactly.

M9 is necessary metadata integrity, not sufficient scientific review. A future
freeze still requires resolving every M7 blocker, reviewing the actual
dataset/model scope, checking the substantive sources, executing the complete
power grid, and obtaining independent statistical sign-off.
