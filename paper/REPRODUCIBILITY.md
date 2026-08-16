# Paper V1 reproducibility guide

## Evidence levels

The repository intentionally separates three evidence layers:

1. `research/protocols/` contains the preregistered protocol, frozen inputs,
   failure policy, analysis plan, and freeze record.
2. Private raw runs remain outside Git. The completed v2 series root is
   `C:\Users\klimi\Documents\Codex\2026-08-12\new-chat\outputs\paper-v1-confirmatory-20260815-v2`
   on the collection machine. Public release requires a separate privacy,
   secret, size, and dataset-license review.
3. `research/publication/paper_v1/` contains the content-hashed primary analysis
   and deterministic, non-secret tables and figures used by the manuscript.

The manuscript is a working research artifact. It is not a journal submission,
arXiv posting, GitHub Release, or Zenodo deposit.

## Exact bindings

- Freeze tag: `paper-v1-experiment-freeze-v2`
- Execution commit: `b4e3da29e1c6b7db42848f7eee82c6a8ef3e9dae`
- Plan ID: `plan_793d85d310427d848a09302c`
- Plan hash: `793d85d310427d848a09302c3c4583870fd3fafb6a8a0590d95e61384edf310f`
- Protocol hash: `de690a4e975781c516b25d2016416bf44e5d416ee61a00d305eae39331ce49c5`
- Reconciliation hash: `805c6dfd3da10a9ae33ea80276d3643df2b2ec17fa64ca83cb781ddbee844e16`
- Analysis result hash: `22ff7ac1ffeb7ce116c384ea32ab23f82984c01aa7b16f19993b8f8f1db3c58a`
- Committed analysis-file SHA-256: `6cbe78eea2de19e85a350cb05521057214b47e44f9282a87cb6658a50b5c2948`

`paper/artifact-manifest.json` binds every tracked input and the generated
manuscript by SHA-256 after normalizing UTF-8 text newlines to LF. This explicit
policy makes the manifest identical on Windows and Linux checkouts.

## Offline verification

From the repository root with the project environment active:

```powershell
python -m research.build_manuscript --check
python -m pytest tests/test_research_build_manuscript.py tests/test_research_report_results.py tests/test_research_analysis.py
python -m pytest
git diff --check
```

To rebuild the generated manuscript after an intentional template change:

```powershell
python -m research.build_manuscript
python -m research.build_manuscript --check
```

The builder validates the canonical analysis hash before rendering and derives
all numeric result prose and tables from `primary-analysis.json`. It fails if the
analysis is incomplete or tampered, a required package input is missing, or the
committed manuscript/manifest is stale.

## What is and is not reproducible

The statistical result and manuscript package are deterministically
reproducible from the committed analysis artifact. Re-running inference exactly
is not independently guaranteed because the two internal model endpoints do not
expose immutable weight revisions. Re-running the full study also requires the
licensed dataset snapshots, private model registry, internal gateway access,
and the frozen sandbox image. No secret or private endpoint credential is
included in the package.

Checklist detector validation is not reproducible yet because human annotation
has not occurred. It is deliberately excluded from manuscript claims.

The committed primary-analysis schema also omits preregistered cached-token,
code/tool-execution, wall-clock, CPU-time, provider-retry, monetary-cost, and
cost-quality-frontier fields. The manuscript reports the available token and
logical-call aggregates and records the rest as a reporting deviation. A future
secondary resource reconstruction must be separately versioned and labeled
post-outcome; it must not be presented as the frozen primary analysis.

## Submission-only gates

Before any external submission or public artifact release, a human must:

- finalize the author name, order, affiliation, acknowledgments, and conflicts;
- verify every citation against the target venue's bibliography style;
- complete or formally defer the two-annotator checklist validation;
- perform privacy, secret, dataset-license, and provider-terms review;
- choose a venue/template and render a PDF under that venue's rules;
- decide whether and how to publish raw trajectories and dataset snapshots;
- approve the AI-assistance disclosure and the final scientific claims.
