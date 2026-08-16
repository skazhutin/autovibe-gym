---
title: "When Does Feedback Pay Off? A Budget-Matched, Failure-Aware Evaluation of LLM Agents for Tabular Machine Learning"
author:
  - "JapanDino (author name and affiliation to be finalized before submission)"
date: "2026-08-16"
bibliography: references.bib
link-citations: true
---

# Abstract

Language-model agents can execute code, observe failures, and revise machine-learning solutions, but feedback loops also consume context and may fail before producing a valid submission. We present a preregistered, budget-matched evaluation of three policies for tabular machine learning: best-of-N independent single-shot attempts (A), stateful iterative execution feedback (B), and the same iterative environment augmented with implicit data-science checklist feedback (C). The frozen design crossed four datasets, two internally served model endpoints, five replicates, and three arms for 120 conditions. All arms shared a global limit of 256,000 accounted tokens, 12 model calls, 20 code executions, 20 tool calls, and 1,800 seconds. The primary endpoint, failure-adjusted normalized utility (FANU), retains agent-caused failures at dummy-model performance and normalizes successful scores against a fixed reference pipeline. The complete preregistered analysis found that iteration underperformed best-of-N by a large margin; the numeric effect, confidence interval, and adjusted test are generated directly from the content-hashed analysis below. The checklist increment was not distinguishable from zero. The iterative arms consumed substantially more tokens and most often terminated through budget exhaustion. These results do not show that feedback is generally harmful. They show that statefulness alone is not a sufficient intervention: under a fixed global budget, an iterative scaffold must convert observations into a valid artifact reliably enough to offset the opportunity cost of longer trajectories. We release an auditable working-manuscript package that binds the protocol, analysis, tables, figures, limitations, and provenance without releasing private raw trajectories.

# 1 Introduction

Large language models increasingly operate as agents: they write programs, call tools, inspect outputs, and update plans. Machine-learning engineering is a natural test bed because success requires more than producing plausible code. An agent must inspect data, select a validation procedure, handle heterogeneous feature types, fit a model, preserve preprocessing at inference time, and deliver a final artifact that can be evaluated on hidden examples. Recent benchmarks have made this end-to-end setting measurable. MLAgentBench gives agents file and execution access across machine-learning experimentation tasks [@huang2024mlagentbench]; MLE-bench evaluates agents on 75 Kaggle competitions and emphasizes real-world engineering outcomes [@chan2025mlebench]; and MLGym studies open-ended AI research tasks in an executable environment [@nathani2025mlgym]. These systems build on the broader idea that reasoning and action can be interleaved with environmental observations [@yao2023react].

The usual intuition is that feedback should help. A stateful agent can see a stack trace, revise preprocessing, compare validation results, and repair a broken model. Yet this intuition leaves an important counterfactual underspecified. If an iterative agent receives more calls or tokens than a one-shot baseline, improved performance may reflect resource scaling rather than feedback. Conversely, when every arm receives the same global budget, iteration may spend much of that budget repeatedly serializing history, interpreting observations, or modifying a solution that never reaches the submission contract. Reliability matters because a high-performing but unevaluable notebook has no hidden-test utility.

This paper asks two focused questions. First, under the same global model-token and execution budget, does stateful execution feedback improve failure-adjusted solution quality relative to best-of-N independent single-shot attempts? Second, does an implicit data-science checklist improve outcomes beyond ordinary runtime, contract, and validation feedback? We answer these questions using a frozen A/B/C experiment over four tabular datasets and two model endpoints. The design pairs arms within dataset, model, and replicate blocks; prevents hidden-test feedback from driving selection; assigns agent failures the frozen dummy baseline; and analyzes all terminal outcomes rather than only successful submissions.

The study contributes four things. First, it defines a budget-matched contrast in which all decision-affecting model calls count against one global episode budget. Second, it treats submission reliability as part of the causal outcome through FANU instead of conditioning the main result on success. Third, it preserves an auditable chain from preregistration and frozen inputs to content-hashed results and generated paper tables. Fourth, it reports a negative result that is operationally informative: the evaluated iterative policies did not merely fail to improve mean quality; they frequently exhausted the budget without a valid terminal artifact.

# 2 Related work

ReAct established a general pattern for interleaving language-model reasoning with actions and observations [@yao2023react]. In software and research settings, this pattern suggests that execution feedback can ground the model and enable repair. The presence of a feedback channel, however, does not guarantee that the agent uses it efficiently or completes the task within a fixed budget.

MLAgentBench evaluates language agents that read and write files, execute code, inspect results, and iterate on machine-learning tasks [@huang2024mlagentbench]. Its reported variation across tasks illustrates that aggregate success can conceal strong task dependence. MLE-bench expands the scale and realism of the evaluation through Kaggle competitions, human leaderboard comparisons, and explicit study of resource scaling [@chan2025mlebench]. MLGym provides a framework for executable AI-research tasks and reports that current agents often improve baselines through conventional adjustments while rarely producing substantial novel advances [@nathani2025mlgym].

AutoVibe Gym differs in estimand rather than claiming broader benchmark coverage. It holds the episode budget fixed, compares stateful feedback with independent attempts, includes failures in the primary endpoint, and adds a common one-shot hidden evaluation boundary. The goal is not to rank frontier systems. It is to isolate whether this feedback scaffold pays for itself in a small, controlled tabular-ML setting.

# 3 Methods

## 3.1 Preregistration and evidence boundary

The hypotheses, failure policy, endpoint, randomization, budgets, model slots, dataset hashes, and analysis were committed before confirmatory outcomes were inspected. The effective freeze is tagged `paper-v1-experiment-freeze-v2`. The v2 amendment corrected numeric serialization of the wall-clock value without changing its effective 1,800-second limit or any scientific factor. Confirmatory execution used commit `b4e3da29e1c6b7db42848f7eee82c6a8ef3e9dae`. The final matrix contained 120 conditions and was randomized in blocks using seed `20260812`.

The unit of analysis was an independent agent episode. A stable condition identifier bound the dataset, split, model endpoint, decoding configuration, arm, replicate, prompt, and budget policy. Attempts were append-only. Infrastructure failures could be censored and replaced under the same condition with a new run identifier; agent-caused failures were terminal and could not be rerun. The final series reconciled every planned condition without unresolved infrastructure censoring.

## 3.2 Experimental arms

Arm A, `budget_matched_best_of_n_single_shot`, made independent solution attempts without stateful execution feedback. Validation score was the only selection signal, and the best valid candidate was submitted. Arm B, `iterative_no_checklist`, used a persistent notebook-style workspace and received runtime, contract, and validation feedback after actions. Arm C, `gym_with_checklist`, used the same iterative mechanism plus selective implicit hints derived from a 12-item data-science process detector. Hints were designed as nudges rather than dataset-specific instructions.

The arm contrast changes the interaction policy, so token use need not be equal ex post. Fairness is enforced by a common upper bound and pre-call accounting rather than by forcing every trajectory to consume the same number of tokens. Each arm had at most 256,000 raw input, raw output, and provider-reported reasoning tokens; 12 logical model calls; 20 code executions; 20 tool calls; and 1,800 wall-clock seconds. Cached input, if any, counted as raw input for the context budget. Post-outcome model summaries were disabled. The client-side paid-provider limit was zero, and fallback to a paid provider was forbidden.

## 3.3 Models and serving

The frozen model slots were `deepseek-v4-flash` and `gemma-4-26b`, served through an authenticated internal OpenAI-compatible LightLLM gateway. Temperature was 0.4 and maximum output length was 4,096 tokens per call. The service did not expose immutable provider weight revisions, so provenance is limited to the recorded model identifiers, decoding configuration, request-level ledgers, and service snapshot dated 2026-08-15. This limitation prevents exact reconstruction from public model weights.

## 3.4 Datasets and tasks

Four public UCI tabular datasets were transformed and frozen. Air Quality was treated as chronological regression for CO concentration, with contemporaneous reference-analyzer columns excluded [@vito2008air]. Diabetes 130-US Hospitals was a patient-group-split binary readmission task with identifiers removed [@clore2014diabetes]. Predict Students' Dropout and Academic Success was a stratified three-class classification task [@realinho2021students]. Bank Marketing was a binary subscription task with call duration removed because it is unavailable before the marketing call concludes [@moro2014bank]. The frozen row counts were 7,674, 20,000, 4,424, and 4,521, respectively.

These datasets were selected to cover regression, binary classification, multiclass classification, chronological splitting, grouped splitting, missing values, and mixed feature types. They are public benchmarks, so pretraining contamination is plausible. The study measures agent behavior on the frozen snapshots; it does not claim novelty or real-world deployment validity for the underlying prediction tasks.

## 3.5 Execution and hidden evaluation

Agent code ran in a Docker-backed notebook environment. The hidden test split was not mounted into the agent workspace or exposed in observations. Before submission, all arms used the same host-controlled validation contract. Host-side autofitting was disabled: the agent had to fit and retain a candidate capable of predicting raw validation rows after a clean replay. Candidate prediction and hidden evaluation ran in a separate ephemeral Docker container with networking disabled, a read-only filesystem, and dropped capabilities. Only a validated scalar prediction vector crossed the evaluation boundary.

Each terminal outcome could trigger at most one hidden evaluation. Hidden scores and diagnostics were never returned to the agent and could not drive selection or another repair turn. A candidate that failed serialization, clean replay, raw-row prediction, prediction-length, non-null output, or hidden evaluation was an invalid submission. Agent code failures, invalid submissions, budget exhaustion, and agent-selected execution timeouts remained in the primary population. Provider, sandbox, and orchestrator failures followed the preregistered censor-and-replace policy.

## 3.6 Endpoint

The primary endpoint was failure-adjusted normalized utility (FANU). For higher-is-better metrics,

```text
FANU = (S_agent - S_dummy) / (S_reference - S_dummy),
```

and the numerator and denominator signs were reversed for lower-is-better metrics. The dummy and fixed reference-pipeline scores were computed and hashed before the confirmatory series. FANU was not clipped, so scores below zero and above one remained possible. Agent-caused failures were assigned $S_{agent}=S_{dummy}$ and therefore FANU zero. Infrastructure-censored attempts did not receive an outcome and instead required replacement. This definition makes reliability part of the estimand and avoids the selection bias of comparing successful submissions only.

H1 compared B minus A. H2 compared C minus B. Pairs were defined by dataset, model endpoint, and replicate. The aggregate statistic assigned equal weight to each of the eight dataset-model strata and equal weight to replicates within a stratum.

## 3.7 Statistical analysis

For each comparison, the analysis reports arm means, the equal-stratum paired mean difference, and the paired median difference. A 95% paired stratified percentile bootstrap interval used 10,000 resamples with seed `20260812`. The two-sided paired permutation test was exact when at most 20 nonzero paired differences were present and otherwise used 100,000 Monte Carlo sign-flip draws with the add-one correction and the same seed. H1 and H2 FANU p-values received Holm family-wise correction at alpha 0.05.

Valid-submission rates use 95% Wilson intervals and exact paired McNemar tests. Dataset-model effects are descriptive heterogeneity summaries; the study was not powered for arbitrary interactions. Successful-only hidden-score summaries are descriptive and selection-biased. Checklist coverage was preregistered as a process outcome only after two-annotator validation of the detector. That annotation has not occurred, so this manuscript makes no empirical checklist-coverage claim.

{{GENERATED_RESULTS}}

# 5 Discussion

## 5.1 What the negative primary result means

The primary result rejects the motivating directional expectation for this frozen configuration. Under the shared global limits, the iterative policy did not convert additional context and observations into more reliable final submissions. Best-of-N independent attempts produced a valid artifact in three quarters of conditions, while the ordinary iterative arm did so in fewer than one sixth. Because agent failures remain in FANU at dummy performance, this reliability gap directly affects the primary utility estimate.

The result should not be simplified to “feedback hurts LLM agents.” The intervention bundled a particular notebook state representation, prompt, observation format, selection policy, model pair, and budget controller. A different scaffold could summarize history more compactly, preserve a known-good candidate, reserve budget for finalization, separate exploration from submission repair, or trigger earlier stopping. The causal conclusion is specific: replacing independent attempts with this stateful policy, while holding the maximum budget and evaluation contract fixed, reduced failure-adjusted utility in the tested conditions.

## 5.2 Why budget exhaustion matters

The failure distribution provides a plausible operational explanation. Most iterative episodes reached a budget boundary without producing a valid candidate, whereas the best-of-N arm more often completed self-contained artifacts. Iteration also used far more input tokens, consistent with repeatedly carrying state and observations through the context. This does not prove that context growth alone caused the failures; prompt strategy, action selection, notebook repair, candidate finalization, and model behavior are entangled. It does identify the next engineering target: completion reliability must be measured and improved before richer feedback can be expected to raise hidden-test quality.

One practical design is a two-budget policy. An exploration allowance can support experimentation, while a protected finalization reserve is unavailable until the agent must clean-replay and validate a candidate. Another is an explicit incumbent contract: after every successful validation, the environment records a restorable candidate, allowing later failed edits to fall back without host-side autofitting. Both changes would alter the intervention and therefore require a new preregistration rather than reinterpretation of the present outcomes.

## 5.3 Checklist feedback

The C-minus-B estimate is imprecise and compatible with moderately negative or modestly positive effects. The checklist arm did not solve the larger completion problem. This may mean the implicit hints were too weak, arrived at unhelpful times, or consumed attention that should have been spent finalizing the model. It may also mean checklist guidance is useful only after the base iterative agent reliably preserves and submits a candidate.

No conclusion is drawn from automatic checklist coverage. The detector relies on code and trace heuristics, and the preregistered human validation has not been completed. This boundary is important: a higher count of recognized pipeline behaviors would be a process proxy, not direct evidence of hidden-test quality.

## 5.4 Implications for agent evaluation

Agent benchmarks should report the complete denominator of attempted tasks, terminal failure categories, and resource use alongside performance. Conditioning on successful episodes can make a brittle system appear strong, while an unconstrained budget can confound scaffolding quality with inference volume. Paired, budget-matched comparisons and failure-adjusted endpoints are useful complements to leaderboard-style evaluations.

The study also illustrates the value of preserving negative results. If only successful trajectories or positive scaffold comparisons are published, engineering teams may overestimate the reliability of interactive agents. Here, the null-to-negative outcome narrows the design space: future AutoVibe Gym variants should first demonstrate valid-submission reliability under the same endpoint before making claims about the value of richer feedback.

# 6 Limitations

First, the experiment covers four tabular datasets, two internally served model endpoints, and five replicates per dataset-model-arm stratum. It cannot establish how feedback behaves for other models, larger samples, modalities, long-horizon research tasks, or alternative agent architectures.

Second, immutable model-weight revisions were unavailable. The service snapshot, endpoint identifiers, decoding configuration, and request ledgers improve auditability but cannot guarantee exact reproduction if the internal service changes.

Third, public UCI datasets may have appeared in model pretraining or online notebooks. Pairing and within-study randomization limit some comparisons, but they do not remove contamination or familiarity effects.

Fourth, FANU depends on frozen dummy and reference pipelines. It is valuable for combining metrics and retaining failures, but it is not a universal utility scale. The Bank Marketing reference denominator is relatively narrow, allowing un-clipped FANU above one; these values are valid under the preregistered formula but can amplify differences.

Fifth, the study isolates an arm-level policy rather than individual feedback components. Runtime output, contract messages, validation feedback, persistent notebook state, and history serialization arrive together in B. The failure distribution motivates follow-up mechanisms but does not separately identify them.

Sixth, human checklist-detector validation is outstanding. Checklist coverage and detector accuracy cannot support a claim in the present manuscript. The nominated second annotator and adjudicator must provide written consent before trajectories are sampled or annotated.

Seventh, a post-collection analysis-reference binding defect was repaired before any successful confirmatory statistic was emitted. The repair aligned the implementation with the already frozen reference file and changed no run, endpoint, estimand, resampling rule, or failure rule. The original failure and repair remain in the audit trail.

Finally, raw trajectories and dataset snapshots are not included in Git because they may contain large generated artifacts, sensitive-looking aggregate task features, provider metadata, and license-restricted data. A future public artifact requires a separate privacy, secret, and license review.

# 7 Reproducibility and ethics statement

The repository contains the protocol, freeze records, deterministic planner and analysis code, a byte-identical content-hashed analysis result, generated CSV tables and SVG figures, a manuscript generator, and an artifact manifest. `paper/REPRODUCIBILITY.md` gives exact offline verification commands and distinguishes Git-tracked evidence from private raw runs. The manifest binds every manuscript input by SHA-256.

No paid provider fallback was used or permitted. Requests were made only through the authenticated internal LightLLM-compatible gateway. Operator-side billing was not independently visible, so the paper reports token volume rather than monetary cost.

The datasets include demographic, socioeconomic, education, and health-related attributes. They are used only for aggregate benchmark evaluation. This study makes no deployment, clinical, educational-intervention, lending, marketing-policy, or individual-level decision claim.

OpenAI Codex was used under human direction to assist with implementation, verification automation, repository maintenance, and drafting. The statistical results were produced by the preregistered deterministic analysis, not by free-form model interpretation. Before any submission, the human author must verify every claim and citation, finalize authorship and affiliation, approve the disclosure language, and accept responsibility for the manuscript.

# 8 Conclusion

In this budget-matched tabular-ML experiment, a stateful iterative feedback scaffold produced lower failure-adjusted utility and fewer valid submissions than independent best-of-N attempts. Adding implicit checklist feedback did not yield a clear incremental improvement. The result is not a verdict against agent feedback; it is evidence that feedback must pay for its own context and coordination costs by reliably producing a valid terminal artifact. Future work should preregister mechanisms that protect finalization budget, preserve incumbents, compress history, and test feedback components separately. The present package provides a reproducible baseline against which those changes can be evaluated.

# References

The citation database is `paper/references.bib`. The working manuscript uses Pandoc citation keys and has not yet been formatted for or submitted to a venue.
