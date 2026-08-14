# Failure, retry, censoring, and replacement policy

Every attempt receives exactly one terminal category. Classification is based on
the first causal failure that prevents a valid terminal agent outcome, with raw
events retained for secondary diagnostics.

## Categories

| Category | Class | Primary-analysis treatment |
|---|---|---|
| `success` | outcome | use the one hidden score |
| `agent_code_failure` | agent | score at dataset dummy baseline for FANU |
| `invalid_submission` | agent | score at dataset dummy baseline for FANU |
| `budget_exhausted` | agent by default | score at dummy unless an infrastructure defect consumed the budget |
| `execution_timeout` | agent by default | score at dummy when generated code exceeded the declared limit |
| `sandbox_failure` | infrastructure | censor attempt and queue replacement |
| `provider_rate_limit` | infrastructure | retry call, then censor/replace if unresolved |
| `provider_capacity` | infrastructure | retry call, then censor/replace if unresolved |
| `provider_auth` | infrastructure | no blind repeated retry; censor and stop for configuration review |
| `provider_other` | infrastructure | retry only if explicitly classified transient; otherwise censor |
| `orchestrator_failure` | infrastructure | censor attempt and queue replacement |

## Classification boundaries

- Syntax/import/runtime errors caused by generated code are
  `agent_code_failure`, even if they occur inside the sandbox.
- A trained artifact that fails the common submission contract, clean replay,
  serialization, raw-row prediction, prediction-length, or non-null-output check
  is `invalid_submission`.
- A timeout from agent-selected code/model search is `execution_timeout`.
- Failure to start, communicate with, or enforce limits in the sandbox is
  `sandbox_failure`.
- A budget controller that stops a valid next call according to policy yields
  `budget_exhausted`; a controller bug is `orchestrator_failure`.
- HTTP/provider identity is not enough by itself: record status code, SDK
  exception, request ID, and retryability evidence before assigning a provider
  category.
- Hidden-test prediction/evaluation is attempted once. Failure is
  `invalid_submission`; the agent receives no repair turn and no second hidden
  attempt.

## Provider retry policy

Transient provider failures may be retried at most three times with exponential
backoff capped by the implementation policy. Each request attempt is appended to
the usage ledger with retry index, timing, request ID when available, finish/error
status, and any reported usage. Attempts are never overwritten.

Retries do not create a new agent outcome or silently reset consumed budget.
Tokens actually reported by a provider count toward resource totals. If the call
cannot complete after the retry limit, the run is infrastructure-censored.

`provider_auth` is treated as configuration failure and is not retried repeatedly.

## Agent reruns

Agent failures are terminal outcomes and are not rerun. The only exception is a
predeclared identical technical retry with evidence that infrastructure—not the
agent trajectory—caused the failure. That retry still receives a new `run_id`,
sets `rerun_of`, states the reason, and preserves the original artifacts.

## Replacement runs

An unresolved infrastructure-censored attempt is replaced under the same stable
`condition_id`, arm, dataset, model, split, replicate, prompt, decoding config,
and budget policy. The replacement receives a new `run_id` and references the
censored attempt through `rerun_of`.

The confirmatory completeness report must reconcile:

```text
planned terminal conditions
= successful agent outcomes
+ agent-failure outcomes
+ unresolved infrastructure-censored conditions
```

Provider retries and superseded infrastructure attempts are reported separately
and cannot be hidden by replacement.

## Adjudication

Ambiguous failures are classified without access to hidden scores or aggregate
arm results. The reviewer records evidence, provisional category, final category,
reviewer, timestamp, and rationale. Any policy amendment requires a versioned
change note and, after freeze, a new protocol tag rather than silent relabeling.
