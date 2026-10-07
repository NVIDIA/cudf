# Skill Benchmark: accelerated-computing-cudf

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `accelerated-computing-cudf`
- Evaluation date: 2026-09-11
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (12 positive, 1 negative)
- Dataset digest: `sha256:307ff81fa3f0d04ee89889dcedc5ac0208fc0eb5dba112e1c8ed515a08fa3ba9` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 84.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 76.9% → 69.2% (-7.7 points) |
| Correctness | Not available | 100.0% → 100.0% (±0.0 points) |
| Discoverability | Not available | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 94.4% → 90.9% (-3.5 points) |
| Efficiency | Not available | 82.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 14,012,571 | 10,980,314 | N/A | N/A | skill 13/14; base 13/39 |
| claude-code | cudf-apply-udf__generic | 1,615,916 | 2,921,909 | -1,305,993 | -44.70% | skill 1/1; base 1/1 |
| claude-code | cudf-csv-etl__generic | 640,979 | 540,152 | +100,827 | +18.67% | skill 1/1; base 1/1 |
| claude-code | cudf-groupby-agg__generic | 1,334,220 | 1,114,950 | +219,270 | +19.67% | skill 1/1; base 1/1 |
| claude-code | cudf-multi-join__generic | 938,388 | 609,377 | +329,011 | +53.99% | skill 1/1; base 1/1 |
| claude-code | cudf-native-stream-handoff-boundary__generic | 753,844 | 676,138 | +77,706 | +11.49% | skill 1/1; base 1/1 |
| claude-code | cudf-null-handling__generic | 573,498 | 698,891 | -125,393 | -17.94% | skill 1/1; base 1/1 |
| claude-code | cudf-parquet-io__generic | 795,742 | 632,812 | +162,930 | +25.75% | skill 1/1; base 1/1 |
| claude-code | cudf-pivot-melt__generic | 969,475 | 636,461 | +333,014 | +52.32% | skill 1/1; base 1/1 |
| claude-code | cudf-string-ops__generic | 934,970 | 723,865 | +211,105 | +29.16% | skill 1/1; base 1/1 |
| claude-code | cudf-timeseries-resample__generic | 585,864 | 624,629 | -38,765 | -6.21% | skill 1/1; base 1/1 |
| claude-code | cudf-window-functions__generic | 2,207,481 | 675,332 | N/A | N/A | skill 1/2; base 1/1 |
| claude-code | negative-deep-learning-training__generic | 458,761 | 642,097 | -183,336 | -28.55% | skill 1/1; base 1/1 |
| claude-code | source-cudf-null-fillna-semantics__generic | 2,203,433 | 483,701 | +1,719,732 | +355.54% | skill 1/1; base 1/1 |
| codex | All cases | 4,399,248 | 3,330,594 | +1,068,654 | +32.09% | skill 13/13; base 13/13 |
| codex | cudf-apply-udf__generic | 366,287 | 333,591 | +32,696 | +9.80% | skill 1/1; base 1/1 |
| codex | cudf-csv-etl__generic | 368,472 | 218,866 | +149,606 | +68.36% | skill 1/1; base 1/1 |
| codex | cudf-groupby-agg__generic | 282,617 | 251,646 | +30,971 | +12.31% | skill 1/1; base 1/1 |
| codex | cudf-multi-join__generic | 246,898 | 177,146 | +69,752 | +39.38% | skill 1/1; base 1/1 |
| codex | cudf-native-stream-handoff-boundary__generic | 322,482 | 324,176 | -1,694 | -0.52% | skill 1/1; base 1/1 |
| codex | cudf-null-handling__generic | 481,827 | 313,356 | +168,471 | +53.76% | skill 1/1; base 1/1 |
| codex | cudf-parquet-io__generic | 294,404 | 206,582 | +87,822 | +42.51% | skill 1/1; base 1/1 |
| codex | cudf-pivot-melt__generic | 269,590 | 280,046 | -10,456 | -3.73% | skill 1/1; base 1/1 |
| codex | cudf-string-ops__generic | 258,274 | 162,821 | +95,453 | +58.62% | skill 1/1; base 1/1 |
| codex | cudf-timeseries-resample__generic | 385,513 | 291,946 | +93,567 | +32.05% | skill 1/1; base 1/1 |
| codex | cudf-window-functions__generic | 444,592 | 281,185 | +163,407 | +58.11% | skill 1/1; base 1/1 |
| codex | negative-deep-learning-training__generic | 268,298 | 220,178 | +48,120 | +21.86% | skill 1/1; base 1/1 |
| codex | source-cudf-null-fillna-semantics__generic | 409,994 | 269,055 | +140,939 | +52.38% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 18,411,819 | 14,310,908 | N/A | N/A | skill 26/27; base 26/52 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/accelerated-computing-cudf/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/accelerated-computing-cudf/SKILL.md`)
- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/accelerated-computing-cudf/SKILL.md`)

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
