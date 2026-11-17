# Skill Benchmark: accelerated-computing-cudf

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `accelerated-computing-cudf`
- Evaluation date: 2026-10-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (12 positive, 1 negative)
- Dataset digest: `sha256:307ff81fa3f0d04ee89889dcedc5ac0208fc0eb5dba112e1c8ed515a08fa3ba9` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
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
| Overall | 76.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 23.1% → 38.5% (+15.4 points) | 38.5% → 92.3% (+53.8 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 79.2% — baseline ran, but no comparable score was available; uplift unavailable | 78.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 92.5% → 94.9% (+2.4 points) | 88.5% → 92.1% (+3.6 points) |
| Efficiency | 71.4% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 14,398,373 | 11,547,678 | +2,850,695 | +24.69% | skill 13/13; base 13/13 |
| claude-code | cudf-apply-udf__generic | 2,450,019 | 2,926,781 | -476,762 | -16.29% | skill 1/1; base 1/1 |
| claude-code | cudf-csv-etl__generic | 517,748 | 912,335 | -394,587 | -43.25% | skill 1/1; base 1/1 |
| claude-code | cudf-groupby-agg__generic | 1,137,553 | 646,888 | +490,665 | +75.85% | skill 1/1; base 1/1 |
| claude-code | cudf-multi-join__generic | 928,964 | 575,088 | +353,876 | +61.53% | skill 1/1; base 1/1 |
| claude-code | cudf-native-stream-handoff-boundary__generic | 814,783 | 683,164 | +131,619 | +19.27% | skill 1/1; base 1/1 |
| claude-code | cudf-null-handling__generic | 859,166 | 682,195 | +176,971 | +25.94% | skill 1/1; base 1/1 |
| claude-code | cudf-parquet-io__generic | 943,643 | 778,017 | +165,626 | +21.29% | skill 1/1; base 1/1 |
| claude-code | cudf-pivot-melt__generic | 789,841 | 942,039 | -152,198 | -16.16% | skill 1/1; base 1/1 |
| claude-code | cudf-string-ops__generic | 1,375,882 | 481,096 | +894,786 | +185.99% | skill 1/1; base 1/1 |
| claude-code | cudf-timeseries-resample__generic | 830,827 | 723,311 | +107,516 | +14.86% | skill 1/1; base 1/1 |
| claude-code | cudf-window-functions__generic | 1,481,049 | 878,393 | +602,656 | +68.61% | skill 1/1; base 1/1 |
| claude-code | negative-deep-learning-training__generic | 636,107 | 611,427 | +24,680 | +4.04% | skill 1/1; base 1/1 |
| claude-code | source-cudf-null-fillna-semantics__generic | 1,632,791 | 706,944 | +925,847 | +130.96% | skill 1/1; base 1/1 |
| codex | All cases | 4,672,480 | 4,359,958 | +312,522 | +7.17% | skill 13/13; base 13/13 |
| codex | cudf-apply-udf__generic | 562,752 | 1,207,294 | -644,542 | -53.39% | skill 1/1; base 1/1 |
| codex | cudf-csv-etl__generic | 312,349 | 199,554 | +112,795 | +56.52% | skill 1/1; base 1/1 |
| codex | cudf-groupby-agg__generic | 326,996 | 186,000 | +140,996 | +75.80% | skill 1/1; base 1/1 |
| codex | cudf-multi-join__generic | 192,398 | 200,815 | -8,417 | -4.19% | skill 1/1; base 1/1 |
| codex | cudf-native-stream-handoff-boundary__generic | 321,351 | 201,040 | +120,311 | +59.84% | skill 1/1; base 1/1 |
| codex | cudf-null-handling__generic | 305,965 | 269,737 | +36,228 | +13.43% | skill 1/1; base 1/1 |
| codex | cudf-parquet-io__generic | 381,162 | 351,579 | +29,583 | +8.41% | skill 1/1; base 1/1 |
| codex | cudf-pivot-melt__generic | 318,688 | 399,785 | -81,097 | -20.29% | skill 1/1; base 1/1 |
| codex | cudf-string-ops__generic | 344,778 | 313,480 | +31,298 | +9.98% | skill 1/1; base 1/1 |
| codex | cudf-timeseries-resample__generic | 491,176 | 363,369 | +127,807 | +35.17% | skill 1/1; base 1/1 |
| codex | cudf-window-functions__generic | 529,585 | 315,999 | +213,586 | +67.59% | skill 1/1; base 1/1 |
| codex | negative-deep-learning-training__generic | 241,846 | 139,900 | +101,946 | +72.87% | skill 1/1; base 1/1 |
| codex | source-cudf-null-fillna-semantics__generic | 343,434 | 211,406 | +132,028 | +62.45% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 19,070,853 | 15,907,636 | +3,163,217 | +19.88% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/accelerated-computing-cudf/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/accelerated-computing-cudf/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/accelerated-computing-cudf/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/accelerated-computing-cudf/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/accelerated-computing-cudf/SKILL.md`)
- 3 additional finding(s) are available in the full evaluation artifacts.

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
