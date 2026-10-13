# Skill Benchmark: accelerated-computing-cudf

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `accelerated-computing-cudf`
- Evaluation date: 2026-10-08
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
| Overall | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 23.1% → 61.5% (+38.4 points) | 69.2% → 53.9% (-15.3 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 78.8% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 95.1% → 94.8% (-0.3 points) | 94.1% → 92.8% (-1.3 points) |
| Efficiency | 72.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 15,380,001 | 9,883,350 | +5,496,651 | +55.62% | skill 13/13; base 13/13 |
| claude-code | cudf-apply-udf__generic | 3,276,979 | 1,632,103 | +1,644,876 | +100.78% | skill 1/1; base 1/1 |
| claude-code | cudf-csv-etl__generic | 929,377 | 626,417 | +302,960 | +48.36% | skill 1/1; base 1/1 |
| claude-code | cudf-groupby-agg__generic | 1,673,299 | 666,999 | +1,006,300 | +150.87% | skill 1/1; base 1/1 |
| claude-code | cudf-multi-join__generic | 656,234 | 583,229 | +73,005 | +12.52% | skill 1/1; base 1/1 |
| claude-code | cudf-native-stream-handoff-boundary__generic | 780,852 | 522,955 | +257,897 | +49.32% | skill 1/1; base 1/1 |
| claude-code | cudf-null-handling__generic | 990,183 | 931,683 | +58,500 | +6.28% | skill 1/1; base 1/1 |
| claude-code | cudf-parquet-io__generic | 910,215 | 682,786 | +227,429 | +33.31% | skill 1/1; base 1/1 |
| claude-code | cudf-pivot-melt__generic | 624,586 | 689,867 | -65,281 | -9.46% | skill 1/1; base 1/1 |
| claude-code | cudf-string-ops__generic | 777,342 | 750,422 | +26,920 | +3.59% | skill 1/1; base 1/1 |
| claude-code | cudf-timeseries-resample__generic | 1,154,409 | 562,653 | +591,756 | +105.17% | skill 1/1; base 1/1 |
| claude-code | cudf-window-functions__generic | 2,417,792 | 807,572 | +1,610,220 | +199.39% | skill 1/1; base 1/1 |
| claude-code | negative-deep-learning-training__generic | 558,752 | 597,992 | -39,240 | -6.56% | skill 1/1; base 1/1 |
| claude-code | source-cudf-null-fillna-semantics__generic | 629,981 | 828,672 | -198,691 | -23.98% | skill 1/1; base 1/1 |
| codex | All cases | 4,143,406 | 3,256,824 | +886,582 | +27.22% | skill 13/13; base 13/13 |
| codex | cudf-apply-udf__generic | 440,428 | 481,525 | -41,097 | -8.53% | skill 1/1; base 1/1 |
| codex | cudf-csv-etl__generic | 205,788 | 187,153 | +18,635 | +9.96% | skill 1/1; base 1/1 |
| codex | cudf-groupby-agg__generic | 278,324 | 167,811 | +110,513 | +65.86% | skill 1/1; base 1/1 |
| codex | cudf-multi-join__generic | 210,799 | 243,024 | -32,225 | -13.26% | skill 1/1; base 1/1 |
| codex | cudf-native-stream-handoff-boundary__generic | 255,481 | 262,663 | -7,182 | -2.73% | skill 1/1; base 1/1 |
| codex | cudf-null-handling__generic | 318,364 | 228,751 | +89,613 | +39.17% | skill 1/1; base 1/1 |
| codex | cudf-parquet-io__generic | 309,807 | 265,751 | +44,056 | +16.58% | skill 1/1; base 1/1 |
| codex | cudf-pivot-melt__generic | 436,604 | 288,134 | +148,470 | +51.53% | skill 1/1; base 1/1 |
| codex | cudf-string-ops__generic | 420,292 | 180,914 | +239,378 | +132.32% | skill 1/1; base 1/1 |
| codex | cudf-timeseries-resample__generic | 403,278 | 260,103 | +143,175 | +55.05% | skill 1/1; base 1/1 |
| codex | cudf-window-functions__generic | 417,147 | 306,835 | +110,312 | +35.95% | skill 1/1; base 1/1 |
| codex | negative-deep-learning-training__generic | 269,474 | 155,081 | +114,393 | +73.76% | skill 1/1; base 1/1 |
| codex | source-cudf-null-fillna-semantics__generic | 177,620 | 229,079 | -51,459 | -22.46% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 19,523,407 | 13,140,174 | +6,383,233 | +48.58% | skill 26/26; base 26/26 |

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
