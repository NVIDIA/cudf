# Skill Benchmark: cudf-query-engine-builder

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `cudf-query-engine-builder`
- Evaluation date: 2026-10-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:247cebc82d646141c1d40cb37bee3138aa3ed88f3d764c44df91933fe13bc270` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 97.7% — baseline ran, but no comparable score was available; uplift unavailable | 94.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 97.5% → 100.0% (+2.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 99.7% → 97.9% (-1.8 points) | 91.2% → 91.5% (+0.3 points) |
| Efficiency | 90.8% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,387,434 | 1,101,532 | +285,902 | +25.95% | skill 8/8; base 8/8 |
| claude-code | async-copy-cleanup | 335,304 | 383,983 | -48,679 | -12.68% | skill 1/1; base 1/1 |
| claude-code | conflicting-query-reference | 163,245 | 75,128 | +88,117 | +117.29% | skill 1/1; base 1/1 |
| claude-code | dataframe-negative | 35,392 | 35,108 | +284 | +0.81% | skill 1/1; base 1/1 |
| claude-code | engine-buffer-read-path | 121,324 | 39,594 | +81,730 | +206.42% | skill 1/1; base 1/1 |
| claude-code | missing-engine-input | 114,061 | 145,217 | -31,156 | -21.45% | skill 1/1; base 1/1 |
| claude-code | native-result-acceptance | 121,890 | 39,090 | +82,800 | +211.82% | skill 1/1; base 1/1 |
| claude-code | owning-result-context | 127,798 | 44,708 | +83,090 | +185.85% | skill 1/1; base 1/1 |
| claude-code | worker-result-handoff | 368,420 | 338,704 | +29,716 | +8.77% | skill 1/1; base 1/1 |
| codex | All cases | 754,883 | 724,097 | +30,786 | +4.25% | skill 8/8; base 8/8 |
| codex | async-copy-cleanup | 287,085 | 298,463 | -11,378 | -3.81% | skill 1/1; base 1/1 |
| codex | conflicting-query-reference | 129,486 | 56,758 | +72,728 | +128.14% | skill 1/1; base 1/1 |
| codex | dataframe-negative | 13,401 | 13,243 | +158 | +1.19% | skill 1/1; base 1/1 |
| codex | engine-buffer-read-path | 30,522 | 15,636 | +14,886 | +95.20% | skill 1/1; base 1/1 |
| codex | missing-engine-input | 44,863 | 27,311 | +17,552 | +64.27% | skill 1/1; base 1/1 |
| codex | native-result-acceptance | 30,433 | 14,059 | +16,374 | +116.47% | skill 1/1; base 1/1 |
| codex | owning-result-context | 32,858 | 15,781 | +17,077 | +108.21% | skill 1/1; base 1/1 |
| codex | worker-result-handoff | 186,235 | 282,846 | -96,611 | -34.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,142,317 | 1,825,629 | +316,688 | +17.35% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cudf-query-engine-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cudf-query-engine-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cudf-query-engine-builder/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/cudf-query-engine-builder/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (470 chars, recommend 50-150) (`skills/cudf-query-engine-builder/SKILL.md`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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
