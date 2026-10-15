## Description: <br>
Build the first native, single-GPU cuDF path in a query engine, or complete its initial adapter, covering libcudf C++ or cuDF Java implementation, buffer ownership, worker readiness and CPU/reference comparison, while excluding standalone dataframe analysis, cuDF mainline changes, production integration and distributed or multi-GPU execution. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and query-engine engineers use this skill to build a native single-GPU cuDF proof of concept in a target engine, including adapter code, buffer ownership and readiness handling, a runnable program, and a correctness comparison against a CPU or reference result. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Complete native work before releasing its owners](references/asynchronous-cleanup.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline code blocks, plus source patches and a runnable program] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports exact build and run commands, comparison policy, observed result, native-execution evidence, and BLOCKED status with named prerequisites when a required step cannot run.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), 1 attempt per task, each run in an isolated k8s sandbox pod and compared with a no-skill baseline. Overall verdict: NEUTRAL. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? (goal accuracy 50% + behavior check 50%) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (tool productivity 50% + token efficiency 50%) <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 97.7% — baseline ran, but no comparable score was available; uplift unavailable | 94.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 97.5% → 100.0% (+2.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 99.7% → 97.9% (-1.8 points) | 91.2% → 91.5% (+0.3 points) |
| Efficiency | 90.8% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
c78664088b (source: git SHA, committed 2026-10-13) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
