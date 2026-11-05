## Description: <br>
Build the first native, single-GPU cuDF path in a query engine, or complete its initial adapter, covering libcudf C++ or cuDF Java implementation, buffer ownership, worker readiness and CPU/reference comparison, including identifying missing POC inputs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and query-engine engineers use this skill to build a first native single-GPU cuDF proof of concept in a target engine, producing working engine or adapter code, a runnable program, and a correctness comparison against a CPU or reference result. It excludes standalone dataframe analysis, cuDF mainline changes, production integration, and distributed or multi-GPU execution. <br>

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
- [Cleanup after asynchronous work](references/asynchronous-cleanup.md) <br>
- [Evaluation guide](evals/EVAL.md) <br>
- [Third-party notices](THIRD_PARTY_NOTICES.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [Source patches and runnable program, with Markdown report including inline shell commands] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Report separates native-execution evidence from numerical agreement and marks blocked steps as BLOCKED] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), one attempt per task, each run in an isolated k8s sandbox pod and compared with and without the skill. Tasks cover repairing and running two C++ host adapters against CPU simulations, four source-level corrections or decisions, prerequisite handling, and out-of-scope routing for a CPU-only dataframe request. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 92.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 98.6% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 87.6% → 93.9% (+6.3 points) | 90.0% → 87.9% (-2.1 points) |
| Efficiency | 89.5% — baseline ran, but no comparable score was available; uplift unavailable | 84.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
939543d1b4 (source: git SHA, committed 2026-10-26) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
