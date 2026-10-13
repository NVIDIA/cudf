## Description: <br>
Official NVIDIA-authored guidance for NVIDIA cuDF GPU DataFrames, pandas acceleration, dask-cuDF, ETL, joins, groupby, CSV/Parquet I/O, nullable semantics, and multi-GPU DataFrame workloads. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers use this skill to have an agent migrate or accelerate pandas DataFrame code on NVIDIA GPUs with cudf.pandas, explicit cuDF, or dask-cuDF, covering ETL, joins, groupby, reshape, nullable semantics, CSV/Parquet I/O, memory management, and CPU/GPU parity validation. <br>

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
- [cuDF API Patterns, Gaps, and Semantic Differences](references/api-patterns.md) <br>
- [cudf.pandas Accelerator — Deep Dive](references/cudf-pandas-accelerator.md) <br>
- [dask-cuDF Patterns](references/dask-cudf-patterns.md) <br>
- [cuDF Documentation](https://docs.nvidia.com/cudf/) <br>
- [dask-cuDF API Reference](https://docs.nvidia.com/dask-cudf/) <br>
- [Installation System Requirements](https://docs.nvidia.com/datascience/install/#system-req) <br>
- [cuDF GitHub](https://github.com/NVIDIA/cudf) <br>
- [cuDF CHANGELOG](https://github.com/NVIDIA/cudf/blob/main/CHANGELOG.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (12 positive, 1 negative), one attempt per task, each run in an isolated k8s sandbox pod and compared against a no-skill baseline. Overall verdict: PASS — recommended for publication (evaluation date 2026-10-08, evaluator version 1.5.6). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from final-answer accuracy. <br>
- Discoverability: Was the right skill loaded when needed? Scored from skill execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal accuracy and behavior check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 23.1% → 61.5% (+38.4 points) | 69.2% → 53.9% (-15.3 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 78.8% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 95.1% → 94.8% (-0.3 points) | 94.1% → 92.8% (-1.3 points) |
| Efficiency | 72.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.4% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
26.12.00 development release (source: repo VERSION file and SKILL.md compatibility section; latest CHANGELOG release 26.10.00, git SHA 18f5bd5bf3 committed 2026-10-08) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
