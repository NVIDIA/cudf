## Description: <br>
Official NVIDIA-authored guidance for NVIDIA cuDF GPU DataFrames, pandas acceleration, dask-cuDF, ETL, joins, groupby, CSV/Parquet I/O, nullable semantics, and multi-GPU DataFrame workloads. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers accelerating tabular data processing with NVIDIA cuDF, including GPU-backed ETL pipelines, pandas code migration, DataFrame joins and aggregations, nullable semantics validation, and multi-GPU workloads via dask-cuDF. <br>

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
- [cudf.pandas Accelerator Deep Dive](references/cudf-pandas-accelerator.md) <br>
- [dask-cuDF Patterns](references/dask-cudf-patterns.md) <br>
- [cuDF Documentation](https://docs.nvidia.com/cudf/) <br>
- [dask-cuDF API Reference](https://docs.nvidia.com/dask-cudf/) <br>
- [cuDF GitHub Repository](https://github.com/NVIDIA/cudf) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (12 positive, 1 negative), 3 attempts per task, each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 72.9% | 84.4% |
| Security | 42.9% → 15.4% (-27.5 points) | 57.1% → 61.5% (+4.4 points) |
| Correctness | 95.7% → 100.0% (+4.3 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 82.5% | 80.8% |
| Effectiveness | 86.4% → 96.0% (+9.6 points) | 90.5% → 96.2% (+5.7 points) |
| Efficiency | 70.6% | 83.4% |

## Skill Version(s): <br>
bdcc16dbe3 (source: git SHA, committed 2026-09-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
