## Description: <br>
Official NVIDIA-authored guidance for NVIDIA cuDF GPU DataFrames, pandas acceleration, dask-cuDF, ETL, joins, groupby, CSV/Parquet I/O, nullable semantics, and multi-GPU DataFrame workloads. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers implementing GPU-accelerated DataFrame operations with NVIDIA cuDF, including pandas-to-GPU migration, ETL pipeline optimization, and multi-GPU workloads with dask-cuDF. <br>

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
- [cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/) <br>
- [dask-cuDF API Reference](https://docs.rapids.ai/api/dask-cudf/stable/api/) <br>
- [cuDF GitHub Repository](https://github.com/NVIDIA/cudf) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (12 positive, 1 negative). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill is found and executed when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and follows expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 74% → 88% (+14 points) | 68% → 83% (+15 points) |
| Security | 85% → 69% (-15 points) | 54% → 46% (-8 points) |
| Correctness | 100% → 100% (±0 points) | 100% → 98% (-2 points) |
| Discoverability | 47% → 89% (+42 points) | 45% → 85% (+40 points) |
| Effectiveness | 96% → 96% (±0 points) | 95% → 94% (-1 points) |
| Efficiency | 44% → 87% (+43 points) | 46% → 89% (+43 points) |

## Skill Version(s): <br>
333911cf41 (source: git SHA, committed 2026-08-26) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
