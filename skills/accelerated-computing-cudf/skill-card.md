## Description: <br>
Official NVIDIA-authored guidance for NVIDIA cuDF GPU DataFrames, pandas acceleration, dask-cuDF, ETL, joins, groupby, CSV/Parquet I/O, nullable semantics, and multi-GPU DataFrame workloads. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers accelerating tabular data processing with GPU DataFrames, migrating pandas code to cuDF, optimizing ETL pipelines, and scaling DataFrame workloads across multiple GPUs. <br>

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
- [NVIDIA cuDF Documentation](https://docs.nvidia.com/cudf/) <br>
- [dask-cuDF Documentation](https://docs.nvidia.com/dask-cudf/) <br>
- [NVIDIA cuDF GitHub Repository](https://github.com/NVIDIA/cudf) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (12 positive, 1 negative), each run with 3 attempts in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks whether the user’s goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks tool-call productivity and token usage efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was selected and decoys were avoided. <br>
- `goal_accuracy`: Verifies whether the user’s goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures tool-call productivity. <br>
- `token_efficiency`: Measures actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 84.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 76.9% → 69.2% (-7.7 points) |
| Correctness | Not available | 100.0% → 100.0% (±0.0 points) |
| Discoverability | Not available | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 94.4% → 90.9% (-3.5 points) |
| Efficiency | Not available | 82.3% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
4ad07b44f1 (source: git SHA, committed 2026-09-10) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
