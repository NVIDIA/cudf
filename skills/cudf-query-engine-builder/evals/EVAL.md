# Evaluation instructions

The distributed evals.json contains two CPU-only checks: ask for missing engine inputs, and avoid activating this skill for a pandas-only request. These check prerequisite handling and skill selection. They do not measure native GPU correctness.

Run each prompt in a fresh workspace with the skill available and without it, using both Claude Code and Codex. Use config.yml for the isolated workspace and resource settings. Record the harness and model version, whether the skill was read, the final answer, token use and elapsed time.

For the missing-input prompt, the answer must request the engine code or build entry point, query or CPU implementation, schema or input fixture, and expected result. It must not invent a completed implementation or execution result. Report skill activation separately from answer correctness.

For the pandas-only prompt, the agent must preserve missing group keys using dropna=False and must not load the query-engine builder or introduce GPU integration work.

## Native GPU results

BENCHMARK.md reports separate team-run native GPU evaluations. Their synthetic C++ starters, GPU dataset and independent checker are not distributed in this package. The reported GPU results cannot be reproduced from these files alone.

To evaluate a new native integration, provide the target engine, workload, CPU reference and compatible GPU environment. Compare results under a stated equality policy, check ownership and readiness across asynchronous work and allocation failures, and collect GPU execution observations separately. Before building or running agent-generated code, review its source, build scripts and proposed commands for unintended file access, network access and destructive operations. Do not execute code that fails this review. Run accepted code only in a disposable, isolated environment with bounded resources and no unrelated files or credentials. Pass program arguments directly rather than interpolating generated text into shell commands. Passing the two distributed CPU-only checks does not replace native integration testing.
