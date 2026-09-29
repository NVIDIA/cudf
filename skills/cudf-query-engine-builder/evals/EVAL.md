# Evaluate query-engine adapter behavior

The dataset contains eight tasks. Two require repairing and executing C++ host adapters, four require concrete source-level corrections or decisions, and two check prerequisite handling and out-of-scope routing. Compare the same task with and without the skill. Preserve baseline successes and skill-assisted failures when reporting results.

## Tasks and checks

| Task | Required result |
|---|---|
| async-copy-cleanup | Repair and run a host adapter that keeps queued consumers safe through normal completion, allocation failure and a later valid call. |
| worker-result-handoff | Repair and run a host adapter that respects worker readiness, result ownership and the actual returned result. |
| owning-result-context | Correct the result's destruction order and exceptional cleanup while retaining an owning result. |
| engine-buffer-read-path | Correct the adapter's host-buffer lifetime while preserving the engine's existing scan interface. |
| conflicting-query-reference | Compute the conflicting outputs and ask which semantics control before choosing the native predicate. |
| native-result-acceptance | Replace a CPU answer plus unrelated CUDA marker with a result-producing native call chain and define evidence needed for acceptance. |
| missing-engine-input | Identify the inputs needed to implement and test the requested native POC without fabricating execution. |
| dataframe-negative | Answer the CPU-only dataframe request without activating this native query-engine skill. |

## Run the C++ fixtures

`run_checks.py` compiles and executes submitted C++ code. Run it in a disposable Linux evaluation environment whose runner restricts filesystem access, network access and resources. Do not mount credentials, the Docker socket or unrelated host directories into the compiler/test environment. Provider credentials needed by an agent must remain outside the environment that executes the submitted code. The checker and dependency Dockerfile do not establish that isolation. Verify the runner configuration before executing code from an untrusted source.

The checker selects the image-supplied compiler at `/usr/bin/g++` or `/usr/bin/clang++`, supplies a small explicit child environment instead of inheriting provider credentials or compiler overrides, and builds in a private temporary directory. It limits compilation to 60 seconds and execution to 20 seconds, constrains CPU time, address space and output files, bounds captured output, and terminates the command process group on completion or timeout. These controls reduce accidental exposure and resource exhaustion. They cannot prevent submitted native code from reading accessible files, opening network connections, or escaping process-group cleanup; the execution runner must enforce those boundaries.

The two fixtures are staged into /workspace/input/async-copy-cleanup and /workspace/input/worker-result-handoff. Each TASK.md defines the editable implementation and protected engine contract. From the relevant directory run:

```sh
# Run CPU ownership checks inside the isolated evaluation environment.
python3 run_checks.py
```

The command compiles the candidate with a C++20 compiler and runs the supplied checks. The checker requires Linux and uses only Python's standard library. evals/environment/Dockerfile provides Python and g++ for CPU runners. File selections are explicit in evals.json so unrelated fixtures are not staged into a task.

These fixtures simulate asynchronous backend work deterministically on the CPU. A passing simulator test establishes only the checked host-side ordering and lifetime behavior. It does not establish CUDA execution, actual libcudf correctness, native allocator behavior or query speed. Native validation must separately build and run a real cuDF adapter, compare the returned values with an independent CPU/reference result, and establish that the result came from the required cuDF operations.

The inline source tasks have no attached native build environment. Their expected result is a concrete source correction or a specific unresolved decision. They must not be scored as an executed native POC.

## Compare the two conditions

Keep the task prompt, task files, model, tools, environment and budget the same. Only skill availability changes. Use fresh isolated workspaces, and do not expose expected_output or assertions to the agent as task inputs. Skill discovery is assessed separately from whether the answer or implementation meets the task's requirements.

Record exact commands and observed outputs for executable tasks. For independent grading, restore protected caller, engine and checker files before rerunning the submitted implementation. Validate the checker with correct implementations and deliberately defective controls; reject defects for the intended violated requirement, not a missing compiler or unrelated build failure. Accept valid synchronous completion, preallocation and safe asynchronous ownership where the interface permits them.

Report paired task outcomes, model/harness versions, elapsed time and token usage when available. The signing pipeline controls its agent/model versions; separately run native trials must identify their own harness and model. Native lab results and CPU simulator results are different measurements. A larger dataset does not guarantee measurable lift, and a single trial per task does not establish a stable improvement estimate.

Do not infer a native pass from a GPU backend label, matching CPU values alone, device availability or unrelated GPU activity. Do not infer a query speedup from correctness, and do not fail a correct first POC solely because it is slower than its CPU reference. Missing profiling may be recorded separately when another reliable observation proves the native result-producing path.

## Independent compiled-fixture grading

The configuration uses SkillEvaluator's `default_plus_custom` mode. The standard graders retain their scores and overall verdict. After those graders run, `evals/grader.py` independently compiles and executes the two C++ adapter repairs and adds `compiled_fixture_correctness` as a separate result. It does not replace the standard verdict or infer native execution. Source-review and routing tasks have no compiled-fixture score.

The custom grader verifies the hashes of the protected fixture files, snapshots the submitted adapter, and reruns the fixed checks in a fresh temporary directory. A changed test or a claimed pass in the agent's answer is insufficient. The expected six check names, individual results and process exit must agree. If a maintainer intentionally changes a protected fixture, update its hash in the grader after validating the revised tests. The runner isolation requirements above also apply to this independent execution.

The custom grader targets Linux containers and writes only direct JSON files under /logs/verifier, which the standard verifier must create first. Other HARBOR_REWARD_JSON directories, including local-mode log directories, are unsupported. Output replacement does not follow symlinks or write through existing hard links.
