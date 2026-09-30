---
name: cudf-query-engine-builder
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: Joe Sack <jsack@nvidia.com>
description: Builds a native cuDF proof of concept for a query engine with no working GPU path. Use when an engineer supplies target-engine or adapter code plus a query, operator sequence, or CPU implementation selected for the POC and needs working libcudf C++ or cuDF Java code, a runnable single-GPU program, and a CPU/reference correctness comparison. Do not use for standalone dataframe analysis, cuDF mainline changes, production or deployment integration,  production fallback or I/O policy beyond the single read path the POC needs, or multi-GPU or distributed work.
---

# cuDF query-engine POC builder

## Outcome and scope

Build the first native cuDF proof of concept in the target engine for a query-engine engineer. Produce working target-engine code, a runnable single-GPU program, and a correctness result, not only advice.

The POC may cover one operator, a sequence of operators, or a query selected for the POC. Do not reduce it to one operator by default. If a smaller increment is needed for feasibility or safe implementation, explain why and preserve the path to the requested POC.

## Gather inputs and choose sources

Before editing, identify:

- the target engine or adapter code and build entry point;
- the data source or fixture, its schema, and how the engine reads it today (its own scan layer, or none);
- the query, operator sequence, or CPU implementation;
- the expected output or CPU/reference result and target-engine semantics; and
- the target-engine language, any existing libcudf C++ or cuDF Java integration code, and the cuDF version when identifiable from the code or build environment.

Determine the cuDF entry and exit interfaces from the supplied engine code. Define schema conversion, buffer ownership and release, lifetime, and producer/consumer readiness as part of the POC. Ask the engineer only when the code and tests cannot establish a required choice.

Ask only when a missing input changes the code, correctness, safety, or ability to build and run. Gather only the environment facts needed for this POC.

If the supplied query, CPU behavior, target-engine semantics, and expected result disagree, ask which behavior controls. Do not choose silently.

Use target-engine code and tests, installed public headers, and version-matched official cuDF documentation and source. Treat partner discussions as problem signals, not technical authority.

## Build the POC

1. Establish the named CPU or reference result. Run the reference implementation when it is executable.
2. Map the requested semantics to public, version-matched cuDF APIs. Handle relevant null, type, order, error, ownership, lifetime, and stream behavior. When native calls can leave GPU work pending, read [cleanup after asynchronous work](references/asynchronous-cleanup.md) before implementing ownership and exception paths.
3. Implement the requested operators and minimal engine or adapter glue without changing cuDF mainline. If the required public cuDF API is missing, preserve any buildable engine-side work, document the smallest cuDF gap, and report `BLOCKED` when the POC cannot proceed safely.
4. Build a runnable program and execute the workload on one GPU. Record exact commands and working directories.
5. State the applicable row-order handling, null and type equality, and floating-point tolerance before comparing the output with the named CPU or reference result.
6. Fix evidenced defects, then rebuild, rerun, and recompare. Continue until the checks pass or a named blocker remains. Report unsafe boundaries, missing prerequisites, unresolved semantics, and larger engine changes.

Before accepting an asynchronous path, inspect every operation that can throw after GPU work is submitted, including output allocations. Arrange ownership and cleanup so exception unwinding cannot release a buffer before its pending consumers finish. For a synchronous POC, keep those owners outside the throwing scope and establish completion before releasing them on both success and failure. Exercise a recoverable allocation failure while work is pending, assert that no owner is released early, and check a subsequent valid call when the engine promises recovery. Treat a failed ownership or readiness check as a failed POC even when normal output values match.

If a required codebase, toolchain, compatible cuDF runtime, GPU, or reference result is unavailable and prevents a required step, report `BLOCKED`, name the missing prerequisite, preserve completed work, and give the exact next command or check. Do not report the blocked step as run or passed.

## POC acceptance criteria and handoff

POC passes only when the source change builds, the program exits successfully, its output matches the named CPU or reference result under the stated comparison policy, and a reliable engine signal or runtime observation confirms that the requested result-producing work ran through cuDF.

Report the native-execution evidence separately from numerical agreement. If reliable confirmation is unavailable, mark native execution unresolved and do not report an overall POC pass. Do not infer native execution from GPU availability or elapsed time. A profiler is not required when another reliable engine signal or runtime observation establishes native cuDF execution.

Return:

- the patch and runnable program path;
- exact build and run commands with their working directories;
- the inputs, selected binding and cuDF version, named CPU or reference result, comparison policy, and observed result;
- any available cuDF-path observation; and
- unresolved questions and larger engine work, with facts separated from inference.

## Keep cuDF and engine responsibilities separate

cuDF provides device-wide algorithms for one GPU. The engine owns query-plan and pipeline orchestration, spill, production fallback, multi-GPU coordination, and distributed execution.

I/O sits on the boundary, and the POC must pick one of two paths and say which. cuDF ships data sources that read local files, remote files, and remote objects directly. The engine can either pass cuDF a path or data source and let cuDF perform the read, or keep its own I/O stack and hand cuDF the bytes in host buffers, usually pinned. Choose the path that matches the supplied engine code: if the engine already has a scan or storage layer that produces host buffers, feed those buffers to cuDF; if it does not, use a cuDF data source. Record the choice, the buffer type (pinned or pageable), and who frees each buffer in the handoff. Caching, prefetch, retries, credentials, and format negotiation stay with the engine and are out of scope.

Write the engine or adapter code needed for the requested POC, including who owns and releases each buffer and how consumers wait for producer work to complete. Report larger engine work separately; do not implement it in the POC.

## Performance and profiling

Profiling is optional for an initial POC and is never a blocker: if Nsight Systems cannot be installed or run, report that and continue. When it is available, capture one nsys profile of the run and check that the GPU is busy in the timed region. Report idle gaps, host-side stalls, and synchronous copies; do not fix them in the POC unless asked. Preserve the exact run command and workload inputs so the run can be profiled later.

When performance work is requested, profiling belongs in the same tuning loop. Benchmark the same logical work under stated conditions. Obvious code changes may be validated with benchmark results, but broader diagnosis and performance claims require profile evidence. If profiling cannot run, report the blocker rather than inferring bottlenecks from code alone.

A GPU result that remains slower than the CPU result is not automatically a failed POC. Preserve the implementation and report the measurements, profile evidence collected, observed bottlenecks, and remaining engine work.
