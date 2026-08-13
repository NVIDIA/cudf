---
name: perf-compare-cudf
description: Benchmark a cuDF branch, WIP changes, or a PR against the `main` branch
---

Use this skill when the user asks to compare libcudf benchmark performance for:
- **the current branch or WIP changes** against `rapidsai/cudf` `main`.
- **a cudf PR link or number** against `rapidsai/cudf` `main`.

# Goal

Run the same selected libcudf NVBench benchmarks on the target (current WIP or cudf PR) and then on `rapidsai/cudf` `main`, then report meaningful differences.

`<cudf-remote>` is the git remote for `https://github.com/rapidsai/cudf` (often `upstream`). Detect it with `git remote -v`.

## Prerequisites

- For PR targets, **`gh` CLI** authenticated — run `gh auth status`. If not authenticated, guide the user to run:
   ```bash
   gh auth login
   ```
  The token needs `repo` scope. Do **not** run `gh auth token` from within the agent.
- Ensure we are in the cudf devcontainer (username `coder`). If not, stop and ask the user for instructions.

## 1. Prepare

- Record the starting branch, `git status --short`, and the exact target (current WIP or cudf PR).
- Run order: Target side first, then `main`.
- Record current timestamp as `ts = <YYYYMMDD_HHMMSS>`
- Create result directories:
  ```bash
  mkdir -p benchmark_compare/<ts>/{target,main}
  ```

## 2. Build Target

- For current-branch or WIP targets: keep target changes applied for the target run.
- For PR targets: Stash any unrelated local changes, record the stash name, and check out the PR:
  ```bash
  gh pr checkout <PR_NUMBER> --repo rapidsai/cudf
  ```
- For PR targets: After switching, check if the PR branch is behind `<cudf-remote>/main` and add a merge commit. DO **NOT** push anything. If there are merge conflicts, stop and guide the user to fix them.

- On the first build for a checkout, force CMake reconfiguration to enable benchmarks:

```bash
configure-cudf-cpp -DBUILD_BENCHMARKS=ON
build-cudf-cpp
```
- Re-run `configure-cudf-cpp -DBUILD_BENCHMARKS=ON` if the build directory is cleaned or CMake options may have changed.
- If needed, refer to the `build-test-cudf` skill for instructions and troubleshooting.

## 3. Choose Benchmarks

- Fetch current main with `git fetch <cudf-remote> main`.
- Infer candidate benchmark suites from:
  ```bash
  git diff --name-only <cudf-remote>/main...HEAD
  ```
- Benchmark binaries live under `cpp/build/latest/benchmarks/*_NVBENCH`.
- Inspect candidate binaries from the target build:
  ```bash
  cpp/build/latest/benchmarks/<BENCH> --list
  cpp/build/latest/benchmarks/<BENCH> --help-axes
  ```
- Confirm benchmark binaries and axis coverage with the user. Use a small, representative axis subset by default; use full coverage only when requested or necessary.
- Record exact `-b` and `-a` options. Reuse them unchanged on both branches.

## 4. Run Target

- Pick an idle GPU with `nvidia-smi`. Do this every time before running anything (target or main run); if the same GPU is no longer idle, pick another one, wait, or ask before continuing.
- Run on one masked device only: `CUDA_VISIBLE_DEVICES=<idx>` and `-d 0`.
- Write target JSON and log files under `benchmark_compare/<ts>/target/`, for example:
  ```bash
  CUDA_VISIBLE_DEVICES=<idx> cpp/build/latest/benchmarks/<BENCH> -d 0 \
    -b <bench_name> -a <axis=...> ... \
    --json benchmark_compare/<ts>/target/<BENCH>.json 2>&1 | tee benchmark_compare/<ts>/target/<BENCH>.log
  ```
- If nvbench emits an end-of-suite segfault after writing results, note it and continue. If a config throws, verify that both branches (main and target) behave the same.

## 5. Switch over to main

- To switch to main, stash any target WIP if needed, record the stash name, and use a clean branch:
  ```bash
  git fetch <cudf-remote> main
  git checkout -B _bench_main <cudf-remote>/main
  ```
- Do not apply any WIP or target changes on `_bench_main`.

## 6. Build and run main

Follow configure, build and benchmark run steps as for the target. Run the same set of benchmarks chosen above, but write JSON and log files to `benchmark_compare/<ts>/main/` instead.

## 7. Compare

Use NVBench's comparison script from the build tree:

```bash
NVBENCH_SCRIPTS=cpp/build/latest/_deps/nvbench-src/python/scripts
test -f "$NVBENCH_SCRIPTS/nvbench_compare.py" || \
  NVBENCH_SCRIPTS=cpp/build/latest/_deps/nvbench-src/scripts
PYTHONPATH="$NVBENCH_SCRIPTS" python "$NVBENCH_SCRIPTS/nvbench_compare.py" \
  --threshold-diff 0.05 --no-color benchmark_compare/<ts>/main benchmark_compare/<ts>/target \
  | tee benchmark_compare/<ts>/COMPARISON.md
```

- The first path is the reference (`main`), the second is the comparison (`target`). Re-run surprising failures once, especially small or noisy configs.

## 8. Restore and report

- Return to the starting branch/state, pop any stash you created, delete temporary branches, and confirm `git status` matches the starting state.
- Remember to note if there were any end-of-suite segfaults or config throws and if the behavior was the same on both branches.
- Use the below template for `COMPARISON.md`, adapting the metric columns to the benchmark. GPU time is always useful, but other metrics such as output file size, throughput, compression ratio, or memory usage are also of interest when they change significantly in target vs main.
- Summarize chat with the headline result (regression, improvement, or within noise), relevant metrics, hardware used, branch SHAs, axis coverage, the summary table from the template, and generated files.

  ```markdown
  # Benchmark Comparison: <cudf-remote>/main vs target (`WIP` or `PR`)

  - Primary metric(s): <GPU time, throughput, output size, memory, etc.>
  - Δ = (target - main) / main. Interpret direction per metric.
  - Significant timing deltas: |Δ| >= 5% AND larger than max(noise) of either side.
  - Hardware: <GPU name + index from nvbench/nvidia-smi>, driver/CUDA if available.
              <CPU name + cores from `lscpu`>, model, architecture, if available.
  - Branches: target `<sha>` vs main `<sha>`. Axis coverage: <skim|full> (list values used).

  ## Summary
  | Benchmark Suite | Primary metric | # Configs | # Meaningful Changes |
  | ... |

  ## Top N Changes
  | Suite / bench | axes | metric | main | target | Δ | noise, if timing |
  | ... |

  ## Per-suite tables
  (one table per benchmark, axes as columns, include all relevant metrics)

  ## Notes
  - Exceptions excluded (same on both branches): ...
  - End-of-suite segfaults ignored.
  - Files generated: list of JSON/log paths + this report
  ```
