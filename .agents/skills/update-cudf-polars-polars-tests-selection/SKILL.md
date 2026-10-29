---
name: update-cudf-polars-polars-tests-selection
description: Regenerates the versioned GPU-executing test selection for the upstream Polars suite. Use after a Polars upgrade, when the minimum supported Polars version is raised, or when the nightly cudf-polars run-list audit reports drift.
---

# Update the upstream Polars GPU test selection

Update `ci/cudf_polars/gpu_test_run_list.json` only from complete default-blocksize
SPMD telemetry. Never rename an old version key or infer membership from
translation attempts: a selected node must have completed a GPU query in its
test body. Telemetry ignores queries run during fixture setup and teardown.

## Preferred process: use nightly evidence

1. Find a nightly run for the target cuDF revision in which every
   `cudf-polars-polars-tests` producer and shard completed. Nightlies are
   `workflow_dispatch` runs of `test.yaml` on `main`. They test their `sha`
   input, not the run's `headSha`, so don't filter with `--commit`; list recent
   runs and confirm the tested revision after downloading in step 2:

   ```bash
   gh run list --repo NVIDIA/cudf --workflow test.yaml --branch main \
       --event workflow_dispatch --limit 10 \
       --json databaseId,attempt,createdAt,conclusion
   ```

   The run's own `conclusion` often reflects unrelated jobs, so check the
   cudf-polars producers directly. This prints any `cudf-polars-polars-tests`
   job that did not succeed in the run's latest attempt; no output means they
   all completed. Keep `per_page` small: larger pages of a nightly's jobs fail
   with HTTP 502.

   ```bash
   gh api --paginate \
       "repos/NVIDIA/cudf/actions/runs/<run-id>/jobs?filter=latest&per_page=25" \
       --jq '.jobs[] | select(.name | startswith("cudf-polars-polars-tests"))
             | select(.conclusion != "success") | "\(.conclusion)\t\(.name)"'
   ```
2. Download its `cudf-polars-fallback-diagnostics-<attempt>` artifact, where
   `<attempt>` is the `attempt` printed by `gh run list`:

   ```bash
   gh run download <run-id> --repo NVIDIA/cudf \
       --name cudf-polars-fallback-diagnostics-<attempt> --dir nightly
   jq .status nightly/run-list-drift.json
   jq -r '[.runs[].cudf_sha] | unique[]' nightly/fallback-diagnostics.json
   ```

   Do not use the artifact if the published status is `incomplete`, or if the
   printed cuDF SHA is not the target revision.
3. Generate a candidate:

   ```bash
   python ci/analyze_cudf_polars_fallbacks.py \
       --diagnostics-file nightly/fallback-diagnostics.json \
       --output-dir diagnostics \
       --audit-run-list ci/cudf_polars/gpu_test_run_list.json \
       --generate-run-list gpu-test-run-list-candidate.json
   ```

   Exit status 1 is expected when the evidence shows drift. It is not acceptable
   when `diagnostics/run-list-drift.json` has status `incomplete`.
4. Review every addition, lost GPU observation, stale node, and unobserved node
   in `run-list-drift.json`. Replace the maintained file with the candidate only
   when the evidence is complete and the differences are understood.

## Bootstrap a new Polars version locally

Use this only when no nightly telemetry exists yet.

1. Use a development environment containing the branch's built `cudf-polars`
   and the exact supported Polars version.
2. Shallow clone https://github.com/pola-rs/polars to a temporary directory with the matching `py-<version>` tag. Record its revision.
3. Unset `CUDF_POLARS_GPU_RUN_LIST`; discovery must run the full suite.
4. Run `ci/run_cudf_polars_polars_tests.sh` with `POLARS_DIR` set to the
   checkout root from step 2; otherwise it uses the repository's `polars/`
   directory. Pass:
   - `--engine spmd`
   - `--inject-gpu-engine-blocksize default`
   - `--run-slow`
   - one `--inject-gpu-engine-report=<path>` per run
   - matching `--cudf-polars-shard-id` and `--cudf-polars-num-shards` values if
     sharded
5. Every shard must use the same cuDF revision, Polars revision, dependencies,
   pytest options, and deselections. Every shard must exit zero. Resolve failures
   before generating the list; do not combine partial runs. If an environment
   requires additional deselections, justify them and apply them uniformly.
6. Preserve the reports under ignored `test-results/` storage, then generate:

   ```bash
   python ci/analyze_cudf_polars_fallbacks.py \
       --reports-dir test-results/gpu-selection-baseline/reports \
       --output-dir diagnostics \
       --audit-run-list ci/cudf_polars/gpu_test_run_list.json \
       --generate-run-list gpu-test-run-list-candidate.json
   ```

The analyzer verifies full-suite mode, zero exit statuses, complete per-test
telemetry, identical collection inventories, all shards, one cuDF revision, and
one upstream revision per Polars version. Do not bypass these checks.

## Remove entries for unsupported Polars versions

The analyzer retains every existing `<polars>|spmd|default` entry, so entries
for old versions remain after cudf-polars raises its minimum supported Polars
version. Remove them in the same PR that raises the lower bound of the `polars`
requirement in `python/cudf_polars/pyproject.toml`:

```bash
python .agents/skills/update-cudf-polars-polars-tests-selection/prune-run-list.py
```

The script reads the minimum supported version from the `polars` requirement
and rewrites `ci/cudf_polars/gpu_test_run_list.json` in place. Without `uv`, run
it with any Python 3.11+ environment that has `packaging` installed.

Entries are only removed for versions below the minimum. Entries for newer
versions than the upper bound are kept. Every version that PR CI can install
must still have an entry, so bootstrap any missing version before removing old
ones.

## Verify the update

1. Validate the JSON and inspect the target `<polars>|spmd|default` entry.
2. Confirm its baseline contains every source report and report SHA-256.
3. Rerun the analyzer against the candidate. It must report `current` with zero
   additions, lost GPU observations, and stale nodes.
4. Run the selected PR configuration by setting
   `CUDF_POLARS_GPU_RUN_LIST=ci/cudf_polars/gpu_test_run_list.json` and invoking
   `ci/run_cudf_polars_polars_tests.sh --engine spmd
   --inject-gpu-engine-blocksize default`.
5. Commit the reviewed manifest and report the selected-node count, full
   collection inventory, Polars version, and upstream revision.
