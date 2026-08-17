---
name: reproduce-ci
description: Reproduce cudf CI failures locally. Provide a GitHub Actions job URL to auto-discover parameters, or supply the container image and CI script directly.
---

# Reproducing cudf CI Failures Locally

For full background, see the [RAPIDS docs on reproducing CI](https://docs.rapids.ai/resources/reproducing-ci/).

## Prerequisites

Verify all of the following before proceeding. Stop and report if any are missing.

1. **`docker` CLI** available and daemon running:
   ```bash
   docker info
   ```

2. **`gh` CLI** authenticated — run `gh auth status`. If not authenticated, guide the user to run:
   ```bash
   gh auth login
   ```
   The token needs `repo` scope. Do **not** run `gh auth token` from within the agent.

3. **Working directory** is a `cudf` checkout with the correct PR commit checked out.

4. **Consistency check** — Read `run.sh` and verify it is still consistent with the [RAPIDS reproducing-CI guide](https://docs.rapids.ai/resources/reproducing-ci/). Key things to check:
   - Environment variables passed to docker (`RAPIDS_BUILD_TYPE`, `RAPIDS_REPOSITORY`, `RAPIDS_REF_NAME`, `GH_TOKEN`)
   - The `docker run` flags (`--pull=always`, `--volume $PWD:/repo`, `--workdir /repo`)
   - The CI script invocation pattern
   - If anything is out of date, modify `run.sh` as needed and inform the user of the changes

---

## Step 1: Determine Parameters

You need three arguments for `run.sh`: **container image**, **CI script**, and **PR number**. Plus an optional `--gpu` flag. Use one of the two options below to obtain them.

### Option A: Discover from a GitHub Actions Job URL

Use this option when the user provides a URL like:
`https://github.com/rapidsai/cudf/actions/runs/<run_id>/job/<job_id>?pr=<pr_number>`

**Parse the Job URL:**

Run the helper script to extract the run ID, job ID, and optional PR number:

```bash
python3 .agents/skills/reproduce-ci/parse-job-url.py "<URL>"
```

This outputs shell-friendly `KEY=VALUE` lines:
```
RUN_ID=XXXXXXXXXX
JOB_ID=XXXXXXXXX
PR_NUMBER=XXXX       # only if pr= query param is present
```

If the script is unavailable, extract manually:
- **Run ID**: path segment after `/runs/`
- **Job ID**: path segment after `/job/`
- **PR number**: `pr` query parameter (may be absent)

**Fetch Job Metadata and Logs:**

Use the extracted IDs to get job details:

```bash
gh api repos/rapidsai/cudf/actions/runs/$RUN_ID/jobs \
  --jq ".jobs[] | select(.id == $JOB_ID)"
```

From the job JSON, note:
- `name` — job name (e.g. `conda-cpp-build / 12.9.1, 3.11, amd64, rockylinux8`)
- `steps[].name` and `steps[].conclusion` — which steps ran and their outcome

Download the full job log:

```bash
gh run view "$RUN_ID" --repo rapidsai/cudf --job "$JOB_ID" --log > /tmp/ci_job_log.txt
```

Read through the log to identify:
1. The **container image** — look for the `Initialize Containers` step or docker pull lines (e.g. `rapidsai/ci-conda:<tag>`)
2. The **CI script** executed (e.g. `./ci/build_cpp.sh`, `./ci/test_python.sh`)
3. The **final outcome** — whether the job passed or failed
4. If failed, the **exact failure** — error messages, test names, assertion text

Now you have the three arguments needed for Step 2.

### Option B: Parameters Provided Directly

Use this option when the user already knows the container image and CI script.

The container image tag corresponds to the current RAPIDS version. Derive it from the `VERSION` file at the repo root:
```bash
RAPIDS_VERSION=$(head -1 VERSION | cut -d. -f1,2)
```

All CI job definitions live in `.github/workflows/pr.yaml`. Each job specifies:
- `container_image`: the Docker image to use
- `script`: the CI script to run
- `node_type`: determines whether a GPU is needed (`gpu-*` → pass `--gpu`)

```bash
grep -A 10 'job-name-here:' .github/workflows/pr.yaml
```

Alternatively, inspect the "Initialize Containers" step in the GitHub Actions job log for the exact image.

Now you have the three arguments needed for Step 2.

---

## Step 2: Run the Reproduction

Invoke `run.sh` with the parameters determined in Step 1:

```bash
.agents/skills/reproduce-ci/run.sh <container-image> <ci-script> <pr-number> [--gpu] [--timeout <minutes>] [--dry-run]
```

Use `--dry-run` first to preview the exact docker command before executing:

```bash
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:${RAPIDS_VERSION}-latest ci/build_cpp.sh 22538 --dry-run
```

Examples:
```bash
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:$(head -1 VERSION | cut -d. -f1,2)-latest ci/test_cmake.sh 22538
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:$(head -1 VERSION | cut -d. -f1,2)-latest ci/test_java.sh 22538 --gpu
.agents/skills/reproduce-ci/run.sh rapidsai/citestwheel:$(head -1 VERSION | cut -d. -f1,2)-latest "ci/cudf_pandas_scripts/pandas-tests/run.sh pr" 22538 --gpu
```

Determine whether `--gpu` is needed: check the `node_type` field in the workflow YAML or job metadata — `gpu-*` values indicate a GPU is required.

The script automatically detects `RAPIDS_SHA` (the PR's head commit) using `gh pr view`.
This is required by CI helper scripts inside the container to locate build artifacts.
If auto-detection fails (e.g., `gh` is not authenticated), set it manually:
```bash
export RAPIDS_SHA=$(gh pr view <pr-number> --repo rapidsai/cudf --json commits --jq '.commits[-1].oid')
```
The script launches a detached container, runs the CI script, and leaves the container running for inspection.
After `--timeout` minutes of idle (default: 30), the container is automatically removed.
```bash
docker exec -it cudf-ci-repro bash   # inspect interactively
docker rm -f cudf-ci-repro           # clean up manually before timeout
```

---

## Step 3: Analyze Output

After `run.sh` completes, analyze the local output against the CI outcome:

1. **Compare results** — did the local run match the CI outcome (both pass, both fail, or diverge)?
2. **Note discrepancies** — if local and CI results differ, call out likely causes:
   - GPU driver version differences (check CI log vs local `nvidia-smi`)
   - Environment differences (OS, CUDA version, conda env)
   - Timing or flaky tests
3. **Summarize** — give the user a clear summary: what ran, what the outcome was, and whether it matched CI.

#### If the job failed (in CI or locally or both)

4. Summarize the **root cause** based on the error output.
5. **Ask the user** if they would like help investigating a fix.
6. If yes, launch a sub-agent (Task tool) with the failure log, the relevant CI script paths, the error context, and any source files implicated in the failure.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GIT_DESCRIBE_NUMBER is undefined` | `git fetch https://github.com/rapidsai/cudf.git --tags` |
| Interactive GitHub auth prompt inside container | Ensure `GH_TOKEN` is set — `run.sh` passes it through automatically via `gh auth token` |
| GPU driver mismatch causing test differences | Note driver version from CI log; compare with local `nvidia-smi` |
| Log download returns empty or 403 | Verify `gh auth status` has `repo` scope; re-auth with `gh auth login` if needed |
| `RAPIDS_SHA` is unbound | The script auto-detects this from the PR via `gh`. If auto-detection fails, set `RAPIDS_SHA=<commit>` in your environment before running. |
