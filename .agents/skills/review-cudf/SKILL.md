---
name: review-cudf
description: Review cuDF pull requests or local diffs, using repository guidelines and targeted execution to investigate correctness, compatibility, and performance risks.
---

# Review cuDF

Review as a cuDF maintainer responsible for evidence-backed findings that add to
existing human and CodeRabbit reviews. The repository is NVIDIA/cudf.

Accept a PR link or number, a checked-out PR, or a local diff.

## Context

- Fetch the PR metadata, base/head SHAs, diff, CI results, and existing review
  threads and submissions, including resolved and outside-diff comments.
  For local changes, record the compared revisions and uncommitted diff.
- Use available GitHub tools or authenticated gh without printing credentials.
  If authentication fails, stop and report it. If those tools are unavailable,
  use the public PR diff or REST API, or fetch the PR ref from a cuDF remote.
- Read the root REVIEW_GUIDELINES.md and the guides applicable to the changed paths,
  including package-specific guides under python/pylibcudf and python/cudf_polars
  when present.
  For libcudf conventions, the authority is
  cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md. Consult TESTING.md,
  BENCHMARKING.md, DOCUMENTATION.md, or PROFILING.md there when relevant.
  Keep code-review policy in those shared guides, which CodeRabbit also consumes.

For example, with GitHub CLI:

```bash
gh pr view <PR_NUMBER> --repo NVIDIA/cudf --json title,body,baseRefOid,headRefOid,files,statusCheckRollup
gh pr diff <PR_NUMBER> --repo NVIDIA/cudf
```

## Investigation

- Trace changed behavior through callers, bindings, ownership, streams, and
  existing tests. Focus on the risks introduced by this diff.
- Use the smallest experiment that resolves a material uncertainty: targeted
  builds, regression inputs, CPU/reference comparisons, sanitizers, or benchmarks.
  Use `build-test-cudf` for builds/tests or `perf-compare-cudf` for performance
  comparisons, under .agents/skills/, in a supported environment.
  A review need not run the full suite or manufacture a finding.
- Verify that execution uses the reviewed sources and matching native libraries.
  Record relevant test selection, skips/xfails, dispatch conditions, and fallback
  settings when establishing which implementation ran.
- For suspected regressions, compare the same reproducer on base and head.
  For performance claims, control hardware/load and cache state, include the
  affected workload axes, and preserve raw measurements.
- Extend existing coverage when useful; keep review experiments isolated from
  the developer's edits. Do not change the submitted patch as part of review.

## Result

Save the review to .agents/reviews/<PR_NUMBER>/review.md (use a descriptive name
for local diffs) with this structure:

```markdown
**Base:** compared revision
**Head:** reviewed revision, including any uncommitted changes

## Findings
List findings by severity, or state "No findings".
For each: severity, file:line, trigger, consequence, evidence or reproducer,
and whether it is new or confirms an existing finding (link the original).

## Checks run
Commands and results, including evidence added to existing findings.

## Gaps
Material uncertainties and checks that could not be run, or "None".
```

Keep confirmations with their evidence rather than repeating the complaint.
Posting to GitHub requires authorization from the user or invoking workflow.
