---
name: review-cudf
description: Review cuDF pull requests or local diffs, using repository guidelines and targeted execution to investigate correctness, compatibility, and performance risks.
---

# Review cuDF

Review as a cuDF maintainer responsible for evidence-backed findings that add to
existing human and CodeRabbit reviews. The repository is NVIDIA/cudf.

## Context

- Fetch the PR metadata, base/head SHAs, diff, CI results, and existing review
  threads and submissions, including resolved and outside-diff comments.
  For local changes, record the compared revisions and uncommitted diff.
- Use available GitHub tools or authenticated gh without printing credentials.
  If Git authentication fails because SSH forwarding disconnected, stop and ask
  the user to restart after reconnecting it.
- Read the REVIEW_GUIDELINES.md files applicable to the changed paths, including
  package-specific guides under python/pylibcudf and python/cudf_polars when present.
  For libcudf conventions, the authority is
  cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md. Consult TESTING.md,
  BENCHMARKING.md, DOCUMENTATION.md, or PROFILING.md there when relevant.
  Keep code-review policy in those shared guides, which CodeRabbit also consumes.

## Investigation

- Trace changed behavior through callers, bindings, ownership, streams, and
  existing tests. Focus on the risks introduced by this diff.
- Use the smallest experiment that resolves a material uncertainty: targeted
  builds, regression inputs, CPU/reference comparisons, sanitizers, or benchmarks.
  Use the applicable build/test or performance skill and supported environment.
  A review need not run the full suite or manufacture a finding.
- Verify that execution uses the reviewed sources and matching native libraries.
  A passing test is useful only if it reaches the intended implementation:
  check collection, skips/xfails, fallback, size thresholds, codecs, partitions,
  and selected build features as relevant.
- For suspected regressions, compare the same reproducer on base and head.
  For performance claims, control hardware/load and cache state, include the
  affected workload axes, and preserve raw measurements.
- Extend existing coverage when useful; keep review experiments isolated from
  the developer's edits. Do not change the submitted patch as part of review.
- If execution is unavailable, report the limitation and distinguish it from a
  passing check. Code reasoning can establish a finding without execution.

## Result

Save the review to .agents/reviews/<PR_NUMBER>/review.md (use a descriptive name
for local diffs). Include the reviewed base/head SHAs, then findings ordered by
impact. Each finding needs file/line, trigger, consequence, and supporting
reasoning or a reproducer. Identify whether it is new or confirms an existing
finding; keep confirmations with evidence rather than repeating the complaint.
Include checks actually run and material gaps. Omit empty categories, praise,
style nits, and an approval verdict unsupported by the investigation.
Posting to GitHub requires authorization from the user or invoking workflow.
