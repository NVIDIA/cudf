# Shared cuDF Review Guidelines

These guidelines apply throughout the repository, alongside the review guidelines
for the affected language or package.

- Report concrete correctness, compatibility, performance, or test coverage risks.
  Give the triggering input or execution path and impact.
- Deduplicate by root cause, including findings in earlier review summaries and
  outside-diff comments.
- Distinguish inspected code, existing CI evidence, and checks actually executed.
- For new APIs, check callers, bindings, and documentation even if unchanged.
- Before requesting another test or benchmark, search for existing coverage and
  identify the missing behavior or measurement; prefer extending existing cases.
- Check that tests reach the claimed path: size thresholds, GPU versus fallback,
  codecs, partition counts, and exception types can allow a test to pass otherwise.
