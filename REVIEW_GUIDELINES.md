# Shared cuDF Review Guidelines

These guidelines apply throughout the repository, alongside the review guidelines
for the affected language or package.

Be direct, concise, and minimal.

- Report concrete correctness, compatibility, performance, or test coverage risks.
  Give the triggering input or execution path and impact.
- Deduplicate by root cause, including findings in earlier review summaries and
  outside-diff comments.
- Distinguish inspected code, existing CI evidence, and checks actually executed.
- For new APIs, check callers, bindings, and documentation even if unchanged.
- Before requesting another test or benchmark, search for existing coverage and
  identify the missing behavior or measurement; prefer extending existing cases.
- Check that tests reach the claimed implementation and assert the intended
  behavior; a successful fallback or an unrelated exception can hide a regression.

## Severity

Use CRITICAL for severe failures such as memory corruption, crashes, or widespread
wrong results; HIGH for substantial correctness, compatibility, or performance
risks; and MEDIUM for bounded issues with a concrete consequence.

The tiers in language-specific guides describe typical impact, not automatic
severity assignments. Regression-evidence checks have no severity of their own:
classify any resulting finding by its demonstrated impact. Skip speculative
concerns and style nits.
