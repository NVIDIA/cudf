# Drafting issues

Every xfail/skip/fallback that stands for a real gap needs an issue the maintainers can act on without asking questions. Draft each as
a markdown file for the user to review; open an issue only when asked, and open issues in other repositories (for example
`pola-rs/polars`) only with an explicit go-ahead. Title format `[BUG] ...` or `[FEA] ...`; labels `bug` or
`feature request` plus `cudf-polars`. Use this structure (modelled on a well-received issue about hanging upstream tests):

1. **Description.** Plain-language symptom first: what happens (wrong result, exception, hang), how it shows up in CI, and what still works
   (other engine, default blocksize, plain Polars CPU), which narrows the cause to cudf-polars. No jargon the reader must decode.
2. **Affected tests.** The exact node ids, in a bullet list, with parameters, and say which are `slow` or otherwise need flags.
3. **How to reproduce.** One copy-pasteable command with every environment variable and flag, run from a stated checkout (for example
   `pola-rs/polars` at tag `py-X.Y.Z`, in `py-polars`), plus the observed result (exit code, error text, the contrast with the passing
   configuration). Prefer a minimal Polars snippet when the failure is not about the test harness.
4. **What I saw in CI** (only if CI has actually run it; for a local pre-merge upgrade write "not yet run in CI" or omit). Workflow run
   id, job names, commit, how many jobs hit it, and where it stopped. State plainly what you did **not** verify (for example "the other hangs may not share one cause"). Do not speculate beyond the evidence.
5. **Root cause / suspected location.** File and function when known (for example a stack dump pointing at a specific actor).
6. **Workaround.** Where the skip/xfail/fallback lives in the repo (file and dict name) and that it should be removed when the issue is fixed.
7. **Caveats.** Anything the workaround may be hiding (for example a skipped hang hides every test behind it).

Link the issue number in the xfail reason (`.../issues/NNNNN`) once the user has opened it. Until then use `TODO: <draft filename>`.

When the user asks you to open them: `gh issue create -R NVIDIA/cudf --title ... --body-file ... --label bug --label cudf-polars`
(use `--label libcudf` for libcudf reader/kernel bugs and `--label "feature request"` for `[FEA]`). End each body with a line linking the
upgrade PR ("Found while upgrading cudf-polars to Polars X.Y in <PR url>"). Verify the labels with `gh issue view <n> --json labels`, then
replace the `TODO` placeholders in the dicts with the issue URLs and commit. Group tests that share a cause into one issue.
