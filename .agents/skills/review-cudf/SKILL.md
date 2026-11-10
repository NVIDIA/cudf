---
name: review-cudf
description: Use this skill to review GitHub pull requests for cudf
---

Use this skill when the user invokes `/review-cudf` with:
- a cudf GitHub PR link
- currently checked out cudf PR
- specified cudf code changes or a diff

cudf GitHub repository is located at: https://github.com/NVIDIA/cudf

# Review cuDF Pull Request

1. **Fetch PR metadata and diff**

```bash
gh pr view <PR_NUMBER> --repo NVIDIA/cudf --json title,body,files,additions,deletions,baseRefName,headRefName
gh pr diff <PR_NUMBER> --repo NVIDIA/cudf
```

Hint: Check if `GH_TOKEN` (or GitHub CLI auth) is already configured in the environment (for example via your secret manager) so `gh` can authenticate and bypass rate limits; do not run `gh auth token` from within the agent. If `gh` auth is unavailable, fall back to GitHub's raw diff/patch URLs, `git fetch` of the PR ref, unauthenticated GitHub REST API with `curl`, or any other available methods.

2. **Fetch review comments already posted** for context on what's already been suggested and need not be repeated.

3. **Read the Developer Guide** (`cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md`) — it is the authoritative reference for libcudf conventions. All rules in the guide apply during review. The checklist below calls out the most review-relevant rules and adds items **not** covered by the guide.

4. **Gather the recent history of the touched files** so the review builds on what reviewers already decided there instead of repeating or contradicting it.

```bash
# Files touched by the PR. The JSON `files` list stops at 100 entries; for larger PRs use
# `gh api repos/NVIDIA/cudf/pulls/<PR_NUMBER>/files --paginate --jq '.[].filename'`.
gh pr view <PR_NUMBER> --repo NVIDIA/cudf --json files --jq '.files[].path'

# The ~10 most recent merged PRs that touched those files (squash-merge subjects carry the PR number).
# Use the remote that tracks NVIDIA/cudf (`upstream` in a fork checkout, otherwise `origin`).
git fetch upstream main
git log --format=%s -n 80 upstream/main -- <touched files> | grep -oE '#[0-9]+' | awk '!seen[$0]++' | head -10

# For each of those PRs: description, review summaries, and inline comments.
gh pr view <N> --repo NVIDIA/cudf --json title,author,mergedAt,body
gh api repos/NVIDIA/cudf/pulls/<N>/reviews --jq '.[] | select(.body != "") | "\(.user.login) [\(.state)]: \(.body)"'
gh api repos/NVIDIA/cudf/pulls/<N>/comments --paginate --jq '.[] | "\(.user.login) \(.path):\(.line // .original_line): \(.body)"'

# Open PRs that touch the same files (one query, about 20 s; the same 100-file cap applies per PR).
gh pr list --repo NVIDIA/cudf --state open --limit 400 --json number,title,isDraft,files

# Commits on main since the PR's base that touch the same files (the branch may predate a relevant fix).
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git log --oneline $(git merge-base pr-<PR_NUMBER> upstream/main)..upstream/main -- <touched files>
```

- If a file has fewer than ~5 PRs in its history, widen the `git log` to its folder.
- Record four things: decisions reviewers already settled on these files (do not reopen them), recurring themes (check this PR against them), open PRs that overlap (ordering or conflict risk), and commits on `main` since the PR's base that touch the same files.

5. **Check every new test against existing coverage.** cudf has thousands of gtest cases and hundreds of pytest files, so a new test often repeats one that already exists in the same file or a sibling file.

```bash
# New C++ test cases and new pytest functions in the diff.
gh pr diff <PR_NUMBER> --repo NVIDIA/cudf | grep -E '^\+\s*(TEST_F|TYPED_TEST|TEST_P|TEST)\(|^\+\s*def test_'
```

For each new test:
- Identify the production function or code path it exercises. Follow shared helpers: a validation function called by several public APIs is usually already tested through one of them.
- Search the base version of the same test file and the other files in that test directory for tests that exercise the same function with the same class of input (empty, nulls, malformed, size mismatch, ...). Matching names or identical literal inputs are leads, not conclusions; read both test bodies before deciding.
- Check whether the case belongs in an existing type list (`TYPED_TEST_SUITE`), `INSTANTIATE_TEST_SUITE_P` list, or `@pytest.mark.parametrize` list instead of a new test function.
- Report a duplicate only when you can cite the covering test by name and `file:line`, and say what, if anything, the new test adds beyond it (keep that part, drop the rest).

6. **Analyze the changes** against the checklist below, reading relevant source files as needed for context. Apply the recurring themes from step 4.

7. **Produce a structured review** using the output format at the bottom.

8. **Dump the structured review** to `.agents/reviews/<PR_NUMBER>/review.md`

---

## Review Checklist

For the detailed review checklist, read these files:
- **C++/CUDA**: `cpp/REVIEW_GUIDELINES.md`
- **Python**: `python/REVIEW_GUIDELINES.md`
- **pylibcudf**: `python/pylibcudf/REVIEW_GUIDELINES.md`
- **cudf_polars**: `python/cudf_polars/REVIEW_GUIDELINES.md`
- **Developer Guide**: `cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md`
- **Testing Guide**: `cpp/doxygen/developer_guide/TESTING.md`
- **Benchmarking Guide**: `cpp/doxygen/developer_guide/BENCHMARKING.md`
- **Documentation Guide**: `cpp/doxygen/developer_guide/DOCUMENTATION.md`

Use all applicable checklist items from those files when reviewing the PR. The Developer Guide is the authoritative reference for libcudf conventions — all rules in the guide apply during review.

---

## Reference Material

| Topic | Path |
|-------|------|
| Developer guide | `cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md` |
| Testing guide | `cpp/doxygen/developer_guide/TESTING.md` |
| Benchmarking guide | `cpp/doxygen/developer_guide/BENCHMARKING.md` |
| Documentation guide | `cpp/doxygen/developer_guide/DOCUMENTATION.md` |
| Profiling guide | `cpp/doxygen/developer_guide/PROFILING.md` |

Online libcudf API docs if needed: https://docs.nvidia.com/cudf/latest/libcudf/

---

## Output Format

Structure your review as follows:

```markdown
## PR Review: <PR title>

**PR:** <link>
**Summary:** <1-2 sentence summary of what the PR does>
**Context:** <merged PRs consulted for the touched files (numbers), open PRs touching the same files, commits on main since the PR base>

### Findings

#### Critical
- **[file:line]** Description of issue that must be fixed before merge.

#### Suggestions
- **[file:line]** Description of improvement to consider.
- **[test file:line]** Redundant test: `<Fixture.Name>` (file:line) already covers this path; keep only the <new aspect> case.

#### Nits
- **[file:line]** Minor style or formatting issue. Keep these minimal, don't suggest adding comments around every line of code or obvious logic.

#### Highlights
- Highlight well-written code, good test coverage, or clever solutions.

### Prior review themes
- <theme reviewers applied on these files> (#<PR>): <how this PR fares against it>

### Verdict
One of: **Approve**, **Request Changes**, or **Comment**
With a brief justification.
```

If there are no findings in a category, omit that category.
