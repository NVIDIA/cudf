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

4. **Analyze the changes** against the checklist below, reading relevant source files as needed for context.

5. **Produce a structured review** using the output format at the bottom.

6. **Dump the structured review** to `.agents/reviews/<PR_NUMBER>/review.md`

---

## Review Checklist

For the detailed review checklist, read these files:
- **C++/CUDA**: `cpp/REVIEW_GUIDELINES.md`
- **Python**: `python/REVIEW_GUIDELINES.md`
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

Online libcudf API docs if needed: https://docs.rapids.ai/api/cudf/nightly/libcudf/

---

## Output Format

Structure your review as follows:

```markdown
## PR Review: <PR title>

**PR:** <link>
**Summary:** <1-2 sentence summary of what the PR does>

### Findings

#### Critical
- **[file:line]** Description of issue that must be fixed before merge.

#### Suggestions
- **[file:line]** Description of improvement to consider.

#### Nits
- **[file:line]** Minor style or formatting issue. Keep these minimal, don't suggest adding comments around every line of code or obvious logic.

#### Highlights
- Highlight well-written code, good test coverage, or clever solutions.

### Verdict
One of: **Approve**, **Request Changes**, or **Comment**
With a brief justification.
```

If there are no findings in a category, omit that category.
