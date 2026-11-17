---
name: contribute-cudf
description: Use this skill before writing or changing code or tests in the cudf repository, and again before opening or updating a pull request, to reuse existing code and keep the change small.
---

# Contribute to cuDF

cudf adds new code only when nothing existing does the job, and keeps what it adds concise. Run steps 1 and 2 on the files you plan to touch before writing code, and steps 3 to 6 on the diff before opening the PR and before each later push.

Push, open the PR, and post or edit PR text and replies only when the user tells you to (see "Automated and AI-assisted contributions" in `CONTRIBUTING.md`).

```bash
# REMOTE is the remote that tracks NVIDIA/cudf: `upstream` in a fork, `origin` in a direct clone.
# For a PR that targets a release branch, replace `main` with `release/X.Y` below.
REMOTE=upstream
git fetch $REMOTE main
# Set both again in each new shell, and $BASE after merging the target: an empty or stale value
# gives a wrong diff.
BASE=$(git merge-base $REMOTE/main HEAD)
git add -N <new files>                      # untracked files are otherwise missing from the diff
git diff --name-only $BASE                  # the touched files
git diff $BASE                              # the diff, including uncommitted work
```

As you work, consult the developer guide for the area (`cpp/doxygen/developer_guide/` for libcudf; the package's `developer_guide` or `developer_docs.md` under `docs/cudf/source/` for Python) and the `REVIEW_GUIDELINES.md` nearest the touched path. Apply them and the steps below with judgment: they guide the change and are not a checklist to satisfy line by line.

## 1. Read the history of the files you will touch

Read about five focused merged PRs that last changed the functions you will change, with their human review comments, and list the open PRs that touch the same files. [references/history.md](references/history.md) has the commands.

- A new file has no history: use the sibling file it is modelled on and the helpers it calls.
- Follow what reviewers settled on this code. Reintroduce something a reviewer removed only when the task needs it, and give the reason in the PR description.
- If an open PR changes the same functions (`gh pr diff <N>`), tell the user before writing.

## 2. Write only what does not exist

For each kernel, loop, functor, helper, type, fixture, or generator you are about to add, name the operation it performs and search for an existing one in the directories [references/reuse-map.md](references/reuse-map.md) lists for cudf, CCCL, and cuco. Do not skip the search: tell the user which directories you searched and what you found.

- A copy or near-copy of existing logic stays only when the existing code cannot give the required semantics, or with a measured performance, build-time, or binary-size gain from an A/B of the branch with and without it (run it as the `perf-compare-cudf` skill does). A fast path added for speed needs the same measurement.
- If the existing code is private to another file, share it when its responsibilities and dependencies fit both callers. Otherwise report the duplication to the user before widening the change.
- Two blocks in your own diff that differ by a constant, type, or flag become one helper or one parameter. Derive each value once.

## 3. Add only the tests that are missing

Keep test code small by extending and reusing what exists. Follow how the tests in each file are organized, and use judgment rather than a single rule.

- Find the existing tests of the function first (`git grep -n '<function under test>' $REMOTE/main -- cpp/tests python`). If one already fails when the new code is wrong, add nothing.
- Extend a test of related behavior (a row, type, or parameter, such as nested or nullable columns in a test of flat ones) before writing a new one, but do not grow a single test until it is hard to read. Move setup that several tests need into a helper.
- A bug fix or behavior change needs a test that fails without it.

## 4. Cut the diff down

Read `git diff $BASE` hunk by hunk, leaving code it only moves as it was.

- Keep the PR to one purpose. A small cleanup in code you are already changing is fine; keep such changes to a few percent of the diff and move anything larger to its own PR. Revert restyled statements, include churn, files your tools left behind, and anything added for a later PR.
- For each guard, fallback, cast, default, optional parameter, and branch the diff adds, name the caller or input that reaches it, and delete it if there is none. Users are the callers of a public API, so its argument checks, defaults, and unsupported-type failures stay. Delete unused includes, parameters, and locals, functions with no caller, and anything left from an approach you abandoned.
- Public APIs need the Doxygen or docstring the guides require. Internal and local helpers need no documentation beyond a comment where the code cannot explain itself, a line or two of why. Delete comments that restate the code or narrate the change, the bug, or the previous code.

## 5. Verify before pushing

- Fetch the target branch and list its new commits on the touched files (`git log --oneline $BASE..$REMOTE/main -- <touched files>`). If there are any, merge the target and repeat the search in step 2.
- Run `/review-cudf` on the diff and fix the findings you can confirm. Leave its `.agents/reviews/` output out of the commit.
- Commit, then run `pre-commit run --from-ref $BASE --to-ref HEAD`, which checks committed files only, and commit its fixes.
- Build and run the tests that exercise the changed functions, not only the suite you edited (see the `build-test-cudf` skill). For a widely included header, cover its nearest users and leave the rest to CI.

## 6. Write the PR

- Keep `.github/PULL_REQUEST_TEMPLATE.md` and its checklist. The title becomes the changelog line: imperative, user-facing names, and no claim beyond what the diff does.
- Description: one short paragraph on what changes, how, and why. Add `closes #N` when an issue exists, every public API or behavior change and whether it is breaking, the existing tests that cover the change if you added none, and overlapping open PRs.
- A measured gain goes in the description in one line. Draft the benchmark table for the user to post as a PR comment, with both commits, the GPU, and the exact command.
- Leave out test and validation logs, agent reports, and tool or co-author lines. Disclose AI assistance in one plain sentence, as `CONTRIBUTING.md` asks.
- Re-read the title and description against the diff before every push.
