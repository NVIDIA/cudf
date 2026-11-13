---
name: retarget-pr
description: Retarget a cuDF PR to a different base branch (e.g. main to release/YY.MM) by rebasing its commits and changing the PR base. Use when asked to retarget a cuDF PR; not for plain rebases onto the same base.
---

# Retarget PR

## Inputs

Require:
- cuDF PR URL or number (a bare number means `NVIDIA/cudf`).
- target branch, for example `release/26.08`. Verify it exists with `git ls-remote --heads <canonical-remote> <target-branch>`.

The canonical remote is whichever remote points at `NVIDIA/cudf` (match the URL, case-insensitively); it is often `upstream`.

## Workflow

1. Inspect the worktree, remotes, PR metadata, current base, head repository and branch, and PR tip SHA.
2. Fetch the current base, target branch, and PR head. Create a local, non-detached branch from the PR head in a separate worktree (`git worktree add -b <local-branch> <path> <pr-tip>`) and run the remaining steps there. This leaves unrelated edits in the current worktree untouched, so no stashing is needed.
3. Confirm signing works without displaying keys or tokens: read `gpg.format`; for `ssh`, probe the configured `user.signingkey` with a temporary `ssh-keygen -Y sign` (the GPG keyring need not hold a secret key); otherwise confirm a usable signing key. If signing cannot work, ask the user whether unsigned rewritten commits are ok. If so, use `--no-gpg-sign` in step 5 and skip the signature check in step 8; if not, help them set up a key. Never run `gh auth login`.
4. Find the fork point `F = git merge-base <canonical-remote>/<current-base> <pr-tip>`. If `git merge-base --is-ancestor F <canonical-remote>/<target-branch>` succeeds, the PR has picked up nothing the target lacks, so no rewrite or force-push is needed. First check that the PR still merges cleanly into the target (`git merge-tree --write-tree <canonical-remote>/<target-branch> <pr-tip>` exits 0); if it conflicts, stop and ask the user to resolve. Otherwise skip to step 10 and only change the PR base.
5. Retarget in a single rebase, replaying only the non-merge PR commits (`F..<pr-tip>`) directly onto the target. Do not first replay onto the current base tip: that can conflict on changes the target branch lacks, then conflict again on the same hunk.

   ```bash
   git rebase --no-rebase-merges --gpg-sign --onto <canonical-remote>/<target-branch> F
   ```

   `--no-rebase-merges` is required because a configured `rebase.rebaseMerges=true` would otherwise keep merges; the result must contain no merge commits.
6. Compare the file-scoped changes of the original and retargeted PR, ignoring `index` and `@@` lines (which legitimately differ when the target's content differs). Using `-U0 --binary` keeps file paths (`diff --git` headers), binary changes, and changed lines, and drops context:

   ```bash
   diff <(git diff -U0 --binary F <original-pr-tip> | grep -v -e '^index ' -e '^@@ ') \
        <(git diff -U0 --binary <canonical-remote>/<target-branch> HEAD | grep -v -e '^index ' -e '^@@ ')
   ```

   Any difference is a change carried only by a dropped merge commit (or lost in conflict resolution); stop and ask the user.
7. If a rebase conflicts, stop and ask the user how to resolve it, including the conflicted files and current commit. Abort only at the user's direction or when the rebase was demonstrably the wrong operation; explain why before retrying correctly.
8. Verify:
   - target branch is an ancestor of `HEAD`;
   - the range has the expected count of PR non-merge commits and zero merge commits;
   - `git diff --check <canonical-remote>/<target-branch>...HEAD` passes;
   - every rewritten commit has a `gpgsig` header (unless unsigned was agreed).
9. Report the worktree path and branch, target base, commit count, conflicts, and signing result. Provide but do not execute a force-with-lease command targeting the PR head repository:

   ```bash
   git push --force-with-lease=refs/heads/<head-branch>:<original-pr-tip-sha> \
     https://github.com/<pr-head-owner>/<repo>.git \
     HEAD:refs/heads/<head-branch>
   ```

   The caller needs write access to the PR head repository; pushing to any other fork does not update the PR.
10. Only after the push succeeds (the user pushed, or asked you to push and it succeeded), confirm the remote PR head equals local `HEAD` with `gh pr view <pr> --repo NVIDIA/cudf --json headRefOid`. A rejected push (for example a failed lease) leaves the old head, so do not change the base then. When step 4 skipped the rewrite, there is nothing to push. Then change the PR base and confirm the commit count:

   ```bash
   gh pr edit <pr> --repo NVIDIA/cudf --base <target-branch>
   gh pr view <pr> --repo NVIDIA/cudf --json baseRefName,commits
   ```

   Remind the user to check the PR's labels and milestone, which often differ for release-branch PRs.
