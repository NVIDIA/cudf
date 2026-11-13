---
name: retarget-pr
description: Retarget a cuDF PR to a different base branch (e.g. main to release/YY.MM) by rebasing its commits and changing the PR base. Use when asked to retarget a cuDF PR; not for plain rebases onto the same base.
---

# Retarget PR

Require a cuDF PR URL or number (bare number means `NVIDIA/cudf`) and a target branch such as `release/26.08`.

## Workflow

Run `.agents/skills/retarget-pr/retarget.sh <pr> <target-branch>` from a cuDF checkout. It fetches, picks the right path, rebases in a separate worktree under the repository's git dir (so local edits are untouched), verifies, and prints the follow-up commands. It never pushes or edits the PR. Act on its exit code:

- **0, `MODE=base-only`**: the PR has nothing the target lacks, so no rewrite or push. Run the printed `gh api -X PATCH` command to change the base (more reliable than `gh pr edit --base`, which can fail on a Projects (classic) GraphQL error).
- **0, `MODE=rewrite`**: report the worktree, branch, commit count, `unsigned_commits`, and `whitespace_errors` (reported, not fatal). Give the user the printed force-with-lease push command (its target is the PR's own head repo and branch, taken from `gh pr view`); do not run it unless asked. The user needs write access to the PR head repository, and pushing to any other fork does not update the PR.
- **1**: usage or environment error (for example the target branch does not exist, or the PR head changed since the first run). Report the script's message.
- **2**: the PR conflicts with the target; the script prints the conflicting files and creates nothing. Show them and ask the user to resolve.
- **3**: rebase conflict; the worktree is left mid-rebase (`git -C <worktree> status` lists the conflicted files and the commit being applied). Report them and ask how to resolve. After the user's resolution and `git -C <worktree> rebase --continue`, rerun with `--resume` to verify and get the push command. Abort (`git -C <worktree> rebase --abort`) only at the user's direction.
- **4**: no working signing key. Ask whether unsigned rewritten commits are ok. If yes, rerun with `--no-sign`; if no, help set up a key. Never run `gh auth login`.
- **5**: verification failed. The retargeted tree must equal the PR's changes merged onto the target; the script prints a `git diff --stat` of any difference (for example a change carried only by a dropped merge commit), or of the whole result when a conflict resolution cannot be checked automatically. Show it to the user and ask. If they accept it, rerun with `--resume --accept-diff`; otherwise stop.

After a push succeeds, confirm `gh pr view <pr> --repo NVIDIA/cudf --json headRefOid` equals the local `HEAD` (it can lag briefly after a push, so re-poll once before concluding the push failed). A rejected push (for example a failed lease) leaves the old head, so do not change the base then. Then run the printed `gh api -X PATCH` command, confirm with `gh pr view <pr> --repo NVIDIA/cudf --json baseRefName,commits`, and remind the user to check the PR's labels and milestone, which often differ for release-branch PRs.

The script leaves its worktree and branch in place: on exit 3 or 5 for inspection, and on `MODE=rewrite` until the push and base change are done. Once the run is finished or the user abandons it, remove them with `git worktree remove --force <worktree>` and `git branch -D <branch>`.
