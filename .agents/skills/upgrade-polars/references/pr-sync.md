# Resuming work on an open PR (keep local and GitHub in sync)

The branch on GitHub can change while you are away: the user squashes or rebases, force-pushes, accepts a review suggestion, edits
the PR in the web UI, or a reviewer pushes. Your local checkout does not know until you fetch. Run this check before you edit,
commit or push anything on an existing PR, and again right before every push.

1. Fetch and compare. `<branch>` is the PR branch:
   ```bash
   git fetch origin <branch>
   git status -sb
   git log --oneline HEAD..origin/<branch>     # commits on GitHub that you do not have (remote ahead)
   git log --oneline origin/<branch>..HEAD     # commits you have that GitHub does not (local ahead)
   ```
2. Act on what you find:
   - **Both empty:** in sync; continue.
   - **Only the remote is ahead:** `git pull --ff-only origin <branch>`, then read what arrived (`git log -p ORIG_HEAD..HEAD`) before
     you build on it.
   - **Only you are ahead:** push normally, with no force.
   - **Both are non-empty (diverged):** find out whether history was rewritten. Test whether the remote still contains the last commit
     you pushed (`git merge-base --is-ancestor <last-pushed-sha> origin/<branch>`; `git merge-base HEAD origin/<branch>` should print
     that sha).
     - **It does** (the user merged `main`, added commits, or accepted a suggestion): nothing was rewritten. Rebase your unpushed commits
       onto the remote with `git rebase -S origin/<branch>` (`-S` keeps them signed), tell the user what arrived, and read the incoming
       diff for files you also touch. Check that your last verification still applies: a merge of `main` can change code your tests cover.
     - **It does not** (a squash, rebase or force-push): stop and tell the user. Do not merge, and do not force-push on your own. With the
       user's agreement: first keep a copy (`git branch backup/<branch>-$(date +%F)`), then move only your new commits onto the remote with
       `git cherry-pick` (your older commits are already inside the squash), resolve conflicts toward the remote side's intent, and check
       the result with `git diff origin/<branch>`.
3. Never `git push --force`. Use `--force-with-lease` only when the user explicitly asks for a history rewrite (for example to squash),
   because it refuses to overwrite a remote that moved since your last fetch.
4. Re-read the PR before changing anything on it: `gh pr view <n> --json headRefOid,body,reviews,comments,reviewDecision`, the inline review
   comments (`gh api repos/<org>/<repo>/pulls/<n>/comments`) and CI (`gh pr checks <n>`). The user may have edited the description, so merge
   your change into the current body; do not overwrite it from an old local file. The same applies to issues (`gh issue view` before
   `gh issue edit`).
5. After pushing, confirm the remote head is your commit (`git ls-remote origin <branch>`). Workflows on a draft PR do not start on their
   own: the user has to comment `/ok to test <sha>` for the new head.
6. After a squash by the user, your old commit hashes are no longer in the PR. Refer to commits by subject, and fix any note that quoted a
   hash. State in the PR which commit the last full test run was on, so later edits can be judged against it.
7. A branch can be checked out in only one worktree, so use the worktree that has it. Do not use bare `git stash` (the stash is shared
   across worktrees); commit instead.
8. When you address a review comment, say what you changed. Replying to or resolving threads is the user's call unless they ask you to.
