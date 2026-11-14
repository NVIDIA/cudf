#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Retarget a cuDF PR to another base branch. Never pushes or edits the PR.
# Usage: retarget.sh <pr-number|url> <target-branch> [--no-sign] [--accept-diff] [--resume]
#   --accept-diff: user reviewed and accepted the exit 5 difference
#   --resume:      verify an existing worktree after the user resolved a rebase conflict
# Exit: 0 ok | 1 usage/env error | 2 PR conflicts with target | 3 rebase conflict
#       4 signing unavailable (ask user, rerun with --no-sign) | 5 verification failed
set -euo pipefail

die() { echo "error: $1" >&2; exit "${2:-1}"; }
trap 'die "unexpected failure at line $LINENO"' ERR
q() { printf '%q' "$1"; }  # shell-quote values (e.g. author-controlled branch names) in printed commands
valid_branch() { [[ $1 != -* ]] && git check-ref-format --branch "$1" >/dev/null 2>&1; }

[[ $# -ge 2 ]] || die "usage: $0 <pr> <target-branch> [--no-sign] [--accept-diff] [--resume]"
[[ $1 =~ ^([0-9]+)$ || ${1,,} =~ github\.com/nvidia/cudf/pull/([0-9]+)(/.*)?$ ]] || die "not a cuDF PR number or URL: $1"
pr=${BASH_REMATCH[1]}; target=$2; shift 2
valid_branch "$target" || die "invalid target branch name: $target"
sign=--gpg-sign; accept_diff=0; resume=0
for flag in "$@"; do
  case $flag in
    --no-sign) sign=--no-gpg-sign ;; --accept-diff) accept_diff=1 ;; --resume) resume=1 ;;
    *) die "unknown flag $flag" ;;
  esac
done
repo=NVIDIA/cudf

remote=$(git remote -v | awk 'tolower($2) ~ /github\.com[:\/]nvidia\/cudf(\.git)?$/ {print $1; exit}')
[[ -n $remote ]] || die "no remote points at $repo"

info=$(gh pr view "$pr" --repo $repo \
  --json baseRefName,headRefOid,headRepositoryOwner,headRefName,headRepository \
  --jq '[.baseRefName,.headRefOid,.headRepositoryOwner.login,.headRefName,.headRepository.name]|join("\u001f")') \
  || die "gh pr view $pr failed"
IFS=$'\x1f' read -r base_branch pr_tip head_owner head_branch head_repo <<<"$info"
[[ $head_owner =~ ^[A-Za-z0-9._-]+$ && $head_repo =~ ^[A-Za-z0-9._-]+$ ]] || die "PR head repository is unavailable (deleted fork?)"
valid_branch "$base_branch" && valid_branch "$head_branch" || die "unexpected branch name from gh"

git fetch -q "$remote" "+refs/heads/$target:refs/remotes/$remote/$target" || die "target branch $target not found on $remote"
git fetch -q "$remote" "+refs/heads/$base_branch:refs/remotes/$remote/$base_branch" || die "cannot fetch base branch $base_branch"
git fetch -q "$remote" "refs/pull/$pr/head" || die "cannot fetch PR head"
[[ $(git rev-parse FETCH_HEAD) == "$pr_tip" ]] || die "PR head changed while running; rerun"
target_ref=$remote/$target
fork_point=$(git merge-base "$remote/$base_branch" "$pr_tip")

if [[ $resume == 0 ]] && git merge-base --is-ancestor "$fork_point" "$target_ref"; then
  git merge-tree --write-tree --name-only "$target_ref" "$pr_tip" >&2 || die "PR conflicts with $target; user must resolve" 2
  echo "MODE=base-only (PR has nothing the target lacks; no rewrite or push needed)"
  echo "NEXT: gh api -X PATCH repos/$repo/pulls/$pr -f base=$(q "$target")"
  exit 0
fi

# Inside this repo's git dir, so other clones and worktrees cannot collide with it.
worktree=$(git rev-parse --path-format=absolute --git-common-dir)/retarget-pr/$pr
branch=retarget-pr-$pr-${target//\//-}
if [[ $resume == 1 ]]; then
  [[ -d $worktree ]] || die "no worktree to resume at $worktree"
  cd "$worktree"
  [[ $(git branch --show-current) == "$branch" ]] || die "rebase still in progress or wrong branch in $worktree" 3
  # Verify against what the first run rebased, and refuse if the PR moved since (the lease would clobber it).
  [[ $(git config "branch.$branch.retargetTip") == "$pr_tip" ]] \
    || die "PR head changed since the first run; remove the worktree and branch, then rerun"
  onto=$(git config "branch.$branch.retargetOnto"); fork_point=$(git config "branch.$branch.retargetFork")
else
  git worktree prune
  [[ ! -e $worktree ]] || die "$worktree exists; rerun with --resume, or remove it and branch $branch"
  ! git show-ref -q --verify "refs/heads/$branch" || die "branch $branch exists from an earlier run; delete it or rerun with --resume"
  if [[ $sign == --gpg-sign ]]; then  # let git itself sign, so the probe uses exactly the rebase's signing config
    git commit-tree -S -m probe "$pr_tip^{tree}" >/dev/null 2>&1 || die "signing unavailable; ask user, then rerun with --no-sign" 4
  fi
  onto=$(git rev-parse "$target_ref")
  git worktree add -q -b "$branch" "$worktree" "$pr_tip"
  git config "branch.$branch.retargetTip" "$pr_tip"
  git config "branch.$branch.retargetOnto" "$onto"
  git config "branch.$branch.retargetFork" "$fork_point"
  cd "$worktree"
  if ! git rebase -q --no-rebase-merges $sign --onto "$onto" "$fork_point"; then
    [[ -n $(git ls-files -u) ]] && die "rebase conflict in $worktree; ask user, then rerun with --resume" 3
    die "rebase stopped without a conflict (see git output above); worktree $worktree kept"
  fi
fi

# The retargeted tree must equal the PR's changes (fork point to tip, merge-only changes included)
# merged onto the target. Any difference is a dropped or altered change.
kept="worktree $worktree and branch $branch kept"
if expected=$(git merge-tree --write-tree --merge-base "$fork_point" "$onto" "$pr_tip" 2>/dev/null); then
  expected=${expected%%$'\n'*}
  if [[ $expected != "$(git rev-parse 'HEAD^{tree}')" ]]; then
    git diff --stat "$expected" HEAD >&2
    [[ $accept_diff == 1 ]] || die "retargeted tree differs from the PR's changes (diff above: expected -> retargeted); $kept; ask user" 5
  fi
else
  git diff --stat "$onto" HEAD >&2
  [[ $accept_diff == 1 ]] || die "PR conflicts with $target, so the resolved result cannot be checked automatically; $kept; ask user to review" 5
fi
[[ $(git merge-base HEAD "$onto") == "$onto" ]] || die "HEAD is not based on $target; $kept" 5
[[ $(git rev-list --merges --count "$onto..HEAD") == 0 ]] || die "merge commits remain; $kept" 5

whitespace=no; git diff --check "$onto" HEAD >/dev/null || whitespace=yes
unsigned=0
for c in $(git rev-list "$onto..HEAD"); do git cat-file commit "$c" | grep -q '^gpgsig' || unsigned=$((unsigned + 1)); done
cat <<EOT
MODE=rewrite worktree=$worktree branch=$branch commits=$(git rev-list --count "$onto..HEAD") unsigned_commits=$unsigned whitespace_errors=$whitespace
PUSH (do not run without user approval):
  git -C $(q "$worktree") push $(q "--force-with-lease=refs/heads/$head_branch:$pr_tip") $(q "https://github.com/$head_owner/$head_repo.git") $(q "HEAD:refs/heads/$head_branch")
THEN (only after push succeeds and headRefOid == $(git rev-parse HEAD)):
  gh api -X PATCH repos/$repo/pulls/$pr -f base=$(q "$target")
EOT
