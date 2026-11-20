---
name: write-cudf-issue
description: Draft or file NVIDIA/cudf issues from bug reports, performance investigations, feature proposals or story plans. Not for PR descriptions or comments on existing issues.
---

# Writing a cuDF issue

Write enough for a maintainer to act or join the discussion. A small bug may need only a few paragraphs and a reproducer. A design proposal may need sections and measurements. Follow the user's requested format.

## Before drafting

1. **Check the current conventions.** Read the matching template in `.github/ISSUE_TEMPLATE/`. Read a few recent open issues of the same kind and area, including the user's issues when available (`--author <username>`). Search by topic as well as title prefix, then read the bodies; do not copy an issue's omissions. The templates define the information to cover; maintainer prose can use a simpler layout.

   ```bash
   gh issue list -R NVIDIA/cudf --state open --search '<topic> sort:created-desc' --limit 5
   gh issue view <number> -R NVIDIA/cudf
   ```

2. **Check for duplicates**, including closed issues and related PRs. Search the topic and the user's issues. If an issue already covers it, link it and ask whether the user wants a comment drafted instead. Do not switch to posting or editing it. If the work belongs under a story, read that story and suggest the relationship in the draft.

   ```bash
   gh issue list -R NVIDIA/cudf --state all --search '<key words>'
   gh pr list -R NVIDIA/cudf --state all --search '<key words>'
   ```

3. **Read the evidence.** Use the supplied branches, notes, result files and CI runs. Check commands, versions, commits and measurements against the claims. If they disagree, explain the discrepancy and resolve it with the user before filing. Mark missing facts in the draft as `TODO(<what is needed>)`. Separate measured results, estimates and suspected causes.

4. **For an issue arising from PR review**, read the PR description and relevant comments or review thread, including later replies. Link the exact comment or thread and the PR. Capture the problem, why it matters, the agreed follow-up scope and any dependencies on the PR. Check what the latest PR changes already address. Distinguish agreed deferred work from suggestions or unresolved feedback; do not assume opening an issue resolves a review thread or makes a blocking concern safe to defer. If the intent is unclear, ask the user what should move into the issue.

## Titles and labels

| Kind | Title prefix | Labels to suggest |
|---|---|---|
| Bug | `[BUG]` | `bug` plus area |
| Feature request | `[FEA]` | `feature request` plus area |
| Documentation | `[DOC]` | `doc` plus area |
| Performance | `[PERF]` | `Performance` plus area |
| Enhancement or design | `[ENH]` | `improvement` plus area |
| Story tracker | `[Story]` | area and a relevant kind label |

The first three prefixes come from the templates; the others are maintainer conventions. Use a specific symptom or change, with a component when helpful: `[BUG] cudf-polars: ...`. Match the prefix style within a story, including `[PERF]:` if that is what it uses.

Verify label names before suggesting them: `gh label list -R NVIDIA/cudf --search <name>`. Areas include `libcudf`, `cuIO`, `Python`, `pylibcudf`, `cudf-polars` and `cudf.pandas`. Leave priority, assignment, milestones and projects to maintainers unless asked.

Use the triage status established by the maintainer or supplied context. An issue may already be triaged before it is opened. Suggest a needs-triage label when review and classification remain necessary; do not add it automatically from the template. Verify the current label convention and show the proposed triage status with the draft metadata. Drafting an issue does not itself complete triage or authorize implementation.

## Body by kind

Use only the sections the issue needs. Short design, performance and small bug reports can be prose: start with the mechanism and identifiers, explain the consequence, then give the proposed fix or reproducer. For CI and test failures, use Description, Affected tests, How to reproduce, What I saw in CI, Root cause, Workaround and Caveats, omitting sections without evidence.

- **Bug:** actual and expected behavior, minimal runnable reproducer, and environment. Follow the bug template's environment fields: install method, bare metal/container/cloud, and `print_env.sh` output when available. Include relevant library versions, CUDA, GPU and the commit tested. For CI failures, link the run and job, name the commit and affected test IDs, and quote the relevant error. State a root cause only as strongly as the evidence supports it. Prefer a pylibcudf reproducer when it isolates the failure; for upstream bugs, try without cuDF. For intermittent failures, give the rate and measurement command.
- **Feature or enhancement:** current behavior and use case, proposed behavior or API, alternatives and tradeoffs. Show the reference library's behavior when compatibility is the point. Describe migration if the API changes. For a missing pandas accelerator function, use `pandas_function_request.md` and include profiler output if available.
- **Performance:** mechanism, baseline, proposed change and rerun command. Give hardware, data size or scale factor, cold/warm conditions, number of runs and variation. Include a profile when available. Distinguish measured prototype gains from estimates and uncontrolled comparisons; do not present a best run as typical.
- **Story:** goal, scope, work items and completion criteria. Add design, phases, risks, rollback and results when needed to review the plan. A tracker does not require a full design document. Identify sub-issues and dependencies; propose the parent relationship separately for approval.
- **Discussion:** context, the question maintainers need to resolve, known evidence, possible directions and open questions. A settled design or implementation plan is not required. Distinguish ideas from agreed decisions.
- **Documentation:** location, what is wrong or missing, how it was checked or searched for, and suggested correction.

Issues can evolve through maintainer discussion. An exploratory issue may be revised into a concrete proposal, bug report or tracker, or lead to separate issues with narrower scopes. When asked to revise it, reflect the current agreement and link the discussion that explains a substantial scope change. Link any follow-up issues in both directions and make clear what each one covers. Show the user proposed edits or follow-up drafts before changing or filing anything on GitHub.

## Style and evidence

- Plain words, short sentences, no em dashes. Name functions and files in backticks. Explain the mechanism and why it matters. Cut anything the reader does not need.
- Link code at a commit or the upstream release tested, using permalinks with line numbers. For local-only code, describe it and mark the missing link for review; publishing a branch needs separate authorization.
- Link related issues and PRs, with a short explanation and their checked state. Fold long logs in `<details>` blocks.
- Include limitations that affect the claim. Tell the user separately about unrelated checks not run and process details.
- Remove secrets, internal hostnames and links, private datasets and local paths from public text. Ask the user to confirm that facts taken from private notes may be public.
- Add one line listing every AI tool involved, for example `Tools: Claude Code, Codex`. Check existing drafts and commit trailers for earlier tools and preserve them.

## AI assistance and author review

AI may draft and revise the whole issue from the supplied evidence. Give it the audience, purpose and relevant context. The author owns the final text: review factual claims against sources, preserve important qualifications when editing, and mark uncertainty rather than inventing evidence.

Keep the `Tools:` disclosure and include this checkbox in AI-assisted drafts:

```markdown
- [ ] I have read this AI-assisted issue and, to the best of my ability, checked that its claims are accurate, its evidence supports them, and any uncertainty is clearly stated.
```

Leave it unchecked until the author explicitly confirms reviewing the final draft. The agent's own checks do not count as the author's confirmation. If substantive claims change afterward, ask the author to review those changes before filing.

## Draft and filing

Write the body to a separate file for `gh issue create --body-file`. Keep the title, suggested labels and parent story outside that file. Show the user the full draft and metadata. A small, well-scoped bug may suit a new contributor; mention that to the user without adding a label yourself.

File only after the user approves the draft and metadata. Check that no `TODO(...)` remains and that claims and links are supported. Create only the approved issue and labels:

```bash
gh issue create -R NVIDIA/cudf --title '<title>' --body-file <draft.md> --label '<label>'
```

If the parent story relationship was also approved, add the new issue as a sub-issue. The API takes the issue's numeric ID, not its issue number:

```bash
id=$(gh api repos/NVIDIA/cudf/issues/<new-number> --jq .id)
gh api -X POST repos/NVIDIA/cudf/issues/<story-number>/sub_issues -F sub_issue_id="$id"
gh api repos/NVIDIA/cudf/issues/<story-number>/sub_issues --jq '.[].number'
```

Verify the posted body and labels and return the links. If a write fails or is denied, report it and stop before retrying; check for a partially created issue before any approved retry.
