---
name: write-cudf-issue
description: Use when drafting or filing a GitHub issue for NVIDIA/cudf (a bug, feature request, performance issue, enhancement proposal or [Story] tracker), or turning notes, a design sketch or an investigation into one. Covers the title, labels, the structure for each kind, the evidence to include, and linking it under a story. Not for PR descriptions or for commenting on an existing issue.
---

# Writing a cuDF issue

The model is the long issues maintainers write for each other, not the short templates in `.github/ISSUE_TEMPLATE/`. Use the templates' title prefixes and labels, but write the body the way the examples in [references/examples.md](references/examples.md) do: the reader should be able to act on the issue without asking you anything.

## Workflow

1. **Collect the facts first.** Code locations (as permalinks at a commit), commands, versions, CI run and job ids, measurements with their conditions. Do not draft from memory. Sources include the user's own branches, commit messages, notes and result files: ask where they are if the user has not said, and read them. If the issue belongs under a story, read the story first and reuse its terms and measurement conditions.
2. **Check the user's claims against the evidence.** If a commit, note or measurement disagrees with what the user told you (a speedup that is within noise, a query that did not improve), tell the user and ask which is right. Do not silently repeat the claim, and do not silently correct it.
3. **Check for duplicates.** `gh issue list -R NVIDIA/cudf --state all --search "<key words>"`. If an issue already covers it, draft a comment for that issue instead. If a story tracks the area, the new issue goes under it.
4. **Pick the kind** (table below) and write the draft (see "The draft file").
5. **Show the draft to the user** and wait. Filing an issue is public. File only when the user says so (see "Filing").

| Kind | Title | Labels to suggest |
|---|---|---|
| Bug | `[BUG] <what goes wrong, where>` | `bug` plus the area |
| Feature request | `[FEA] <what to add>` | `feature request` plus the area |
| Performance | `[PERF] <what to make faster, how>` | `Performance` plus the area; `improvement` when it changes behaviour beyond speed |
| Enhancement or design proposal | `[ENH] <proposal>` | `improvement` plus the area |
| Story (tracker for several issues) | `[Story] <goal>` | the area, plus `feature request` or `Performance` |
| Documentation | `[DOC] <what is wrong or missing>` | `doc` plus the area |

Areas used as labels: `libcudf`, `cuIO`, `Python`, `pylibcudf`, `cudf-polars`, `cudf.pandas`, `Java`, `tests`, `ci`. In a title, a component prefix after the tag helps when the area is not obvious from the rest: `[PERF] cudf-polars: ...`. Older issues also use `[PERF]:` with a colon; either is fine, but use one form within a story. Check that a label exists with `gh label list -R NVIDIA/cudf --search <name>` before suggesting it. Maintainers set triage and priority labels themselves.

## What every issue needs

- **A title that states the problem or the change**, specific enough to tell it apart in a list: "multi-key `sort_by(...).first()` in `group_by` panics with Polars 2.1 instead of falling back to CPU", not "group_by bug".
- **The first paragraph says what is wrong or what you want, and why it matters.** Someone who reads only that paragraph should know whether the issue is theirs.
- **Evidence over assertion.** Exact commands, versions and commits; numbers with what was measured, on what hardware, at what scale and how many runs; profiles or plots for performance claims. Say which claims you measured and which you reasoned through.
- **Exact references.** Link code with permalinks at a commit (`https://github.com/NVIDIA/cudf/blob/<sha>/<path>#L10-L20`), including upstream code (Polars, rmm, kvikio) at the released tag you used. Name functions and files in backticks.
  Code that exists only on a local branch cannot be linked. Ask the user whether to push the branch to their fork (then link it at a commit) or to describe it and add the link later.
- **Related issues and PRs** listed with one line each on how they relate, and their state: "builds on #24199 (merged)", "depends on #25034 (open)".
- **Long output folded.** Put logs, sanitizer output and full tracebacks in `<details><summary>...</summary>` blocks with only the relevant lines quoted in the body.
- **Nothing private.** No secrets, internal hostnames, internal dashboards or links, local paths, or private bucket and dataset locations in commands. Public issues are permanent. When you take facts from the user's private notes, list them for the user so they can confirm they may be public.

## The draft file

Write the body to its own file, because `gh issue create --body-file` posts the whole file. Keep the title, labels and parent story out of it: give them to the user in your message, next to the path of the body file.

Mark every fact still to be filled in with a visible `TODO(<what is needed>)` in the text, never an HTML comment, which would be invisible once filed. Before filing, `grep -n TODO <body-file>` must print nothing.

## Structure by kind

The section lists below are a starting point, not a form. Leave out sections that would be empty, and add what the issue needs: a performance issue with a prototype may need "Prototype and results", "Open questions and risks" and "How to rerun" from the story structure. A short issue can be a few paragraphs with no headings (#25132 is).

### Bug

```
## Description        what happens, what should happen, which versions; one paragraph
## How to reproduce   minimal, self-contained code; versions (cudf, polars/pandas, CUDA, GPU)
## What I saw in CI   (if from CI) run id, job, commit, pass/fail counts, the error line
## Affected tests     (if any) test node ids
## Root cause         (if known) the chain from trigger to failure, with permalinks
## Workaround         (if any)
## Caveats            what you tried that did not work, what you are unsure of
```

Reduce the reproducer until removing anything makes the bug go away. If the bug is upstream (Polars, rmm, CCCL), reproduce it without cuDF where possible and say where the fix belongs. For intermittent failures, give the failure rate and how you measured it (for example "3 of 200 runs with `pytest --count=200`").

### Feature request or enhancement

```
tl;dr               (for long ones) the proposal in one or two sentences
## Overview / Context        the status quo, with runnable examples of current behaviour
## Problems                  numbered, each concrete
## Proposal                  what to build; API sketch if it adds or changes API
## Alternatives considered   and why not
## Further reading           related issues, other libraries' behaviour, papers
```

Show current behaviour with code and its output, and the behaviour of the reference library (pandas, Polars, Spark) next to it when compatibility is the point. For API changes, say what is deprecated and how users migrate.

### Performance

```
## Problem         where the time goes, with the measurement: hardware, scale factor or data size,
                   iteration (cold/warm), number of runs, and a profile or plot
## Proposed fix    the change and the expected effect, with the reasoning behind the estimate
## Related         issues and PRs
```

Give the baseline and the measured improvement if you prototyped it, and how noisy the measurement is. Name the benchmark command so someone else can rerun it. Do not quote a single best run as the result.

### Story

A story is a tracker. Its body is the design and the plan; its sub-issues carry the work.

```
## Goal / Problem      what the end state is, why now (what changed: hardware, users, measurements)
## Background          (if needed) how things work today, with links
## Proposal            the design, in enough detail to review
## Plan / Phases       staged so each step can merge and be measured alone; how a step is turned
                       on (feature flag, environment variable) and how it is rolled back
## Risks               and how each is handled
## State of the work   results so far (a table works), and what the measurements show
## Tracking            sub-issues and PRs; when an item counts as done
```

Keep the "State of the work" table current as results change, and relabel numbers that turn out to be measured differently than stated rather than silently replacing them. Link each piece of work as a sub-issue (see "Filing") rather than only listing it in the body.

## Writing style

- Plain, direct sentences. Define a term the first time you use it; use one name per thing.
- Explain the reasoning, not only the conclusion: "the panic cannot become a fallback because `PanicException` is a `BaseException` and poisons the plan lock", not "cannot be fixed here".
- Be honest about uncertainty: "reasoned through, not run", "measured once", "I have not checked Hopper".
- Length follows content. A one-line bug with a three-line reproducer is a good issue; a design proposal may need several thousand words. Cut anything the reader does not need to act.

## Filing

Only after the user approves the draft:

1. Check that `grep -n TODO <draft.md>` prints nothing, then `gh issue create -R NVIDIA/cudf --title "<title>" --body-file <draft.md> --label "<label>" ...`
2. To put it under a story, add it as a sub-issue (the API takes the issue's id, not its number):

   ```bash
   id=$(gh api repos/NVIDIA/cudf/issues/<new-number> --jq .id)
   gh api -X POST repos/NVIDIA/cudf/issues/<story-number>/sub_issues -F sub_issue_id="$id"
   ```

3. Check it: `gh api repos/NVIDIA/cudf/issues/<story-number>/sub_issues --jq '.[].number'`.
4. Give the user the links. Do not assign people, set milestones or add to projects unless asked.
