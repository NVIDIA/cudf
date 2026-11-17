# Repository queue report

Collect repository evidence, preview independently owned actions, and configure
the associated GitHub Project without running a GitHub Action. The scripts use
Python's standard library. Collection and preview are read-only; Project setup
requires an explicit `--apply` flag to write fields, views, and documentation.

## Run locally

Collection requires Python 3.8+ and an authenticated GitHub CLI that supports
`gh api --paginate --slurp`. Use an existing login session or supply `GH_TOKEN`
through your environment. The collector only invokes `gh api --method GET`.
The repository must be readable; checks may require additional read access.

From the repository root:

```sh
python3 ci/repo_queue/collect.py \
  --repo NVIDIA/cudf --sample-size 25 --output /tmp/cudf-queue/snapshot.json

python3 ci/repo_queue/report.py /tmp/cudf-queue/snapshot.json \
  --output /tmp/cudf-queue/report.md

python3 ci/repo_queue/report.py /tmp/cudf-queue/snapshot.json \
  --format json --output /tmp/cudf-queue/report.json
```

To investigate particular items, replace `--sample-size 25` with
`--numbers 20188 24176 23695`. Use `--all` to collect every open issue and PR;
this paginates the repository's issue listing rather than using search, and
can consume considerably more API requests. All timeline, review, check-run,
inline review comment, and status collections are paginated too.
Use `--label Python` to restrict a sample or the full listing to that label.
Unlabeled issues are excluded by that selection and require component
classification before they can be included in the Python pilot.

The sample selects approximately equal numbers from five search strata:
untriaged issues, approved PRs, changes-requested PRs, PRs without reviews,
and drafts. It selects the oldest updated candidates within each stratum,
deduplicates them, and fills short strata from other open PRs. Search is only
candidate selection; classification uses the retrieved metadata and events.
The sample is deliberately biased toward old work, and its counts cannot
estimate repository-wide prevalence.

Collection saves the raw responses and records its scope and timestamps. A
failed mandatory request aborts collection without replacing an existing
snapshot; permission failures reading optional CI data become explicit
warnings. A snapshot is assembled over time, so changes during collection can
produce inconsistent observations. Recollect an item before acting on it.

## Tune and replay offline

Edit a copy of [policy.json](policy.json) to change thresholds, triage owner, bot accounts,
explicit author/deferred labels, or suggested component routing:

```sh
python3 ci/repo_queue/report.py /tmp/cudf-queue/snapshot.json \
  --policy /tmp/my-policy.json --output /tmp/cudf-queue/tuned.md
```

Rendering defaults to the snapshot's collection time, making replay
deterministic. Use `--now 2026-11-20T12:00:00Z` to assess aging at a later time;
this extrapolates from the saved state and does not refresh GitHub metadata.
Machine-readable output includes the applied policy, per-reviewer request
timestamps, evidence, observations, and uncertainty.

The initial policy uses these thresholds:

| Queue | Threshold |
| --- | --- |
| Needs triage | More than 3 business days with the triage label |
| Needs reviewer assignment | Immediately after becoming ready |
| Waiting for first or another review | More than 3 business days |
| Waiting for author | More than 7 calendar days |
| Has approvals; inspect blockers | More than 2 business days since the earliest surviving approval |

Business time accumulates Monday–Friday in UTC and excludes weekends, but
does not account for holidays. Missing waiting-since events produce unknown
ages. `updated_at` is never a waiting clock. Re-requests begin new reviewer
clocks; draft-to-ready and reopening transitions bound review waiting time.

Human reviews exclude the author, GitHub Bot accounts, `[bot]` logins, and
configured bot accounts. Comment-only reviews preserve previous decisions;
dismissed decisions stop counting. A later author comment or later-dated
commit after a changes-requested review leaves the handoff unclear unless an
explicit label or re-request establishes it. Commit dates do not establish
push timestamps.

Drafts have their own queue and no inactivity threshold. Deferred work and
explicit author handoffs require configured labels; both lists start empty
so the report does not invent repository policy. Component routing is a
suggestion based on existing labels and a configurable team mapping, not an
assignment or a complete CODEOWNERS evaluation. Issue assignees are not
assumed to be triage owners.

CI observations stay separate from lifecycle state. The report does not
evaluate branch rules, required reviewer counts, unresolved review threads,
or ops-bot merge barriers. Surviving approvals do not establish merge
readiness; approvals on other head commits are explicitly marked.

## Verify

```sh
python3 -m unittest discover -s ci/repo_queue -v
ruff check ci/repo_queue
ruff format --check ci/repo_queue
```

These tests run without cuDF, CUDA, GitHub credentials, or network access.

## Project setup and pilot configuration

[pilot.json](pilot.json) records the agreed NVIDIA Project, Python scope,
`vyasr` as primary/escalation contact, Slack destination, and reminder policy.
Slack IDs are identifiers, not posting credentials. Delivery is disabled.
The Project is private; its intended audience needs read access before Slack
messages link people to the queue.

[project_schema.json](project_schema.json) defines the custom fields, statuses,
and five saved views. Native draft-item assignees power the `assignee:@me`
view, independently from the underlying issue or PR assignment.
[PROJECT.md](PROJECT.md) supplies the Project's README.

```sh
python3 ci/repo_queue/project.py --output /tmp/cudf-queue/project-plan.json
python3 ci/repo_queue/project.py --apply --output /tmp/cudf-queue/project-state.json
```

Project setup uses the GitHub GraphQL API and needs the `project` scope. It
paginates schema reads, preserves existing option IDs and additional options,
and verifies the result by reading the Project back. Reapplying an already
configured schema performs no mutations. Setup does not change visibility,
grant access, create cards, or send messages.

## Action and message previews

```sh
python3 ci/repo_queue/collect.py --label Python --sample-size 12 \
  --output /tmp/cudf-queue/python-snapshot.json
python3 ci/repo_queue/actions.py /tmp/cudf-queue/python-snapshot.json \
  --project-state /tmp/cudf-queue/project-state.json \
  --output /tmp/cudf-queue/actions.md
python3 ci/repo_queue/actions.py /tmp/cudf-queue/python-snapshot.json \
  --format json --output /tmp/cudf-queue/actions.json
```

`actions.py` emits a proposal; it has no publication or delivery capability.
Each requested individual reviewer gets an independent action and clock.
Team-only Python requests route to the primary contact to name a reviewer;
an existing named reviewer prevents a redundant team-routing action.
An outstanding reviewer action remains visible while a different person owes
an author response on the same PR.

Stable action keys identify a handoff episode. New commits do not reset a
review request; an explicit re-request starts a new episode. The output retains
exact timestamps and source SHA alongside Project date-field proposals.
The preview exposes missing Slack identities rather than guessing mentions.
Identity keys are GitHub logins, matched case-insensitively.

Optional `--history FILE` accepts a JSON object keyed by action key. Each
record may contain `first_notified_at`, `last_notified_at`, and `escalated_at`.
These must describe successful deliveries. The reminder proposal uses a
3-business-day cooldown and escalates 7 calendar days after the first
successful notification. Without a delivery record, escalation dates stay
empty even for very old imported work. Owner and escalation mentions are
consolidated when they identify the same person.

## Remaining implementation and rollout

The live Project schema and private previews are the current pilot stage.
Project card synchronization, activity-driven completion/supersession,
durable delivery history, a Slack publisher, and scheduled execution remain
to be implemented. Previews are not a running repository monitor.

No scheduled workflow or publisher is included while the report is being
tuned privately. A future workflow should only provide the trigger, read
permissions, runtime/authentication setup, invoke these commands, and retain
their outputs. Sampling, event interpretation, policy, and rendering belong
in the scripts. Any publication or reminder step should be a separate script
that consumes the structured report and records successful deliveries.

Unattended posting needs an approved Slack app credential supplied through
the execution environment; the connected Slack tools in this chat do not
provide credentials to a GitHub Action. GitHub Project updates also need a
suitable workflow credential. No credential belongs in these configuration
files.

The release-sync pagination issue and existing Project status automation
remain separate follow-up changes; this prototype does not alter them.
