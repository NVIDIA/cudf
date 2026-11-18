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
| Waiting for author | More than 14 calendar days without author activity; remind component lead |
| Has approvals; inspect blockers | More than 2 business days since the earliest surviving approval |

Business time accumulates Monday–Friday in UTC and excludes weekends, but
does not account for holidays. Missing waiting-since events produce unknown
ages. `updated_at` is never a waiting clock. Re-requests begin new reviewer
clocks; draft-to-ready and reopening transitions bound review waiting time.

Human reviews exclude the author, GitHub Bot accounts, `[bot]` logins, and
configured bot accounts. Comment-only reviews preserve previous decisions;
dismissed decisions stop counting. Changes-requested reviews count even when
the reviewer was not explicitly requested. Author replies and revisions leave
ownership with the author until explicit re-review requests establish the
next handoff. A request to a different reviewer does not clear the original
reviewer's requested changes. Independent reviewer cards remain visible.

Author comments, inline replies, and attributed timeline activity reset the
inactivity clock. When a changed head revision is attributed to the PR author,
its first observation resets that clock; repeated observations do not. Commit
dates do not establish push timestamps, and the first imported revision is
not assumed to be new activity. Reviewer/bot activity and `updated_at` do not
reset author inactivity. Without complete reply history the author still owns
the action, but automated inactivity follow-up is suppressed.
Until activity is observed after the handoff, the initial author handoff or
first confirmed conflict observation supplies the clock's baseline.

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

The action-routing policy sends GitHub-confirmed merge conflicts to the PR
author as Resolve conflicts actions, replacing generic primary-contact
inspection/clarification cards. Requested reviewers retain their independent
actions. Failed CI alone does not identify the next owner. The underlying
report retains its lifecycle classification and lists CI/conflict observations
separately; the action planner determines the concrete work and owner.

The conflict clock starts at the first confirmed observation and uses the
14-calendar-day author inactivity threshold. Project metadata preserves this episode
across runs and temporarily unknown mergeability; approval dates are not
conflict start dates. For offline replay across snapshots, pass a prior pilot
run's `routing-history.json` using `actions.py --routing-history FILE`. A later
confirmed conflict-free result completes the author action. A new conflict
after that resolution starts a new episode.
If blocker inspection is still needed after conflict resolution, it also
starts a new handoff episode rather than inheriting the old fallback's
notification history.

Stable action keys identify a handoff episode. New commits do not reset a
review request; an explicit re-request starts a new episode. The output retains
exact timestamps and source SHA alongside Project date-field proposals.
The preview exposes missing Slack identities rather than guessing mentions.
Identity keys are GitHub logins, matched case-insensitively.

Optional `--history FILE` accepts a JSON object keyed by action key. Each
record may contain `first_notified_at`, `last_notified_at`, and `escalated_at`.
These must describe successful deliveries. Reviewer/routing notifications
use a 3-business-day cooldown and escalate 7 calendar days after the first
successful notification; without a delivery record their escalation dates
stay empty even for old imported work.

Author-owned Respond and Resolve conflicts actions instead notify the
component lead after strictly more than 14 calendar days without author
activity, independently of previous notifications. The author remains assigned
to the card. Last author activity and Escalate on fields expose that deadline.
Lead follow-ups repeat after a 3-business-day cooldown. New author activity
postpones follow-up for another 14 calendar days without changing the action
key or reviewer responsibility. Owner and lead mentions are consolidated when
they identify the same person.
Only the lead's Slack mapping is required for author-inactivity follow-up;
an external author without a mapping is identified by a GitHub profile link.

## Test Slack delivery in a DM

The pilot contact can test the installed bot without posting to the team
channel. This targets the contact's Slack user ID from `pilot.json`, never
the configured production channel. First preview the payload:

```sh
python3 ci/repo_queue/slack_dm_test.py
```

To send a parent message and a threaded reply, run interactively:

```sh
python3 ci/repo_queue/slack_dm_test.py --send
```

The script prompts for the bot token without displaying it, or reads
`SLACK_BOT_TOKEN` from the environment. It does not read the token from GitHub
Actions secrets, which cannot be retrieved after saving. Never pass a token
as a command-line argument or commit it. The Actions secret is named
`NV_SLACK_CUDF_WORK_TRACKING_BOT_TOKEN`.

Use `--text-file /tmp/message.txt` to test a rendered action body, including
real owner mentions. People outside the bot's DM with the pilot contact
will not receive mention notifications. Messages carry an explicit test
heading. The script prints delivery identifiers and does not update Project
cards or successful-notification history. It does not retry ambiguous sends;
inspect the DM before rerunning after a connection failure.

## End-to-end DM pilot

Preview a small, explicit selection using fresh GitHub evidence on each run:

```sh
python3 ci/repo_queue/pilot.py --numbers 24176 24230 23772 \
  --output-dir /tmp/cudf-work-tracking/e2e
```

Add `--apply` to synchronize Project-only draft action cards, or add both
`--apply --send-dm` to synchronize and send notifications to the primary
contact's bot DM. The token prompt and `SLACK_BOT_TOKEN` behavior match the
smoke test. The production channel is never a delivery target for this command.
Real owner mentions are retained. At most ten source issues/PRs can be selected.
Run from one machine and use the same output directory for each pilot run;
the local lock prevents concurrent runs using that directory. Cross-machine
concurrent writers are not supported.

Each managed card has a stable action key, a native GitHub assignee, source
and handoff links, and a managed body section. Maintainer notes outside that
section survive updates. Duplicate action keys or truncated card reads abort
synchronization. Unchanged cards require no mutations, and a second Project
read verifies the result after applying changes.

Only explicitly selected sources are reconciled. Submitted reviews complete
that reviewer's old card; new requests create new episodes. Replacement
handoffs supersede old cards with an explanatory disposition. Closure completes
cards, while draft/deferred sources defer them. Losing a component label or
an ambiguous absence does not establish completion. Manual In progress,
Deferred, and Complete statuses survive refreshes of the same current action.
Archived cards are not changed or notified. The underlying issues and PRs
are not modified.

DM delivery history is durable in each draft card's managed metadata, keyed
separately from any future production destination. Subsequent due reminders
and test escalations use the original DM thread. Test timestamps do not set
the Project's production Follow-up or Escalate on dates. Updating a card after
GitHub activity currently does not publish a completion message to Slack.

Before sending, the script saves a pending delivery record in the Project.
After success it saves a local receipt to `dm-receipts.json`, then acknowledges
the delivery in the Project. A rerun can recover the acknowledgment from that
receipt without resending. An uncertain send with no local receipt blocks
further delivery for inspection; this is deliberately not an exactly-once
delivery guarantee. Preserve the receipt file until all pending records have
been acknowledged. A pending record without a receipt currently requires
manual reconciliation after inspecting the DM.

The command saves the refreshed snapshot, DM plan, delivery receipts, and
result under the output directory. These files contain no bot token.

## Remaining implementation and rollout

The Project schema, explicit-source synchronization, and DM delivery form a
manual pilot. Scheduled execution, production-channel delivery, automated
reconciliation of ambiguous sends, and broader component classification remain
to be implemented. Generic prose handoffs still require maintainer judgment;
the classifier does not infer owners from arbitrary comments.

No scheduled workflow is included while the report is being tuned privately.
A future workflow should only provide the trigger, permissions and concurrency,
runtime/authentication setup, invoke these commands, and retain
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
