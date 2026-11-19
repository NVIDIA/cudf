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
classification before they can be included in the Python scope.

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
timestamps, evidence, observations, and uncertainty. Pass
`--routing-history /path/to/routing-history.json` from a prior tracking run to
retain previously confirmed author activity. History newer than the snapshot
is rejected, including when `--now` extrapolates to a later date. Without
history, previously confirmed activity is unavailable. Reports and action cards share the same handoff normalization
and author-clock calculation; aggregate reports and individual reviewer
cards remain separate presentations.

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
Historical requests still count after GitHub removes them from the current
reviewer list: a submitted re-review, including a comment-only review, consumes
the handoff. Withdrawing a request before a response returns work to the author.
A current request with a missing timeline timestamp keeps an unaged reviewer
action and marks the handoff uncertain.

Author comments, inline replies, and attributed timeline activity reset the
inactivity clock. A changed head alone does not reset that clock: commit
authorship and commit dates establish neither the push actor nor the push
time. A maintainer rebase may preserve the original author's identity. The
last confirmed activity is retained, and an unattributed head change is marked
uncertain. Reviewer/bot activity and `updated_at` do not
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

## Project setup and configuration

[config.json](config.json) records the agreed NVIDIA Project, Python scope,
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
actions. A bot-authored changes-requested PR routes to the primary contact
to identify a human responder, avoiding unassignable bot logins.
Failed CI alone does not identify the next owner. The underlying
report retains its lifecycle classification and lists CI/conflict observations
separately; the action planner determines the concrete work and owner.

The conflict clock starts at the first confirmed observation and uses the
14-calendar-day author inactivity threshold. Project metadata preserves this episode
across runs and temporarily unknown mergeability; approval dates are not
conflict start dates. For offline replay across snapshots, pass a prior tracking
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

The primary contact can test the installed bot without posting to the team
channel. This targets the contact's Slack user ID from `config.json`, never
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
real owner mentions. People outside the bot's DM with the primary contact
will not receive mention notifications. Messages carry an explicit test
heading. The script prints delivery identifiers and does not update Project
cards or successful-notification history. It does not retry ambiguous sends;
inspect the DM before rerunning after a connection failure.

## End-to-end DM delivery

Preview a small, explicit selection using fresh GitHub evidence on each run:

```sh
python3 ci/repo_queue/track.py --numbers 24176 24230 23772 \
  --output-dir /tmp/cudf-work-tracking/e2e
```

Add `--apply` to synchronize Project-only draft action cards, or add both
`--apply --send-dm` to synchronize and send notifications to the primary
contact's bot DM. The token prompt and `SLACK_BOT_TOKEN` behavior match the
smoke test. The production channel is never a delivery target for this command.
Real owner mentions are retained. At most ten source issues/PRs can be selected.
Run from one machine and use the same output directory for each tracking run;
the local lock prevents concurrent runs using that directory. Cross-machine
concurrent writers are not supported.

Each managed card has a stable action key, a native GitHub assignee, source
and handoff links, and a managed body section. Maintainer notes outside that
section survive updates. The body is re-read immediately before writing so
notes added during collection are retained; concurrent changes to managed
metadata abort the write. GitHub offers no conditional draft-body mutation,
so a small race remains between that final read and write. Duplicate action
keys or truncated card reads abort
synchronization. Unchanged cards require no mutations, and a second Project
read verifies the result after applying changes.

Only explicitly selected sources are reconciled. Submitted reviews complete
that reviewer's old card; new requests create new episodes. Replacement
handoffs supersede old cards with an explanatory disposition. Closure completes
cards, while draft/deferred sources defer them. Losing a component label or
an ambiguous absence does not establish completion. Manual In progress,
Deferred, and Complete statuses survive refreshes of the same current action.
Archived cards are not changed or notified. Automatic draft/closed
dispositions are reversed when the same action becomes active again; manual
Deferred, Complete, and In progress overrides remain in place. Automatic status
provenance is stored in the managed metadata. Pending Status transitions
record both the old and intended values so a partial write can be retried
without mistaking the automation's own changes for a manual override.
Closing a conflicting PR completes its card without recording conflict
clearance; reopening the same conflict resumes that episode. Confirmed
clearance is recorded even on manually completed cards. Older records use their
recorded disposition to recognize automatic transitions. Missing or damaged
metadata on a card with a managed Action key stops the run before creating
cards. The underlying issues and PRs are not modified.

DM delivery history is durable in each draft card's managed metadata, keyed
separately from any future production destination. Subsequent due reminders
and test escalations use the original DM thread. Test timestamps do not set
the Project's production Follow-up or Escalate on dates. Updating a card after
GitHub activity currently does not publish a completion message to Slack.

Before sending, the script saves a pending delivery record in the Project.
After success it saves a local receipt to `dm-receipts.json`, then acknowledges
the delivery in the Project, retrying that acknowledgment up to three times
within the same run. A rerun using the same output directory can recover the
acknowledgment from that receipt without resending. Definite Slack rejections
(`ok: false` or HTTP 429) clear the pending intent and do not start delivery
clocks. An uncertain send with no local receipt blocks
further delivery for inspection; this is deliberately not an exactly-once
delivery guarantee. Preserve the receipt file until all pending records have
been acknowledged. A pending record without a receipt currently requires
manual reconciliation after inspecting the DM.

The command saves the refreshed snapshot, DM plan, delivery receipts, and
result under the output directory. These files contain no bot token.

## Scheduled DM delivery

[scheduled.json](scheduled.json) limits unattended runs to the three agreed
tracked sources. It is an explicit allowlist, including closed sources so their
cards can complete. Editing it is the way to expand the selection. No sampling or
backlog-wide collection happens in the scheduled job.

The workflow in `.github/workflows/repo-queue.yml` runs daily at 17:17 UTC
and can be dispatched manually. It only runs in `NVIDIA/cudf` on its default
branch. GitHub schedules become available once the workflow reaches that
branch, and scheduled executions can be delayed. See
[GitHub's schedule documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
The schedule refreshes GitHub evidence, synchronizes the board, and sends only
due notices to the primary contact's bot DM. It does not post to the team
channel. The existing policy thresholds are unchanged.

The same entry point works locally:

```sh
python3 ci/repo_queue/scheduled.py --mode preview --output-dir /tmp/cudf-queue/daily
python3 ci/repo_queue/scheduled.py --mode refresh --output-dir /tmp/cudf-queue/daily
python3 ci/repo_queue/scheduled.py --mode dm --output-dir /tmp/cudf-queue/daily
```

`preview` is read-only, `refresh` writes the board, and `dm` also delivers due
DMs. The unattended entry point requires `SLACK_BOT_TOKEN` for delivery and
never prompts for a credential. For interactive prompting, use `track.py`
directly. Missing Slack credentials stop the scheduler before any board writes.

The workflow supplies `NV_SLACK_CUDF_WORK_TRACKING_BOT_TOKEN` as
`SLACK_BOT_TOKEN`, and the existing `ADD_TO_PROJECT_GITHUB_TOKEN` as `GH_TOKEN`.
The latter must read repository issues, PRs and users and edit the private
organization Project, including its draft items and assignees. Its access to
this particular Project must be verified by a manual `refresh` run before
relying on the schedule. An ordinary `GITHUB_TOKEN` cannot access organization
Projects; see [GitHub's authentication guidance](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/automating-projects-using-actions).
No credential belongs in the configuration or command-line arguments.

Workflow concurrency serializes its executions without cancelling an active
delivery. Do not run another local writer while the workflow is active;
the file lock does not coordinate different machines. Project metadata keeps
the authoritative notification history across fresh runners. Repeated runs
do not resend notices before the next policy deadline.

The public workflow uploads **no artifacts**. The scheduler captures child
stdout/stderr into `diagnostics.log` inside its private output directory and
prints only a fixed success message or an allowlisted failure stage and exit
code; exception text stays private. Snapshots, private Project
history, rendered plans, and Slack DM receipts must not be copied into public
logs or repository artifacts. The local output directory is restricted to its
owner; existing successful notification history remains in the private Project.

For a local failure after Slack succeeds but before Project acknowledgment,
rerun `track.py` with the same output directory to recover from
`dm-receipts.json`. Do not discard receipts or switch directories during
recovery. Hosted runners are ephemeral: their local diagnostics and receipts
are not retained after a job ends. If acknowledgment fails there, the pending
record in the private Project blocks further delivery, and an operator must
inspect the bot DM and reconcile the uncertain attempt before retrying. This
tradeoff keeps recovery data private without adding another storage service;
it does not provide automatic recovery after an ambiguous hosted-runner send.
Pause the workflow before local recovery and resume it afterward.

## Accelerated reminder rehearsal

Exercise an initial notice, the three-business-day reminder, and the
seven-calendar-day escalation immediately, using the same notification policy
and delivery code as the live tracking run:

```sh
python3 ci/repo_queue/rehearse.py --output-dir /tmp/cudf-queue/rehearsal
python3 ci/repo_queue/rehearse.py --output-dir /tmp/cudf-queue/rehearsal --send-dm
```

The first command previews all three payloads offline. The second prompts
for the bot token or uses `SLACK_BOT_TOKEN`. It sends one parent and two replies
in the same DM thread, with the owner mention retained and the lead mentioned
on escalation. All three messages identify the simulated time and say no
action is required. The closed test PR is only a reference link.

Rehearsal state, clocks, pending sends, and receipts stay in local files;
there are no GitHub requests or Project writes. Preview and actual delivery
use separate subdirectories, and neither changes the live DM or production
notification histories. Rerunning in the same directory does not resend a
completed sequence. Keep that directory for recovery if a send is uncertain;
use a new directory only when deliberately starting another rehearsal.

The workflow's manual `rehearsal` mode performs the same sequence using its
Slack secret, with no Project access required. Each new workflow execution
starts a new rehearsal; inspect the DM before retrying a failed rehearsal.
Its hosted-runner files are not retained. A rehearsal never runs on the daily
schedule.

## Remaining implementation and rollout

Production-channel delivery, automated reconciliation of ambiguous sends, and
broader component classification remain to be implemented. Generic prose
handoffs still require maintainer judgment; the classifier does not infer
owners from arbitrary comments.

The workflow only supplies triggers, permissions, concurrency, credentials,
and invocation. Selection, event interpretation, policy, rendering,
and delivery live in scripts that run from a terminal.

The release-sync pagination issue and existing Project status automation
remain separate follow-up changes; this prototype does not alter them.
