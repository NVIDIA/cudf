# cuDF work tracking

This Project tracks **the next action and its owner**, separately from the
underlying issue or pull request. Each action is a Project-only draft item;
it is never converted into a repository issue by this automation.

The initial pilot covers Python-related work. The Project also has a C++
component for future expansion. Vyas Ramasubramani (`vyasr`) is the pilot's
primary routing contact and escalation contact.

## Find your work

- **My actions:** outstanding cards assigned to the signed-in GitHub user.
- **Outstanding actions:** the shared queue across owners.
- **Overdue actions:** cards beyond their action-specific threshold.
- **Needs routing:** cards that need an explicit recipient or handoff.
- **Action board:** the same outstanding cards arranged by status.

Each independently responsible reviewer gets a separate card. Native
assignees identify the action owner; assignment of the underlying PR or
issue is unaffected. Team review requests go to the primary contact to
select a person; teams are retained as routing evidence.

Concrete author work goes to the PR author. A merge conflict confirmed by
GitHub creates a **Resolve conflicts** action asking the author to update
the branch and resolve conflicts. This replaces a generic blocker-inspection
or unclear-handoff action for the primary contact. Explicit reviewer actions
remain independent. CI failures alone do not establish author responsibility;
unknown requirements and ownership still go to the primary contact.

Conflict clocks start when automation first confirms the conflict, using the
7-calendar-day author threshold. They are not backdated to an approval or a
commit. Refreshes preserve the clock, including while GitHub temporarily
reports unknown mergeability. A later confirmation of no conflicts completes
that action without claiming the PR satisfies other merge requirements.

## Handoffs and clocks

**Awaiting action** means responsibility has been established but the
requested action has not been observed. **In progress** records explicit
work without claiming completion. **Deferred** requires a recorded reason
and return date. **Complete** means the action was fulfilled, explicitly
superseded, or made unnecessary by the source item closing.

GitHub activity is the response channel for the pilot. Slack carries
notifications and links to these views; Slack replies do not change state.
Ambiguous discussion is routed for clarification. Labels, bot updates, and
unrelated edits do not restart an action's clock.

Initial thresholds are 3 business days for triage/review, immediate routing
for a ready PR without reviewers, 7 calendar days for author response, and
2 business days to inspect a PR with approvals. Business days currently
exclude weekends in UTC and do not account for holidays.

After the initial Slack notification, repeat after 3 business days and
escalate to `vyasr` after 7 calendar days if the action remains outstanding.
Escalation starts from successful notification, not from importing old work.
If the contact and escalation recipient are the same person, consolidate
their notification rather than sending two mentions.

## Pilot state

Production Slack delivery is disabled during preview. The manual DM pilot
delivers test messages only to the primary contact and keeps its history
separate from production notification clocks. Planned production messages
target `#swrapids-etl-scrum`, with one thread per action card. Draft item assignments
do not provide GitHub assignment notifications; Slack delivery will supply
the notification once the pilot is enabled.

Evidence, exact handoff timestamps, source revisions, and observations belong
in the card body. Date fields support the board's daily views; automation
must retain timestamp precision for reminders. CI failures are observations
and do not prove that merge requirements are satisfied or blocked.
