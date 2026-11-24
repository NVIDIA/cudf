# cuDF work tracking

This Project tracks **the next action and its owner**, separately from the
underlying issue or pull request. Each action is a Project-only draft item;
it is never converted into a repository issue by this automation.

The scope covers Python and C++ work. Vyas Ramasubramani (`vyasr`) is the automation's
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
14-calendar-day author inactivity threshold. They are not backdated to an approval or a
commit. Refreshes preserve the clock, including while GitHub temporarily
reports unknown mergeability. A later confirmation of no conflicts completes
that action without claiming the PR satisfies other merge requirements.
Closure does not prove that conflicts were cleared: reopening an unresolved
conflict resumes the same episode and preserves prior manual status. Confirmed
clearance is recorded even when someone manually completed the card.

## Handoffs and clocks

**Awaiting action** means responsibility has been established but the
requested action has not been observed. **In progress** records explicit
work without claiming completion. **Deferred** requires a recorded reason
and return date. **Complete** means the action was fulfilled, explicitly
superseded, or made unnecessary by the source item closing.

GitHub activity is the response channel for work tracking. Slack carries
notifications and links to these views; Slack replies do not change state.
Ambiguous discussion is routed for clarification. Labels, bot updates, and
unrelated edits do not restart an action's handoff clock.

After a changes-requested review, the author keeps ownership until explicitly
requesting re-review. This includes reviews from people who were not originally
requested. Author replies and revisions count as activity, without implying
a reviewer handoff. A request to another reviewer does not clear outstanding
requested changes from the original reviewer. A submitted re-review consumes
the request even when it is comment-only and no longer appears in GitHub's
current reviewer list. Withdrawing a request before a response returns
ownership to the author. Missing request timestamps leave reviewer aging
unknown until the evidence can be refreshed.

For author-owned work, notify the component lead after **more than 14 calendar
days without author activity**, while keeping the author assigned. The clock
uses author comments/replies and other activity with an identified actor.
A new head alone does not count: commit authorship cannot identify who pushed
or rebased it. The last confirmed activity is retained, with uncertainty
recorded for an unattributed head change. Bot activity,
reviewer comments, and generic PR update timestamps do not reset this clock.
The **Last author activity** and **Escalate on** fields show the activity time
and follow-up deadline. New author activity postpones the lead follow-up.
Until activity is observed after the handoff, the initial author handoff (or
first confirmed conflict observation) supplies the clock's baseline.

Initial thresholds are 3 business days for triage/review, immediate routing
for a ready PR without reviewers, 14 calendar days of author inactivity, and
2 business days to inspect a PR with approvals. Business days currently
exclude weekends in UTC and do not account for holidays.

For reviewer/routing actions, after the initial Slack notification, repeat after 3 business days and
escalate to `vyasr` after 7 calendar days if the action remains outstanding.
These reviewer/routing escalation clocks start from successful notification,
not from importing old work. Author-owned actions use the activity-based
14-day lead-follow-up rule above, regardless of whether notifications were
previously sent. Their follow-up reminders repeat after 3 business days.
If the contact and escalation recipient are the same person, consolidate
their notification rather than sending two mentions.

## Delivery state

Channel delivery is disabled during preview. DM test messages go only
to the primary contact, with their delivery history kept
separate from production notification clocks. Planned production messages
target `#swrapids-etl-scrum`, with one thread per action card. Draft item assignments
do not provide GitHub assignment notifications; Slack delivery will supply
the notification once channel delivery is enabled.

Evidence, exact handoff timestamps, source revisions, and observations belong
in the card body. Date fields support the board's daily views; automation
must retain timestamp precision for reminders. CI failures are observations
and do not prove that merge requirements are satisfied or blocked.
