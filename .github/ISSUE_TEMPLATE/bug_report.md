---
name: Bug report
about: Create a bug report to help us improve cuDF
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue you're having and resolve the issue more quickly.
<!-- Give code that runs as written. Prefer a pylibcudf reproducer when it isolates the failure, and try without cuDF if the bug may be upstream. For a failure in CI, link the run and job, name the commit and the affected tests, and quote the relevant error. For an intermittent failure, give how often it fails and the command that measured it. -->

**Actual behavior**
What happens, with the full error message or output.
<!-- Say what you measured and what you suspect separately. State a root cause only as strongly as the evidence supports. -->

**Expected behavior**
A clear and concise description of what you expected to happen.
<!-- Show the behavior of a reference library, such as pandas or Polars, if compatibility is the point. -->

**Environment overview (please complete the following information)**
 - Environment location: [Bare-metal, Docker, Cloud(specify cloud provider)]
 - Method of cuDF install: [conda, Docker, or from source]
   - If method of install is [Docker], provide `docker pull` & `docker run` commands used

**Environment details**
Please run and paste the output of the `cudf/print_env.sh` script here, to gather any other relevant environment details

**Additional context**
Add any other context about the problem here.
