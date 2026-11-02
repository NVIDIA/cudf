---
name: upgrade-polars
description: Use when raising the supported Polars version range for cudf-polars (for example 2.0 to 2.1). Covers the version pins, adapting the translator to new IR/expression changes, the cudf-polars unit tests with 100% coverage, the upstream polars test suite (xfail/skip lists), and the narwhals plugins.
---

# Upgrading the supported Polars version in cudf-polars

This skill describes how to move the `polars` range that cudf-polars supports (for example
`>=2.0,<2.1` to `>=2.1,<2.2`). It is distilled from the previous upgrade PRs, weighted toward
the Polars 2.0 upgrade (rapidsai/cudf#24549). A bump is done when all of these pass locally:

1. The cudf-polars unit tests, with **100% coverage** (CI uses `--cov-fail-under=100`).
2. The upstream polars test suite with the injected GPU engine (`in-memory` and `spmd`).
3. The narwhals run for cudf-polars (`narwhals_cudf_polars_test_plugin`, the second block of `ci/test_narwhals.sh`). The cudf and
   cudf.pandas runs do not exercise polars. They only matter if Narwhals itself is bumped, which the `update-narwhals-testing`
   skill covers (it requires all three to pass).
4. `pre-commit` (including mypy and `ci/check_cudf_polars_ir.py`).

Do the main work against the **newest** polars you support: coverage and the upstream polars tests only run on it in CI. Then run the
cudf-polars unit tests on the **oldest** supported polars too (Step 4). CI's wheel job (`wheel-tests-cudf-polars`, `POLARS_VERSIONS=earliest`
in `ci/test_python_cudf_polars.sh`) does exactly that, and a test, xfail or assertion changed for the new version often breaks there.

## Prerequisites

Ask the user for:
- a working GPU (check `nvidia-smi`). Build the environment yourself in Step 2 the way CI does; only ask the user for one if that is
  not possible;
- the target Polars version (default: the newest release, `pip index versions polars`);
- the target Narwhals version, only if narwhals must change.

Ground rules:
- One Polars minor version per PR. Do not bundle unrelated fixes; split them out (a reviewer
  asked for this in #24549).
- Never leave a test failing and never hide wrong results behind an xfail. Prefer making the
  translator raise `NotImplementedError` (CPU fallback) over an xfail when the GPU result would be wrong.
- Small root-caused fixes go in the upgrade PR. Larger work gets a drafted issue, linked from the
  xfail/skip reason or a `TODO`. Draft issues for the user to review before opening them.
- Record which tests were run and their outcome as you go; the PR description needs it.
- On an existing PR branch, sync with GitHub first (see "Resuming work on an open PR"). Never force-push.
- Install the target version into the dev environment first, then confirm with
  `python -c "import polars; print(polars.__version__)"`. If a test environment variable or
  plugin is missing, say so instead of silently skipping.

## Practical notes

- Run long suites in the background with output to a log file, and check them later rather than waiting inline.
- The narwhals pin lives in `dependencies.yaml` (`depends_on_narwhals`); read it there instead of assuming a version.
- **CI under-reports failures.** Every test job stops at its first failure (`-x`, or xdist "stopping after N failures"), so a red PR
  shows only a handful of tests. In the Polars 2.1 upgrade CI showed 3 unit and 4 upstream failures per job; the full local runs
  had 37 unit failures and 153 (in-memory) / 65 (spmd) upstream failures. Treat CI as a smoke signal and triage from full local
  runs (`--maxfail=300`, `--maxfail=100000`). Read CI logs with
  `gh pr checks <pr>` and `gh run view --job <job-id> --log-failed`. PR CI also runs only the upstream tests selected by the GPU run
  list (see "GPU test selection in PR CI" in Step 5), so a green PR says nothing about the unselected tests.
- **Shared GPUs.** Check `nvidia-smi` before every run and never disturb other people's work. Use one GPU per process
  (`CUDA_VISIBLE_DEVICES`) and keep concurrent jobs on different GPUs. The streaming engine (and the xdist plugin runs) abort with
  `No GPUs found on current NUMA node N` when the process runs on CPUs of the other NUMA node. Find the GPU's node with
  `nvidia-smi topo -m` and pin: `numactl --cpunodebind=<node> --membind=<node> python -m pytest ...`. The `in-memory` engine is unaffected.
- `pytest-cov` can under-report: lines such as `raise NotImplementedError(...)` that a test provably executes may show as missed in a
  full run. Before adding a pragma or a test, confirm a line is really unreachable by tracing it:
  run the query under `sys.settrace` (or with `pytest <that test file> --cov`) and see whether the line executes. Only lines that are
  not reached by any query on the new polars need a test or `# pragma: no cover`.
- Upstream tests marked `write_disk` are deselected unless you pass `-m ""` (CI passes `-m "not slow"`); remember this when running
  a single upstream test.
- The narwhals repo keeps its package under `src/narwhals`. To test the pinned version without installing it, put `<clone>/src` (not the
  clone root) first on `PYTHONPATH`, and print `narwhals.__file__` to confirm which copy is imported.
- **Parallel work.** Independent failure clusters can be handed to subagents. Give each its own `git worktree` and branch off the
  current PR commit, its own GPU, a list of exact test ids, and the rule "fix and test on your branch, or recommend an xfail with a
  drafted issue". Agents must not edit shared dicts like `inject_gpu_engine.py`; you merge their commits with `git cherry-pick` and
  resolve the (usually append-only) conflicts. Give each agent a private scratch directory; shared scratch scripts get overwritten.

## Resuming work on an open PR (keep local and GitHub in sync)

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

## Step 1. Survey what changed upstream

Start by shallow-cloning Polars at the target tag into a temporary directory and keep it for the whole upgrade. It is the
reference for this survey, for the exact logic behind a behaviour change, and for the upstream test suite in Step 5:

```bash
git clone --depth 1 --branch py-X.Y.Z https://github.com/pola-rs/polars.git <tmp>/polars
git -C <tmp>/polars fetch --depth 1 origin tag py-OLD   # only needed to diff against the old version
git -C <tmp>/polars diff py-OLD py-X.Y.Z -- crates/polars-python/src/lazyframe/visitor py-polars/src/polars/_plr.pyi
```

Read the Polars release notes for every version between the old and new pin
(`gh api repos/pola-rs/polars/releases/tags/py-X.Y.Z -q .body`; they are long and mostly irrelevant to cudf-polars, so skim for
the items below) and, when a note is unclear, read the code in the clone (the visitor in `crates/polars-python`, optimizer rules in
`crates/polars-plan`). List: removed or renamed
APIs, new IR nodes or `Function`/`BooleanFunction`/`StringFunction`/`StructFunction`/
`TemporalFunction` enum members, changed option payloads, changed error types/messages,
changed dtype semantics, and new deprecation warnings (the test suite runs with
`filterwarnings = error`). Keep the list; it drives steps 3 to 5.

Also read the Polars upgrade guide (https://docs.pola.rs/releases/upgrade/), which lists the breaking changes and
deprecations for each release. Check each item against cudf-polars; behaviour changes that the GitHub release notes only mention
briefly are often described properly there.

Beyond release notes, the most reliable signals are the tests themselves: `test_from_polars_all_names` for enum drift, and
the diffs of `crates/polars-python/src/lazyframe/visit.rs` and `py-polars/src/polars/_plr.pyi` between the two tags (in the clone) for IR changes.

Also run `git log` and grep the tree for stale guards that a merge may have reintroduced:
`grep -rn "POLARS_VERSION_LT\|POLARS_LOWER_BOUND\|polars_compat" python ci dependencies.yaml`.

## Step 2. Bump the pins

All the places that carry the range (edit the exact same string everywhere):

| File | What |
| --- | --- |
| `dependencies.yaml` | `run_cudf_polars` list: `polars>=X,<Y` |
| `dependencies.yaml` | `test_cudf_polars_compat`: the `polars_compat_version` list under `files:` and the matching `polars==X.*` entries under `dependencies:` (one block per supported minor, plus the fallback entry). CI scripts read this matrix, so they usually do not need edits |
| `python/cudf_polars/pyproject.toml` | **generated** from `dependencies.yaml` by `rapids-dependency-file-generator` (the list carries a "generated" marker); do not hand-edit, confirm it changed after the generator run |
| `conda/recipes/cudf-polars/recipe.yaml` | `polars >=X,<Y` |
| `.pre-commit-config.yaml` | mypy hook `additional_dependencies` |
| `conda/environments/all_cuda-*_arch-*.yaml` | **generated** by the same hook; run `pre-commit run --all-files rapids-dependency-file-generator` rather than hand-editing |
| `python/cudf_polars/cudf_polars/utils/versions.py` | `POLARS_LOWER_BOUND` only if the minimum moves |

Default for a new minor: **raise the upper bound only** (`>=2.0,<2.1` becomes `>=2.0,<2.2`) and append the new
minor to the `test_cudf_polars_compat` matrix (`["2.0", "2.1"]`, plus a `polars==2.1.*` block next to the existing one).
Do not drop the previous minor in the same PR. Policy for the lower bound: support roughly the last 5 or 6 minor
versions, but cloud defaults (Colab, AzureML, Kaggle) can justify a lower minimum. Confirm the intended range with the
user before editing, and ask before raising the minimum. When the upgrade adds support for a new **major** Polars version (for
example 2.0), the lower bound moves too: raise it to the first minor of that major, update `POLARS_LOWER_BOUND`, and remove the
guards, code and tests that only supported the older major (see below).

Regenerate the generated files (the conda environment files and the `pyproject.toml` dependency list) with the env's own `pre-commit` (it is usually not on `PATH`):
`export PATH=<env>/bin:$PATH; pre-commit run --all-files rapids-dependency-file-generator`. The first run "fails" because it
modified files; that is expected. Run it again to confirm it is clean. It needs network access to initialise the hooks.
When the minimum moves, delete every guard for versions below it (and the code, tests, `request`
fixtures and xfail conditions that only existed for them).

Do **not** reintroduce `POLARS_VERSION_LT_*` flags speculatively. Add one to `versions.py` only
when the code must behave differently on two supported versions, and delete it when the lower
bound passes it. Parenthesize conditional expressions (`x == (A if cond else B)`); an unparenthesized
one was a real bug in an earlier upgrade.

### Remove run-list entries for dropped versions

PR CI selects upstream tests from `ci/cudf_polars/gpu_test_run_list.json`, which has one entry per supported Polars version
(`<version>|spmd|default`). Regenerating the list keeps entries for old versions, so when the lower bound moves, remove the entries
below it in the same PR. After the `pyproject.toml` regeneration above, run from the repo root:

```bash
python .agents/skills/update-cudf-polars-polars-tests-selection/prune-run-list.py
```

The script reads the minimum supported version from the `polars` requirement in
`python/cudf_polars/pyproject.toml`, prints each entry it removes, and rewrites
the list in place. Check that `git diff ci/cudf_polars/gpu_test_run_list.json`
only removes entries for versions below the new minimum.

The newest supported version also needs an entry, or PR CI fails before running any tests. Generate it with the
`update-cudf-polars-polars-tests-selection` skill once the upstream suite passes (Step 5).

### Build the environment the way CI does

Do this right after the pin and the generator run, so the environment matches what CI will use:

1. Edit the polars pin in `dependencies.yaml` (the `run_cudf_polars` list and the `test_cudf_polars_compat` matrix, see the table).
2. Run `rapids-dependency-file-generator` (the pre-commit hook above) so the generated files pick up the change.
3. Create an environment from `dependencies.yaml` or from a generated file. CI builds it with
   `rapids-dependency-file-generator --output conda --file-key <key> --matrix "cuda=<ver>;arch=<arch>;py=<ver>;dependencies=<latest|oldest>"`
   followed by `mamba env create -f` (`ci/test_python_common.sh`); the cudf-polars job uses the keys `test_python_other` and
   `test_cudf_polars_compat` (`ci/test_python_cudf_polars.sh`, which also reads the polars version from the matrix). For a local
   environment, `conda env create -n <name> -f conda/environments/all_cuda-<ver>_arch-<arch>.yaml` is the same set of packages.
   If the new polars release is not on conda-forge yet, `pip install polars==X.Y.Z` into the environment and run `pip check`.
4. Build or install cudf-polars and the related libraries into it: `pip install -e python/cudf_polars`, with pylibcudf and libcudf
   from the nightly channel unless you are changing C++. Confirm `python -c "import polars, cudf_polars; print(polars.__version__)"`.

To reproduce a failing CI job in its own container instead, use the `reproduce-ci` skill.

## Step 3. Adapt the translator and expressions

Run the code against the new polars and fix in this order.

1. **IR version gate.** `Translator.translate_ir` in `cudf_polars/dsl/translate.py` rejects IR versions
   it has not been validated against. The error message prints the version of the polars you are *adopting*
   (for example `(16, 0)`), which is a version you must now support, **not** the value to put in the gate. Once the
   code works against it, set the gate to the next minor, `>= (16, 1)`. Keep the `# pragma: no cover` on the raise.
2. **Enum drift.** `tests/dsl/test_serialization.py::test_from_polars_all_names` compares polars' function
   name enums with ours exactly. For each new member add `Name.X = auto()` (alphabetical) in
   `dsl/expressions/{boolean,string,struct,temporal}.py`. If it is unsupported, ensure it raises
   `NotImplementedError` so it falls back to CPU, and add a translation-raises test. Track unimplemented
   new features in a drafted issue.
3. **Changed IR/option payloads.** When a node gains or reshapes a field (for example `MergeSorted.key`
   becoming a list, a new `HConcat` slot, `Sort.slice` becoming a 3-tuple), update the translate function and
   the IR class. When an IR class signature changes, grep every constructor call (translator, `streaming/`,
   tests). `ci/check_cudf_polars_ir.py` validates `do_evaluate` signatures.
4. **New fallbacks.** If a new option cannot be honoured on GPU, raise `NotImplementedError` from the translator
   (or the node constructor) with a clear message. Add `assert_ir_translation_raises(q, engine, NotImplementedError)`
   tests; where the *upstream* test cannot pass on GPU, this is the preferred fix.
5. **Semantic changes.** Fix when small (with a unit test). If it needs libcudf work or a redesign, fall back to CPU
   and draft an issue.
6. **Pass the stream.** Pass `stream=df.stream` to every pylibcudf call you add or touch (reviewers sweep for it).
   Prefer `Column.astype` over hand-rolled cast checks, and drop `hasattr` guards for APIs that now always exist.
7. **Imports.** Use `from polars import _plr as plrs` for the private Rust bindings. mypy needs the overrides already in
   the root `pyproject.toml` for `polars._plr`.

8. **Polars constant-folds literals into scans.** In 2.1 `select(pl.lit(<struct/enum>))` becomes a `DataFrameScan`, so the translator
   never sees a `Literal` and a test that expects `NotImplementedError` from that query stops raising. Keep the raise test but build
   the query with `with_columns` (or `when/then`), which still reaches the `Literal`, and add a small `assert_gpu_result_equal` test for
   the folded `select` form. Check what a changed test now exercises (print `translator.errors`), not just that it passes.
9. **Option payloads can change shape.** 2.1 added a join type whose `options[0]` is a 2-tuple (`("Left", "cross_and_filter")`) where
   code unpacked a 3-tuple. Look at the kind of the payload before unpacking, and raise `NotImplementedError` for kinds you do not support.
10. **A polars panic cannot be caught.** If a visitor call panics (for example `get_schema()` on a `BinaryOffset` column, which polars
    maps to `unimplemented!()`), the exception is a `BaseException` and the visitor's arena lock is poisoned, so neither a catch nor the CPU
    fallback works. Do not add a try/except. Xfail the affected tests with a reason and draft an issue for `pola-rs/polars`.
11. **Optimizer rewrites that hide behind a visitor limit.** 2.1 lowers `x.sort_by(k).first()` in `group_by` to `min_by` with a "total
    order" that the Python visitor refuses to expose, so such queries fall back to CPU. Where the rewrite is skipped (for example
    `maintain_order=True`) keep the GPU path covered by passing that option in tests, and add a canary test that the default form raises.
12. **Version-specific code.** The supported range spans two polars minors, but CI measures coverage on the newest. Code reachable
    only on the older version gets `# pragma: no cover; polars 2.0 only` on its `if`/`raise` line; code now unreachable on the newer
    version one gets a note naming that version. Re-run the coverage check after every polars-induced plan change.
13. **mypy.** `DataType.polars_type` is the generic `pl.DataType`; narrow it with `assert isinstance(t, pl.Decimal)` before reading
    `.precision`/`.scale`. Run `mypy --config-file=pyproject.toml cudf_polars tests` from `python/cudf_polars` and ignore the
    `import-untyped`/`_expr_nodes` noise that comes from not having the hook's isolated environment.

Keep comments about polars behaviour short and link the polars issue/PR when there is one.

## Step 4. Run the cudf-polars unit tests with coverage

Use the environment from Step 2. If you add packages by hand, the unit tests need the test extras (`rich`, `pytest-httpserver`, `zstandard`, `psutil`, `cloudpickle`, `ray`,
`rapids-dask-dependency`, `dask-cuda`) or ray/dask-parametrized tests error at import. Installing them can silently move
`pytest` past the repo's `<9.1.0` pin, `pandas` past its pin, and even `polars`. Freeze the important pins into a constraints
file and use it for every later `pip install`:
`pip freeze | grep -iE '^(polars|polars-runtime-32|pandas|numpy|pyarrow|pylibcudf|narwhals|pytest|pytest-env)==' > constraints.txt`.

Run from the repo root with the target polars installed. Mirror CI (`ci/test_python_cudf_polars.sh`):

```bash
./ci/run_cudf_polars_pytests.sh -x --numprocesses=4 --dist=worksteal \
    --cov-config=./pyproject.toml --cov=cudf_polars --cov-fail-under=100 \
    --cov-report=term-missing:skip-covered
```

(Equivalent: `cd python/cudf_polars && python -m pytest tests ...`. `pytest-xdist` plus coverage only works from that
directory.) Notes:
- The suite runs with `filterwarnings = error` and `xfail_strict = true`, so a new `DeprecationWarning` fails a test and a
  passing xfail fails the run.
- Tests are parametrised over executors (`in-memory`, `spmd`, `ray`, `dask`). Install the extras above so all of them run. If
  one cannot, exclude it with `-k` and **say so in the PR**; never exclude a failure you do not understand.
- To triage everything in one pass use `--maxfail=300` instead of `-x`. A full run takes about 5 minutes with 4 xdist workers on
  one GPU. Use at most one GPU per concurrent job (`CUDA_VISIBLE_DEVICES`).
- For every failing test, find the root cause. If it is a small fix, fix it and keep the test. If it is a larger change, make
  the translator fall back, mark the test with a reason and a link to a drafted issue, and list it for the user.
- Delete tests that no longer assert anything, and rewrite legacy canary tests (for example "unsupported feature" tests) when the
  feature becomes supported. Do not relax `check_exact`/`check_row_order` unless unavoidable.

### Run the unit tests on the oldest supported polars too

The supported range spans two minors, and CI tests both ends. Without this check, a PR that is green on the newest polars can fail the
wheel job on the oldest, as the Polars 2.1 upgrade did (4 failures in CI, 71 in a full local run). Overlay the old polars on the same
environment so nothing else changes (no `ray`/`dask` extras are lost, and the environment stays intact):

```bash
pip install --no-deps --target <tmp>/polars-old "polars==X.Y.Z" "polars-runtime-32==X.Y.Z"
PYTHONPATH=<tmp>/polars-old:$PWD python -m pytest tests --maxfail=300 ...   # from python/cudf_polars
python -c "import sys; sys.path.insert(0, '<tmp>/polars-old'); import polars; print(polars.__version__)"   # confirm
```

What usually breaks, and the fix:

- **A test that asserts new behaviour** (a plan shape, a message, a broadcast rule): `pytest.mark.skipif(POLARS_VERSION_LT_NN, reason=...)`,
  or assert per version when both versions matter (an explain test that expects merged scans on 2.1 and a `UNION` on 2.0).
- **An xfail you deleted because it XPASSes on the new version** still fails on the old one. Keep it, conditional on the old version:
  `pytest.mark.xfail(POLARS_VERSION_LT_NN, reason=...)`. Strict xfail only fires when the condition is true.
- **A new xfail for new behaviour** needs `not POLARS_VERSION_LT_NN` in its condition, or it XPASSes (and fails) on the old version.
- Define the flag next to the others in `cudf_polars/utils/versions.py` and delete it when the lower bound passes it. Write it as
  `POLARS_VERSION_LT_NN = parse("X.Y") > POLARS_VERSION`: the pinned ruff flags `POLARS_VERSION < parse("X.Y")` as a Yoda condition (SIM300),
  and pre-commit.ci fails on it (older flags in this file predate that rule).
- Failures that mention a libcudf feature newer than the nightly in your environment (for example a test from a just-merged C++ change) are
  an environment limit, not a regression. CI builds libcudf from the branch.

### Coverage

- `cudf_polars` must stay at 100%, except the files in `[tool.coverage.report] omit` of `python/cudf_polars/pyproject.toml`.
- Code that is dead on the newest polars (kept only for older supported versions) gets `# pragma: no cover` with a short reason,
  placed on the `if` line so the whole block is excluded. Code that is now live loses its pragma.
- Defensive raises that polars validates first use `# pragma: no cover; polars validates this ahead of time`.
- New behaviour gets a real test. Fallbacks get `assert_ir_translation_raises`; parity gets `assert_gpu_result_equal`;
  error parity gets `assert_collect_raises`.
- Read the `term-missing` output and cover every reported line; do not lower the threshold.

## Step 5. Run the upstream polars test suite

CI (`ci/test_cudf_polars_polars_tests.sh` and `ci/run_cudf_polars_polars_tests.sh`) clones polars at the tag matching the installed
version and runs its `py-polars/tests` with the cudf-polars plugin injected. Use the shallow clone from Step 1 (it must be at the
installed version's tag, `python -c 'import polars; print(f"py-{polars.__version__}")'`) as `<dir>/polars`:

```bash
# install the test requirements, applying the sed patches from ci/test_cudf_polars_polars_tests.sh
# (it removes polars-cloud and pins pyarrow/duckdb; check whether each is still needed)
pip install -r <dir>/polars/py-polars/requirements-dev.txt -r <dir>/polars/py-polars/requirements-ci.txt \
    "polars-iceberg>=0.1.0,<1.0.0"
POLARS_DIR=<dir>/polars ./ci/run_cudf_polars_polars_tests.sh --engine in-memory
POLARS_DIR=<dir>/polars ./ci/run_cudf_polars_polars_tests.sh --engine spmd
POLARS_DIR=<dir>/polars ./ci/run_cudf_polars_polars_tests.sh --engine spmd --inject-gpu-engine-blocksize small
```

The script hardcodes `-x` (fail fast) because failures pollute state. To triage the whole suite locally pass `--maxfail=100000`
(a later option wins, and the deselection files and warning filters are kept), then rerun without it at the end to confirm a clean
pass. Install the requirements with the constraints file from Step 4 (`pip install -c constraints.txt -r ...`) or the resolver can
move polars/pandas/pytest. Check whether each `sed` pin in the CI script is still needed (for example `duckdb<2` once polars
supports duckdb 2). Long runs: use a timeout and the default `-m "not slow"`; run `--run-slow` once at the end.

### GPU test selection in PR CI

PR CI does not run the whole upstream suite. It runs only the tests listed in `ci/cudf_polars/gpu_test_run_list.json`, under the key
`<exact polars version>|spmd|default` (`CUDF_POLARS_GPU_RUN_LIST` in `.github/workflows/pr.yaml`). A selected test that passes must have
executed a query on the GPU in its test body. Nightlies run the full suites and audit the list. To run the PR configuration locally:

```bash
CUDF_POLARS_GPU_RUN_LIST=ci/cudf_polars/gpu_test_run_list.json POLARS_DIR=<dir>/polars \
    ./ci/run_cudf_polars_polars_tests.sh --engine spmd --inject-gpu-engine-blocksize default
```

What this means for an upgrade:

- **The new Polars version needs its own entry, including a patch release.** Without one, PR CI stops with
  `Cannot select GPU tests for 2.2.0|spmd|default`. Generate it with the `update-cudf-polars-polars-tests-selection` skill. Use complete
  nightly telemetry for the target revision when it exists. Only when none exists, bootstrap the entry with a full default-blocksize SPMD
  run with `--run-slow` on every shard, all exiting zero. In that case, triage the failures below first and regenerate the list last.
- **Renamed or removed upstream tests** fail PR CI with `Stale GPU run-list nodes`. Update the list in the same PR.
- **Turning a GPU execution into a CPU fallback fails the selected test** with `Selected GPU test passed without observed GPU
  execution`. When a fallback is the right fix (triage step 2 below), also remove that node from the run list.
- Skipped or xfailed items are not checked for GPU execution, so the xfail and skip dicts keep working as before.
- When the PR raises the minimum supported Polars version, prune the entries for dropped versions in the same PR (see "Remove run-list
  entries for dropped versions" in Step 2).

### Triage each failure

1. Real, small bug: fix it in cudf-polars and add a unit test.
2. Unsupported feature: make the translator fall back to CPU (the test then passes). Preferred over xfail. If the test is in the GPU run
   list, remove its node from the list as well.
3. Larger fix needed: add the test to `EXPECTED_FAILURES` (or the streaming variants) in
   `python/cudf_polars/cudf_polars/testing/inject_gpu_engine.py` with a reason **and an issue link**.
4. Flaky, non-deterministic, hangs, only checks CPU logs, or would XPASS on one engine and fail on another: `TESTS_TO_SKIP`
   (or the streaming skip dicts).

Conventions for the dicts (`EXPECTED_FAILURES`, `TESTS_TO_SKIP`, `STREAMING_ENGINE_EXPECTED_FAILURES`,
`STREAMING_ENGINE_TESTS_TO_SKIP`, and the `*_SMALL_BLOCKSIZE` variants):
- Key: pytest node id relative to `py-polars` with exact parameter ids, `"tests/unit/io/test_csv.py::test_x[param]": "reason"`.
  Polars often re-parametrizes tests, so stale ids must be updated.
- Reason: a cudf issue URL, a polars issue/PR URL, or the established phrases ("Debug output on stderr doesn't match",
  "Casting that raises not supported on GPU", ...). Streaming engines wrap errors in an `ExceptionGroup`, which has its own phrase.
- Conditional entries use plain Python guards after the dict (not tuples).
- **Audit the existing entries.** xfails are strict: remove entries whose test now passes (XPASS fails the run) or no longer exists.
  The Polars 2.0 audit removed 76 and added about 30.
- Wrong-result failures should become fallbacks, not xfails. In the 2.1 upgrade several "failures" were real wrong results
  (integer-to-`Date` casts relabelled an `INT64` buffer as `INT32`; mixed-scale `Decimal` comparisons wrapped in libcudf; non-strict
  narrowing casts wrapped instead of returning null). Compare the GPU result with the CPU result before assuming a test is flaky.
- When a fix turns a fallback into correct GPU execution (or the reverse), the matching xfail becomes an XPASS: rerun the whole
  upstream suite after merging fixes and delete the stale entries.
- SPMD/streaming failures where the right polars error arrives wrapped in an `ExceptionGroup` go in
  `STREAMING_ENGINE_EXPECTED_FAILURES` with the established phrase ("Correct polars.exceptions.X raised but it's in an ExceptionGroup").

If the new polars adds an optional dependency that its tests need, install it too: read the `[project.optional-dependencies]` of the cloned
`py-polars/pyproject.toml` (2.1 needs `polars-iceberg`, without which ~110 Iceberg/catalog tests fail with a `PerformanceWarning`). Add it to
`ci/test_cudf_polars_polars_tests.sh` unconditionally: CI only runs the upstream tests against the installed Polars version.

If an upstream test requirement breaks (a new dependency release or a warning under `-W error`), patch the requirement with `sed -i`
in `ci/test_cudf_polars_polars_tests.sh` or add a `-W ignore:` filter in `ci/run_cudf_polars_polars_tests.sh`, with a comment that
links the upstream fix so it can be removed later. Tests that cannot run in CI go in `ci/cudf_polars/polars_test_deselections*.txt`.

## Step 6. Narwhals

For a polars upgrade only the **cudf-polars** variant has to pass (`narwhals_cudf_polars_test_plugin` in `ci/test_narwhals.sh`). The
cudf and cudf.pandas variants test cudf and pandas, not polars. If Narwhals itself has to be bumped, load and follow the
`update-narwhals-testing` skill, which requires all three variants to pass.

1. Check whether the pinned Narwhals (`dependencies.yaml`, `depends_on_narwhals`) supports the new polars. Only bump it if it does not,
   or if the user asks.
2. Run the cudf-polars variant with the pinned Narwhals checked out and installed in the environment, exactly as `ci/test_narwhals.sh`
   does (`NARWHALS_POLARS_GPU=1`, `--constructors=polars[lazy]`, `-p narwhals_cudf_polars_test_plugin`). It needs the NUMA pinning from
   the practical notes.
3. For each failure, root-cause it. Typical causes are a changed polars error message, a removed API, a new deprecation warning or a
   polars panic. Add an `EXPECTED_FAILURES` entry with a reason to `ci/narwhals_cudf_polars_test_plugin.py` and link an issue.
4. Remove entries that now pass (they are strict xfails) or whose tests no longer exist.
5. Local-environment noise (for example a newer duckdb API breaking the duckdb tests, or the version test) is not a regression: compare
   with CI before touching anything.

## Step 7. Final checks

- `pre-commit run --all-files` (mypy, ruff, the dependency-file generator, `ci/check_cudf_polars_ir.py`). The mypy hook installs
  its own polars from `.pre-commit-config.yaml`, so confirm that pin was updated.
- `git diff --stat` after any merge (an earlier PR accidentally pulled in unrelated Java changes).
- Run the sync check from "Resuming work on an open PR" before every push.
- Run the cudf-polars unit tests on the oldest supported polars (Step 4) and confirm the upstream suites ran on the newest.
- Confirm `ci/cudf_polars/gpu_test_run_list.json` has an entry for the new Polars version, has no stale or fallback-only nodes, and that
  the PR configuration command in Step 5 passes locally.
- Grep again for stale version guards and for `TODO`s that mention the old version.
- Write the PR description: old and new range, the user-visible behaviour changes, every new fallback, every new xfail/skip with its
  issue, the commands you ran and their results, and the follow-up issues drafted.

## Drafting issues

Every xfail/skip/fallback that stands for a real gap needs an issue the maintainers can act on without asking questions. Draft each as
a markdown file for the user to review; open an issue only when asked, and open issues in other repositories (for example
`pola-rs/polars`) only with an explicit go-ahead. Title format `[BUG] ...` or `[FEA] ...`; labels `bug` or
`feature request` plus `cudf-polars`. Use this structure (modelled on a well-received issue about hanging upstream tests):

1. **Description.** Plain-language symptom first: what happens (wrong result, exception, hang), how it shows up in CI, and what still works
   (other engine, default blocksize, plain Polars CPU), which narrows the cause to cudf-polars. No jargon the reader must decode.
2. **Affected tests.** The exact node ids, in a bullet list, with parameters, and say which are `slow` or otherwise need flags.
3. **How to reproduce.** One copy-pasteable command with every environment variable and flag, run from a stated checkout (for example
   `pola-rs/polars` at tag `py-X.Y.Z`, in `py-polars`), plus the observed result (exit code, error text, the contrast with the passing
   configuration). Prefer a minimal Polars snippet when the failure is not about the test harness.
4. **What I saw in CI** (only if CI has actually run it; for a local pre-merge upgrade write "not yet run in CI" or omit). Workflow run
   id, job names, commit, how many jobs hit it, and where it stopped. State plainly what you did **not** verify (for example "the other hangs may not share one cause"). Do not speculate beyond the evidence.
5. **Root cause / suspected location.** File and function when known (for example a stack dump pointing at a specific actor).
6. **Workaround.** Where the skip/xfail/fallback lives in the repo (file and dict name) and that it should be removed when the issue is fixed.
7. **Caveats.** Anything the workaround may be hiding (for example a skipped hang hides every test behind it).

Link the issue number in the xfail reason (`.../issues/NNNNN`) once the user has opened it. Until then use `TODO: <draft filename>`.

When the user asks you to open them: `gh issue create -R NVIDIA/cudf --title ... --body-file ... --label bug --label cudf-polars`
(use `--label libcudf` for libcudf reader/kernel bugs and `--label "feature request"` for `[FEA]`). End each body with a line linking the
upgrade PR ("Found while upgrading cudf-polars to Polars X.Y in <PR url>"). Verify the labels with `gh issue view <n> --json labels`, then
replace the `TODO` placeholders in the dicts with the issue URLs and commit. Group tests that share a cause into one issue.

## Reference: where things live

- Pins: `dependencies.yaml`, `python/cudf_polars/pyproject.toml`, `conda/recipes/cudf-polars/recipe.yaml`, `.pre-commit-config.yaml`
- Version helpers: `python/cudf_polars/cudf_polars/utils/versions.py`
- Translator: `python/cudf_polars/cudf_polars/dsl/translate.py`; IR in `dsl/ir.py`; expressions in `dsl/expressions/`
- Test helpers: `cudf_polars/testing/asserts.py` (`assert_gpu_result_equal`, `assert_ir_translation_raises`, `assert_collect_raises`)
- Upstream test lists: `python/cudf_polars/cudf_polars/testing/inject_gpu_engine.py`
- CI scripts: `ci/test_python_cudf_polars.sh`, `ci/run_cudf_polars_pytests.sh`, `ci/test_cudf_polars_polars_tests.sh`,
  `ci/run_cudf_polars_polars_tests.sh`, `ci/test_narwhals.sh`, `ci/narwhals_cudf_*_test_plugin.py`
- GPU test selection: `ci/cudf_polars/gpu_test_run_list.json`, `ci/analyze_cudf_polars_fallbacks.py`,
  `python/cudf_polars/cudf_polars/testing/{observation,fallback,fallback_report}.py`, and the
  `update-cudf-polars-polars-tests-selection` skill
- Coverage config: `[tool.coverage.report]` in `python/cudf_polars/pyproject.toml`
