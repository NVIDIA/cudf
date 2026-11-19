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
  list (see `references/upstream-tests.md`), so a green PR says nothing about the unselected tests.
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

## Resuming work on an open PR

The PR branch on GitHub can change while you are away (squash, force-push, accepted suggestions, web edits). Before you edit,
commit or push, and again before every push, fetch and compare with the PR branch on the remote that hosts the PR head (usually your fork). If history was rewritten, stop and ask the user.

- Re-read the PR before changing it (`gh pr view <n> --json headRefOid,body,reviews,comments`, `gh pr checks <n>`). The user may have
  edited the description, so merge your change into the current body instead of overwriting it.
- CI does not start on its own for a draft PR: the user has to comment `/ok to test <sha>` for the new head.
- State in the PR which commit the last full test run was on, so later edits can be judged against it.

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

Run the code against the new polars and fix in this order: the IR version gate in `Translator.translate_ir`, enum drift
(`test_from_polars_all_names`), changed IR/option payloads, new CPU fallbacks, then semantic changes. Read
`references/translator-fixes.md` before starting; it lists each case with the fix that worked in earlier upgrades.

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

Then run the unit tests on the **oldest** supported polars too and bring coverage back to 100%; follow
`references/unit-tests-oldest-and-coverage.md`.

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

PR CI runs only the tests in `ci/cudf_polars/gpu_test_run_list.json`, and the new Polars version needs its own entry. For the
run-list rules, how to triage each failure, and the conventions for the xfail/skip dicts, follow `references/upstream-tests.md`.

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
- Run the cudf-polars unit tests on the oldest supported polars (`references/unit-tests-oldest-and-coverage.md`) and confirm the upstream suites ran on the newest.
- Confirm `ci/cudf_polars/gpu_test_run_list.json` has an entry for the new Polars version, has no stale or fallback-only nodes, and that
  the PR configuration command in `references/upstream-tests.md` passes locally.
- Grep again for stale version guards and for `TODO`s that mention the old version.
- Write the PR description: old and new range, the user-visible behaviour changes, every new fallback, every new xfail/skip with its
  issue, the commands you ran and their results, and the follow-up issues drafted.

## Drafting issues

Every xfail/skip/fallback that stands for a real gap needs an issue. Draft each as a markdown file for the user to review, and open
issues only when asked. Follow `references/drafting-issues.md` for the structure, labels and `gh` commands.

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
