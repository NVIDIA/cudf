# Example issues

Long issues by cuDF maintainers (vyasr, wence-, Matt711) that show each structure well. Read the one that matches the kind of issue you are writing before you draft, with `gh issue view <number> -R NVIDIA/cudf`. Each entry says what to take from it.

## Stories

- [#23965](https://github.com/NVIDIA/cudf/issues/23965) (vyasr), *Rewrite parquet decode kernels around a staged, page-global preprocessing pass*. Problem that explains why the old design made sense once and why it no longer does; a proposal with the data structure it introduces; phases behind one environment variable so each step merges and is measured alone; risks and rollback.
- [#16251](https://github.com/NVIDIA/cudf/issues/16251) (vyasr), *Enabling prefetching of unified memory*. Problem statement tied to users (cudf.pandas OOMs); a short-term proposal separate from long-term plans; the plan split by component (pointer accessors, rmm data structures, cuIO).
- [#21456](https://github.com/NVIDIA/cudf/issues/21456) (Matt711), *JIT Compilation in cudf-polars*. Motivation, background on the three libcudf entry points and their tradeoff, tracking PRs, then one section per area with benchmarks and `[TODO]` for the areas not started.
- [#16480](https://github.com/NVIDIA/cudf/issues/16480) (wence-), *Full coverage of stringfunction methods in cudf polars*. A coverage tracker: the goal in two sentences, then a table with one row per Polars method, its status and notes.

## Feature requests and enhancements

- [#17578](https://github.com/NVIDIA/cudf/issues/17578) (wence-), *Clean up rolling window interfaces*. Survey of the current API and of what each consumer (Polars, pandas, Spark) needs, the implementation problems that follow, then numbered concrete requests.
- [#13544](https://github.com/NVIDIA/cudf/issues/13544) (wence-), *Reimagining cudf user-input boundaries*. tl;dr first; status quo with examples; numbered problems; a proposal grounded in a named design approach; further reading.
- [#11990](https://github.com/NVIDIA/cudf/issues/11990) (wence-), *actually inplace updates in `__setitem__` and friends*. Context, why it happens, why the status quo is a problem, possible solutions.
- [#13744](https://github.com/NVIDIA/cudf/issues/13744) (vyasr), *Expose public stream-ordered C++ APIs*. A large API change broken into lists per namespace so it can be split across PRs.

## Bugs

- [#24843](https://github.com/NVIDIA/cudf/issues/24843) (Matt711), *multi-key `sort_by(...).first()` in `group_by` panics with Polars 2.1*. Description, affected tests, reproducer with and without cuDF, CI run and job ids, root cause traced into Polars at a release tag, why it cannot be worked around in cuDF, caveats.
- [#24740](https://github.com/NVIDIA/cudf/issues/24740) (Matt711), *Polars tests hang under the SPMD engine with a small blocksize*. The same structure for a hang.
- [#14229](https://github.com/NVIDIA/cudf/issues/14229) (wence-), *Potential for use-after-free on libcudf/cudf interface boundaries*. A design-level bug: background on how each layer works, with permalinks into rmm, then the failure and how to fix it.
- [#12667](https://github.com/NVIDIA/cudf/issues/12667) and [#13748](https://github.com/NVIDIA/cudf/issues/13748) (wence-). Sanitizer and intermittent failures: the CI run that failed, how to reproduce locally, the tool output, and what was ruled out.

## Performance

- [#23526](https://github.com/NVIDIA/cudf/issues/23526) (Matt711), *Cache parquet metadata at the session level*. Problem with a profile from a named instance and scale factor, a proposed fix with the cache key and why it is safe, related issues.
- [#25132](https://github.com/NVIDIA/cudf/issues/25132) (Matt711), *cudf-polars: separate fetching a scan task's bytes from decoding them*. Opens with the measurement conditions (dataset, instance, iterations, noise), then the mechanism measured directly (bytes held, peak memory, when), then the proposal and prior art.
