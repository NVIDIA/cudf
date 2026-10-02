# Measuring libcudf build and size changes

## Record enough to repeat the comparison

Capture base and candidate SHAs, dirty patch if any, CUDA/host compiler and CCCL versions,
dependency pins, CMake cache, generated compile commands, CPU/RAM limits, Ninja job count,
nvcc internal parallelism,
build type, SASS/PTX architecture list, compression/LTO/RDC settings, and enabled targets.
For GPU work, record GPU UUID/PCI identity and driver; check that the selected device is
idle. Do not assume a CUDA ordinal identifies the same GPU across containers.

Use separate build directories with matching settings. Avoid reconfiguring someone else's
benchmark directory. Pre-provision matching dependencies so download/configure time is
not silently counted as CUDA compile time. If measuring an end-to-end build, define
whether configuration, dependencies, tests, benchmarks, and linking are included.

For iteration, an explicit single GPU architecture can isolate a source change. `NATIVE`
may resolve to multiple architectures on a multi-GPU host; record the generated SASS/PTX
flags rather than calling it a single-architecture build. For shipped-size claims use the
supported production matrix, or clearly label projections as unverified estimates. CI
metrics help only if their compiler-cache state, architecture set, and flags are known.

## Uncached compiler measurements

Clear inherited launcher environment variables and configure each measurement directory
with the project's usual options plus:

```sh
-DCMAKE_C_COMPILER_LAUNCHER= \
-DCMAKE_CXX_COMPILER_LAUNCHER= \
-DCMAKE_CUDA_COMPILER_LAUNCHER= \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Inspect `CMakeCache.txt`, `compile_commands.json`, and the generated Ninja commands for
`sccache`, `ccache`, distributed launchers, or compiler wrappers. A cache-disable variable
alone is not sufficient evidence that direct compilation occurred. Preserve complete
commands, exit codes, and logs; a failed compile does not produce a useful timing sample.

The review of [#17234](https://github.com/NVIDIA/cudf/pull/17234) caught timings distorted
by sccache hits. Its later all-architecture table also showed several CUDA compiles slowing
down despite smaller objects: reducing device dispatch or binary bytes is not sufficient
evidence of faster compilation.

Start with a fresh measurement log and force the intended object set to compile in both
build directories. Include *all* replacement TUs when a source is split, and all removed
objects on the baseline side. Repeat paired runs when variation could explain the result;
alternate ordering and avoid competing CPU/GPU jobs.

Run the report utility **from the build directory containing the objects**, and keep its
input log in that same directory. The current parser checks object existence relative to
the working directory, then reads size relative to the log directory. Running at the repo
root can silently report zero bytes; relocating the log alone can also cause an exception.

Set absolute paths for one side of the comparison, then snapshot it immediately after the
measured build succeeds, before rebuilding or removing any objects:

```sh
cudf_source=/absolute/path/to/cudf
cudf_build=/absolute/path/to/base-build
cudf_report=/absolute/path/to/results/base
mkdir -p "$cudf_report"
(
  cd "$cudf_build" || exit 1
  python3 "$cudf_source/cpp/scripts/sort_ninja_log.py" .ninja_log --fmt csv \
    > "$cudf_report/compile.csv" &&
  python3 "$cudf_source/cpp/scripts/sort_ninja_log.py" .ninja_log --fmt html \
    > "$cudf_report/compile.html" &&
  cp .ninja_log "$cudf_report/ninja-log.txt"
)
```

Repeat with the candidate build/results directories. Compare the two saved CSVs; their
`time` values are milliseconds and `size` values are bytes. The copied raw log is archival:
do not feed it back to the parser at its new location expecting valid object sizes.
Snapshot CSVs before overwriting artifacts because logs do not store historical sizes.
Avoid cross-tree `--cmp_log` for size accounting: baseline existence checks use the current
working directory, so removed/renamed objects can be misreported.

For an isolated iteration, rebuild the complete affected object set without caches;
there is no need to rebuild unrelated objects after every edit. For a full clean-build
wall-time claim, run comparable clean builds at fixed concurrency. Record both Ninja job
count and nvcc internal parallelism; do not overwrite existing CUDA flags just to fix one.

The parser resets its selected records when end timestamps decrease and deduplicates by
command hash with `setdefault`. A mixed incremental log is therefore not a reliable record
of an arbitrary experiment, and a repeat is not guaranteed to replace an older entry.
Use a dedicated measurement directory and fresh log for each round, archiving earlier
reports first. Check that all expected compile outputs appear, that object sizes are
nonzero for existing nonempty files, and that removed baseline TUs remain in the comparison.
Do not sum all log rows blindly: link and other build edges are not compiler invocations.
Report:

| Metric | Meaning and boundary |
| --- | --- |
| Longest affected TU | Worst individual compiler invocation; not necessarily the entire build bottleneck |
| Summed affected compile durations | Total elapsed compiler-task seconds, potentially overlapping |
| Parallel target wall time | Measured start-to-finish time with a fixed job count and defined target set |
| Full clean build wall time | Includes the actual build dependency graph; separate from an isolated target experiment |
| Compiler peak RSS | Measure expensive invocations consistently, for example with `/usr/bin/time -v`; not aggregate concurrent build memory |
| Incremental rebuild cost | Explicitly name the edited header/source and invalidated targets |

#23322 is the useful counterexample: longest TU fell from 662s to 103s while summed CUDA
compilation stayed around 662s versus 665s. #10756 is a compiler-memory precedent:
14.6 GB to 2.4 GB peak RSS. These are historical PR-reported results, not current baselines.

## Binary and kernel accounting

Record exact file bytes rather than rounded `ls -h` values. Compare like-for-like artifacts:

```sh
stat --format='%s %n' /path/to/libcudf.so
readelf -SW /path/to/libcudf.so
nm -D --defined-only -C /path/to/libcudf.so
cuobjdump --list-elf /path/to/libcudf.so
```

Use the installed toolkit's `cuobjdump --help` to select extraction/disassembly options.
Compare kernel identities and code within the same architecture; repeated images for
different supported architectures are intentional. Match names, code, and launch/type
semantics before deciding two kernels are interchangeable. Preserve local artifacts and
hashes when correlating source, objects, fatbins, and runtime measurements.

Distinguish summed object bytes, host ELF sections, compressed `.nv_fatbin`, extracted
uncompressed device code, final `.so`, and packaged wheel/JAR size. Host linker deduplication
does not establish that equivalent kernels in separate fatbins were removed. Conversely,
an object-size reduction does not establish a linked-library reduction.

Check the exported ABI before/after (especially missing strong/public symbols), CMake
source membership, link success, and runtime launches. Do not strip production symbols,
drop supported architectures, or change compression/assertion settings merely to make
an instantiation comparison look better. #7583 and #18755 concern compression/build
policy; those are separate mechanisms and require their own compatibility measurements.

A single-architecture prototype is useful for attribution, but extrapolated full-matrix
savings remain estimates. #23419 explicitly distinguishes such estimates from subsequent
all-architecture results. Do not add percentage savings from PRs with different baselines.

## Correctness and runtime coverage

Choose validation from what the change can affect, not merely its label as a cleanup.
Keep required repository checks and measure the compilation/size metric being claimed.

| Change | Validation to start with | When to add runtime comparisons |
| --- | --- | --- |
| Documentation or skill guidance | Check links, structure, and executable examples that changed | No GPU benchmark is needed |
| Include cleanup, host-only moves, or mechanical instantiation ownership changes | Build affected consumers and configurations; link; run affected existing tests; check symbols when ownership/visibility changes | When generated device code, launch behavior, or equivalence is uncertain |
| Dispatch/type erasure, inlining, materialization, algorithm, preprocessing, or allocation changes | Affected tests covering the changed dimensions, plus paired benchmarks and relevant scratch/peak-memory measurements | Required for paths whose runtime behavior can change |

Capture the relevant runtime baseline before editing when a comparison is needed.
Do not run an unrelated GPU suite for every header or documentation edit. If source moves
change the compiler's optimization scope, treat them as potentially affecting runtime
rather than assuming that moved code generates identical kernels.

Select existing suites using the current build's available targets. Depending on the path,
cover joins/search, groupby/streaming groupby, reductions, rolling, sorting, strings, and
affected test utilities. Add focused regression cases only for uncovered semantics.
Check nullable and non-nullable, allocated-all-valid masks, NaNs and signed zero, nested
child nulls, dictionary remapping, supported decimals/chrono types, empty inputs, and
operation-specific boundaries. For explicit instantiations, test every supported selector
and the actual group/block parameters. Compile Debug or multiple supported CUDA versions
when the change depends on inlining, template lookup, or compiler behavior.

When runtime comparisons are needed, run paired benchmarks on identical datasets/seeds
and GPU conditions. Include setup,
preprocessing, scratch allocation, and synchronization if changed. Separate reusable-state
and one-shot paths when relevant. Report median/variation and material outliers, not just
a geometric mean; distinguish noise in tiny cases from repeatable regressions. Profile
registers, occupancy, launches, and traffic when needed to explain a tradeoff. A compiler
speedup does not justify silently accepting runtime or peak-memory regressions.

The proposed global-memory dispatcher and reduction materialization in #21973 regressed
runtime despite compiling faster. The report itself asks for independent revalidation.
Its claim that splitting always improves wall time is not a guarantee under limited jobs,
RAM pressure, duplicated parsing, or a different dependency critical path.

## Decide whether to keep the change

Agree on the target metric and use any runtime/memory constraints already set for the task.
Do not invent a universal speedup threshold or silently trade away supported behavior.

- **Keep:** comparable measurements show a repeatable improvement in the stated metric,
  appropriate correctness/link checks pass, runtime and memory costs satisfy the task's
  constraints, and the added maintenance cost is justified by the benefit.
- **Rework:** the compile/size benefit is real but a particular path regresses, scratch
  grows too much, or the implementation adds unnecessary complexity. Narrow the change,
  retain the original path where appropriate, and remeasure the changed cases.
- **Discard:** the apparent gain disappears under comparable conditions, the change breaks
  required behavior, or its costs outweigh the benefit. Restore only the experiment's own
  edits; preserve unrelated work and record the useful negative result.
- **Inconclusive:** noise, missing coverage, or unavailable hardware prevents a decision.
  State the limitation and needed evidence; do not present the candidate as a demonstrated
  improvement. Do not keep repeating equivalent tests without a reason they will resolve it.

## Result format

Lead with the problem and final behavior. Use only applicable metrics; this template is a
reporting aid, not a requirement to publish or fill every field. Build-time or size work
alone does not authorize commits or a PR.

```markdown
<Operation/TU> compiled <redundant combinations or repeated implementation>.
This change <mechanism and ownership change>, preserving <relevant behavior>.

Compared <base SHA> with <candidate SHA/patch> using <CUDA, host compiler, dependencies>,
<build mode>, <exact SASS/PTX architectures>, <Ninja jobs/nvcc threads>, no compiler cache.

| Metric | Base | Candidate | Scope |
| --- | ---: | ---: | --- |
| Longest affected CUDA compile, s | ... | ... | Named TU set |
| Summed affected compile durations, s | ... | ... | Includes removed/replacement TUs |
| Parallel build wall time, s | ... | ... | Exact targets; clean or incremental |
| Affected object bytes | ... | ... | Complete object set |
| libcudf.so / .nv_fatbin bytes | ... | ... | Identical compression and architectures |
| Compiler peak RSS / runtime scratch | ... | ... | Where relevant |

Validation: <existing suites and focused cases>; <benchmark cases, GPU UUID,
paired-run variation, material outliers and regressions>; <ABI/link checks>.
Evidence: <commands/configuration/log paths>. Limitations: <untested or estimated scope>.
```

Separate historically reported measurements from newly reproduced evidence. Do not call a
proposal successful solely because it compiles, its code is shorter, or an earlier PR used
it. Do not turn a fixed percentage threshold into proof of noise or a universal acceptance
criterion: investigate repeatable regressions in the user's relevant workload, including
small deltas with large absolute cost. Explain tradeoffs and leave unresolved results visible.
