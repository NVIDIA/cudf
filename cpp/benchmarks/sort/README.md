# Experimental single-column string sorting

This branch keeps prefix merge as the default and exposes the faithful segmented
implementation for manual experiments. These are internal controls, not public cuDF
API. Multi-column sorting is outside this experiment.

## Runtime controls

| Environment variable | Supported values | Default |
|---|---|---|
| `LIBCUDF_STRING_SORT_ALGORITHM` | `0`: prefix merge; `1`: segmented; `2`: segmented with terminal exact-duplicate elimination; `3`: tuned R8 LRB refinement | `0` |
| `LIBCUDF_SEGMENTED_STRING_SORT_LEXIC_PRECISION` | Maximum number of eight-byte radix passes, `1..255` | `1` |
| `LIBCUDF_SEGMENTED_STRING_SORT_RADIX_RUN_MIN` | Minimum tied-run length eligible for another radix pass, `2..1048576` | `512` |
| `LIBCUDF_SEGMENTED_STRING_SORT_TRACE` | `0`: off; `1`: configuration and per-pass/finish statistics on stderr | `0` |

The segmented controls do not affect selector `0`. Settings are cached on first use;
start a fresh process for every configuration. Unset, invalid, or out-of-range values
use their defaults, without an error. Check the effective configuration with trace
before timing, then turn trace off: its additional readbacks perturb measurements.
Degenerate identity fast paths may return before emitting trace.

Precision is a **pass budget**, not a promise to execute every pass. Resolved runs
stop early; small runs and short-string/zero-padding collisions go to safe comparison
finishing. Proven-prefix skipping is always enabled. Keys are eight bytes and comparison
chunks are 512 items; chunk size and extraction launch parameters are compile-time
constants. Older six-byte, percentage-depth, adaptive-RLE, and compact-finish/radix
environment controls are no longer supported and have no effect.

## Radix LRB comparison path

Selector `3` builds on the measured R8 configuration from `string-sort-prefix-variants`:
8-byte prefix keys with separate 32-bit row IDs, bounded funnel extraction and 8-byte suffix
comparisons, exact run-length encoding, logarithmic binning, hybrid warp refinement (groups
2/4/8 use shared bitonic comparisons, 16/32 use CUB WarpMergeSort), tiered block refinement,
1024-row giant tiles, hierarchical merges, and a launch cap of 32 warps per SM. The measured
kernel implementation is retained in `sort_string_radix_refine.cuh`; its wrapper and comparison
helpers are isolated in `string_sort_lrb.cuh`. The PR's selectors `0`, `1`, and `2` retain their
original algorithms and tuning defaults. Prefix merge remains the default.

Native R8 refinement also uses typed 32-/64-bit offsets and a three-way suffix comparison
for stable warp sorting. Runs longer than 32,768 rows may receive an exact tiled
common-prefix/duplicate scan. Binning compares a reference with three sampled rows (near
one-third, two-thirds and the end); samples only decide whether the exact scan is worthwhile.
Promotion requires at least 24 common bytes, sampled duplicates no longer than 128 bytes,
or an equal-length short-tail candidate spanning the input. The full scan checks every row.
Proven identical runs skip sorting and final copying; other promoted runs compare after the
minimum proven common prefix. If a single non-null run covers the whole input, has equal
string lengths and a remaining suffix of 1–8 bytes, a stable global suffix radix sort finishes
it using the original key buffers. Otherwise the original tile/merge workers finish the run.

Native scheduling reads bin counts once. It reads one additional 16-byte proof record only
when a full-input shortcut is possible. Sample promotion is folded into binning, so an
unpromising input adds no proof allocation or proof kernels. These are fixed R8 tunings;
there is no prototype experiment selector or cardinality sketch.

`LIBCUDF_RADIX_LRB_STRING_SORT_SCHEDULE=0` uses native metadata scheduling (default).
Set it to `2` for guarded device scheduling and CUDA graph capture/replay. The guarded
path keeps metadata on device and uses the original comparison finish without the new
proof/suffix-radix shortcuts. Other values use `0`.
PR segmented precision/cutoff/trace settings affect only selectors `1` and `2`. The historical
`CUDF_STRING_SORT_VARIANT` and radix tuning variables no longer select algorithms; the old
branch is preserved as `backup/string-sort-prefix-variants-before-pr24498-20261008`.

The shared benchmark helpers support `CUDF_STRING_SORT_BENCH_STABLE=1` and
`CUDF_STRING_SORT_BENCH_DESCENDING=1` for every selector. Defaults are unstable ascending.
`sorted_order_strings_segments` adds deterministic shuffled prefix-tie runs spanning singleton,
subgroup, block, logarithmic, skewed, and giant cases, with 0/64 additional shared suffix bytes.
It checks the output permutation and ordering before timed runs, and equal-row order when stable.
Keep permutation generation (`sorted_order`) separate from end-to-end string gathering (`sort`).

```bash
# Same binary, inputs, and stable/direction controls; fresh process per configuration.
LIBCUDF_STRING_SORT_ALGORITHM=1 LIBCUDF_SEGMENTED_STRING_SORT_LEXIC_PRECISION=32 \
  cpp/build/conda/cuda-13.3/latest/benchmarks/SORT_NVBENCH -d 0 \
  -b sorted_order_strings_segments -a num_rows=262144 -a shared_suffix=64 \
  -a segment_profile=logarithmic --min-samples 20 --target-samples 20 \
  --stopping-criterion sample-count --no-batch
LIBCUDF_STRING_SORT_ALGORITHM=3 \
  cpp/build/conda/cuda-13.3/latest/benchmarks/SORT_NVBENCH -d 0 \
  -b sorted_order_strings_segments -a num_rows=262144 -a shared_suffix=64 \
  -a segment_profile=logarithmic --min-samples 20 --target-samples 20 \
  --stopping-criterion sample-count --no-batch
```

Use an external timeout when testing PR precision `1` on giant unresolved runs. Its comparison
finish can be expensive. Report timed-out cases separately; do not substitute a timeout for a
completed timing or silently omit it from a full-suite ranking.

## Bounded comparison

Run from the repository root inside the `coder` devcontainer, using an already-built
`cpp/build/latest/benchmarks/SORT_NVBENCH`. Before any rebuild, require working credentials,
a healthy sccache-dist scheduler and available server, reset statistics, run the full
incremental `ninja -C cpp/build/latest`, and verify zero distributed compilation failures.

List GPUs and replace `GPU_UUID` below with an idle physical GPU's UUID. Device `0`
inside NVBench then refers to that masked GPU. Recheck occupancy before each process;
an idle precheck is not an exclusive reservation, so monitor other clients during runs.

```bash
nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used --format=csv
export CUDA_VISIBLE_DEVICES=GPU_UUID
artifact_dir=$(mktemp -d /home/coder/.cache/string-sort-tuning.XXXXXX)
precision=1
cutoff=512
rows=32768
width=32
workload=normal

for run in 1 2 3; do
  for mode in 0 1 2; do
    test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
      --query-compute-apps=pid --format=csv,noheader,nounits)" || break 2
    name="$artifact_dir/mode${mode}-p${precision}-cutoff${cutoff}-run${run}"
    env LIBCUDF_STRING_SORT_ALGORITHM="$mode" \
      LIBCUDF_SEGMENTED_STRING_SORT_LEXIC_PRECISION="$precision" \
      LIBCUDF_SEGMENTED_STRING_SORT_RADIX_RUN_MIN="$cutoff" \
      LIBCUDF_SEGMENTED_STRING_SORT_TRACE=0 \
      timeout --signal=TERM --kill-after=10s 180s \
      cpp/build/latest/benchmarks/SORT_NVBENCH -d 0 \
        -b sorted_order_strings_workload \
        -a num_rows="$rows" -a min_width=0 -a max_width="$width" \
        -a workload="$workload" \
        --stopping-criterion sample-count --min-samples 20 --target-samples 20 \
        --timeout 120 --json "$name.json" --csv "$name.csv" \
        > "$name.log" 2>&1 || break 2
  done
done
```

Keep artifacts outside the worktree. Record the SHA, binary hashes, GPU/compiler identity,
exact commands/environment, and cache statistics alongside JSON/CSV/logs. Use identical
axes and the same binary for every selector; these generators use deterministic seeds.
Use `sort_strings_workload` with the same axes for end-to-end sorting, and report it
separately from permutation generation. Existing benchmark families remain unchanged.

Require exactly 20 samples in each completed case. A timeout, skipped case, or partial
output is not a valid comparison. Repeat noise above 2%, conflicting directions, and
deltas within measured noise. Compare all three pairs with NVBench's JSON comparison
tool (`compare.py`, or `python/scripts/nvbench_compare.py` in newer NVBench checkouts),
recording its revision. Report peak memory as well as GPU time; benchmark peak memory
excludes the pre-generated input but includes allocations such as the returned permutation.

## Avoid runaway experiments

Start with one small case, then estimate larger-run costs before starting a matrix.
With precision `1`, long common prefixes can leave a huge unresolved run. The all-sibling
chunk finish has quadratic chunk-pair work; the 2.1M-row, width-256 shared-prefix case
takes about 35 minutes **per sort** on V100. Twenty samples alone would take nearly
12 hours. NVBench's timeout is not a reliable mid-kernel interrupt; keep an external
process limit and verify the process exits and the GPU clears after termination.

For a bounded depth experiment, try `rows=262144`, `width=256`,
`workload=shared_prefix`, and `precision=32`. This family has 248 common leading bytes,
so 32 passes can reach the distinguishing suffix; the planned precision `1/2/4/8`
sweep cannot. This is a diagnostic setting, not a new default or a universal solution.
Longer prefixes can require more depth, while extra passes cost radix work and
synchronization. Duplicate elimination helps exact duplicates, not distinct strings
with a shared prefix. Do not launch a full sweep merely because a bounded case improves.

Before interpreting a new setting, run focused correctness in a fresh process:

```bash
env LIBCUDF_STRING_SORT_ALGORITHM=1 \
  LIBCUDF_SEGMENTED_STRING_SORT_LEXIC_PRECISION="$precision" \
  LIBCUDF_SEGMENTED_STRING_SORT_RADIX_RUN_MIN="$cutoff" \
  LIBCUDF_SEGMENTED_STRING_SORT_TRACE=0 \
  cpp/build/latest/gtests/SORT_TEST --gtest_filter=StringSort.*
```

Repeat for selectors `0` and `2`. Run full `SORT_TEST` before accepting implementation
changes. Keep source-default results
(precision `1`, cutoff `512`, chunk size `512`) separate from tuned results.
