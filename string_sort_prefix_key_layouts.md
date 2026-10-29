> Current branch status: rebased onto PR #24498 (`a257d75`). Runtime selection now uses
> `LIBCUDF_STRING_SORT_ALGORITHM`: PR prefix merge `0`, PR segmented `1`, PR segmented with
> terminal exact-duplicate elimination `2`, and the best measured R8 LRB recipe `3`.
> The key-layout experiments below are historical; their original source and controls remain
> available in `backup/string-sort-prefix-variants-before-pr24498-20261008`.
> See `cpp/benchmarks/sort/README.md` for the current comparison commands and graph schedule.

# Carrying the cached prefix with the row index in string sorts

Context: https://github.com/NVIDIA/cudf/pull/24267 (cached 8-byte prefixes for
single-column string `sorted_order`). Status: idea only, not built or measured.

## The problem

In the PR, the sort runs `cub::DeviceMergeSort::{Stable,}SortKeysCopy` over the
row indices `0..N-1`, with a comparator that does `prefixes[lhs]` and
`prefixes[rhs]`.

After the first few merge passes the indices being compared are scattered, so
each `prefixes[idx]` is a random 8-byte read that costs a 32-byte sector.
Coalescing tricks such as the funnel-shift loads in
`cpp/include/cudf/strings/detail/gather.cuh` do not help, because the access
pattern depends on the data.

The fix is to make the prefix travel with the index, so the comparator reads
its operands from the sequentially staged sort keys instead of gathering them
from `prefixes[]`.

## The idea

Sort a key type that holds both the prefix and the row index:

```cpp
struct prefix_row {
  uint64_t prefix;
  cudf::size_type row;
};
```

`sizeof(prefix_row)` is 16 bytes (4 bytes of padding). CUB's merge sort stages
keys through shared memory in coalesced tiles, and the comparator sees both
fields without any global gather. Only the tie path (equal prefixes) touches
the string data, via `row`.

Comparator sketch:

```cpp
template <bool has_nulls>
struct prefix_row_comparator {
  __device__ bool operator()(prefix_row const& l, prefix_row const& r) const
  {
    if (l.prefix != r.prefix) { return ascending ? l.prefix < r.prefix : l.prefix > r.prefix; }
    return tie_break(l.row, r.row);
  }
  column_device_view const d_column;
  bool ascending;
  null_order null_precedence;
};
```

`tie_break` is the existing post-prefix logic from `string_prefix_comparator`
(length check, then suffix compare starting at byte 8).

## Implementation plan

Work in `cpp/src/sort/sort_column_impl.cuh`, on top of the PR branch.

1. Add `prefix_row` and `prefix_row_comparator` next to
   `string_prefix_comparator`.
2. Change `string_prefix_extractor` to output `prefix_row{prefix, row}` into a
   `rmm::device_uvector<prefix_row>` (replacing the `uint64_t` array).
3. Call `cub::DeviceMergeSort::SortKeysCopy` (or `StableSortKeysCopy` when
   `method == sort_method::STABLE`) with `prefix_row` input and output
   buffers. The existing `merge_sort` helper takes the key iterators, so it
   needs generalizing from `size_type` keys to a key type template.
4. Write the sorted rows back to `indices` with a `thrust::transform` that
   projects `.row`.
5. Nulls: the comparator still needs the null check before the prefix compare,
   so keep the `has_nulls` specialization and check `d_column.is_null(row)` on
   both operands. This reads the null mask via `row`, which is a gather, but a
   cheap one because the mask is 1 bit per row. An alternative is to partition
   nulls out first and sort only the valid rows, which removes the check from
   the comparator. That is a bigger change, so try it second.
6. Keep the PR's early exits (size < 2, all null, all empty) unchanged.

Memory cost: the key array and the sort output are each 16 B/row, against the
PR's 8 B prefix plus 4 B indices. Expect roughly 32 B/row for keys plus CUB
temporary storage, versus about 16 B/row now. Report this alongside any speedup.

### Alternative to evaluate afterwards: radix sort on the prefix

`cub::DeviceRadixSort::SortPairs` (`SortPairsDescending` for descending) with
`uint64_t` keys and `size_type` values is stable and avoids comparator calls
entirely. It then needs a second step that finds runs of equal prefixes and
sorts those runs with the full string comparator, for example with
`cub::DeviceSegmentedSort`. Nulls and the length rule for zero-padded prefixes
complicate this, so it is the larger experiment. Do it only if the merge-sort
variant above shows the gather was the bottleneck.

## Validation

Correctness comes before timing.

- Run the PR's `cpp/tests/sort/string_sort_prefix_test.cpp` and the existing
  sort tests unchanged against the new implementation. The PR's 342-line test
  file covers shared prefixes, embedded zero bytes, nulls, and both sort
  orders.
- Add cases that stress ties: many rows with identical 8-byte prefixes and
  differing suffixes, and strings that differ only by trailing zero bytes.
- Compare `sorted_order` output of the new path against the PR's path on the
  same random inputs. Stable sorts must match exactly; unstable sorts must
  match as orderings of the string values.

## Benchmark plan

Benchmarks live in `cpp/benchmarks/sort/sort_strings.cpp`:
`sort_strings` (end to end), `sorted_order_strings`, and
`sorted_order_strings_multi`. The PR adds 20 more `sorted_order` cases
(low cardinality, shared prefixes, variable length, 0/50/100% nulls).

1. Build the benchmarks on three commits: `main`, the PR head, and the PR
   head plus this change. Use the repo's normal build flow (for example
   `./build.sh libcudf benchmarks`, or your existing build directory) and run
   the same GPU for all three.
2. Run each with the same NVBench settings the PR used so numbers are
   comparable, for example:

   ```
   SORT_NVBENCH -b sorted_order_strings --json base.json
   ```

   The executable is the `SORT_NVBENCH` target, found under the build
   directory's `benchmarks/`.
3. Compare with NVBench's `nvbench_compare.py` (shipped with NVBench, not in
   this repo): `nvbench_compare.py base.json new.json`.
4. Report geometric mean over the original cases and over the new cases
   separately, as the PR does. The PR notes the new cases are biased toward
   large gains.
5. Report peak memory for at least the largest case.

### Profile before investing

Run Nsight Compute on the PR head for one large case to decide whether this is
worth doing:

```
ncu --set full -k regex:"DeviceMergeSort|prefix" SORT_NVBENCH -b sorted_order_strings --run-once
```

Look at:
- time share of the extractor kernel versus the merge-sort kernels;
- L2 sector efficiency (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld` versus
  requests) on the merge-sort kernels, which shows how scattered the
  `prefixes[]` reads are;
- fraction of comparator calls that reach the string tie path (count via a
  temporary device counter in a scratch build, not committed).

If the merge-sort kernels are not limited by those prefix loads, this change
will not pay for its extra memory.

## Open questions

- CUB version in the build may differ in which `DeviceMergeSort` entry points
  exist (for example whether there is a stable pairs-with-copy variant). Check
  the headers before choosing between a struct key and a keys/values pair.
- The 4 bytes of padding in `prefix_row` could carry something useful, such as a
  null flag or the string length clamped to 8, which would let the comparator
  skip the null-mask read and the length lookup on ties.

## Variants under evaluation

All variants keep the PR's early exits and its tie semantics (zero-padded
prefix, length rule, suffix resume after the cached bytes).

| Id | Key layout (16 B) | Cached bytes | Null handling in comparator | Tie threshold |
|----|-------------------|--------------|-----------------------------|---------------|
| P  | PR head: `uint64_t prefixes[]` gathered by row id | 8 | `d_column.is_null(row)` | `size < 8` |
| A  | `{uint64 prefix; int32 row; 4 B pad}` | 8 | `d_column.is_null(row)` | `size < 8` |
| B  | `{uint64 hi; uint32 lo; int32 row}` | 12 | `d_column.is_null(row)` | `size < 12` |
| C  | `{uint64 prefix; int32 row; uint32 is_null}` | 8 | `is_null` field, no mask read | `size < 8` |

P is the reference, not a candidate. Comparing A to P isolates the effect of
carrying the prefix with the row. B and C are each compared against A, so each
isolates one change.

### Details that differ per variant

- B: `hi` is bytes 0-7 and `lo` is bytes 8-11, both big-endian. The comparator
  checks `hi`, then `lo` only when `hi` is equal. The suffix compare resumes at
  byte 12. Strings shorter than 12 bytes are zero-padded and resolved by length.
- C: the extractor writes `is_null = 1` and `prefix = 0` for null rows. The
  comparator checks both operands' `is_null` fields before comparing prefixes, so
  the null mask is never read in the sort. The `has_nulls == false`
  specialization still skips the check entirely. Rows with `is_null = 0` follow
  the A path.
- All keys are 16 B with 8 B alignment. Verify with `static_assert(sizeof(...) == 16)`
  in each variant so none silently grows.

### Implementation approach

Make the key type a template parameter of the extractor, the comparator and
`prefix_sorted_order_impl`, with one struct and one comparator per variant (A, B,
C). Select the variant with a compile-time constant while evaluating, for
example a macro or a template argument set in
`column_sorted_order_fn::operator()`. Do not add a runtime switch. The goal is to
pick one variant, so the losing variants and the selection mechanism are deleted
before the PR.

### Evaluation matrix

Run each variant on each of these axes. P is run on the same matrix.

- Existing `sorted_order_strings` cases from `sort_strings.cpp` (typical data).
- The PR's 20 added cases: low cardinality, shared prefixes, variable length,
  and 0%, 50%, 100% nulls.
- Add two targeted cases if the PR's set lacks them:
  - strings sharing exactly 8-11 leading bytes and diverging at byte 12 or
    later, which is where B should win and A and C should lose;
  - nullable columns at 50% nulls with short distinct strings, which is where C
    should win by avoiding mask reads.
- Both `sorted_order` (unstable) and `stable_sorted_order`, and both ascending
  and descending, since the comparator paths differ.

Report for each variant versus A: geometric mean over original cases, geometric
mean over added cases, the single worst regression, and peak memory on the
largest case. Expected outcomes to confirm or refute, not assumptions:

- B improves only on the 8-11 byte shared-prefix cases and is flat or slightly
  slower elsewhere.
- C improves only on nullable cases and is flat elsewhere.
- A versus P shows whether the struct key helps at all. If A is not faster than P,
  B and C are not worth pursuing in this form.

### Decision rule

Keep a variant only if it improves its target cases by a margin outside
run-to-run noise (use NVBench's reported noise; rerun if the noise exceeds 1%)
without regressing the original cases. If B and C each help their own cases,
a combined layout is not possible in 16 B with 12 B of prefix; the combination
would need a 24 B key or a null flag packed into the length-clamped field, which
is a separate experiment.

## Results

Measured 2026-10-06 on an RTX 3070 Ti (8 GB) under WSL. Code: PR head
`338b29de0d` merged with upstream/main, plus the variants in
`sort_column_impl.cuh`, selected at runtime with `CUDF_STRING_SORT_VARIANT`
(0 = upstream comparator, 1 = P, 2 = A, 3 = B, 4 = C). This runtime switch is
for evaluation only.

The A, B and C keys are sorted in place with `DeviceMergeSort::{Stable,}SortKeys`,
then `.row` is projected into `indices`. They use the same `memcpy` + `byteswap`
extraction and `size <= cached_bytes` length rule as the PR head. All five
configurations pass `SORT_TEST` (1627 tests). That count includes an added
randomized differential test around the 8- and 12-byte boundaries, with
embedded NULs and nulls, in both orders, stable and unstable.

Protocol:
- 20 samples per case:
  `--min-samples 20 --stopping-criterion stdrel --max-noise 100 --min-time 0`.
- 4 rounds with the variants interleaved. Each case uses the median across
  rounds of NVBench's mean GPU time.
- Percentages are `1 - geomean(t / t_ref)`, the same formula as the PR description.
- The 40 cases are `sorted_order_strings` (20, "original") plus
  `sorted_order_strings_distribution` and `sorted_order_strings_nulls`
  (20, "new"), matching the PR's comparison.
- Median per-case relative stdev is 6-17%, so differences under about 10% on a
  single case are not meaningful.

### Against upstream

| Variant | 40-case | Original | New |
|---------|--------:|---------:|----:|
| P       | 62.12%  | 27.36%   | 80.24% |
| A       | 69.97%  | 54.75%   | 80.07% |
| B       | 72.00%  | 57.91%   | 81.37% |
| C       | 72.74%  | 56.47%   | 82.92% |

P reproduces the PR's figures (H100 57.64 / 20.42 / 77.45, V100
62.86 / 27.13 / 81.08).

### Against P

| Variant | 40-case | Original | New | New, excluding fast-path cases |
|---------|--------:|---------:|----:|-------------------------------:|
| A       | 20.72%  | 37.71%   | -0.90% | 10.54% |
| B       | 26.08%  | 42.06%   | 5.71%  | 12.24% |
| C       | 28.03%  | 40.07%   | 13.56% | 11.26% |

Five of the new cases take the early exit (all 100%-null cases, plus
`cardinality_1_width_32` at 262144 rows). They run in tens of microseconds and
execute the same code in every variant, so their differences are noise. The
last column leaves them out.

### Per-variant findings

- A versus P: the struct key helps, so the gather was a bottleneck.
  - About 2.4x faster on `max_width=8` at 2M and 16M rows.
  - About 1.9x faster on `max_width=32` at 2M and 16M rows.
  - 3.3x faster on every `num_rows=32768` case with width 32 or more.
- B versus A: wins its target, the PR's `sorted_order_strings_prefixes` with
  `prefix_width=8, prefix_cardinality=1` (all rows share exactly 8 bytes):
  5.97 ms against 49.4 ms at 2M rows. Contrary to the expectation, it is also
  about 7% faster than A on the original cases instead of flat.
- C versus A: wins its target. It is 10-12% faster at 50% nulls with 2M rows and
  flat at 262144 rows. It is also about 4% faster on the original cases.

### Regressions

All struct-key variants are slower than P on heavy-tie inputs:

| Case | P | A | B | C |
|------|--:|--:|--:|--:|
| `shared_prefix_64`, 262144 rows | 54.1 ms | 166.2 ms | 152.8 ms | 163.3 ms |
| `cardinality_64_width_128`, 262144 rows | 12.9 ms | 18.8 ms | 16.9 ms | 18.3 ms |
| `cardinality_64_width_128`, 2M rows | 101.0 ms | 127.9 ms | 116.2 ms | 124.8 ms |

Cause: CUB's merge-sort tuning scales items per thread by key size
(`nominal_4B_items_to_items(17, key_size)`).
- 16 B keys get 4 items per thread, against 17 for P's 4 B row ids. Tiles are
  about 4x smaller.
- The sort therefore needs more merge passes: 8 against 6.
- The merge-path search costs more comparator calls per item.
- When nearly every comparison falls through to a 64-byte string compare, that
  overhead dominates.

Nsight Systems on `shared_prefix_64` at 262144 rows shows the merge kernels at
111.6 ms over 8 launches for A, against 18.2 ms over 6 launches for P.

Tuning partly fixes this. A with 8 items per thread
(`cuda::execution::tune` with a custom `MergeSortPolicy`, selected by
`CUDF_STRING_SORT_IPT=8`, 2 rounds only):
- 72.31% / 55.58% / 82.74% faster than upstream (40-case / original / new);
- 26.90% faster than P overall, 33.15% with the fast-path cases excluded;
- `shared_prefix_64` at 262144 rows: 96 ms, still 1.8x P;
- `shared_prefix_64` at 2M rows: 971 ms against P's 1728 ms.

With 11 items per thread, the original cases are about 4% slower than A's
default tuning. That is likely occupancy: 256 threads x 11 items x 16 B is about
45 KB of shared memory per block. B and C were not tuned.

### Memory

Peak memory for `sorted_order_strings` at 16M rows, `max_width=256`:

| Upstream | P | A, B, C |
|---------:|--:|--------:|
| 128 MiB | 256 MiB | 576 MiB |

That is about 36 B/row for the struct-key variants: 16 for the keys, 16 for
CUB's temporary key buffer, and 4 for the output indices.

### Decision-rule outcome

B and C each improve their target cases by more than the noise, and A beats P
on the original cases. No variant meets "without regressing", because of the
heavy-tie cases above and the 2.25x memory relative to P.

Next steps:
- Tune items per thread for B and C, and check whether the heavy-tie gap closes.
- Test a combined B + C layout, which needs a 24 B key or a packed null flag.
- Weigh the memory cost against the gain on typical data.

## Variant D: packed 8-byte key

D packs everything into one `uint64_t`:
- bits 63-32: a 4-byte big-endian prefix;
- bit 31: the null flag;
- bits 30-0: the row index, masked with `0x7FFF'FFFF`.

Row indices fit in 31 bits because `size_type` is a signed 32-bit integer.

The comparator works in three steps:
1. For nullable inputs, check bit 31 on both keys, so the null mask is not read during the sort.
2. Compare the 4-byte prefixes.
3. On a tie, fall back to the shared tie-break, which resumes at byte 4 with a
   `size <= 4` length rule.

The extractor uses the same byte loop as main. For an 8-byte key, CUB's default
tuning gives 8 items per thread (`17 * 4 / 8`). That is the same tile shape as
"A ipt8", with half the shared memory.

Code: branch `string-sort-prefix-variants` (commit `a35037ddf2`, on upstream/main
`7032a7b2c1`, which includes the merged PR). Select D with
`CUDF_STRING_SORT_VARIANT=5`. All variants pass `SORT_TEST` (1628 tests), which
includes a new 4-byte boundary test with zero-padded ties.

All variants were remeasured on this base with the same protocol as above:
4 interleaved rounds, RTX 3070 Ti. The table above is from the PR head, so the
numbers here differ from it slightly.

### Against upstream

| Variant | 40-case | Original | New |
|---------|--------:|---------:|----:|
| P (main) | 61.20% | 27.04% | 79.37% |
| A        | 71.10% | 56.17% | 80.95% |
| B        | 71.91% | 56.30% | 81.94% |
| C        | 71.63% | 55.71% | 81.83% |
| D        | 74.83% | 51.99% | 86.81% |
| A ipt8   | 72.87% | 55.64% | 83.41% |

### Against P

| Variant | 40-case | Original | New | New, excluding fast-path cases |
|---------|--------:|---------:|----:|-------------------------------:|
| A       | 25.51% | 39.93% | 7.64%  | 2.65%  |
| B       | 27.59% | 40.10% | 12.47% | 7.96%  |
| C       | 26.88% | 39.30% | 11.92% | 2.92%  |
| D       | 35.12% | 34.20% | 36.03% | 40.60% |
| A ipt8  | 30.08% | 39.20% | 19.58% | 23.58% |

### Selected cases

Times in ms, median of 4 rounds.

| Case | P | A | C | D | A ipt8 |
|------|--:|--:|--:|--:|-------:|
| `nulls` 50%, fixed_8, 2M | 8.37 | 5.68 | 5.23 | 3.31 | 5.65 |
| `nulls` 50%, variable_128, 262144 | 0.83 | 0.82 | 0.72 | 0.44 | 0.88 |
| `nulls` 0%, fixed_8, 262144 | 0.73 | 0.88 | 1.02 | 0.46 | 0.69 |
| `variable_128`, 2M | 11.80 | 5.74 | 5.64 | 4.25 | 5.09 |
| `sorted_order_strings` w=8, 2M | 14.66 | 6.35 | 6.19 | 5.97 | 6.41 |
| `sorted_order_strings` w=32, 16M | 256.8 | 137.6 | 137.8 | 170.8 | 134.8 |
| `sorted_order_strings` w=256, 262144 | 22.04 | 23.31 | 23.15 | 16.31 | 17.44 |
| `cardinality_64_width_128`, 262144 | 12.30 | 18.24 | 18.12 | 12.26 | 12.09 |
| `cardinality_64_width_128`, 2M | 96.56 | 122.80 | 122.82 | 115.83 | 69.91 |
| `shared_prefix_64`, 262144 | 51.78 | 160.04 | 159.95 | 129.60 | 94.43 |
| `prefixes` width 8, 2M | 65.95 | 47.58 | 47.66 | 109.25 | 35.69 |

### Findings

- D has the best 40-case and new-case geomeans of any variant. It is 35% faster
  than P overall and 41% faster on the non-trivial new cases. Most of that comes
  from nullable and short or variable-length inputs, where D is 25-45% faster
  than C. C was the previous best on nullable data.
  - Bit 31 avoids null-mask reads, as C's `is_null` field does.
  - The smaller key moves half the data.
  - The key size also gets a larger default tile.
- D is weaker than A, B and C on the original cases (34% vs about 40% faster
  than P). Its losses are on wide strings with many rows: `max_width` 32-256 at
  2M and 16M rows is 10-25% slower than A. With only 4 cached bytes, more
  comparisons tie and fall through to the string data.
- D loses badly on its worst case: strings that share 4-8 leading bytes. On
  `prefixes` width 8 it is 1.66x slower than P and 2.3x slower than A at 2M rows.
- D does not regress `cardinality_64_width_128` at 262144 rows (it matches P).
  It is 20% slower than P at 2M rows.
- D still regresses `shared_prefix_64` at 262144 rows (2.5x P). That is better
  than A, B and C (about 3x), but worse than A ipt8 (1.8x).
- Memory: D peaks at 320 MiB at 16M rows x 256 B, about 20 B/row (8 keys,
  8 CUB temp, 4 indices). That is 1.25x P (256 MiB) and much less than the
  16-byte variants (576 MiB).

### Updated decision-rule outcome

No variant beats P on every case. The candidates trade off as follows:

- **D** gives the largest overall gain for the least extra memory. Its weak spots
  are strings sharing 4-8 leading bytes and the heavy-tie case.
- **A ipt8** has the most even profile. Its worst case is a 1.8x regression on
  `shared_prefix_64` at 262144 rows, but it needs 2.25x P's memory.

Worth trying next:
- D with tuned items per thread.
- An unpadded 12-byte key: three `uint32_t` holding an 8-byte prefix and a
  row/null word, with 4-byte alignment. It would keep 8 cached bytes at 12 B/key.
  Padded to 16 B, this is just C.

## H100 results and 12-byte variant (2026-10-07)

Measured on GPU 0 of `viking-prod-206`, an H100 80GB HBM3 with 132 SMs, in the existing CUDA 13.3 cuDF devcontainer. Branch `string-sort-prefix-variants`, base `be51629c4fb29c84e8426d2e35c944421bc51914`. The user's `build-all -j0 -DBUILD_BENCHMARKS=ON` completed before the experimental changes were built and measured. The earlier RTX/WSL results above remain historical; H100 results differ.

### Variant E: eight prefix bytes in a 12-byte key

E is `{uint32_t hi, lo, row_and_null;}` with size 12 and alignment 4, checked by static assertions. The first two words hold big-endian, zero-padded prefix bytes 0–7. The low 31 bits of the last word hold the row index; bit 31 records nullness. Comparison handles null precedence first, compares the prefix halves, then resolves ties from byte 8. Output projection masks off the null bit. Embedded NULs, short strings, duplicate prefixes, nulls, stable/unstable sorting and both directions pass the prefix correctness tests.

`CUDF_STRING_SORT_VARIANT=6` selects E. `CUDF_STRING_SORT_IPT` supports default 0 and 1/2/3/4/5/6/8/11 for carried layouts, plus 16 for the 8-byte D key. The gathered-index P path now supports 1/2/3/4/5/6/8/11/16/17, keeping its existing helper at default 0. All custom policies use 256 threads and warp-transpose loads/stores. CUB defaults are P=17, A/B/C=4, D=8, E=5. IPT 16 is rejected for 12/16-byte carried keys because it exceeds the 48 KiB block limit and otherwise triggers a 64-thread/one-item fallback.

### Protocol and confirmed results

Screened 58 layout/tuning configurations in two randomized rounds using the earlier 20-sample protocol. 10 finalists were confirmed in four randomized rounds with at least 30 requested samples, a 1% relative-noise target, 0.05 s minimum GPU time, and a 10 s per-measurement timeout. Reported times are medians of round means; lower time ratios are better. Stable and descending main/targeted checks use one round each. The unstable/ascending 36-case targeted matrix was also screened across all 58 configurations in two randomized rounds; leading expanded-suite configurations were then confirmed in four rounds on both matrices. End-to-end sort, multi-column controls, and true constant nonempty strings were measured separately.

| Layout | IPT | Fixed 40 time reduction / P | Original 20 reduction / P | Pooled 76 reduction / P | Peak auxiliary MiB |
|---|---:|---:|---:|---:|---:|
| C | 1 | 30.13% | 43.91% | 22.90% | 576.5 |
| E | 1 | 28.97% | 41.64% | 21.68% | 448.5 |
| D | 1 | 28.89% | 41.34% | 15.50% | 320.5 |
| A | 1 | 27.88% | 43.88% | 21.62% | 576.5 |
| P | 2 | 27.52% | 47.49% | 15.15% | 256.1 |
| B | 1 | 27.26% | 43.95% | 35.04% | 576.5 |
| E | 0 | 22.53% | 33.37% | 17.54% | 448.1 |
| B | 2 | 21.76% | 41.01% | 26.69% | 576.3 |
| B | 3 | 19.92% | 37.14% | 27.29% | 576.2 |
| P | 0 | 0.00% | 0.00% | 0.00% | 256.0 |

**Decision:** C IPT 1 is fastest on the fixed 40-case suite and wins all four rounds; on its 34 nontrivial cases it reduces time by 34.60%. E IPT 1 is about 1.67% slower than C in the fixed-suite aggregate but uses 448.5 rather than 576.5 MiB at 16M rows (22% less auxiliary memory). D IPT 1 is effectively tied with E and uses less memory still, but caches only four bytes and loses on the targeted shared-prefix matrix. P IPT 2 is best for the original random-string cases and uses the original storage layout.

**B IPT 1 is the winner if all 76 main plus targeted cases receive equal weight (35.04% lower time than default P).** Its 12 cached bytes help strings diverging at bytes 8–11; it uses roughly 64.5% less time than A IPT 1 on those targets. C IPT 1 uses 27.3% less time than A IPT 1 on the four 50%-null cases and is flat versus A on the original cases. There is no universal winner: C/E regress about 15% on the small shared-prefix-64 case versus default P; P IPT 2 regresses about 47% on the 2M-row shared-prefix-64 case. Keep the suite definitions explicit instead of pooling controls or silently reweighting distributions.

### Tuning and measurement limits

The H100 prefers small tiles in these matrices. Nsight Systems on 32768 rows / width 32 shows default P using eight initial blocks, 40 registers/thread, 17424 shared bytes/block and three merge passes; P IPT 2 uses 64 blocks, 32 registers/thread, 2064 shared bytes and six passes. C/E IPT 1 use 128 blocks, 32 registers/thread, 4112/3088 shared bytes and seven passes. This supports the parallelism-versus-pass-count explanation; it does not directly prove achieved occupancy or DRAM bandwidth. Nsight Compute counters are unavailable under the driver policy (`ERR_NVGPUCTRPERM`). Profile timings include tracing/initialization overhead and are excluded from rankings.

Both cardinality-1/width-32 cases produce empty strings here and take the early exit, along with four all-null cases. Peak memory equal to output int32 indices identifies these six early exits; retain them in the fixed 40 but exclude them from the active 34. The targeted matrix has two further cardinality-1/width-32 early exits; its width-128 cardinality-1 cases are nonempty. A separate fixed nonempty 32-byte string test confirms real all-equal ties: B IPT 1 is fastest at 2M rows, and E IPT 1 beats C there. E default is faster than E IPT 1 at 262144 rows.

The tiny 32768-row/width-8 case remains mildly noisy for some layouts even after stricter reruns. Most retry medians are below 1%, but C remains around 1.02%; preserve the noise and cross-round spread rather than selecting a favorable run. Clocks were observed rather than locked; GPU 0 was reserved and selected explicitly, while another GPU later became active on the same node. The reservation thread also notes concurrent large CPU benchmarks, so close comparisons should be repeated on a quiet node.

### Validation and artifacts

All seven default layouts passed the full SORT_TEST suite: 1628 passed and five existing skips per run. Final B/C/E IPT 1 and P IPT 2 configurations also passed the full suite. All 58 configurations passed the 12 prefix-focused tests. The final build and clang-format/diff checks passed. Changes remain uncommitted evaluation controls, with default P unchanged.

See [the detailed H100 report](work/string-prefix-h100/report.md), [main-case timings](work/string-prefix-h100/timings.csv), [all matrix timings](work/string-prefix-h100/all-timings.csv), and [profile metadata](work/string-prefix-h100/profile-summary.csv). The work directory also contains raw NVBench JSON, logs, GPU telemetry, Nsight traces, reproducer scripts, and the source patch. The report includes commands for E IPT 1 and direction/stability controls.

## October 7 follow-up: radix refinement, word suffixes and F

Implemented R8/R12: stable radix-prefix sorting, boundary discovery or NonTrivialRuns RLE, scanned per-segment tile counts, block sorts and merge-path refinement only inside nonnull prefix-tie segments larger than one. Tested tiles 128/256/1024. Added bounded four/eight-byte suffix comparisons and F: exactly sixteen bytes with twelve prefix bytes plus packed null bit/31-bit row ID.

Four randomized native confirmation rounds, fourteen finalists, five warmups and seven samples per case, no batch. Across the 68 active original cases, F/IPT1/word4 is fastest: 44.84% lower time than P/default and 10.29% lower than B/IPT1/byte. R12/tile128/word4/RLE is 42.47% lower than P; R8 is 42.25% lower. RLE improves their matched boundary variants by about 3.0% and 2.8%. Two additional nonempty-constant controls retain F as aggregate winner.

Radix wins individual workloads: medium random strings R8/128 1.563 ms vs F 2.077; 16M random strings R12/1024 10.610 vs F 13.600; long shared prefix R12/128 23.533 vs F 27.768. Prefix-width-8 ties and true constants regress. Word4 reduces F's long-prefix case by 35.4% but increases prefix11 time about 9.7%. Packed nullness reduces the 50%-null case by 28.8% versus B with matched word4, while its aggregate gain is only about 1.4%.

R8/R12 currently allocate 16-byte prefix keys, with peak logged allocation about 966.1 MiB at 16M rows, versus F 576.5 MiB; figures include output. The radix path synchronizes host metadata and cannot yet support CUDA graph capture. Default P remains selected; controls are experimental.

Select R8/R12 with CUDF_STRING_SORT_VARIANT=7/8, CUDF_STRING_SORT_RADIX_TILE=128/256/1024, CUDF_STRING_SORT_RADIX_RLE=0/1, CUDF_STRING_SORT_WORD_BYTES=0/4/8. Select F with VARIANT=9 and IPT=1. All selector names use CUDF_STRING_SORT_ prefix.

All fourteen finalists passed full SORT_TEST: 1629 passed and five existing skips each. Five Compute Sanitizer configurations passed thirteen prefix tests with zero errors. Twenty-two Nsight traces show long-prefix refinement merging still dominates (81-82% GPU busy time); hardware counters remain permission-blocked. Further directions: specialize narrower radix keys, short-run warp refinement, eliminate host readbacks, optimize giant-run merging, and first-byte rejection before word loads.

See [the follow-up report](work/string-prefix-radix/radix-refinement-report.md), [confirmation cases](work/string-prefix-radix/confirmation-cases.csv), [pool summaries](work/string-prefix-radix/confirmation-summary.csv), and [Nsight phases](work/string-prefix-radix/profiles/phase-summary.csv).

### Synchronization improvement, October 7

Commit experiment: keep the compacted segment count on device while preparing tile counts and scanning the n/2 upper bound. Gather segment count, tile count and maximum segment length in one 12-byte device-to-host copy, with one scheduling synchronization instead of two. CUDF_STRING_SORT_RADIX_DEVICE_META=1 enables this RLE path; 0 retains staged metadata as a matched control. Starts, ends and tile offsets remain device arrays.

Three randomized rounds on nine representative cases, both R8/R12 and tiles 128/1024 (216 measurements), gave optimized/control geometric time ratios 0.9864, 0.9905, 0.9886 and 0.9873 respectively. Small32 improved 5.7-9.1%; large cases were mostly flat, with worst measured regression below 0.7%. These are workload-specific improvements, not a reranking of the full 68-case suite. Raw data: work/string-prefix-sync/metadata-timings.csv.

All eight prefix configurations passed thirteen actual StringPrefixSort tests. All four optimized configurations passed the full suite (1629 tests, five existing skips). R8/R12 memcheck each passed thirteen tests with zero errors. Four Nsight traces verified four sorted_order stream synchronizations in the staged control versus three in the optimized path: two common waits precede radix scheduling, which itself drops from two to one. Each trace has five warmups and seven measured calls. Installed CUB 3.6 DeviceSegmentedSort supports numeric keys only and exposes no suffix comparator, so it cannot directly refine string-prefix ties.

### Fused segment compaction and scheduling

CUDF_STRING_SORT_RADIX_COMPACT=1 replaces RLE filtering and the tile-count scan with a bounded grid of block scans. Each block reserves segment IDs and tile offsets in one packed 64-bit atomic addition; whole descriptors may reorder, but their internal row order is retained. Block-reduced maximum lengths and packed counts are read together once (16 bytes). Only nonnull runs longer than one enter refinement. All starts, ends and tile offsets stay on device. COMPACT=0 retains the scanned DEVICE_META=1 path as control.

Three randomized rounds, the same nine cases and four prefix/tile choices (216 measurements): geometric optimized/control ratios 0.9850 R8/128, 0.9829 R8/1024, 0.9854 R12/128, 0.9869 R12/1024. Small32 improves 4.7-5.9% beyond the first synchronization improvement. Eight configurations passed all thirteen prefix tests; all four compacted configurations passed 1629 full sort tests with five existing skips; R8/R12 memcheck passed thirteen tests with zero errors. Four Nsight traces verify the separate DeviceSelect and DeviceScan kernels disappear, while the complete call retains three waits (one for radix scheduling). Data: work/string-prefix-sync/fused/metadata-timings.csv.

### Direct string storage removes device-view waits

CUDF_STRING_SORT_RADIX_FLAT_VIEW=1 passes chars, offsets, null mask, parent row offset and offset-width flag directly as kernel arguments. It avoids allocating/copying generic device child views for R8/R12. Both 32-bit and 64-bit offsets are supported, including sliced nullable columns. Other layouts keep the generic view path.

Three randomized paired rounds over nine cases/four prefix-tile combinations (216 measurements) give optimized/control geometric ratios 0.9732 R8/128, 0.9727 R8/1024, 0.9831 R12/128 and 0.9711 R12/1024. Small32 improves 6.2-10.5%; 16M random R12/1024 improves 10.601 to 10.154 ms. All eight configurations pass fourteen prefix tests, including the added INT64-offset representation/slice differential test. Four optimized configurations pass 1630 full sort tests with five existing skips; both memchecks pass fourteen tests with zero errors. Four Nsight traces verify complete sorted_order stream waits drop from three to one. Data: work/string-prefix-sync/flat/metadata-timings.csv.

### Device-only refinement schedules and graph replay

CUDF_STRING_SORT_RADIX_SCHEDULE=0 retains the fastest native schedule with one 16-byte metadata copy/wait. Mode 1 separates a freely scheduled block sort from a cooperative persistent merge kernel; mode 2 launches a bounded grid of ordinary merge kernels for the known row-count upper bound. Each mode-2 kernel checks the actual device merge depth and skips inactive levels; a device finalizer selects the real parity. Both modes pass compact device segment/tile arrays directly and perform no metadata readback. They require RLE=1, COMPACT=1 and FLAT_VIEW=1 for a complete call without stream waits. COOPERATIVE=1 is a compatibility selector for mode 1; SCHEDULE overrides it. Mode 1 requires cooperative launch support.

Three randomized paired rounds over nine cases/four prefix-tile choices (216 measurements per experiment) reject zero waits as the throughput default. An initial fused cooperative block-sort/merge design is 28-35% slower in aggregate; separating block sorting reduces that to 19-26%, while making small32 4-7% faster. Guarded launches reduce the aggregate penalty to 9-14% (time ratios 1.1351 R8/128, 1.0921 R8/1024, 1.1072 R12/128, 1.0897 R12/1024). Large R12/1024 is 4.6% slower; long-prefix R12/128 is approximately flat. Keep mode 0 for native throughput and mode 2 when capture/no host synchronization is required. These measurements precede narrower radix keys.

The device-scheduled graph test captures stable and unstable sorting in both directions and replays a 4097-row fixed-width column through one giant tie segment, 128 small segments, no tie segments and then the giant segment again. It checks exact stable CPU row IDs and unstable permutation/value order. Its explicit monotonic device arena keeps captured temporaries alive; this validates algorithm replay, not arbitrary allocator capture compatibility. The async allocation-node harness passed normally but failed under memcheck; a pool-based harness invalidated capture. Both were replaced rather than accepted as passing validation.

Mode 2 passes all four full SORT_TEST configurations (1631 tests, five existing skips); R8/R12 memcheck passes all fifteen prefix tests with zero errors. Four Nsight traces verify zero complete-call stream waits in device scheduling versus one in mode 0. Data: work/string-prefix-sync/device-schedule/metadata-timings.csv and sync-validation.json. Separate cooperative-mode validation is recorded in work/string-prefix-sync/cooperative-split.

### Narrow nonnullable radix keys

R8 now uses a native uint64 key (64 radix bits) for nonnullable columns; R12 uses an exact 12-byte three-uint32 prefix (96 bits). Nullable columns retain the 16-byte null-rank key and 65/97 bits. Row IDs remain separate values. Explicit extractor key types avoid CUDA host/device auto-return deduction discrepancies.

Three randomized paired rounds over nine cases/four prefix-tile combinations (216 measurements), alternating saved wide and current narrow libraries with the same default-stream interceptor: geometric time ratios 0.9355 R8/128, 0.9345 R8/1024, 0.9630 R12/128 and 0.9575 R12/1024. At 16M rows, R8/1024 improves 10.472 to 8.992 ms; R12/1024 improves 10.159 to 9.213 ms. Peak allocation drops from 966.1 MiB to 578.9 R8 and 772.3 R12. The 50%-null negative control retains the same allocation and is approximately flat.

Eight paired prefix configurations and eight generic/staged RLE/boundary configurations pass fourteen prefix tests. Four optimized native configurations pass 1630 full tests, with the graph-only test additionally skipped (six total skips). Native R8/R12 memchecks pass; device-scheduled R8/R12 memchecks additionally pass all fifteen tests including changing-pattern graph replay. All four memchecks report zero errors. Data: work/string-prefix-sync/narrow/timings.csv; library hash/source provenance is preserved alongside the control.

### Sequential items per merge thread

CUDF_STRING_SORT_RADIX_MERGE_ITEMS=1 makes each four-item thread search the stable merge path once, produce four adjacent values sequentially, then use CUB BlockExchange to preserve coalesced stores. It affects tile1024; one-item tiles retain the original path. Mode 0 keeps four independent searches.

Three paired randomized rounds/nine cases (216 measurements): optimized/control geometric ratios 0.9160 R8/1024 and 0.9453 R12/1024. Large shared-prefix cases improve 29.0% and 28.3% (29.042 to 20.644 ms R8, 27.656 to 19.824 ms R12). True constant strings regress 22.4% and 18.5%; retain mode 0 for that workload. Tile128 controls are approximately flat. Four full native configurations pass 1630 tests/six skips. Three additional tile1024 memchecks exercise the changed path and graph replay: R8/word4, R12/word4 and R12/word8 each pass fifteen tests with zero errors. Data: work/string-prefix-sync/merge-items/metadata-timings.csv.

### Final optimization confirmation

Four randomized native rounds, fifteen finalists, 78 cases per finalist, five warmups/seven samples (4680 means). R8/128/word4 with RLE, fused compaction and direct storage wins the 68 active original cases: 51.47% lower time than P/default, 11.58% lower than F/IPT1/word4 and 21.09% lower than B/IPT1/byte. It reduces its matched original wide-key R8/128 by 15.43%. With the two nonempty constant controls included (70 active cases), it remains the winner, 51.26% lower than P and 11.24% lower than F. R8/1024 with sequential merging is very close on the original active68 (0.4% slower), but loses more on constants. Default P remains selected; experiments use explicit environment controls.

Maximum logged allocation in this matrix, including output but excluding pre-existing input, is 578.9 MiB R8, 772.3 R12, 576.5 F and 256.0 P. Nullable R8/R12 still use sixteen-byte keys. Native optimized radix performs one 16-byte scalar metadata copy/wait; all segment starts, ends and tile offsets remain on device. SCHEDULE=1/2 removes that copy/wait at the measured throughput costs described above.

Fifteen final Nsight traces cover F, optimized R8/128, optimized R12/1024, original R12/1024 and guarded R12/1024 on small, large and long-prefix cases. Runtime synchronization counts are respectively two, one, one, four and zero in every measured call. Kernel activity is attributed through launch correlation IDs, including work executing after asynchronous API return. Profile timings are excluded from the native ranking. Hardware performance counters remain unavailable; register/shared-memory metadata does not prove achieved occupancy or bandwidth.

Recommended general configuration: VARIANT=7, RADIX_TILE=128, WORD_BYTES=4, RADIX_RLE=1, RADIX_DEVICE_META=1, RADIX_COMPACT=1, RADIX_FLAT_VIEW=1, RADIX_SCHEDULE=0. Use RADIX_MERGE_ITEMS=1 for variable-suffix tile1024 workloads and 0 for true constants. All names have CUDF_STRING_SORT_ prefix. Confirmation data: work/string-prefix-sync/confirmation/{raw,cases,summary}.csv; final profile phases: work/string-prefix-sync/final-profiles/phases.csv.

### Funnel loads and dispatch-statistics experiment, October 8

Prefix extraction now reconstructs unaligned words from aligned uint32 loads with funnel shifts and endian reversal. Device-visible character bounds permit adjacent-row reads within the allocation, followed by masking beyond the current string. Bounds derive from terminal offsets, including sliced INT32/INT64 storage, without another host wait. The suffix reader reuses overlapping words; at least four cached bytes make its first aligned load safe inside the original string. Only complete logical words enter that reader, with guarded final physical-word reads and a byte tail. FUNNEL_PREFIX/SUFFIX=1 enable these paths and default on; 0 retains the original loading path within the new library. WORD_BYTES remains 0/4/8. TMA is not used.

Three randomized native rounds, fifteen configurations and 78 cases (3510 means, five warmups/seven samples) favor R8/tile1024/funnel8. On active68 it reduces time 54.57% versus P/default, 5.86% versus previous R8/128/memcpy4 and 17.14% versus F/IPT1/memcpy4. It also wins active70 including nonempty constants (6.71% below old R8). F/funnel8 wins the active34 targeted pool; R8 wins the main34. Tile1024 regresses on small random inputs, so the aggregate is not a universal dispatch rule. R8/128/funnel4 is 10.44% slower than old R8; width/tile tuning matters. New P/byte is flat against old P. Maximum allocation remains 578.9 MiB R8, 772.3 R12 and 576.5 F, including output. Original shared-library controls preserve compiled baseline resources; disabling a new runtime flag does not remove register cost.

Eighteen final Nsight traces confirm one native radix wait and two F waits. On 2M shared-prefix-64, R8/1024/funnel8 merging still takes 74.7% of GPU busy time, block sorting 23.6%, and prefix extraction only 0.47%. Comparator specialization and giant-run merging remain useful next experiments. Trace timings do not enter rankings; launch resources do not establish achieved occupancy/bandwidth.

A separate nonnullable CUDA prototype fuses exact length summaries and two 256-register prefix HLL sketches. Across 20 small/large shapes, length-only/full-HLL/sampled-HLL time ratios to extraction alone are 1.0668/1.6494/1.1637. At 2M rows length summaries add 0.56–0.79 us (1.0–1.7%); all-row sketches add about 7.5–73 us, sampled sketches about 1.8–7 us. These are synthetic kernel-only costs excluding initialization, final estimation/reduction/readback and allocation, not production end-to-end dispatch overhead.

The cardinality-sketch dispatch proposal is superseded by the exact-run LRB experiment below. The HLL work remains a separate historical cost prototype; it was never integrated into sorting or automatic dispatch. Prefix extraction and suffix comparisons do not compute a cardinality estimate.

Eight final native configurations pass 1631 full tests each with six skips. Four suite memchecks report zero errors, including R8/word8 and R12/word8 changing-pattern graph replay (16 prefix tests). The current-header standalone allocation-boundary probe passes all 264 shapes and memcheck. Added integration coverage includes all four alignments, lengths 0–65, embedded zeros/high bytes, mismatches, slices, both directions and stable row IDs. Full report and raw data: work/string-prefix-funnel/funnel-report.md and confirmation/{raw,timings,summary}.csv; statistics: statistics{,-summary}.csv; profiling: profiles/{calls,phases,phase-summary,kernels}.csv.

## Logarithmic radix binning refinement (2026-10-08)

LRB uses exact nontrivial RLE run lengths, excluding null runs. Histogram/prefix/scatter bins whole run descriptors by ceil(log2(length)); row ranges remain in radix order. Packed counters combine segment and initial-tile counts. Warp-aggregated histogram and block-aggregated scatter write tile offsets directly, with one end sentinel per bin. No intermediate descriptor-ID buffer or cardinality sketch is needed.

CUDF_STRING_SORT_RADIX_LRB=0 retains the existing scheduler and remains the default. Mode 1 bins with the selected fixed RADIX_TILE (128/256/1024); mode 2 uses 2/4/8/16/32-lane bitonic sorting for lengths 2–32, 128-thread blocks for 33–128, 256-thread blocks for 129–256 and 256-thread/4-item tiles above 256. Stable equal values use original row IDs even in descending order. Merge stages visit only bins requiring that depth. Finalization copies scratch-owned completed runs using each bin's parity; no whole-column scratch initialization/copy-back occurs. Null/singleton ranges stay untouched.

Native SCHEDULE=0 reads back 256 bytes of counts with one host stream wait and launches occupied tiers. Guarded SCHEDULE=2 uses device counts with fixed bounded grids, empty-range guards and no host metadata readback; CUDA graph replay changes distributions safely. LRB requires RLE and rejects cooperative SCHEDULE=1. Existing funnel loads and R8/R12/F layouts are preserved.

Eighteen configurations and 114 states, three randomized rounds, five warmups/seven samples produce 6156 native means. Fourteen configurations run in the main phase (including independently compiled original R8/R12 libraries); four tile256 configurations run in a later randomized supplement with matched tile256 controls. Timings cover unstable ascending sorting. Structured benchmarks additionally validate every output permutation/order before timing. The 36 new states cover nine tie-size distributions, 32K/2M rows and 0/64 shared suffix bytes. Combined 104 equals previous active68 plus these 36; true nonempty constants and eight early exits remain separately reported.

Adaptive R8 wins combined104: time/P=0.42415, 6.94% less time than original R8/1024 and 7.58% less than rebuilt R8/1024. On structured36 it saves 20.25% versus original R8. On original active68 it is 0.99% slower; original R8/1024 remains fastest. These equally weighted pools do not imply a universal workload dispatch rule. Fixed LRB and wider keys are useful controls but do not win the combined pool. Maximum logged allocations remain 578.9 MiB R8 and 772.3 MiB R12, including output and excluding pre-existing input.

All seven configured architectures compile. Twelve native prefix configurations and two generic-view/word4 configurations pass. Eight full LRB configurations pass 1632 tests each with six expected skips. Four guarded R8/R12 x fixed/adaptive prefix suites pass graph replay and memcheck with zero errors. A current-header standalone probe passes sixteen direction/mode/schedule/merge combinations with mixed bin parities and untouched null runs, also with zero memcheck errors. New tests cover boundaries through 32769, slices/null orders, stable CPU row IDs and unstable values/permutations.

Source and measured data: work/string-prefix-lrb/lrb-report.md, confirmation/{raw,timings,summary,comparisons}.csv and profiles/{calls,phases,phase-summary,kernels}.csv. Trace durations are excluded from rankings. Suggested follow-ups use exact bin counts to choose a refinement policy, specialize/cache giant-bin metadata, tune warp-sort thresholds and try selective wider radix refinement. No automatic dispatcher is added.

### Warp refinement follow-up (2026-10-08)

Adaptive LRB now has `CUDF_STRING_SORT_RADIX_WARP_SORT`: 0 original bitonic,
1 one comparison per bitonic pair with a shuffled swap flag, 2 CUB warp merge
sorting, and 3 a hybrid (default within adaptive LRB): shared comparisons for
2/4/8 lanes and CUB for 16/32 lanes. Each physical warp owns its scratch union;
a warp barrier protects reuse when the next task changes bins. Stable ties keep
original row order in either requested direction. LRB itself remains opt-in.

On H100, 11 configurations and 126 states in three randomized rounds (five
warmups, seven samples, 4158 measured means) select R8/hybrid. On the previous
combined104 pool it saves 4.28% versus the compiled previous adaptive R8 and
7.67% versus plain R8/tile128. All-CUB is within 0.43%. The new combined116 pool
adds tiny4/8/16 cases and saves 5.61% versus previous adaptive R8. At 2M rows and
64 shared suffix bytes, tiny32 improves 3.370 to 1.644 ms and tiny-plus-giant
4.702 to 3.204 ms. Giant-dominated cases remain largely unchanged; logarithmic
runs still favor plain tile128, motivating a separate large-tile/grid sweep.

All seven configured architectures build. R8/R12 with each of the three new
warp selectors pass the full sort suite (1632 passes, six expected skips each).
Six guarded graph memchecks and four synchronization/race checks pass with no
errors or hazards; native prefix and generic-view/word4 coverage also pass.
Raw data and reproducible scripts are in `work/string-prefix-warp`.

### Giant tile and persistent grid follow-up (2026-10-08)

Adaptive giant bins now honor RADIX_TILE=128/256/1024. Medium tiers retain
128/256-row local sorts. Merge run depth advances independently from the active
bin floor, preserving early repeated giant stages and completed-run parity.
CUDF_STRING_SORT_RADIX_LRB_GRID_WARPS=32/64/128 sets the requested warp budget
per SM (default 64); each worker converts that budget using its thread count.
This is a launch cap, not measured achieved occupancy.

Eighteen configurations and 126 states in three randomized rounds (five warmups,
seven samples, 6804 means) select R8/hybrid/tile1024/grid32 for combined104.
It saves 6.17% versus saved previous LRB, d0c686, 1.77% versus the compiled phase-A
hybrid, ea0eae, and 9.41% versus plain R8/tile128. Combined116 saves 7.20% versus
previous LRB and 13.49% versus plain128. Active68 favors grid128 only 0.04% over
grid32, within noise; structured36/48 favors tile128/grid128. No universal
dispatch or global LRB-default change is inferred from these test-state weights.

At 2M rows and 64 shared suffix bytes, grid32 changes logarithmic 13.421->8.387 ms
(plain128 still 7.336 ms), one-segment 16.206->13.688 ms, hot90 15.490->13.273 ms,
and block256 4.303->3.233 ms versus previous LRB. Tiny32 is 3.381->1.662 ms;
phase-A hybrid was 1.640 ms. The three combined104 round ratios to previous LRB
span 0.9364-0.9426. Peak workspace remains 578.9 MiB for R8.

All seven architectures build. Twenty validation logs include six full suites
(R8/R12 x three tiles, 1632 passes and six skips each), six guarded graph
memchecks, generic word4/views, fixed-bin smoke tests, alternate-grid graph
replay, and synchronization/race checks with zero errors or hazards. Twenty-five
Nsight traces compare selected native/guarded, compiled controls and plain128;
trace durations are excluded from ranking. Sources and data are in
work/string-prefix-warp-retune.

A read-only subagent review of PR #24498, head a257d75a7b5a4bb5b4ee392861d8d466e950d297,
recommends retaining LRB/hierarchical merging and testing a capped giant-only
exact common-prefix proof before selective additional radix passes. Cached
string metadata and one three-way stable suffix comparison are separate
hypotheses. The PR all-sibling finish may receive giant unresolved runs; no
matched PR performance comparison was run. Exact duplicate completion must
bypass sort, merge and scratch finalization.
