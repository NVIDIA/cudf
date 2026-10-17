# cuDF PR #24498 versus tuned radix-prefix LRB refinement

> Historical baseline measurements for R8 before selective large-run proofs and suffix radix.
> The [current PR benchmark comment](https://github.com/NVIDIA/cudf/pull/24518#issuecomment-6074998344)
> contains the updated recipe, matched measurements, plots and regression checks.

R8 is 2.250× faster than the best fixed PR segmented/RLE setting in the complete repeated matrix across the 206 active permutation states, and 1.863× faster on the separate 16-state end-to-end sort pool. R8 wins by more than 2% in 162 of those 206 permutation states; PR mode 2 wins in 41, with three within 2%. These are equal-weight synthetic-suite results, not a universal dispatch decision.

Rebased `string-sort-prefix-variants` onto PR head `a257d75a7b5a4bb5b4ee392861d8d466e950d297`. The original `f6ba458cd0d0c399f5bf514cfa4914addc092f89` is preserved as `backup/string-sort-prefix-variants-before-pr24498-20261008`. Integration source was measured at `6e05c89801`; the complete SHA and file/binary hashes are in `provenance.json`. The branch was committed locally. No force-push was performed.

The PR segmented implementation and the saved LRB kernel/load headers are byte-identical to their controls. PR selectors 0/1/2 retain their algorithms and defaults. Selector 3 uses the prior measured winner: R8 prefix + separate row ID, bounded funnel loads, exact RLE, tiered LRB, hybrid warp sorting, 1024-row giant tiles, 32-warps-per-SM launch cap, and hierarchical merging. It has its own wrapper/namespace in `string_sort_lrb.cuh`. The historical A–F/R12 runtime selectors are archived on the backup branch. Prefix merge remains the default.

Reference: [NVIDIA/cuDF PR #24498](https://github.com/NVIDIA/cudf/pull/24498), pinned to the head above. This comparison covers the experimental single-string-column path; it does not select a new multi-column algorithm.

| Selector | Algorithm | Settings used in the main comparison |
|---|---|---|
| 0 | PR prefix merge | Unchanged default |
| 1 | PR segmented | Precision 32; cutoff 512 |
| 2 | PR segmented with terminal exact-duplicate elimination | Precision 32; cutoff 4096 |
| 3 | Tuned R8 LRB | Fixed prior winning recipe; native schedule 0 |

H100 SXM 80GB GPU 0 on `viking-prod-206`, CUDA 13.3, one serialized workload at a time. The primary comparisons use one freshly built binary and identical deterministic inputs. Every main state/configuration has three independently warmed processes with 20 samples and five warmups; job order is randomized. The reported time is the median of the three unprofiled GPU-event means. Nsight timings do not enter the rankings. The old-library control uses the same benchmark binary with `LD_PRELOAD`, and includes the upstream rebase as well as refactoring.

The preliminary tuning sweep tested both PR modes at precision 1/8/10/16 and 32, including cutoffs 128/512/4096 at precision 32, over ten representative cases (five samples each). The full repeated matrix then tested all three cutoffs at precision 32. That complete-depth budget covers the longest fixed-width shared prefixes. The table selects each PR mode's best fixed cutoff on the combined suite; it is not a per-case oracle or a universal optimum. Default precision-1 results and bounded failures are reported separately.

| Pool | States | Prefix / R8 | Best PR 1 / R8 | Best PR 2 / R8 |
|---|---:|---:|---:|---:|
| original-active | 68 | 2.249 | 1.835 | 1.688 |
| segments | 48 | 3.168 | 3.920 | 3.947 |
| pr-new | 90 | 2.438 | 2.429 | 2.072 |
| combined | 206 | 2.523 | 2.475 | 2.250 |
| end-to-end | 16 | 2.445 | 2.218 | 1.863 |
| all-sorted-order | 216 | 2.438 | 2.431 | 2.218 |

Ratios are geometric means of each state's time divided by the matched R8 time; values above 1 favor R8. `original-active` excludes identity exits and true constants from the prior original suite. `segments` contains all 48 controlled distributions. `pr-new` includes PR workloads, source-parity inputs, diagnostics, and small zero-padding collisions. `combined` excludes end-to-end gathering and the ten original identity/constant cases; `all-sorted-order` includes those ten. End-to-end `sort` is separate. These synthetic pools weight states equally, not by an application's frequency.

| 2,097,152 rows, 64 shared suffix bytes | R8 ms | PR 1 ms | PR 2 ms |
|---|---:|---:|---:|
| tiny32 | 1.662 | 13.204 | 13.862 |
| block256 | 3.227 | 10.346 | 10.817 |
| logarithmic | 8.389 | 10.538 | 13.284 |
| tiny_plus_giant | 3.207 | 24.215 | 24.820 |
| hot90 | 13.283 | 138.082 | 138.260 |
| few_giants | 3.061 | 15.432 | 16.025 |
| one_segment | 13.724 | 154.444 | 154.525 |

| Peak allocated memory: 2M rows, 64 shared suffix bytes | R8 MiB | PR 1 MiB | PR 2 MiB |
|---|---:|---:|---:|
| tiny32 | 72.365 | 87.313 | 87.258 |
| logarithmic | 72.365 | 93.074 | 92.439 |
| one_segment | 72.365 | 86.428 | 86.373 |

Peak allocation counters exclude the pre-generated input and include the output permutation and temporary allocations. Every state is retained in `pr24498-timings.csv`; allocator measurements are separate from GPU-event timings.


Saved-library control: new R8 / old R8 geometric mean = 1.0003 over 126 states. 5 states were above +2% and 5 below -2%; 2% is an attention threshold, not a confidence interval. This control cannot separate the effect of unrelated upstream changes from the wrapper refactor.


Follow-up checks use three new processes of 50 samples each for all ten saved-control deviations above 2%, the nine main cases with near ties within 5% or recorded sample noise above 20% and R8 time above 0.1 ms, and the four strongest distinct PR-default wins. These checks do not replace or selectively alter the primary rankings. Round ranges and sample noise for every main state are preserved in the timings CSV. Differences near 2% should not be treated as established statistical wins.

| Follow-up state | Primary/screen ratio | New 50-sample ratio |
|---|---:|---:|
| control (old-r8-control): sorted_order_strings_cardinality, {"cardinality": "1", "max_width": "32", "num_rows": "262144"} | 0.969 | 0.938 |
| control (old-r8-control): sorted_order_strings_cardinality, {"cardinality": "0", "max_width": "128", "num_rows": "262144"} | 1.028 | 1.017 |
| control (old-r8-control): sorted_order_strings_prefixes, {"num_rows": "262144", "prefix_cardinality": "1", "prefix_width": "4", "suffix_width": "32"} | 1.027 | 1.013 |
| control (old-r8-control): sorted_order_strings_prefixes, {"num_rows": "262144", "prefix_cardinality": "64", "prefix_width": "4", "suffix_width": "32"} | 1.040 | 0.974 |
| control (old-r8-control): sorted_order_strings, {"max_width": "8", "min_width": "1", "num_rows": "32768"} | 0.973 | 1.004 |
| control (old-r8-control): sorted_order_strings, {"max_width": "32", "min_width": "1", "num_rows": "32768"} | 0.979 | 1.001 |
| control (old-r8-control): sorted_order_strings_distribution, {"num_rows": "2097152", "profile": "cardinality_1_width_32"} | 0.970 | 1.019 |
| control (old-r8-control): sorted_order_strings_nulls, {"null_percent": "100", "num_rows": "262144", "profile": "fixed_8"} | 0.967 | 1.021 |
| control (old-r8-control): sorted_order_strings_segments, {"num_rows": "32768", "segment_profile": "pairs", "shared_suffix": "0"} | 1.022 | 1.000 |
| control (old-r8-control): sorted_order_strings_segments, {"num_rows": "32768", "segment_profile": "tiny4", "shared_suffix": "0"} | 1.020 | 1.001 |
| main (pr2-p32-c4096): sorted_order_strings_source_parity, {"num_rows": "262144", "profile": "duplicates_32"} | 8.001 | 8.059 |
| main (pr2-p32-c4096): sorted_order_strings_workload, {"max_width": "32", "min_width": "0", "num_rows": "262144", "workload": "normal"} | 0.997 | 1.012 |
| main (pr2-p32-c4096): sorted_order_strings_workload, {"max_width": "128", "min_width": "0", "num_rows": "262144", "workload": "duplicates"} | 2.618 | 2.602 |
| main (pr2-p32-c4096): sorted_order_strings_workload, {"max_width": "64", "min_width": "0", "num_rows": "2097152", "workload": "duplicates"} | 0.996 | 0.987 |
| main (pr2-p32-c4096): sorted_order_strings_segmented_diagnostics, {"num_rows": "262144", "profile": "duplicates_12"} | 0.951 | 0.958 |
| main (pr2-p32-c4096): sort_strings_workload, {"max_width": "32", "min_width": "0", "num_rows": "262144", "workload": "normal"} | 1.020 | 1.004 |
| main (pr2-p32-c4096): sorted_order_strings_segments, {"num_rows": "32768", "segment_profile": "tiny8", "shared_suffix": "64"} | 3.489 | 3.479 |
| main (pr2-p32-c4096): sorted_order_strings_cardinality, {"cardinality": "0", "max_width": "32", "num_rows": "2097152"} | 1.626 | 1.649 |
| main (pr2-p32-c4096): sorted_order_strings, {"max_width": "32", "min_width": "1", "num_rows": "262144"} | 1.001 | 1.028 |
| default-win (pr2-p1-c512): sorted_order_strings_segmented_diagnostics, {"num_rows": "2097152", "profile": "duplicates_12"} | 0.287 | 0.285 |
| default-win (pr2-p1-c512): sorted_order_strings, {"max_width": "64", "min_width": "1", "num_rows": "16777216"} | 0.326 | 0.327 |
| default-win (pr2-p1-c512): sorted_order_strings_source_parity, {"num_rows": "2097152", "profile": "raw_duplicates_40"} | 0.336 | 0.332 |
| default-win (pr2-p1-c512): sorted_order_strings, {"max_width": "128", "min_width": "1", "num_rows": "16777216"} | 0.349 | 0.348 |

Control ratios are new R8 / old R8; other ratios are the indicated PR configuration / R8. See `noise-recheck/` in the archive for commands, rounds and logs.

| PR defaults (precision 1, cutoff 512) | Completed states | Timed out at 30 s | Failed |
|---|---:|---:|---:|
| pr1-p1-c512 | 214 | 18 | 0 |
| pr2-p1-c512 | 217 | 15 | 0 |

Default checks use one warmed process with 20 samples per completed state. A process timeout includes input setup, validation, warmups and timed samples; it is not a per-sort timing or a speedup bound. Timeouts are censored and excluded from any timing ratio. See `defaults/results.json` for every state, command and result.

| Fastest observed PR-default cases relative to R8 | PR ms | R8 ms | PR / R8 |
|---|---:|---:|---:|
| pr2-p1-c512: sorted_order_strings_segmented_diagnostics, {"num_rows": "2097152", "profile": "duplicates_12"} | 0.591 | 2.058 | 0.287 |
| pr2-p1-c512: sorted_order_strings, {"max_width": "64", "min_width": "1", "num_rows": "16777216"} | 5.185 | 15.922 | 0.326 |
| pr2-p1-c512: sorted_order_strings_source_parity, {"num_rows": "2097152", "profile": "raw_duplicates_40"} | 1.070 | 3.184 | 0.336 |
| pr2-p1-c512: sorted_order_strings, {"max_width": "128", "min_width": "1", "num_rows": "16777216"} | 10.270 | 29.453 | 0.349 |
| pr2-p1-c512: sorted_order_strings_segmented_diagnostics, {"num_rows": "2097152", "profile": "duplicates_32"} | 0.896 | 2.506 | 0.358 |
| pr2-p1-c512: sorted_order_strings, {"max_width": "32", "min_width": "1", "num_rows": "16777216"} | 3.729 | 10.395 | 0.359 |
| pr2-p1-c512: sorted_order_strings_segmented_diagnostics, {"num_rows": "2097152", "profile": "duplicates_24"} | 0.772 | 2.072 | 0.373 |
| pr2-p1-c512: sorted_order_strings_segmented_diagnostics, {"num_rows": "2097152", "profile": "duplicates_6"} | 0.373 | 0.986 | 0.379 |

These default wins should guide further investigation; their single-process screening is distinct from the three-round main ranking.

| Stable/direction supplement | PR 1 / R8 | PR 2 / R8 |
|---|---:|---:|
| stable=True, descending=False; 4 cases | 2.684 | 2.726 |
| stable=True, descending=True; 4 cases | 2.679 | 2.725 |
| stable=False, descending=True; 4 cases | 2.977 | 3.029 |

The supplement uses three rounds of 20 samples for duplicates, giant-long, tiny32-long and half-null inputs at 2M rows. It is a separate small pool, not a replacement for the primary ascending matrix.

| Nsight sample | Kernel launches | Stream synchronizations | Profiled kernel ms |
|---|---:|---:|---:|
| pr1-p32-c512: duplicates64 | 140 | 37 | 6.344 |
| pr1-p32-c512: logarithmic | 147 | 39 | 9.940 |
| pr1-p32-c512: one_segment | 120 | 30 | 154.402 |
| pr1-p32-c512: tiny32 | 57 | 12 | 13.044 |
| pr2-p32-c4096: duplicates64 | 140 | 37 | 6.339 |
| pr2-p32-c4096: logarithmic | 129 | 30 | 12.541 |
| pr2-p32-c4096: one_segment | 120 | 30 | 154.454 |
| pr2-p32-c4096: tiny32 | 57 | 12 | 13.712 |
| radix-lrb: duplicates64 | 34 | 1 | 5.944 |
| radix-lrb: logarithmic | 34 | 1 | 8.308 |
| radix-lrb: one_segment | 39 | 1 | 13.555 |
| radix-lrb: tiny32 | 27 | 1 | 1.601 |

Nsight reports medians across seven sampled calls, after warmup. CUDA-event total times and traced kernel sums measure different quantities; do not use this table to rank implementations. `profiles/kernels.csv` retains launch dimensions, registers, shared memory and every kernel duration.

| Large zero-padding collision probe | Rows | Result | GPU ms (completed only) |
|---|---:|---|---:|
| prefix: zero_collision_pass1 | 262144 | complete | 0.279 |
| pr1-p32-c512: zero_collision_pass1 | 262144 | complete | 5.232 |
| pr2-p32-c4096: zero_collision_pass1 | 262144 | complete | 6.197 |
| radix-lrb: zero_collision_pass1 | 262144 | complete | 0.273 |
| prefix: zero_collision_pass2 | 262144 | complete | 0.727 |
| pr1-p32-c512: zero_collision_pass2 | 262144 | complete | 6.733 |
| pr2-p32-c4096: zero_collision_pass2 | 262144 | complete | 7.700 |
| radix-lrb: zero_collision_pass2 | 262144 | complete | 0.385 |
| prefix: zero_collision_pass1 | 2097152 | complete | 0.906 |
| pr1-p32-c512: zero_collision_pass1 | 2097152 | complete | 254.750 |
| pr2-p32-c4096: zero_collision_pass1 | 2097152 | complete | 265.815 |
| radix-lrb: zero_collision_pass1 | 2097152 | complete | 0.999 |
| prefix: zero_collision_pass2 | 2097152 | complete | 2.517 |
| pr1-p32-c512: zero_collision_pass2 | 2097152 | complete | 273.916 |
| pr2-p32-c4096: zero_collision_pass2 | 2097152 | complete | 285.608 |
| radix-lrb: zero_collision_pass2 | 2097152 | complete | 1.884 |

These bounded single-process probes use 20 samples on completion. Mixed short/zero-padded values cannot prove a full next radix word; a large pass budget alone does not bound the PR comparison finish. Timeouts refer to the complete process, not a single sort.


Validation: six full SORT_TEST configurations passed 1,658 tests each (five existing unsupported-list skips plus the explicitly gated graph test). The guarded LRB string-only configuration passed 43 tests including graph replay. Additional PR cutoff configurations passed their string suites. LRB memcheck, synccheck and racecheck reported zero errors/hazards. Repository hooks and the seven-architecture build passed. The saved-library compatibility smoke passed three LRB edge/long-suffix tests. Raw logs document exact counts and skips.


Use `LIBCUDF_STRING_SORT_ALGORITHM=1` or `2` for the PR backend, and `3` for tuned R8 LRB. PR precision/cutoff settings affect only modes 1/2. Set `LIBCUDF_RADIX_LRB_STRING_SORT_SCHEDULE=2` for the LRB graph path; native schedule 0 remains the measured default. Settings are cached, so start a fresh process after changing them. Current commands are in `cpp/benchmarks/sort/README.md`.


The giant PR profile spends 152.24 of 154.40 ms (98.6%) in a CUB segmented-sort fallback with grid=1, block=256, 179 registers/thread and 33856 shared bytes. Its tiny32 profile spends 90.7% of kernel time in a 512-thread finish for 32-row runs. Native R8 uses distributed hierarchical merging and the warp tier respectively. The giant PR call has 30 stream synchronizations versus one for R8; the giant kernel itself dominates the cost.


Further directions, in order of measured relevance:

1. Add a separately selected experimental fast path for a single continuing run covering every valid row: reuse the PR global radix sorter instead of the single-block segmented fallback. The existing exact continuing-row count supplies the safety condition; inactive ranges must retain their output ownership. Measure this narrow change before generalizing to multiple large runs.
2. Try an exact, tiled common-prefix/duplicate proof for large LRB runs. Reduce the minimum proven common-prefix advance across all eligible rows, preserve short/embedded-zero length distinctions, and bypass both sorting and scratch finalization for proven identical runs. This could remove the repeated long-suffix comparisons that dominate R8 giant merging. No cardinality sketch is needed.
3. Port the LRB subgroup/block/hierarchical finish into a new hybrid backend after prefix advancement. The PR tiny32 finish is oversized for its run size; its giant comparison finish can remain expensive after partial-word or zero-padding collisions. Keep selectors 1/2 unchanged as reference controls.
4. Evaluate a three-way suffix comparator where stable warp refinement currently makes reverse comparisons, then cached or typed offset/length metadata in comparison-heavy workers. Measure register/shared-memory costs and end-to-end time rather than assuming wider records help.

These are proposals, not implemented or measured new optimizations.
