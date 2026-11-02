# tall-a3-fusedengine implementation notes (fused short-string warp-tile engine)

As-of 2026-11-02, worktree `worktrees_tmp/cudf___tall-a3-fusedengine___impl`, branch
`batch-gather/tall-a3-fusedengine`, base = tip `ff874145af` (fresh off tip, same base as a5's
design map). Design sources: `results/optimization-hunt/round-argon/opportunities.md` §A3;
`results/optimization-hunt/strings-round/opportunities.md` item 2; `results/ab-tall-round12.md`
(str5's flat verdict: barrier share ~0 — the engine's win is the WALK elimination, not the
barrier count). LOCAL EDITS ONLY — no build/test/bench, no commit (executor rounds own those).

## Mechanism (what landed)

`string_fused_short_engine_kernel<block_size>` (batch_gather.cu, after
`string_routed_metadata_kernel`): ONE warp owns 32 consecutive rows of ONE column per
iteration (per-column tile walk, tiles never straddle a column boundary). Per tile iteration,
barrier-free at CTA scope:

1. **Resolve + offsets + staging** — lane-parallel, one row per lane: one map read per lane,
   the row's gathered size, validity. The lane writes its offsets entry straight from the scan
   scratch (dense publication — no staged boundaries, no shared group publication) and the
   last-live-row lane writes the column's N+1 sentinel (scratch[num_rows] = the column total).
   Row 0's entry IS its +0 scratch value (child's first entry needs no special case). The
   resolved (src_row, len) pair stages into warp-private shared slots (`warp_src`/`warp_len`).
2. **Masks + counts** — `__ballot_sync(0xffffffff, src >= 0)` over the FULL warp (dead lanes
   vote false; tail words exact); lane 0 writes the mask word wholesale (never OR'd into stale
   storage) and pops the count into the column's atomic counter. The ballot ALSO publishes the
   staged slots warp-wide (writes precede it in program order), so no separate `__syncwarp`
   is needed before the copy phase.
3. **Chars** — 4-lane subgroups copy short rows: subgroup `sub` takes rows sub, sub+8,
   sub+16, sub+24 (8 rows in flight per round, 4 rounds per tile); each subgroup reads its
   row directly from the staged slots — NO per-byte binary search, NO shared group
   publication. Rows above `kBlockRowCopyThreshold` (32 B) fall to the stock full-warp
   `copy_string_span` stride path (never truncated; long outliers safe on the engine).
   Subgroup loop bounds differ per subgroup — safe: nothing inside is a warp collective.

Per 32 rows: ONE warp-level barrier (the ballot) replaces the walk's TWELVE CTA barriers
(3 per 8-row group × 4 groups), and the per-byte binary-search fan disappears entirely.

## Admission (host-side, `assemble_strings`)

- Engine admitted ONLY when `not chars_routing_active` (num_rows < 524'288 — below the str1
  routing gate) and every live column's average gathered row is at or under
  `kBlockRowCopyThreshold` (bytes / num_rows <= 32). The AVERAGE does not bound the maximum:
  long outlier rows are SAFE on the engine (full-warp copy), so the short-average check alone
  admits. Above the gate, the unrouted cohort is long-row by construction → engine never
  admitted there; the group walk stays authoritative for the unrouted cohort.
- When admitted, `memcpy_unrouted` moves into `engine_cols` and the launch site swaps:
  `unrouted_rows == 0` + `engine_active` → `string_fused_short_engine_kernel` launch
  (`engine_cols.size() * ceil(num_rows/32)` warp slots, ≤65535 blocks of 8 warps).
- The scan/carve/metadata pipeline (scan → totals readback → carve → items upload → counts
  readback) is UNTOUCHED — the engine consumes exactly the same `string_work_item` array,
  scratch, offsets children, masks, and count slots as the payload kernel it replaces.
  `carve_strings_buffers` stays the allocation path; zero-size allocations still return NULL
  (skip carve). Sliced sources: destination size = gather_map.size() (num_rows), unchanged.

## Seam gate change (the band revival)

The seam predicate dropped the `[kStringFallbackRowThreshold, kBatchMemcpyRowThreshold)`
exclusion: strings rejoin the fused path for `num_rows < kBatchMemcpyRowThreshold` — i.e. the
whole below-gate domain, including the 262144×100 fallback-band cell where the engine's value
concentrates (the unrouted cohort is EMPTY at the headline 1048576×100 cell, so A3's win is
entirely in the band + sub-gate cells). `kStringFallbackRowThreshold` remains as a
documentation constant (comment updated: the historical A/B numbers describe the group-walk
path, not the engine). A band call whose rows are NOT short-average skips the engine and
falls to the payload kernel's group walk — the pre-engine fused behavior; that shape's tall
loss is the known admission risk the falsifier's regression checks watch.

## Geometry choices (deviations from item 2's text, with why)

- **8 subgroup rounds in flight (4-lane subgroups), not 8-row-per-round lanes:** item 2's
  "4/8-lane subgroups copying 8 rows per round" maps to 4-lane × 8 subgroups; 8-lane
  subgroups (4 in flight) would idle 4 lanes of the warp per round on ≤16 B rows.
- **Warp-private staged slots (`__shared__ [8][32]` per block) instead of shuffles:** lane r
  resolves its OWN row in the resolve phase (one map read per lane per tile = 1 read/row —
  same traffic as the walk), then subgroups read staged (src, len) directly. A lane-0-resolve
  + `__shfl_sync` broadcast per row would serialize 32 broadcasts per tile.
- **`copy_string_span` full-warp fallback for >32 B rows:** item 2C's mandate (never reuse the
  scalar head/tail with 4/8 threads) is honored by NOT using subgroups on long rows at all —
  the 4-lane scalar path handles ≤32 B rows (4 lanes × 4 bytes × 2 rounds covers 32 B
  exactly), everything above rides the proven 32-lane stride form.
- **One ballot for validity AND slot publication:** str5 proved the barrier share is ~0, so
  the win is the walk elimination; folding publication into the ballot's ordering keeps the
  engine at ONE barrier per 32 rows with no extra sync.

## Compose-conflict notes for the future tip rebase

- `string_flat_layout`/`scan_layout` NOT passed to the engine (it uses `num_rows` directly and
  indexes `dst_scratch[row]` per column via the cohort table `engine_cols`) — if a1/b12
  rebase changes the scratch indexing contract, re-check the engine's offsets writer
  (`*(item.dst_offsets + row) = item.dst_scratch[row]`).
- The engine shares `warp_src`/`warp_len` shared arrays sized `[BlockSize/32][32]` — any
  block_size change re-sizes automatically; shared bytes: 2 × 8 × 32 × 8 B = 4 KB/block
  (vs the payload kernel's 1296 B — acceptable in the below-gate regime where occupancy is
  launch-bound, not shared-memory-bound).
- If str3's tile-prefix pipeline lands on the same tip, its early-masks/counts writer and the
  engine's mask/counts phase are the SAME cohort (below-gate) — sequence them, do not merge
  both writes into one phase without a fresh gate.

## Falsifier (for the executor/orchestrator gate)

- [ ] At 262144×100 the payload kernel's launches GONE (zero group-walk launches on that
      cell; the engine replaces it — nsys should show `string_fused_short_engine_kernel`, no
      `string_payload_kernel`).
- [ ] Kernel µs drop on 262144×100 and 65536×100 cells; end-to-end win at the fallback-band
      cells (target +15–40% at 262144×100 per the design's prediction band).
- [ ] No regression on the routed headline cell (1048576×100: payload already empty there;
      the engine is unreachable above the gate — cell should be TIE vs control).
- [ ] All `StringFusedEngineParity` tests green (10 new tests, below).
- [ ] All pre-existing tests stay green (the routed-metadata suite, the seam suite, the
      mixed suite — the engine path is a new branch, nothing old moves except the band).

## Test list (`StringFusedEngineParity`, cpp/tests/copying/batch_gather_tests.cpp)

1. `SeamBandShortRowsSingleColumn` — 262161 rows (1 mod 32 tail), nullable, the band's
   headline shape; one column so the tile walk exercises its per-column restart.
2. `MultiColumnTilesAndSentinels` — 300000 rows (band interior), 2 short string columns
   (different strides) + i32; per-column tiles, per-column sentinels, per-column masks.
3. `ShortAverageColumnWithLongOutliers` — every 32nd row 300–349 B (average ~15 B: admitted),
   the long rows ride the full-warp path byte-exactly; subgroup loop skips them.
4. `RowLengthClassBoundaries` — 128 rows cycling {0, 1, 3, 4, 16, 17, 32, 33} B lengths:
   subgroup boundary (16), full-warp boundary (17), gate (32), above-gate (33), empty-but-
   valid (0). The item-2 test-list edge sizes 0/1/3/4/15/16/17/31/32/33/129/4097 collapse
   onto these classes (the subgroup form is byte-count-uniform).
5. `NullifyDeadRowsEngine` — NULLIFY + OOB entries: dead rows contribute zero bytes and zero
   validity through the engine's ballots; null_count matches stock exactly.
6. `NegativeIndexWrapEngine` — ALLOWED negatives through the shared `resolve_gathered_row`.
7. `AllEmptyRowsEngine` — zero-byte rows: subgroup loop skips (len 0), offsets all-zero with
   sentinel, masks all-valid, count == num_rows.
8. `SlicedSourceEngine` — sliced source: mask_bit_offset + offsets-view offset indexing.
9. `RepeatedSourceRowsEngine` — repeated map entries (dense aliasing): bytes land at each
   row's own scratch offset, no aliasing.
10. `BandLongRowsKeepPayloadKernel` — 300000 rows of ~543 B average (`long_strings`): engine
    NOT admitted, payload kernel's group walk serves it, parity with stock.

Stale-comment updates: `GateBandStaysFallback`, `AtThresholdStringsFallbackPinned`,
`BoundaryBelowByOneUsesFastPath` — the band's route changed (fused engine, not fallback);
the tests themselves stay (same row counts, still parity-vs-stock).

## Risks

- The engine's warp slots (`blockIdx.x * 8 + warp`) cover `num_engine_cols * tiles_per_col`
  tiles; a tail slot's dead lanes vote false and store nothing — the ballot's dead-lane path
  is the same discipline the routed kernel's ballots use (verified in its tail-tile test).
- Shared-memory growth (4 KB vs 1296 B) could reduce occupancy on 256-thread blocks; the
  below-gate regime is launch/alloc-bound, and the falsifier's kernel-µs check catches it.
- `__syncwarp` removal relies on `__ballot_sync` ordering the staged-slot writes before the
  subgroup reads WITHIN one warp's iteration; a different compiler could theoretically
  reorder around the ballot (it is a memory barrier for warp-shared state per the PTX
  memory model — the routed kernel relies on the same property for its warp_counts writes).
- The engine is NOT taken when `memcpy_unrouted` is empty but `chars_routing_active` is false
  and no live columns exist (`memcpy_unrouted.empty()` guard) — the launch site's
  `else if (engine_active)` handles the empty cohort (no launch, no allocation, parity with
  the payload path's own empty-cohort skip).

## Compose onto the A4 staging-carve tip (2026-11-02)

Cherry-picked onto `batch-gather/tall-a3-fusedengine2` off tip `f91c7f5596` (A4 staging carve).
`git cherry-pick` auto-merged CLEAN — zero conflicts — despite the pre-pick expectation of
overlap at the staging-arena sizing. Why clean: the two diffs touch DISJOINT hunks of
`batch_gather.cu`. A4 (a) added `carve_device_vector` at the file's helper region and swapped
five `make_device_uvector_async` staging calls to it, (b) inserted the `staging_total` sizing +
`staging_carve_mr` creation ahead of `d_items`. A3 (a) inserted the engine kernel after
`string_routed_metadata_kernel`, (b) added engine admission AFTER the unrouted-classification
loop, (c) added the `else if (engine_active)` launch arm AFTER the `unrouted_rows > 0` block,
(d) rewrote the seam predicate + band comments. The only shared REGION is the launch site —
and A3's engine arm attaches after the payload launch A4 left untouched, while A4 changed only
the `d_unrouted` allocation INSIDE that block (now `carve_device_vector` into
`staging_carve_mr`).

Composed semantics verified on the merged tree (branch `batch-gather/tall-a3-fusedengine2`,
tip `8247341464`):

- The staging arena still sizes/serves whatever remains on the engine path: `d_items`,
  `d_payload_items`, `d_memcpy_live`, `d_routed_items` are all `carve_device_vector` carves
  from `staging_carve_mr`. The engine's own cohort upload (`d_engine_cols`) still rides
  `make_device_uvector_async` with `temp_mr` — A3's original allocation pattern for the
  below-gate cohort, unchanged by the carve conversion. Its size is
  `engine_cols.size() * sizeof(size_type)`, bounded by the arena's reservation math only if
  moved onto the arena — deliberately NOT moved (the engine cohort is below-gate only, so it
  is disjoint from the routed worst case the reservation comment computes; putting it on the
  arena would need that bound re-derived for the unrouted-cohort maximum).
- The carve machinery survives intact: `carve_strings_buffers`, the destination carve classes,
  and the zero-size→NULL skip are untouched by the cherry-pick (A3 never touched them).
- The engine's below-gate cohort keeps its own allocation pattern; the arena's
  `staging_total` reservation math (`3 * string_work_item + 2 * size_type` segments) is
  exactly A4's original text — no re-derivation needed since the engine arm allocates
  OUTSIDE the arena.
- Seam band + `StringFusedEngineParity` tests landed verbatim (10 tests at
  `cpp/tests/copying/batch_gather_tests.cpp`); `kStringFallbackRowThreshold` retains its
  updated documentation comment with the band-revival note.

Next bench composes BOTH: A4's staging-carve (7 staging mallocs → 0) + A3's engine. The A3
band-cell wins were measured against pre-A4 controls, so the next gate's control must be the
`f91c7f5596` tip, not `ff874145af`.

## Seam-guard mitigation (2026-11-02, slot a3-guard)

The composed round-22 grid proved the engine's win is real (262144×100 co0 m0 −69.5%, 3.3×;
65536×100 co0 m0 −59.1%, 2.4×) but 8 tall cells regressed (+0.98% to +42.73%, every delta
reproduced 4/4 ABBA reps, A/A ≤2%). The round-22 report's mechanism attribution ("band revival
re-admits ≥262144-row strings into the fused-path group walk") was WRONG: the composed base's
predicate (`>= kBatchMemcpyRowThreshold`) already excluded ≥524288-row strings from the fused
path, so the tall regression was the composed tree LOSING the adopted tall-str1 fused-routed
route (round-10 win, +26.7% headline) — the composed tall numbers (7.28-class) match round-10's
PRE-str1 stock control (7.54), not the routed path (5.5-class). Verified by the step-gate
reviewer against `git show` of all five predicate generations.

**Guard (corrected band-form restore):** the classification walk in `batch_gather_impl`
restores the carve tip's band predicate —
`not(STRING and num_rows >= kStringFallbackRowThreshold and num_rows < kBatchMemcpyRowThreshold)`
(262'144 / 524'288). The engine serves ONLY below-band string tables (65536–262143 rows win
band); the seam band [262144, 524288) takes the stock fallback; ≥524288-row strings stay fused
on the routed-metadata path (tall-str1), which the engine does not disturb (unreachable above
the routing gate). Pure host-side branch: no device code changes, the engine kernel untouched.

**Evidence (round 22, `results/ab-tall-round22.md`, bench sha 49f0b430):** the 65536×100 win
(2.4×) sits below the band → preserved under the engine. The 262144×100 band-cell win WAS
engine-vs-stock (round-22 falsifier: engine kernel 4 launches in K, 0 in C; C = fresh carve-tip
control running the stock band route) — under the band predicate the band cell returns to that
stock route, so the engine no longer serves it and the win is forfeited (both arms stock →
expected tie). The 8 tall cells ran stock on the composed tree and fused-routed on the carve-tip
control → restoring the band predicate restores the tall-str1 route → expected ties.

**Expected post-guard grid (band form, from existing measurements only):** 1 win (65536×100,
2.4×), 9 ties (8 tall cells back on the fused-routed path both arms; band cell stock-routed
both arms), 0 losses.

**Tests:** the `StringFusedEngineParity` suite's row counts re-scoped from 262144/300000 to
262143/262142 (largest below-band counts preserving the same shapes: tail tiles, band interior,
long outliers) so the tests still exercise the ENGINE rather than silently degrading to the
stock fallback under the band gate; `BatchGatherTallSeamTest::SeamGuardExcludesBandStringFromFusedPath`
added at exactly 262144 rows pinning the host-side band routing (short-average engine-shaped
strings at the band's lower edge go to the stock fallback). The `GateBandStaysFallback`
routed-metadata pin likewise re-scoped to 262143. clang-format dry-run -Werror clean on both
edited files. LOCAL EDITS ONLY — no build/test/bench in this slot (executor rounds own those).
