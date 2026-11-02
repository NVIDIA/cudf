# a3-guard step gate review — Design conformance + Optimality + Correctness

Reviewer: glm step-gate reviewer (slot a3-guard), distinct from implementer.
As-of: 2026-11-02. Scope: UNCOMMITTED diff only (`git diff`) on worktree
`worktrees_tmp/cudf___tall-a3-fusedengine2___impl`, branch `batch-gather/tall-a3-fusedengine2`,
base `deb8534b8e` (verified: `git log -1` = `deb8534b8e [Docs] Record the A3 compose…`; HEAD
unchanged → LOCAL EDITS ONLY, no commit ✓). Files touched: `cpp/src/copying/batch_gather.cu`,
`cpp/tests/copying/batch_gather_tests.cpp`, `results/tall-a3-impl-notes.md` — exactly the
allowed set (`git status --short`: 3 `M` entries, nothing else).

Design doc (quoted contract): MV-18 amendment, mixed-verdict-registry + round-22 report
(`results/ab-tall-round22.md`): "restore the ≥262144-row string exclusion; engine serves only
65536–262143"; envelope: change the classification-walk row gate in `batch_gather.cu`; update
`results/tall-a3-impl-notes.md` with a guard section (condition, evidence, expected post-guard
grid: 2 wins hold, 8 tall losses → ties); add a guard-pin test; clang-format both edited files;
LOCAL EDITS ONLY.

---

## DESIGN-CONFORMANCE

**Literal predicate instruction: CONFORMANT.**

- The classification-walk predicate is changed from `num_rows >= kBatchMemcpyRowThreshold` back
  to `num_rows >= kStringFallbackRowThreshold` — `cpp/src/copying/batch_gather.cu:2751-2752`:
  `not(src.type().id() == type_id::STRING and num_rows >= kStringFallbackRowThreshold)`.
  Base form at `deb8534b8e` line 2746 was `… and num_rows >= kBatchMemcpyRowThreshold)`
  (verified via `git diff` context and `git show deb8534b8e:…`).
- `[[maybe_unused]]` removed — `cpp/src/copying/batch_gather.cu:570`:
  `constexpr size_type kStringFallbackRowThreshold{262'144};` (no attribute; base line 569
  carried `[[maybe_unused]]`).
- Engine kernel body untouched: the diff on `batch_gather.cu` has exactly 4 hunks —
  `@@ -565` (comment + attribute removal), `@@ -2405` (comment), `@@ -2741` (comment),
  `@@ -2746` (predicate). `string_fused_short_engine_kernel` (`batch_gather.cu:1118`) and its
  launch (`:2581`) are outside all hunks. Pure host-side branch ✓.
- Guard-pin test exists: `BatchGatherTallSeamTest.SeamGuardExcludesBandStringFromFusedPath`
  (`cpp/tests/copying/batch_gather_tests.cpp:1326`), `num_rows{262'144}` at `:1328`,
  short-average engine-shaped strings (`periodic_strings(num_rows, 17)`, `:1329`), parity vs
  `run_reference` (`:1347-1348`); the test comment at `:1322-1325` notes the routing is a
  host-side branch — acceptable per the design doc ("routing is host-side so a comment noting
  the routing is acceptable").
- Notes guard section exists: `results/tall-a3-impl-notes.md:195-222` ("Seam-guard mitigation
  (2026-11-02, slot a3-guard)") with condition (`:204-208`), evidence (`:210-213`), expected
  post-guard grid (`:213`), tests (`:215-221`).
- clang-format: `clang-format --dry-run -Werror` on BOTH edited files → clean (ran it:
  `FORMAT_OK_batch_gather`, `FORMAT_OK_tests`; clang-format 20.1.8).

**Semantic contract: VIOLATED — see blocking finding F1.** The design doc's envelope says
"pre-existing seam handles >=262144" and acceptance check 3 requires "the >=262144 string
exclusion matches the pre-A3 carve tip semantics". It does not (evidence below).

## OPTIMALITY

**Verdict: SUBOPTIMAL** (evidence-only, no new bench required, per the acceptance check).

Predicate genealogy (all verified via `git show <commit>:cpp/src/copying/batch_gather.cu`):

| Commit | Predicate | ≥524288 strings |
|---|---|---|
| `4265d8a208` (original seam) | `>= kStringFallbackRowThreshold` (line 1990) | stock fallback |
| `f191be6160` (tall-str1, round-10 ADOPT-ELIGIBLE) | band `>= 262144 and < 524288` (lines 2132-2133) | **fused** → routed-metadata path |
| `f91c7f5596` (pre-A3 carve tip, this step's base parent) | band (lines 2558-2559) | **fused** → routed-metadata |
| `deb8534b8e` (composed A3 base) | `>= kBatchMemcpyRowThreshold` (line 2746) | stock fallback |
| this diff (restored) | `>= kStringFallbackRowThreshold` (`batch_gather.cu:2752`) | stock fallback |

The pre-A3 carve tip did NOT route all ≥262144 strings to stock — it excluded only the band
[262144, 524288); strings at ≥524288 stayed on the FUSED path, served by the adopted tall-str1
routed-metadata mechanism (`string_routed_metadata_kernel`, `batch_gather.cu:1018`; engaged
only when `chars_routing_active = num_rows >= kBatchMemcpyRowThreshold`, `:2194`, branch at
`:2384`). That path is the measured round-10 win: fused+routed 5.5231 ms vs pre-str1 stock
7.5383 ms at the 1048576×100 headline (+26.7%), 7/8 tall cells WIN, verdict ADOPT-ELIGIBLE
(`results/ab-tall-round10.md:32-39,55`). Round-18 confirms the carve tip's tall cells sit at
5.22–5.33 ms (`results/ab-tall-round18.md:63-70`).

The composed base's regression at the 8 tall cells (round-22: K 7.2790 vs C 5.3301 ms, +36.56%
headline; `results/ab-tall-round22.md` grid) is therefore NOT "band revival re-admitting tall
strings into the fused-path group walk" as the round-22 report's attribution (§Step-3 item 3)
states. At ≥524288 the composed predicate (`>= kBatchMemcpyRowThreshold`, `deb8534b8e`
`batch_gather.cu:2746`) EXCLUDES strings from the fused path — they ran the stock per-column
fallback, i.e. the composed tree LOST the tall-str1 fused-routed route. The numbers corroborate:
composed K 7.28 ≈ round-10 pre-str1 stock CTRL 7.54 (not the ≥10 ms a 1M-row group walk would
show); control C 5.33 ≈ round-10 STR1 5.52 / round-18 5.32 (fused-routed).

**Consequence:** this diff's only behavioral delta vs the composed base is inside
[262144, 524288). At the 8 tall (1048576) cells the guarded tree behaves identically to the
round-22 K arm (stock fallback, 7.28-class) while the carve-tip control runs fused-routed
(5.33-class) → **the 8 tall losses PERSIST (+26…+37%), they do not become ties.** The recorded
mitigation's expected outcome is unachievable with this predicate. It also structurally orphans
`string_routed_metadata_kernel`: only string columns feed the routed path (`string_work_item`
items built from `pending_strings`, `:2208`; routed branch `:2384`; `routed_items` `:2599`), and
strings can now be fused only when `num_rows < 262144` (`:2752`) while the routed path needs
`num_rows >= 524288` (`:2194`) — disjoint, so the adopted tall-str1 mechanism is dead code on
the guarded tree.

**Strictly dominating in-envelope alternative (host-side, same one-line shape):** restore the
carve-tip band predicate — `not(STRING and num_rows >= kStringFallbackRowThreshold and
num_rows < kBatchMemcpyRowThreshold)`. Expected grid vs the carve-tip control, from existing
measurements only: band cells (262144×100) → both arms stock → tie (same as this diff);
65536×100 → engine vs group walk → win holds (same as this diff); 8 tall cells → both arms
fused-routed (5.3-class, rounds 10/18/22 convergence) → **ties** (this diff: persisting
losses). Net: 1 win + 9 ties vs this diff's 1 win + 1 tie + 8 losses. Engine-side changes are
out of scope per the design doc; the band-form restore is not.

## CORRECTNESS

Passes:
- Non-string descriptor-path columns unaffected: the guard conjoins `src.type().id() ==
  type_id::STRING` (`batch_gather.cu:2752`), so fixed-width/struct/dictionary columns keep the
  base routing at every row count (`is_descriptor_path_candidate`, `:111-124`; STRING returns
  true at `:116`). Dictionary32 string-key columns are DEFERRED upstream of this walk (base
  behavior, untouched by this diff).
- Re-scoped engine tests exercise the ENGINE under the restored gate: `SeamBandShortRowsSingleColumn`
  262'143 (`batch_gather_tests.cpp:2040`; 262143 % 32 = 31 → one tail tile ✓), `MultiColumnTilesAndSentinels`
  262'142 (`:2061`), `ShortAverageColumnWithLongOutliers` 262'143 (`:2090`), `NullifyDeadRowsEngine`
  262'143 (`:2150`), `NegativeIndexWrapEngine` 262'143 (`:2180`), `AllEmptyRowsEngine` 262'143
  (`:2215`), `SlicedSourceEngine` 262'143 (`:2236`), `RepeatedSourceRowsEngine` 262'143
  (`:2258`), `BandLongRowsKeepPayloadKernel` 262'143 (`:2289`, long-average → engine rejects →
  payload group walk, still fused since 262143 < 262144 ✓). All counts < 262144 → fused under
  the restored gate (`batch_gather.cu:2752`), `chars_routing_active` false (< 524288, `:2194`),
  short-average vs `kBlockRowCopyThreshold{32}` (`:551`) → engine admission fires
  (`:2415-2427`). ✓
- Boundary pins coherent: `BoundaryBelowByOneUsesFastPath` 262'143 (`:1373`, fused) vs
  `AtThresholdStringsFallbackPinned` 262'144 (`:1305`, stock — 262144 >= 262144 excluded ✓);
  `AboveThresholdStringsFallbackPinned` 300'000 (`:1396`, stock ✓).
- clang-format dry-run `-Werror`: PASS both edited files (ran it; output above).
- LOCAL EDITS ONLY, no commit: HEAD = base `deb8534b8e`; 3 modified files only. ✓

Fails:
- **The ≥262144 exclusion does NOT match the pre-A3 carve tip semantics** (acceptance check 3):
  carve tip `f91c7f5596` excluded only [262144, 524288) — ≥524288 strings stayed fused
  (routed-metadata tall-str1 path). The restored wholesale form routes them to stock fallback.
  See OPTIMALITY for the measured consequence.

Non-blocking (comment accuracy, fix with F1):
- `batch_gather.cu:2406-2408`: "Above the gate the payload kernel serves only the UNROUTED
  cohort (long-row columns) … the group walk stays authoritative there" — under the restored
  guard no string column can be fused at ≥524288, so the described scenario is unreachable
  (the string payload/routed paths only serve fused string columns).
- `batch_gather_tests.cpp:2002` `GateBandStaysFallback` name is historical (the test now pins a
  262143 engine-routed call); its comment (`:1997-2001`) is accurate, name only.

## Blocking findings

**F1 — Restored guard deviates from the pre-A3 carve-tip seam semantics and cannot deliver the
recorded mitigation's expected outcome** (violates the design doc's envelope "pre-existing seam
handles >=262144" + acceptance check 3 "matches the pre-A3 carve tip semantics").
`cpp/src/copying/batch_gather.cu:2751-2752` excludes ALL ≥262144 strings from the fused path;
the carve tip (`f91c7f5596` `batch_gather.cu:2558-2559`) excluded only [262144, 524288) and
kept ≥524288 strings on the fused routed-metadata path — the adopted round-10 tall-str1 win
(+26.7% headline, `results/ab-tall-round10.md:32-39,55`). Evidence the 8 tall losses persist
rather than tie: the guarded tree is behaviorally identical to the composed base at ≥524288, and
round-22 measured exactly that configuration vs the carve-tip control at the 1M cells (K
7.2790/0.7893/0.6332/0.8035/0.6358/5.6702/7.4715/5.7194 vs C 5.3301/0.6334/0.6270/0.6275/
0.6087/4.8213/5.2348/4.7279 — all 8 LOSS, every delta 4/4 reps, `results/ab-tall-round22.md`).
Fix: band-form restore `… and num_rows >= kStringFallbackRowThreshold and num_rows <
kBatchMemcpyRowThreshold` (see OPTIMALITY).

**F2 — The notes' guard section states a false expected post-guard grid and a false
"preserved" claim** (violates the design doc's notes requirement: condition + evidence +
"expected post-guard grid: 2 wins hold, 8 tall losses → ties").
`results/tall-a3-impl-notes.md:210-213`: "win cells 262144×100 (3.3×) and 65536×100 (2.4×)
both sit at/below the gate → preserved … Expected post-guard grid: 2 wins hold, 8 tall losses
→ ties." Both halves are contradicted by the predicate the same section documents:
(a) 262144 >= 262144 is EXCLUDED by the `>=` gate — the step's own tests pin 262144-row
strings to stock (`batch_gather_tests.cpp:1305-1310`, `:1326-1328`) and the re-scoped engine
tests avoid 262144 for exactly this reason (`:2040` comment "largest below-gate count") — so
the 262144×100 win is forfeited (both arms stock → tie), only the 65536 win holds;
(b) per F1 the 8 tall cells persist as losses, not ties. Root cause: the round-22 report's
mechanism attribution ("composed tall columns re-enter the fused path … execute the payload
kernel's group walk", `results/ab-tall-round22.md` §Step-3 item 3) is contradicted by the
composed base's own predicate (`deb8534b8e` `batch_gather.cu:2746` excludes ≥524288 from
fused); the tall regression was the LOSS of the tall-str1 fused-routed route, not group-walk
re-admission. The notes must be rewritten once F1's fix lands (expected grid under band-form:
1 win (65536×100), 9 ties).

Both findings are against the same root cause; fixing F1 (band-form predicate) and rewriting
the notes evidence/expected-grid accordingly resolves both. Per the step gate rule, no commit
and no bench until fixed; after the fix, rebuild + retest before any benchmark.

## Summary table

| Check | Result |
|---|---|
| Predicate changed to `>= kStringFallbackRowThreshold` (262'144), `[[maybe_unused]]` removed | PASS (`batch_gather.cu:570`, `:2751-2752`) |
| Engine kernel body untouched (pure host-side branch) | PASS (4 comment/predicate hunks only; kernel `:1118` untouched) |
| Guard-pin test (262'144 rows) | PASS (`batch_gather_tests.cpp:1326-1348`) |
| Notes guard section present | PASS present (`tall-a3-impl-notes.md:195-222`) — content FALSE (F2) |
| Non-string descriptor columns unaffected | PASS (`batch_gather.cu:2752`) |
| ≥262144 exclusion matches pre-A3 carve tip semantics | **FAIL** (F1) |
| Re-scoped engine tests (262143/262142) exercise the engine | PASS (`batch_gather_tests.cpp:2040,2061,2090,2150,2180,2215,2236,2258,2289`) |
| clang-format dry-run -Werror, both edited files | PASS (ran: both clean, clang-format 20.1.8) |
| LOCAL EDITS ONLY, no commit | PASS (HEAD = `deb8534b8e`) |
| OPTIMALITY | **SUBOPTIMAL** — band-form restore strictly dominates (evidence-only, rounds 10/18/22) |

VERDICT: FAIL — 2 blocking findings (F1 carve-tip semantic mismatch / tall losses persist; F2 false expected-grid evidence in notes) — do not commit or bench; fix predicate to band-form + rewrite notes evidence/expected grid, then rebuild + retest.

---

## Re-check (fix delta)

Reviewer: glm step-gate reviewer (slot a3-guard), same reviewer, FIX-DELTA scope only. As-of
2026-11-02 (post-fix). Base unchanged: HEAD = `deb8534b8e` (`git log -1`), same 3 modified
files + this review file (`git status --short`) → the fix is LOCAL EDITS on top of the
prior-reviewed diff, no commit ✓. Not re-audited: everything that passed the prior review and
is untouched by the fix delta (re-scoped test counts, guard-pin test, non-string column
routing, engine kernel body, boundary pins).

### F1 — wholesale ≥262144 exclusion: RESOLVED

- `cpp/src/copying/batch_gather.cu:2754-2756` is now `not(src.type().id() == type_id::STRING
  and num_rows >= kStringFallbackRowThreshold and num_rows < kBatchMemcpyRowThreshold)` —
  semantically identical to the carve tip `f91c7f5596` `batch_gather.cu:2558-2559` (verified
  via `git show f91c7f5596:cpp/src/copying/batch_gather.cu`: same three conjuncts, same
  constants `262'144`/`524'288` at `:558`/`:571`). ≥524288 strings stay fused, so the
  routed-metadata path (`chars_routing_active = num_rows >= kBatchMemcpyRowThreshold`,
  `batch_gather.cu:2196`) is reachable again — F1's dead-code orphaning of
  `string_routed_metadata_kernel` is gone.
- `[[maybe_unused]]` removed at `:571`; the constant is used at `:2755` → no -Werror risk.
- Comment blocks updated to the band rationale: gate constant `:560-570`, engine admission
  `:2404-2413`, classification walk `:2738-2753`. The prior non-blocking inaccuracy at the old
  `:2406-2408` ("Above the gate the payload kernel serves only the UNROUTED cohort") now
  describes a REACHABLE scenario under the band form (≥524288 fused strings; routed metadata
  owns routed columns; payload kernel serves the unrouted long-row cohort) — accurate.
- clang-format dry-run `-Werror` on both edited files: re-ran in this re-check — clean
  (clang-format 20.1.8). ✓

### F2 — false expected-grid / "preserved" claims in notes: RESOLVED

- `results/tall-a3-impl-notes.md:222-224`: expected grid now "1 win (65536×100, 2.4×), 9 ties,
  0 losses" — matches the band-form analysis from the prior review's OPTIMALITY section and is
  measurement-supported: the 65536×100 win is engine-vs-stock (`ab-tall-round22.md:76`: K
  0.5282 vs C 1.2910 ms); the band-cell both-arms-stock tie is directly measured
  (`ab-tall-round18.md:71`: 4.0059 vs 4.0153, +0.24% TIE); the tall both-arms-routed tie class
  is `ab-tall-round18.md:63-70` (ties at the 5.3 ms class).
- The false "win cells preserved" claim is gone, and the tall-loss mechanism attribution is
  corrected at `tall-a3-impl-notes.md:199-205`: verified against the composed base's own
  predicate (`git show deb8534b8e:cpp/src/copying/batch_gather.cu:2746` = `not(STRING and
  num_rows >= kBatchMemcpyRowThreshold)`) — ≥524288 strings were stock on the composed tree,
  so the tall regression was the LOSS of the tall-str1 fused-routed route, not group-walk
  re-admission. Matches the prior review's analysis. ✓

### NEW F3 — BLOCKING: the rewritten evidence paragraph asserts a false provenance for the
round-22 band-cell win

`results/tall-a3-impl-notes.md:216-218`: "The 262144×100 'band-cell win' was measured
composed-vs-composed (both arms running the group-walk band), not engine-vs-stock". Both
halves are contradicted by the very report the paragraph cites (`ab-tall-round22.md`, bench
sha 49f0b430):

- K ran the ENGINE at that cell, not the group walk: kernel census `ab-tall-round22.md:48`
  (`string_fused_short_engine_kernel<256>`: K 4 launches / C 0) and `:89` ("the falsifier
  PROVES the engine is absent at 262144+ capture shape in C AND that K launches it only at
  the band cell"); stock-chars launches C 200 vs K 40 (`:49`). Round 19's census agrees
  (`ab-tall-round19.md:58`, `:60`, `:67`).
- C was the FRESH CARVE-TIP CONTROL, not composed: `ab-tall-round22.md:59` ("K = composed A3,
  C = fresh carve-tip control"); on the carve tip the band routes to stock
  (`f91c7f5596:2558-2559`), and C's band-cell time 4.0473 ms (`:75`) is the stock class
  (r18 control 4.0153, `ab-tall-round18.md:71`). The −69.5%/3.3× band-cell win WAS
  engine (K 1.2346 ms) vs stock (C 4.0473 ms), `:75`.

Same defect class as F2 (false evidence in the contract-required notes guard section), and it
matters operationally: false evidence in these notes already produced one blocking-scale error
in this step (round-22's wrong §Step-3 item-3 attribution is what motivated the wholesale
guard of F1 in the first place). It also corrupts the adopt-decision record — the honest trade
is that the guard forfeits a REAL, falsifier-proven 3.3× engine win at 262144×100 to recover
the 8 tall losses; the notes instead record the win as a comparison artifact.

Fix (one-sentence rewrite, notes-only, no code change): state the true record — the band-cell
win was a real engine-vs-stock win (3.3×, falsifier-proven in rounds 19 and 22), consciously
forfeited by the band guard; the post-guard band-cell tie is supported directly by round 18's
stock-vs-stock tie (`ab-tall-round18.md:71`) and round 22's C-arm stock time (4.0473 ms).

### NEW non-blocking (same root cause as F3 — fix in the same pass)

- `batch_gather.cu:567-568` ("re-admitting the band re-produced the tall loss") and `:2745`
  ("at ≥262144 rows the fused group walk re-produced the measured tall loss") repeat the
  debunked round-22 attribution and contradict the notes' own corrected mechanism
  (`tall-a3-impl-notes.md:199-205`) and the composed predicate (`deb8534b8e:2746`): the
  measured tall losses (1048576 cells, `ab-tall-round22.md:67-74`) occurred with the composed
  K arm running the STOCK fallback at ≥524288 — the group walk never ran there; band
  re-admission produced the band-cell WIN, not the tall loss. Comment-accuracy class per the
  prior review's precedent (non-blocking), but required with F3 (comment-only .cu edit).

### Verified-unchanged (spot checks against the prior-reviewed state)

- Test row counts identical to the prior review's recorded state: 262'144 pins
  (`batch_gather_tests.cpp:1306`, `:1329`), 262'143 (`:1374`, `:2007`, `:2041`, `:2091`,
  `:2151`, `:2181`, `:2216`, `:2237`, `:2259`, `:2290`), 262'142 (`:2062`), 300'000
  (`:1352`) — the test-file fix delta is comment wording only, as claimed. ✓
- Notes guard condition (`tall-a3-impl-notes.md:207-213`) quotes the predicate verbatim
  (`batch_gather.cu:2754-2756`). ✓

Per-finding verdicts: **F1 RESOLVED**, **F2 RESOLVED**, **F3 NEW BLOCKING** (false
evidence provenance in the notes' guard section), **N1/N2 NEW NON-BLOCKING** (comment
accuracy, same root cause as F3). The predicate itself is now correct and carve-tip-identical;
no behavioral or test defect was introduced; the expected post-guard grid (1 win, 9 ties) is
sound. The blocking finding is evidence-integrity only, in the same notes section F2 targeted:
fix the one sentence (+ the two comment clauses), then this gate clears without a rebuild
(comment-only .cu edit; bench still gated on rebuild + retest per the step rule).

VERDICT: FAIL blocking=1

## Re-check 2 (F3 delta)

Reviewer: glm step-gate reviewer (slot a3-guard), same reviewer, F3-DELTA scope only. As-of
2026-11-02. Scope: the three F3 fix sites only (notes evidence paragraph + two .cu comment
clauses) against `results/ab-tall-round22.md` (falsifier table + grid) and the band-form
predicate; clang-format dry-run on the edited source file. Nothing else re-audited.

### Site 1 — `results/tall-a3-impl-notes.md:215-221` (evidence paragraph): CONSISTENT

- "band-cell win WAS engine-vs-stock (round-22 falsifier: engine kernel 4 launches in K, 0 in
  C; C = fresh carve-tip control running the stock band route)" — matches the falsifier table
  (`results/ab-tall-round22.md:46-48`: Control C `string_fused_short_engine_kernel<256>`
  launches 0, Composed K 4, 600.9 µs avg) and the grid header (`ab-tall-round22.md:59`:
  "K = composed A3, C = fresh carve-tip control"). No "composed-vs-composed" claim remains.
- "the win is forfeited (both arms stock → expected tie)" — consistent with the band predicate
  (below) sending 262144×100 to the stock fallback on both arms, and with the round-22 C-arm
  stock time 4.0473 ms (`ab-tall-round22.md:75`).
- Expected post-guard grid (notes:223-225: 1 win 65536×100 2.4×, 9 ties, 0 losses) — consistent
  with the grid: 65536×100 sits below 262144 → engine serves it (−59.08% win, `ab-tall-round22.md:76`);
  8 tall cells back on fused-routed both arms; band cell stock both arms.

### Site 2 — `batch_gather.cu:565-570` (gate-constant comment): CONSISTENT

- "the engine wins the band cell when admitted (262144x100 −69.5%, engine-vs-stock
  falsifier-confirmed)" — matches `ab-tall-round22.md:75` (262144×100 **−69.50%** WIN 3.3×)
  and the falsifier (`:48`, `:56-57` "Falsifier: MET").
- "The band exclusion trades the band win for the tall-str1 route" — consistent with the band
  predicate keeping ≥524288 strings fused-routed (tall-str1, the round-10 +26.7% measured win).

### Site 3 — `batch_gather.cu:2742-2751` (classification-walk comment): CONSISTENT

- "engine wins the band cell when admitted (262144x100 −69.5%, engine-vs-stock)" — same
  evidence as Site 2 ✓. "band exclusion trades that one-cell win for keeping ≥524288-row strings
  on the routed-metadata path (tall-str1), the measured tall win (round-10, +26.7% headline)" —
  consistent with `results/ab-tall-round10.md` as cited in the prior review.
- Comment matches the predicate at `batch_gather.cu:2752-2754`:
  `not(STRING and num_rows >= kStringFallbackRowThreshold and num_rows < kBatchMemcpyRowThreshold)`
  with `kStringFallbackRowThreshold{262'144}` (`:571`) and `kBatchMemcpyRowThreshold{524'288}`
  (`:558`) → exclusion band exactly [262144, 524288); 262144×100 excluded, 65536×100 admitted,
  ≥524288 fused-routed. Comment and code agree.

### clang-format

`clang-format --dry-run -Werror cpp/src/copying/batch_gather.cu` (worktree root) → exit 0,
output `FMT_OK_batch_gather`. Only one source file is touched by the F3 delta (the other two
edits are markdown); no drift.

F3 evidence-integrity defect is cured at all three sites; no new finding in scope.

VERDICT: PASS blocking=0
