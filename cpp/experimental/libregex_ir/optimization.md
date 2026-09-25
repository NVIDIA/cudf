# Optimization guide

This document describes the performance decisions made by Regex IR's current
compiler and NVVM renderer, the limits that preserve regex semantics, and the
remaining optimization opportunities. It is a description of the checked-in
implementation, not a promise that every pattern takes the fastest path.

The core library compiles one pattern and one regex API operation at a time. It
does not own a CUDA context, a cuDF column, module caching, allocation, or kernel
launch policy. The cuDF integration and benchmark/test adapters provide those
pieces. Consequently, performance has three distinct layers:

1. compile-time IR simplification and executor selection;
2. generated per-row device code; and
3. integration policy, including compilation caching, row scheduling, output
   allocation, and kernel launch geometry.

Optimizations must preserve leftmost-first matching, ordered alternation,
greedy/lazy priority, captures, zero-length progress, UTF-8 boundaries, and the
selected API's result shape. A transformation that is valid for boolean
existence is not automatically valid for find, extract, replace, or split.

## Optimization pipeline

The compiler uses the following pipeline:

```text
pattern
  -> ordered Thompson Automata IR
  -> operation-specialized Instruction IR
  -> graph and capture simplification
  -> executor analysis and DFA/position construction
  -> operation-specific textual NVVM IR
  -> libNVVM LTO IR at -opt=3 -gen-lto
  -> RTCX/nvJitLink device LTO
  -> linked CUDA cubin
```

The optimizer is deliberately split between Instruction IR passes and NVVM
executor selection. Instruction IR stays inspectable and target-independent;
the NVVM renderer can then choose a code shape based on which results are
observable.

## Instruction IR passes

The internal optimizer applies these passes in order and verifies the graph before
and after the sequence:

| Pass | Current action | Performance purpose | Semantic constraint |
|:---|:---|:---|:---|
| unobserved capture removal | removes capture writes for boolean, find, count, and split; replacement retains only referenced captures; extract retains all captures | reduces stores, snapshots, live state, and register pressure | a capture is removed only when the selected API cannot observe it |
| epsilon-jump folding | bypasses linear instruction-free one-successor blocks, stopping at cycles | removes dispatch and empty control-flow plumbing | nullable cycles are not traversed indefinitely |
| literal fusion | combines a single-incoming linear chain of singleton predicates into one `match_literal`, up to 64 code points by default | exposes bounded peeks and literal-specialization opportunities | branch joins and cycles stop fusion |
| second epsilon fold | removes empty blocks exposed by literal fusion | shortens the final graph | same rule as the first fold |
| unreachable removal | compacts reachable blocks and rewrites dense block IDs | reduces generated code and analysis cost | reachability starts at the operation entry block |

These passes do not currently perform DFA minimization, common-subexpression
elimination, data-dependent planning, or row-length analysis. After optimization,
the NVVM planner separately extracts fused ASCII literals whose blocks dominate
the accept block.

## Executor selection

The NVVM renderer analyzes the optimized IR and selects the first safe path in
the following conceptual order:

```text
generated string-operation or word-run plan?
  yes -> emit the specialized operation
  no  -> exact ASCII expression?
           yes -> direct compare, byte finder, packed finder, or long-literal scan
           no  -> exact non-ASCII UTF-8 expression?
                    yes -> guarded-pivot packed finder and/or byte-domain KMP finder
                    no  -> assertion-free, non-nullable boolean graph with at most 64 positions?
                             yes -> compare a bit-parallel Glushkov plan with the existential DFA
                             no  -> continue to deterministic construction
                           deterministic construction succeeds?
                             boolean -> existential or assertion-aware DFA
                             capture-free span/global result -> streaming prioritized Glushkov or ordered DFA
                             extract with an unambiguous one-pass capture history -> tagged DFA
                             otherwise -> ordered recursive Thompson fallback
```

The generated module comments identify the selected executor and, for a
deterministic machine, its state and alphabet-class counts. Glushkov modules
report position, alphabet-class, shift, and exception counts. This is useful
when correlating a benchmark case with PTX, SASS, or an Nsight report.

### Generated string-operation plans

Before constructing an automaton, the renderer recognizes linear boolean ASCII
expressions whose explicit anchors make them equivalent to a fixed string
operation. It emits a dedicated function for:

- `begins_with(literal)`;
- `ends_with(literal)`;
- equality with `literal`;
- end-of-line matching against `literal` at end of input or immediately before
  the final LF accepted by the non-extended `$` semantics; and
- the corresponding anchored equality with that final-LF rule.

These functions perform only the necessary size checks, fixed-position
8/4/2/1-byte literal comparisons, and final-LF check. They do not call cuDF
string APIs and do not allocate or traverse DFA/Thompson state. The recognition
is deliberately restricted to boolean result shapes, linear singleton/literal
IR, ASCII text, and anchor modes whose exact behavior the generated operation
implements. Other cases continue through normal executor selection.

The non-multiline, case-sensitive ASCII form `literal.*$` also has a generated
final-line finder for boolean, count, and find. Default `$` semantics restrict
the only possible match to the final logical line. The generated function
walks backward to that line, seeks the literal there, and returns the span
ending at input end or immediately before the final LF. It does not construct
or call a general regex automaton.

### Boundary-delimited word runs

The exact source form `\b\w{N,}\b`, for positive decimal `N`, has a dedicated
maximal-word-run scanner for boolean, count, and find. ASCII-class mode uses
inline digit/letter/underscore tests and byte positions. Unicode-class mode
decodes code points and calls a generated predicate with an ASCII fast path
and constant-memory binary search over Unicode word ranges. This replaces the
former hundreds-of-comparisons linear range chain. A short run is skipped as a
unit, so the executor does not restart the regex machine at each character
inside a run. Other spellings or more complex word-boundary expressions are
not rewritten into this plan.

### Bit-parallel Glushkov NFA

Assertion-free, non-nullable graphs with at most 64 consuming positions can
use a Glushkov position automaton. Boolean `contains` uses the existential
form. Capture-free count, find/findall, replacement, and split use a streaming
prioritized form that advances all candidate starts together, records the
greedy accepted end, then performs a bounded rescan to recover the earliest
winning start. Lazy quantifiers remain on the ordered DFA/Thompson paths. The
canonical public IR remains ordered Thompson IR; the Glushkov machine is a
private NVVM code-generation plan derived with iterative epsilon closure.

Span/global selection also retains the ordered executor when a mandatory
ASCII start prefix is available, because its generated restart seeker skips
non-candidates more cheaply than continuously injecting starts. Replacement
templates that reference captures retain one-pass capture propagation rather
than paying for span recovery. Alternation is tracked structurally in the IR:
small alternatives that fit the 64-position representation may stream, while
larger alternatives rebuild the fallback DFA with priority preservation. This
avoids both treating an escaped `\|` as alternation and retaining the
speculative unordered DFA when Glushkov construction exceeds its position
limit.

One `uint64_t` is the complete per-row regex state. Bit `i` means that position
`i` matched the preceding logical character. For the next character `c`, the
generated loop computes:

```text
next = reach(c) & (first | follow(active))
```

`contains` injects `first` on every character, so all possible starts are
processed in one forward pass. `matches` injects it only at byte zero and tests
acceptance at end of input. Accepting positions are another compile-time mask.
Because these APIs expose only existence, no two-phase rescan, match-start
tracking, alternative-priority kill, or capture history is needed. Those
features would be required before a position automaton could implement find,
extract, replace, or split semantics.

Follow edges with the same positive position delta are combined into at most
eight masked shifts. Back edges, zero-delta edges, and less frequent positive
deltas become explicit exception masks. Alphabet-equivalent input characters
map to precomputed reach masks. Small reach sets are emitted as immediate
selects; larger sets use constant storage. The JIT embeds this read-only data
in the generated module, so the cuDF proposal's cooperative shared-memory copy
was not adopted: for the accepted plans there is no large generic program
object to stage, and adding launch-time shared-memory policy would violate the
core library's integration boundary.

Glushkov is selected only when profiling predicts a material advantage:

- the competing DFA transition table exceeds 32 KiB and the position graph has
  at most five exception source positions; or
- the graph has at least 32 positions, exactly one forward shift, and no
  exceptions.

Otherwise the existing specialized literal or DFA path remains selected.
Assertions, nullability, more than 64 positions, and result shapes that expose
spans, priority, or captures automatically use the established paths. This is
a performance fallback, not a semantic restriction or a user-facing option.

### Boolean existential DFA

For `contains` and `matches`, capture history and greedy/lazy path priority do
not change the boolean result. The renderer therefore uses an existential
subset construction when it is representable within the resource limits.

For scanning `contains`, the start closure is injected into every next state.
This recognizes a match beginning at any input position in one forward pass;
it does not invoke a matcher separately at every candidate position. The state
encoding uses one acceptance bit and 15 state bits, permitting at most 32,767
encoded states.

For a beginning-anchored boolean expression, the renderer removes the
already-proven start assertion and builds a non-scanning machine. A
non-scanning machine rejects immediately on entry to its dead state rather
than reading the remainder of a known-failing row.

### Assertion-aware boolean DFA

Boolean zero-width assertions are included in deterministic epsilon closure.
The transition is indexed by both an input alphabet class and the relevant
position context, which can include:

- beginning or end of input;
- beginning or end of line;
- word or non-word boundary; and
- CRLF-sensitive extended-newline context.

This avoids the recursive fallback for boolean word boundaries and line
anchors. It increases the transition-table dimensions, so construction still
falls back when the state/table limit would be exceeded.

### Ordered DFA

Find, count, replacement, split, and capture-free span operations expose the
preferred match. Their subset states therefore preserve Thompson-thread order.
Each 16-bit transition stores 14 state bits, an acceptance bit, and a
`stop-before-accept` bit used to retain ordered alternation and greedy/lazy
priority. The state limit is 16,383.

Acceptance is an ordered state marker, not an unordered property of the DFA
subset. When an earlier Thompson thread can continue past a match from a later
thread, the later acceptance remains deferred in the next state. A transition
updates the accepted end position only when that transition discovers a new
higher-priority acceptance; merely carrying a deferred acceptance must not
overwrite it. If none of the earlier threads can consume the next character,
`stop-before-accept` returns the deferred match without consuming that
character. This distinction is required for lazy repetition, ambiguous
alternation, and nullable branches in find, count, replacement, and split.

The repair was checked by replaying all 43,358 span-producing cases from the
fixed-seed 45-minute GPU differential campaign across two RTX A6000 devices;
the replay produced zero CPU-oracle mismatches.

A later nullable fix prevents scan-mode construction from injecting a new
start state on a stop-before transition. For a nullable expression that
injection would turn the deferred empty match into an accepting consuming
transition.

The ordered finder normally retries from successive logical-character
boundaries when a candidate fails. Unlike boolean `contains`, it cannot fold
all possible start positions into an existential state because it must retain
the winning start and path priority. Two compile-time proofs reduce that cost:

- a non-nullable machine with at most 16 possible starting bytes in at most two
  ASCII ranges receives an inline start-range predicate before initialization;
  and
- if every initial consuming class enters the same non-accepting self-loop
  state, those classes loop in that state, and no other state can re-enter it,
  a failed prefix run is skipped as one unit.

The latter rule preserves leftmost-first behavior because every later start
inside the skipped run reaches the identical state at the failure position.
Machines that accept the prefix, have stop-before transitions, multiple prefix
states, or another incoming edge do not use the acceleration.

For count, an ASCII start-filter miss advances the start by the already-known
one-byte width instead of decoding that byte again. Find, extract, replace, and
split retain the shared advance path because profiling found that the extra
control-flow edge worsened libNVVM lowering inside their larger executors.

### Tagged DFA

Capture extraction uses deterministic transitions only when analysis proves:

- no deterministic state has more than one matching consuming thread for an
  alphabet class;
- acceptance is terminal rather than mixed with continuing alternatives; and
- every route to a deterministic state has the same capture-update history.

When those conditions hold, each transition carries a small capture action
program and extraction runs without recursive Thompson calls. Ambiguous
capture histories fall back to preserve exact capture semantics.

### Recursive Thompson fallback

The fallback is the general correctness path for non-boolean internal
assertions and ambiguous capture histories. One recursive dispatcher executes
the cyclic Instruction IR block graph. It has a size-derived step limit to
bound nullable cycles.

The fallback retains several optimizations:

- an entry singleton or fused literal can supply a required first ASCII byte;
- candidate starts whose first byte does not match are skipped;
- `llvm.expect` marks a required-prefix hit as unlikely;
- literal predicates receive specialized helpers;
- leaf helpers carry `alwaysinline`, `readonly`, `readnone`, and `nounwind`
  attributes where valid; and
- only capture slots live for the selected result are initialized or updated.

The fallback uses two complementary filters. An entry literal supplies the
candidate-start byte check. Separately, the planner selects the longest fused
ASCII literal whose block dominates acceptance; a packed row-level search can
then reject inputs that cannot match before any recursive retries begin. The
row filter runs only for the initial search, so global and materializing APIs do
not rescan the suffix for every match. Literals that are merely common across
separate alternative blocks are not yet recovered. Historical SASS and Nsight
analysis found recursive call frames, local-memory traffic, repeated decoding,
and divergent backtracking to be the dominant costs, so deterministic paths
are preferred whenever semantics and table bounds permit.

### Candidate seeking, restart, and row rejection

The current implementation has four related but distinct literal filters:

1. If a scanning expression's entry IR requires one ASCII byte, the generated
   `seek_prefix_byte` helper searches eight input bytes per load using the
   standard zero-byte mask and `cttz`, verifies candidate bytes, and handles
   the tail bytewise. Recursive and ordered finders jump directly to those
   candidate positions instead of attempting every UTF-8 boundary. This is the
   start-prefix byte seeker; it is not a whole-literal match and is disabled
   when another specialized executor already owns the scan.
2. A deterministic machine can instead receive an inline sparse start-byte
   predicate when at most 16 byte values in at most two ASCII ranges leave the
   initial state alive. This avoids initializing the ordered matcher for clear
   non-candidates. COUNT advances an ASCII miss directly because its width is
   already known; larger span/materializing control-flow graphs retain the
   shared advance path based on profiling.
3. An ordered DFA can prove that all initial consuming classes enter one
   non-accepting, self-looping prefix state with no other incoming edge. On a
   failed attempt, its restart logic carries that state across the run and
   skips candidate starts that would reach the identical failure state. The
   proof excludes acceptance, stop-before transitions, and ambiguous prefix
   states.
4. Independently, the recursive fallback finds the longest fused ASCII literal
   whose block dominates acceptance. A packed whole-row presence test rejects
   impossible rows before recursive candidate retries. Because dominance is
   required, this filter does not yet recover literals that occur separately
   in every alternative.

The first two mechanisms filter possible starts, the third removes equivalent
ordered restarts, and the fourth rejects an entire row. They should not be
described interchangeably as "prefix filtering."

### Large boolean alternations

For a boolean graph with at least 80 blocks and an empty multi-way entry, the
renderer can compile top-level alternatives as separate recursively optimized
DFA functions. A short-circuit wrapper calls them in sequence.

This bounds state-product growth and allows one difficult branch to fall back
without forcing every branch into the same large machine. The tradeoff is
re-reading a row when multiple alternatives fail. It is intentionally limited
to boolean results, where alternative priority is not observable.

## Alphabet and transition representation

Deterministic construction partitions Unicode into equivalence classes: two
code points share a class when every consuming NFA predicate treats them the
same. The machine therefore stores `states x classes`, not `states x 1,114,112`,
transitions.

The generated representation has:

- a 256-entry constant-memory table for byte-to-class mapping;
- a generated comparison/select chain for non-ASCII Unicode intervals;
- 16-bit transition entries with state and control flags packed together; and
- a hard cap of 4,194,304 transition items during construction.

Transition tables up to 32 KiB use NVVM constant address space. Larger tables
use read-only global storage so multiple tables cannot exhaust the device's
64-KiB constant segment. The renderer emits ordinary loads and does not force a
cache operator; libNVVM and device linking choose the cache policy.

ASCII input avoids full UTF-8 decoding. Non-ASCII input decodes a code point,
computes its byte width, then classifies it. When the 256-byte alphabet differs
from its most frequent class in no more than two contiguous intervals, the
renderer emits inline range comparisons and selects instead of a byte-class
table load. More fragmented alphabets retain the constant table. Every DFA
still performs a transition lookup per consumed character, so transition-loop
instruction count remains a target for larger machines.

## Literal and anchor specialization

Several local analyses avoid general regex machinery:

- A complete linear exact-ASCII expression is recognized after IR
  optimization. One-byte global operations receive a direct raw-byte finder;
  multi-byte span/global operations use a first-byte guard and packed 8/4/2/1
  byte comparisons. ASCII bytes cannot occur inside a valid UTF-8 continuation
  sequence, so logical-boundary semantics are preserved.
- Anchored boolean exact matches compare the fixed literal directly. Scanning
  boolean literals of 2–15 bytes use the optimized DFA because PGO found it
  superior on early high-selectivity matches. Literals of at least 16 bytes
  scan eight possible first bytes per load with a byte-equality mask and invoke
  the packed verifier only for candidates.
- A complete linear exact expression containing non-ASCII code points operates
  on its canonical UTF-8 bytes. A leading match byte cannot be a continuation
  byte, so complete encoded equality preserves code-point boundaries. Literals
  with a selective ASCII punctuation/space byte use that byte as a fixed-offset
  pivot, reject candidates with an 8-byte guard, and verify the survivor with
  packed 8/4/2/1-byte loads. Count uses this path directly. Other APIs dispatch
  to byte-domain KMP for rows of at least 256 bytes, retaining linear behavior
  on repeated-prefix inputs, and use the guarded pivot below that threshold.
  Returned spans remain byte offsets and use the normal find-from ABI.
- The cuDF integration compiles an additional warp-per-string kernel when
  libregex_ir reports that the optimized expression is an exact ASCII literal.
  cuDF does not re-parse the source pattern. The launch follows the existing
  cuDF scalar-search policy and selects that kernel only above 64 average bytes
  per non-null row.
  A warp first probes 32 contiguous candidate positions, then processes four
  consecutive candidates per lane in 128-position rounds. Literal verification
  first compares the leading byte and only then issues the remaining packed
  8/4/2/1-byte loads. This prevents near-universal wide loads when the first
  byte is rare. The first probe avoids the early-hit regression of an
  unconditional four-candidate inner loop.
- Beginning anchors on boolean expressions become non-scanning control flow.
- Non-scanning deterministic machines stop at their dead state.
- Fused literals reduce helper and cursor operations in the fallback.
- A replacement reference or extract capture proven to cover the whole match
  reuses the match span instead of allocating and maintaining separate capture
  state. Extract aliases require exactly one begin and one end write and copy
  the completed whole-match span in the operation adapter.

The failure-function executor is deliberately limited to complete non-ASCII
literal expressions and is paired with the selective-pivot plan above. Exact
ASCII search retains its packed and warp-parallel paths because an earlier
general KMP experiment did not improve that workload.
General recursive expressions use dominator-proven fused literals as
variable-offset rejection filters, but do not yet derive a common literal from
separate alternatives or use a failure function for an extracted filter.

`compile_result::exact_ascii_literal` carries the semantic exact-literal proof
from libregex_ir to the integration layer. This keeps launch specialization
independent of raw regex spelling: escaped literals and compiler normalization
need not be rediscovered by cuDF.

The warp-per-string choice follows established precompiled cuDF string kernels,
not a regex-only scheduling model. Existing examples include character counts,
case-conversion sizing, URL decoding, `LIKE`, scalar `find`/`rfind` and
`contains`, `contains_multiple`, `find_instance`, long-string slicing, and the
string-parallel gather/copy path. Most use an average-width threshold; URL
decode and `find_instance` use their warp mapping directly. These kernels are
compiled into libcudf, whereas the regex literal kernel is emitted and linked
for one compiled pattern.

### 2026-09-25 literal-search follow-up

Both literal paths were gated on an RTX A6000 with a controlled same-session,
same-GPU cuDF baseline, Release compilation, 30-sample NVBench runs, focused
cross-backend tests, and the complete strings test binary.

- The final ASCII literal matrix covered 64/128/256-byte rows, 262,144 and
  2,097,152 rows, and 50%/100% hit rates. Warp-selected cases improved by a
  1.527x geometric mean, with a 0.996x–3.094x range and no greater-than-5%
  regression. The 64-byte controls retain the existing row-per-thread kernel.
- The UTF-8 KMP gate covered contains, count, and find at 128 and 512 bytes.
  The six exact-literal cases improved by a 1.263x geometric mean, ranging from
  1.036x to 1.505x. Overlapping-prefix tests additionally cover replacement
  and split and verify that find returns character rather than byte indices.
- Delaying the remaining packed loads until after the first-byte comparison
  improved the targeted exact-ASCII matrix by a further 1.172x geometric mean
  against the frozen complete-corpus baseline. The 512-byte rare-literal case
  improved by 1.987x; all targeted states remained within the 5% regression
  gate.
- Extending the general start seeker to verify the complete fused prefix was
  rejected. It improved `find` on `fn (?:is|as)_(\w+)` by 3.7--7.6%, but
  regressed 128-byte `count` by 6.5% and improved the six-state geometric mean
  by only 1.010x. The retained general seeker therefore performs an eight-byte
  vectorized search for the structurally proven first byte and leaves complete
  verification to the selected executor.

The final 1,488-state, 20-sample replay covered all legacy regex benchmarks and
all added corpora for both backends. JIT retained a 2.719x geometric-mean
speedup over the interpreter: 643 of 744 pairs were more than 5% faster, 23
were neutral, and 78 were more than 5% slower. Against the frozen pre-campaign
JIT run, the full-matrix geometric mean was 0.997x while the interpreter control
was 0.992x, indicating session-level drift rather than a broad JIT regression.
The intended exact-ASCII `contains` cases improved materially: 512-byte rare
literal improved 1.828x, 512-byte subtitle literal 1.732x, 128-byte subtitle
literal 1.662x, and 128-byte rare literal 1.423x.

### 2026-09-25 residual-loss executor campaign

The remaining large interpreter wins were traced to repeated candidate starts,
linear Unicode classification, whole-match capture bookkeeping, and direct
literal strategy rather than a single launch-policy defect. Controlled
20-sample RTX A6000 runs produced these end-to-end changes:

- streaming prioritized Glushkov reduced `pcre2/ff-off-range` count from about
  43.8 ms to 8.7 ms at 512-byte rows and changed `findall` from 24.5 ms to
  2.0 ms at 128-byte rows;
- aliasing a sole outer capture reduced `extract_all_record` from 29.6 ms to
  2.1 ms at 128-byte rows and from 426.8 ms to 31.5 ms at 2,097,152 by
  256-byte rows;
- the generated final-line `literal.*$` executor reduced comment-tail count
  from 7.45 ms to 0.16 ms at 128-byte rows and from 20.29 ms to 0.58 ms at
  512-byte rows;
- constant-memory Unicode word-range search reduced `pcre2/long-word`
  contains from 5.85 ms to 0.40 ms and from 20.53 ms to 1.34 ms for 128- and
  512-byte rows; and
- guarded-pivot/KMP UTF-8 literal execution made all six Russian-literal
  contains/count/find states faster than the same-run interpreter, ranging
  from 1.02x for contains to 3.52x for find.

Streaming split retains the selective bounded-span cache policy. On the legacy
late-match pattern, that combination was 1.74x and 1.90x faster than streaming
without caching for the two 256-byte split states. Nullable graphs that cannot
use the streaming position executor now rebuild the comparison DFA with
priority preservation. Together with suppressing start-state injection on a
stop-before transition, this restores empty-match counting, zero-range
counting, zero-length/zero-range replacement, and zero-length limited split.
The broad regex correctness sweep passes 234 of 235 interpreter/JIT typed
tests; the configured deep-nesting limit is the sole remaining known failure.
The formerly failing alternation and large-replacement cases also pass.

The first complete replay exposed two overly broad streaming choices that the
targeted cases did not cover: whole-match backreference replacement had lost
its capture-aware executor, and prefix-accelerated or oversized-alternation
patterns retained a speculative unordered fallback after the position plan
failed. The final selector excludes capture-substitution replacement and
mandatory-prefix searches, records alternation structurally, and rebuilds a
priority-preserving DFA when an alternation exceeds 64 positions. Focused
replays restored backreference replacement to 5.5--7.0x faster than the
interpreter, `hyperscan/user-agent` to 11--23x, and the large subtitle
alternation to 28--42x while preserving enumeration and prefix-free streaming
wins.

The refined 1,488-state, 20-sample replay measured a 3.169x geometric-mean JIT
speedup over the interpreter. Of 744 paired configurations, 704 were more than
5% faster, 20 were within 5%, and 20 were more than 5% slower; 19 of the slow
pairs exceeded the combined-noise screen. This reduces the pre-campaign 83
slow pairs to 20. Against the frozen pre-campaign run, JIT improved by a 1.154x
geometric mean while the interpreter control measured 0.986x, so the aggregate
gain is not explained by session drift. The residual losses are concentrated
in pre-existing short/common-literal searches, `rebar/rust-functions`, a few
small replacement cases, and 64-byte launch-overhead states rather than the
new streaming, Unicode-word, final-line, enumeration, or UTF-8-literal paths.

## API specialization

Only the selected API is emitted into a module. There is no runtime regex
opcode switch and no generic result union.

libregex_ir proves from optimized Instruction IR when every accepted match must
begin at input position zero. Such COUNT operations have cardinality at most
one, so the renderer emits a boolean executor and a generated adapter that
zero-extends the result to the COUNT ABI. FIND can use the same lowering when
its caller declares that the match end is unobservable; the adapter then writes
the known begin offset and deliberately omits the end offset. The default FIND
contract still writes a complete begin/end span.

The proof, executor selection, internal symbol naming, and adapter generation
therefore live in libregex_ir and honor custom codegen symbol options. cuDF only
sets the start-only FIND contract that matches its public API and retains the
column/offset wrappers, kernel compilation and retention, launch geometry,
memory-resource policy, and result-column construction. It does not parse raw
regex syntax or rewrite textual NVVM IR to select this optimization.

| API | Per-row generated behavior | Important cost consideration |
|:---|:---|:---|
| `contains` | returns on the first accepting state | existential scanning DFA can cover all starts in one pass |
| `matches` | starts at byte zero and requires the operation's anchor/end semantics | dead-state rejection avoids a known-useless tail scan |
| `find` | stores the first preferred begin/end pair | ordered candidate retries may dominate on long near-misses |
| `count` | enumerates non-overlapping matches and handles zero-length progress | calls the selected finder repeatedly |
| `extract` | writes the whole-match and numbered capture spans | tagged DFA is used only for capture-safe one-pass graphs |
| `replace` | copies unmatched ranges and a compile-time replacement template | null output sizes; non-null output materializes; referenced captures remain live |
| `split` | counts or writes field spans around non-overlapping matches | null output sizes; non-null output writes spans |

Replacement constants live in constant storage, and range copies use
`llvm.memcpy`. Replace and split expose sizing and emission through the same
device function by accepting a null output pointer.

At the cuDF-column integration layer, variable-size replace and split use a
sizing kernel, a device prefix scan/allocation, and an emission kernel. The
direct emission path matches rows again, avoiding pessimistic output
allocation. Enumeration, replacement, and split can instead select the bounded
span cache described below: sizing retains a small number of matches or fields
per row, emission consumes cached records, and only overflow rows rematch. The
program's `AUTO`, `OFF`, or `FORCE` cache policy, sampled match density,
executor cost, and the 64-MiB temporary-memory cap decide which path runs.

Split avoids several former integration costs. Forward limited split passes
`maxsplit` into the generated enumerator and stops once the limit is reached.
Unlimited and forward-limited paths use one count/offset sequence; only reverse
limited split builds the additional full-count offsets and enumerates the final
matches it needs. Record split materializes one final list column from field
pairs. Table split reuses those pairs to construct each output string column
directly, without first building a temporary list-of-strings column and then
extracting every list position.

## Compiler and linker optimization

The core API returns textual NVVM IR and does not invoke CUDA tools itself. The
cuDF integration uses:

- libNVVM verification for the selected compute architecture;
- libNVVM compilation of the assembled matcher and generated column wrapper
  with `-opt=3 -gen-lto`;
- the resulting in-memory LTO IR fragment as input to RTCX/nvJitLink with
  device LTO and `cudf_kernel_entry` retention; and
- module loading of the linked cubin.

The context RTCX cache keys both the NVVM fragment and linked cubin by source,
architecture, toolkit/runtime inputs, and JIT bundle identity. No CUDA C++
frontend is needed for the pattern-specific module: Regex IR already emits the
typed low-level control flow, constants, and wrapper ABI that libNVVM accepts.

Every Regex IR benchmark state also reports an uncached JIT-ready interval.
It starts at the source regex, disables nvJitLink's cache, and stops after the
linked module is loaded and every required kernel function is resolved. Input
construction, output allocation, and the first launch are excluded.

Symbol prefixes isolate generated internals, and the public execute-function
name lets the stable wrapper call the specialized matcher. Production
integrations should cache linked cubins by at least pattern, operation,
compile/codegen options, GPU architecture, and CUDA toolkit version. JIT
specialization targets repeated workloads; uncached one-shot compilation is
not expected to beat a precompiled interpreter's setup latency.

## Current launch and column policy

The cuDF integration consumes STRING columns and produces owning cuDF columns.
General generated kernels map one CUDA thread to one input row. Complex ordered
count, replacement, and split executors use 256-thread blocks; boolean,
extract/find, and direct literal executors use 1,024-thread blocks. The separate
exact-ASCII `contains` kernel uses 256 threads as eight warps, one warp per row,
when average bytes per non-null row exceed 64. The imported-corpus benchmark
adapter uses 128-thread blocks for its smaller grids. These gates come from the
profile campaigns below. Consecutive rows otherwise retain input order and are
addressed through the cuDF offsets column.

Column-level policy currently uses two inexpensive summaries. Exact-ASCII
`contains` uses total character bytes and non-null row count for its warp gate.
Selective span caching samples at most 2,048 rows to estimate average bytes,
matches, and overflow. Executor selection itself remains compile-time and does
not depend on input-column statistics.

The current integration does not:

- sort or bucket rows by length;
- transpose strings into a pivoted layout;
- use a persistent work queue for long-tail rows;
- assign multiple lanes to general non-literal regex executors;
- use input statistics to change the compiled executor;
- vector-load input in the general matcher; or
- combine several regex programs into one multi-pattern automaton.

These omissions are policy choices and future opportunities, not claims that
the techniques are universally unhelpful. Their preprocessing and temporary
storage must be included in end-to-end measurements.

### Offset and long-string support

The integration retains compiled wrapper variants for both 32-bit and 64-bit
input string offsets and selects from the actual offsets child at execution.
Generated matcher cursors, match spans, cached spans, and byte counts use
64-bit values. Variable-width output construction uses cuDF's offsets-child
helper, then selects a 32-bit or 64-bit emission wrapper from the returned
offset type. Replacement sizing separately reports a per-row overflow before
allocation. Thus columns whose total character storage requires large-string
offsets do not truncate addresses inside the generated regex code, while
ordinary fixed-width API outputs retain cuDF `size_type` semantics.

## Profile-guided findings

The detailed measurements are retained in the README's profile-guided
optimization section. The conclusions that currently guide the code are:

- recursive Thompson execution was limited by call-frame/local-memory traffic,
  repeated UTF-8 work, low active-lane efficiency, and register pressure;
- ordered and tagged determinization removed that traffic and substantially
  improved occupancy and active lanes;
- a bounded Glushkov position plan removes large boolean transition-table
  loads when the follow graph is sparse, while exception-heavy graphs remain
  faster as DFAs;
- direct one-byte global operations were previously paying full ordered-DFA
  overhead; the byte finder reduced instruction count by roughly eightfold in
  the profiled split kernels;
- non-scanning anchored failures were reading dead states to end-of-row; early
  dead-state rejection removed that work;
- current direct-byte kernels are memory-throughput limited; and
- profiled complex complete-corpus DFA kernels are compute/instruction limited,
  with low DRAM utilization, high branch efficiency, no spills, and remaining
  costs in classification, transitions, and divergent row completion.

Branch-free code is not an objective by itself. A specialized DFA still needs
loop, end-of-input, ASCII/UTF-8, and acceptance control flow. The useful goal is
to remove input-dependent regex dispatch and backtracking while keeping the
remaining branches uniform or strongly biased.

## Relation to the Sitaridi and Ross paper

[GPU-accelerated string matching for database
applications](https://doi.org/10.1007/s00778-015-0409-y) studies exact
single- and multi-pattern database search on C2070 and K40 GPUs. The authors'
[thesis chapter](https://www.cs.columbia.edu/~eva/gpu_thesis.pdf) gives the
full algorithms and evaluation. Its central lessons are still relevant:

- independent threads can create excessive L2 footprint and uncoalesced
  accesses;
- similar-length grouping reduces idle lanes at row completion;
- splitting work into stages can compact unfinished rows and reduce
  divergence;
- segmenting or pivoting input can improve cache-line reuse and coalescing;
- wide loads reduce load instructions and cache traffic;
- regular KMP access is more robust than data-dependent skipping on
  adversarial inputs; and
- algorithm, thread-group size, layout, selectivity, and device generation
  must be tuned together.

The paper is not a direct design for this project. Most of its GPU evaluation
is substring/SQL-LIKE matching over preprocessed layouts, while Regex IR must
preserve general regex priority, assertions, captures, and materializing API
semantics on ordinary cuDF columns. Its pivoting cost can be amortized over a
database column and repeated queries; it may lose end-to-end on a one-shot
regex call. Its KMP result applies directly to literal-only or extracted
literal filters, not as a replacement for a general tagged or ordered regex
automaton.

## Assessment of common optimization proposals

| Proposal | Current status | Assessment |
|:---|:---|:---|
| JIT/operation specialization | implemented | core design; warm execution benefits, while linked cubins should be cached |
| fixed string-operation lowering | implemented for safe linear boolean forms | emits begins-with, ends-with, equality, and final-LF-aware variants without automaton state |
| boundary-delimited word runs | implemented for exact `\b\w{N,}\b` syntax | scans maximal ASCII or Unicode word runs for contains, count, and find instead of retrying within a run |
| exact UTF-8 literal search | implemented for complete non-ASCII literal expressions | guarded fixed-offset pivots handle short/common rows; byte-domain KMP handles long or repeated-prefix rows; complete encoded equality and lead-byte starts preserve UTF-8 boundaries |
| warp-per-string ASCII literal scan | implemented in the cuDF integration | selected above 64 average bytes per non-null row; a contiguous first probe plus four candidates per lane balances early-hit latency and long-miss throughput |
| bit-parallel Glushkov NFA | implemented for gated boolean and capture-free span/global plans | existential boolean execution and two-phase prioritized span recovery avoid candidate restarts; lazy, assertion-heavy, capture-bearing, and exception-heavy cases retain other executors |
| start-prefix seeking and ordered restart removal | implemented behind structural proofs | eight-byte ASCII seeking skips impossible starts; sparse DFA start ranges and self-loop restart proofs remove additional retries without changing leftmost-first results |
| selective bounded span caching | implemented for enumeration, replacement, and split | samples large inputs, caches up to four bounded records per row within 64 MiB, and rematches only overflow rows |
| profile occupancy, memory use, and divergence | implemented as a development practice | current complex DFA cases are more instruction/lane limited than DRAM limited; direct-byte cases are bandwidth limited |
| Aho-Corasick | not implemented | useful for many exact literals or a bank of extracted literals, not a general regex replacement; dense transition storage, not automaton state count alone, is the main GPU memory risk |
| cheap filter then full regex | partially implemented | the recursive fallback has entry-byte filtering and a fused row-level filter for dominator-proven ASCII literals; alternative-intersection and selectivity-aware planning remain gaps |
| group strings by length | not implemented | promising for high length variance and repeated scans, but binning/permutation/scatter cost must be amortized |
| pivot strings | not implemented | potentially useful for stable, repeatedly scanned fixed/coarsely bucketed columns; too costly as an unconditional cuDF API step |
| staged unfinished-row compaction | not implemented | promising for selective contains and highly skewed rows, but extra kernels/global traffic can outweigh divergence savings |

Aho-Corasick is specifically a multi-literal trie plus failure transitions. It
does not make arbitrary regex matching a DFA, and its number of states is
normally linear in the total literal length. A full regex DFA can suffer
subset-state explosion; an Aho-Corasick implementation more commonly suffers
from a large alphabet-by-state transition representation. The two concerns
should not be conflated.

## 2026-07-05 highest-value-opportunity campaign

All eight opportunities below were exercised on an RTX A6000 with CUDA 13.2,
cuDF 26.08, Release compilation, libNVVM `-opt=3`, and nvJitLink `-lto -O3`.
Normal timings came from NVBench; Nsight Compute full-set replay was used only
for diagnosis. The primary gate was the existing 2,097,152-row,
`StringBytes=128` API matrix. Output allocation and owning cuDF-column
construction remained inside the timed region.

The final accepted code changed the geometric-mean latency of that gate as
follows. Small sub-millisecond contains differences have higher relative noise;
count and materializing wins are substantially larger than the measured noise.

| API | Cases | Before (ms) | After (ms) | Speedup |
|:---|---:|---:|---:|---:|
| contains | 22 | 0.747 | 0.708 | 1.056x |
| count | 7 | 4.347 | 3.381 | 1.286x |
| extract | 3 | 7.914 | 7.922 | 0.999x |
| replace | 14 | 16.818 | 12.089 | 1.391x |
| split | 7 | 22.800 | 17.030 | 1.339x |

The largest individual accepted improvement was `[a-z]+Z`: count fell from
6.828 to 3.689 ms, plain replacement from 20.493 to 11.121 ms, backreference
replacement from 19.797 to 11.189 ms, and split from 19.895 to 12.930 ms.

### Opportunity outcomes

| Opportunity | Experiment | Outcome |
|:---|:---|:---|
| literal planner | exact-ASCII graph recognition, packed comparisons, 8-byte first-byte masks, and short-literal DFA A/B | accepted; short scanning literals use the DFA, fixed/non-boolean and long literals retain direct paths |
| selectivity-aware filtering | contains sweep at 1%, 5%, 10%, 50%, and 100% hits | fused plan retained; it already sustained about 156–253 GB/s on low-to-moderate selectivity, leaving no one-shot margin for a second scan plus compaction/scatter |
| ordered restart removal | sparse start ranges and a proved self-loop prefix-run skip | accepted; count improved 16–46% across the seven transform expressions |
| materialization reuse | temporary one-pass staged-row replacement versus current exact-size rematch | rejected; staged allocation and compaction increased latency 26–157% |
| deterministic-loop reduction | inline byte-class ranges for alphabets with at most two exceptional intervals | accepted; several ordered count cases improved another 4–27%, with boolean cases neutral within noise |
| length scheduling | temporary physically length-sorted copy of the existing normal-width corpus | matcher-only upper bound was 7–16% faster, but sorting/gather/scatter was excluded; not enabled for one-shot APIs |
| pivoted representation | temporary fixed-width 2,097,152×128 literal probe | cached scan was 4.45x faster, but the 21.05 ms pivot plus 0.68 ms scan was 7.1x slower than the 3.03 ms row-major one-shot scan; keep as a cached-column design only |
| multi-pattern filter | temporary four-literal corpus producing four predicate arrays | one fused scan took 2.91 ms versus 10.30 ms for four scans, a 3.54x win; no single-pattern ABI change until a real multi-pattern consumer exists |

The complete 58-case large-corpus Regex IR gate also improved: geometric mean
latency fell from 0.366 to 0.340 ms, a 1.076x speedup. Boost cases 1, 2, and 6,
Leipzig cases 1 and 11, and OpenResty cases 1 and 18 improved by 23–36%.
The only apparent slower state in the first sweep was a 0.169 ms OpenResty
case; a longer 0.5-second rerun measured 0.170 ms, within 0.6% of its baseline.

### Temporary workload definitions

The temporary probes were removed after measurement. They are described here
so the conclusions are reproducible without mistaking them for shipped
benchmarks:

- **staged replacement:** 262,144 variable-length rows capped at 128 bytes,
  the existing transform patterns 0, 2, 5, and 6, and a per-row temporary
  stride of `2 * StringBytes`. The generated replacement ran once into the
  temporary, sizes were scanned, then a second kernel compacted bytes into the
  owning cuDF STRING child. Existing rematch times were 2.17–3.00 ms; staged
  times were 2.74–5.73 ms.
- **length ordering:** the existing seeded normal row-length distribution at
  2,097,152 rows, widths 128 and 256, and count patterns 2, 5, and 6. Rows were
  physically reordered by length before upload. This deliberately measured a
  cached-layout upper bound; permutation construction and output restoration
  were not timed.
- **pivot:** fixed 128-byte rows, 50% containing `0987 5W43` at byte 59,
  15 median event-timed samples after two warm-ups. Both layouts produced the
  same boolean output. The one-shot result includes the row-major-to-pivot
  transpose.
- **multi-pattern:** fixed 128-byte rows with `error:`, `https://`, `@host.`,
  `192.168.`, or no literal in a five-row cycle. Four separate specialized
  kernels and one fused kernel both wrote four boolean arrays; 15 median
  samples followed two warm-ups.

Two Nsight Systems CUDA traces completed the benchmark process but exceeded
120- and 180-second limits while finalizing their reports in this environment,
so no partial `.nsys-rep` is treated as evidence. Nsight Compute full profiles
completed normally. The optimized `[a-z]+Z` count kernel used 78 registers per
thread, achieved 43.37% warp occupancy, executed 1.287 billion SASS
instructions, and had 89.69% uniform branch targets. Its low 6.29 active-thread
ratio explains the cached length-ordering headroom, but the end-to-end
reordering cost still prevents enabling it unconditionally.

## 2026-07-07 Glushkov experiment

The experiment used the design and implementation discussion in
[rapidsai/cudf#21936](https://github.com/rapidsai/cudf/pull/21936) as a reference,
then adapted the idea to Regex IR's JIT boundary. The cuDF proposal stores a
general position program and cooperatively caches it in shared memory. Regex IR
instead specializes follow and reach masks into NVVM constants and immediate
operations for one pattern. It also limits the position engine to boolean
existence, where leftmost-first start and capture reconstruction are
unobservable.

The first forced-plan sweep ran all 74 complete-corpus Regex IR cases. It
validated every result against the existing cuDF setup oracle and measured a
1.021x geometric-mean speedup, but it also exposed 11 regressions above 2%.
Exception-heavy bounded graphs were the wrong fit: Boost/GCC's names-near-river
case fell to 0.647x, the quoted-string case to 0.754x, and IPv4 to 0.912x.
Keeping a universally selected Glushkov path would therefore have failed the
experiment despite the positive average.

The retained cost gate was evaluated with baseline/branch ABBA ordering because
the available account could not lock RTX A6000 clocks. Twenty-five or more
NVBench samples and a 0.5-second minimum sampling interval were used per state.
Representative accepted results were:

| Corpus case | Existing DFA (ms) | Glushkov (ms) | Speedup | JIT-ready effect |
|:---|---:|---:|---:|:---|
| OpenResty 23, `[a-q][^u-z]{13}x` | 0.497 | 0.358 | 1.388x | about 92–103 ms to 48–50 ms |
| Rust Leipzig 6, same expression | 0.394 | 0.288 | 1.367x | about 91–99 ms to 48–55 ms |
| OpenResty 8, 57-position fixed alternatives | 0.564 | 0.457 | 1.234x | neutral, because its DFA was already small |
| OpenResty 7, same expression on an early-result corpus | 0.197 | 0.198 | 0.999x | neutral |
| mariomka IPv4 negative control | 0.266 | 0.266 | 1.000x | DFA retained |

The bounded expression previously produced 16,385 DFA states and a read-only
global transition table. Glushkov represents it with 15 position bits, one
shift, no exception edges, and no transition table. A full Nsight Compute
replay of OpenResty case 23 explained the normal-timing gain:

| Metric | DFA | Glushkov |
|:---|---:|---:|
| profiled kernel duration | 65.92 us | 46.37 us |
| global-load instructions | 274,118 | 138,083 |
| long-scoreboard cycles per issued instruction | 11.40 | 5.10 |
| eligible warps per scheduler | 0.157 | 0.258 |
| registers per thread | 40 | 40 |
| achieved occupancy | 24.07% | 24.48% |
| branch efficiency | 96.73% | 96.73% |

Executed instructions increased from 4.72 million to 4.99 million; the win is
not fewer arithmetic instructions, but eliminating the large data-dependent
transition-table load. The profile also found an LTO device-call frame in the
benchmark wrapper. Marking the Glushkov public entry `alwaysinline` was tested
and rejected: controlled normal timings were neutral, so the attribute was not
retained without evidence that it changes linked code profitably.

Temporary diagnostic output used to record position, class, shift, exception,
and DFA-state counts was removed after the gate was chosen. Baseline
executables, normal NVBench JSON, and full `.ncu-rep` files were kept outside
the source tree during the campaign; no profiler-replay duration is used in
presentation tables.

## 2026-07-07 all-benchmark PGO campaign

This campaign began from commit `9bcd4fc` on an isolated branch and froze the
branch-point benchmark executables and JSON before changing source. Normal
timings used GPU 1 of the two RTX A6000 devices because GPU 0 had unrelated
activity. The final gate replayed all 1,764 Regex IR cuDF-API states, all 88
imported-corpus states, and all eight warm/cold states. Unchanged cuDF states
from the immediately preceding same-machine run were retained as the
comparison baseline. Nsight Compute full-set replay and an Nsight Systems CUDA
timeline were used only for diagnosis.

The profiles covered contains, count, extract, both replacement kernels, both
split kernels, and representative OpenResty, Leipzig, Boost, and mariomka
cases. They confirmed that no one launch or executor policy fits every result
shape: count is instruction/divergence limited, extract and emission phases
move substantially more data, and the small-grid corpus runs expose different
launch-geometry tradeoffs from the multi-million-row API matrix.

### Accepted changes

An ordered DFA with a sparse ASCII start filter used to load a rejected ASCII
byte in the filter and then load it again in `advance` solely to move the start
by one byte. Count now takes a direct filter-miss edge using the already-known
ASCII width. The direct edge is deliberately limited to `MATCH_COUNT`.
Controlled ABBA measurements found that placing the same edge inside the
larger replace/split control-flow graph changed libNVVM's lowering and slowed
large replacement states, even though it removed source-level work.

The full count matrix improved by 1.027x. A higher-precision large-state ABBA
gate measured 1.126x for `\d+` and 1.093x for `[a-f]+|[0-5]+`; replacement
returned to 0.999x after the operation gate. The profile explains the count
gain:

| Metric, 2,097,152 rows × 128 bytes, `\d+` | Before | After |
|:---|---:|---:|
| profiled kernel duration | 2.544 ms | 2.252 ms |
| executed SASS instructions | 1.241 billion | 1.083 billion |
| global-load instructions | 57.31 million | 39.26 million |
| registers per thread | 39 | 39 |
| achieved occupancy | 90.27% | 90.46% |

The corpus adapter now uses 128-thread blocks while the large API wrappers
retain 256. The corpus-only ABBA gate improved OpenResty by 1.020x, Leipzig by
1.005x, Boost by 1.009x, and mariomka by 1.004x. The exhaustive corpus replay
improved the large families by 1.013–1.021x, compact Boost by 1.018x, and
scalar Boost by 1.035x. This is integration policy rather than a public codegen
option.

### Rejected changes

| Experiment | Measured result | Decision |
|:---|:---|:---|
| force `alwaysinline` on boolean and extract entrypoints | linked SASS retained the same 72/80-byte backend-outlined call frames; API and corpus timings were neutral | removed |
| widen the sparse start filter from 16 to 32 candidate bytes | transform geomeans fell to 0.711x for count, 0.753x for replace, and 0.800x for split because passing candidates paid a duplicate load | removed |
| jump directly from an unfiltered first dead transition back to search | the additional inner-loop backedge produced a less favorable libNVVM CFG; geomeans fell to 0.640x, 0.699x, and 0.792x | removed |
| use 128-thread blocks for every wrapper | representative contains fell to 0.398x and count/extract/replace/split lost 1–5%, despite the small-grid corpus wins | retained only for corpus integration |
| use the direct filtered edge for all span/materializing APIs | large replacement regressed 5–8% in ABBA measurements | gated to count |

The rejected 32-byte-filter and direct-dead-edge experiments used every
transform expression at 2,097,152 rows and 128 bytes. The launch experiment
used one representative from each API and all four corpus families. The final
replacement gate used patterns 0 and 3, plain and backreference replacement,
1,048,576–8,388,607 rows, and 256-byte rows. These temporary matrices are
documented here but are not additional shipped benchmark registrations.

## 2026-07-14 Glushkov comparison and launch PGO campaign

This campaign compared the JIT at commit `8816f571fd` with
`lingyany/glushkov-nfa` at `8eaf307991` on an RTX A6000 (SM 8.6). Normal
NVBench timings exclude compilation. Nsight Compute 2026.1 full-set replays
were used only to explain representative kernels; their replay durations are
not mixed into the normal benchmark geomeans.

The comparison intentionally profiled both a Glushkov-winning late-failure
case and a JIT-winning negative control:

- count with `.+[0-9]`, 262,144 rows, and widths from zero to 64; and
- contains with `[A-Z ]+\d+ +\d+[A-Z]+\d+$`, 262,144 128-byte rows, and a
  50% hit rate.

Nsight embedded the final SASS but could not associate PTX or source with the
runtime-linked JIT cubin. The JIT PTX was therefore captured separately by
passing the same assembled NVVM module through libNVVM with
`-arch=compute_86 -opt=3`, omitting only `-gen-lto`. The branch PTX was
extracted from its SM 8.6 fatbin. This gives an exact view before their
different link paths, while the NCU SASS and dynamic counters remain the
authoritative comparison of executed code.

### Late-failure count profile

| Metric | JIT, 1,024 threads | JIT, 256 threads | Glushkov branch |
|:---|---:|---:|---:|
| NCU replay duration | 767.20 us | 597.15 us | 376.58 us |
| executed SASS instructions | 182,884,681 | 182,884,681 | 139,648,028 |
| static SASS instructions reported | 472 | 472 | 2,080 |
| registers per thread | 39 | 39 | 40 |
| theoretical occupancy | 66.67% | 100.00% | 100.00% |
| achieved occupancy | 35.67% | 57.16% | 76.55% |
| eligible warps per scheduler | 1.51 | 2.70 | 3.01 |
| average active threads per warp | 6.87 | 6.87 | 15.12 |
| branch efficiency | 96.66% | 96.66% | 93.97% |

The launch-only JIT change leaves the SASS, instruction count, register count,
branch efficiency, and active-lane count unchanged. Its gain comes from making
more of that same work schedulable. In the hot JIT loop, a byte load,
transition address arithmetic, and a constant-memory 16-bit transition load
each execute about 3.11 million warp-times with only eight threads active. The
Glushkov loop's corresponding byte-processing sequence executes about 490,000
warp-times with 18 threads active. It wins this case despite having more than
four times as much static SASS, because it executes 24% fewer dynamic
instructions and keeps more than twice as many lanes useful.

The PTX explains the code-shape difference. The JIT embeds the small DFA and
its classes as pattern constants in a self-contained kernel. The branch uses a
generic `gkprog_device`, loads program metadata, and performs cooperative
setup and a block barrier. Final SASS register usage is nevertheless nearly
identical, so static PTX size and virtual-register counts do not explain the
timing; dynamic traversal does.

### Contains negative control

| Metric | Specialized JIT | Branch fallback (`reprog_device`) |
|:---|---:|---:|
| normal NVBench time | 0.307 ms | 6.618 ms |
| NCU replay duration | 199.78 us | 6.33 ms |
| executed SASS instructions | 44,140,370 | 1,269,388,949 |
| static SASS instructions reported | 448 | 1,776 |
| registers per thread | 38 | 56 |
| achieved occupancy | 57.95% | 56.41% |
| eligible warps per scheduler | 0.74 | 0.51 |
| average active threads per warp | 24.77 | 18.25 |
| branch efficiency | 98.48% | 87.56% |

Here the JIT is 21.5x faster in normal timing and about 31.7x faster under NCU.
The external branch does not select `gkprog_device` for this expression; its
`reprog_device` fallback executes 28.8x as many instructions. The hottest SASS
sequence executes about 7.10 million warp-times and repeatedly performs local
loads, index arithmetic, and indirect transition loads. The JIT loop executes
about 1.05 million warp-times with 28 active lanes. This negative control does
not characterize Glushkov itself. It shows that adopting the branch wholesale
would lose the JIT's specialized executor coverage, so plan selection matters
as much as the best individual executor.

### Accepted launch policy

Complex ordered count, replacement, and split kernels now launch with 256
threads. Boolean kernels, extract/find kernels, and the single-byte and packed
ASCII literal executors retain 1,024 threads. The executor gate uses the
generated module's executor annotation, so regex syntax is not parsed a second
time in the cuDF integration.

The selected thread count is a policy ceiling, not an unchecked launch
requirement. After linking, cuDF queries
`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` for the generated kernel and clamps
the launch to that device-reported limit, rounded down to a full warp. This is
normally a no-op, but prevents register-heavy generated extract kernels from
failing with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` while preserving the faster
1,024-thread geometry for kernels that support it.

The later exact-ASCII `contains` warp kernel is a separate 256-thread launch
selected by column width; it does not change this row-per-thread block-size
gate.

The final replay covered every affected parameterized state:

| Suite | States | Before geomean | After geomean | Speedup | >5% wins / neutral / >5% regressions |
|:---|---:|---:|---:|---:|---:|
| count | 42 | 1,193.172 us | 1,004.524 us | 1.188x | 36 / 6 / 0 |
| extract control | 36 | 436.848 us | 436.605 us | 1.001x | 0 / 36 / 0 |
| replace | 84 | 3,726.696 us | 3,141.476 us | 1.186x | 72 / 12 / 0 |
| split | 42 | 3,540.833 us | 3,059.631 us | 1.157x | 35 / 7 / 0 |
| all affected states | 204 | 1,998.126 us | 1,744.145 us | 1.146x | 143 / 61 / 0 |

A controlled replay of the profiled count case improved from 776.872 us to
661.643 us, or 1.174x. The final 256-thread JIT remains slower than the branch
on that particular late-failure expression because launch geometry cannot
remove its extra transitions or ordered restart work.

Applying 256 threads to every non-boolean executor was rejected. Although that
experiment improved the 204-state geomean by 1.131x, the direct-literal
256-byte cases regressed to 0.325x for count, 0.585x for replace, and 0.488x
for split, and some large extract cases lost about 19%. These scans benefit
from the lower cache pressure of a 1,024-thread block. Executor-aware gating
keeps their original geometry and removed every greater-than-5% regression in
the final replay.

### Profile-derived opportunities

| Priority | Opportunity | Evidence and constraint |
|:---:|:---|:---|
| 1 | Inline tiny DFA transitions as selects or immediate logic | The late-failure JIT spends about 3.11 million warp-times on per-byte address arithmetic and `LDC.U16`; its machine has only a few states and classes. Gate by generated code size because larger machines need compact tables. |
| 2 | Build an ordered streaming or bit-parallel plan for capture-free span/global operations | The branch executes 24% fewer instructions and more than doubles active lanes for `.+[0-9]`. Any plan must still reproduce the exact leftmost start, greedy/lazy end, zero-length progress, and non-overlapping restart semantics used by count, replace, and split. |
| 3 | Bin or persistently schedule skewed rows | Only 6.87 lanes are active in the JIT count loop. A coarse offset-derived length histogram or work queue could reduce tail effects, but its construction, indirect row access, and result scatter must be included in timing. |
| 4 | Cache bounded match spans selectively between sizing and emission | Subsequently implemented for enumeration, replacement, and split with sampling, a four-record bound, 64-MiB cap, and overflow-only rematching. Unconditional staging remains rejected. |
| 5 | Add diagnostic line information without changing release cubins | Runtime-linked LTO cubins prevented NCU PTX/source correlation. A profiling-only cache option that retains line information would make future SASS attribution easier without burdening production code or cache keys. |

## Current optimization status and prioritized opportunities

The following sections retain the original priority order while recording work
that has since landed, particularly literal planning and bounded span caching.
Remaining ideas still need end-to-end benchmarks over API, row count, row
width, length variance, hit rate, regex complexity, and reuse count.

### 1. Extend the literal planner

Exact ASCII expressions, exact non-ASCII UTF-8 KMP, sparse first-byte ranges,
and dominator-proven fused ASCII literals in general recursive expressions are
implemented. Remaining work includes deciding whether long ASCII literals need
a failure-function or Two-Way path and intersecting mandatory substrings across
separate alternatives.
Useful specializations still include:

- a benchmark-justified regular linear search for adversarial long ASCII
  literals;
- common-substring extraction across alternatives; and
- selectivity estimates that can decide when the row filter repays its extra
  input pass.

A mandatory literal at a variable regex offset can reject a row but cannot in
general identify the match start. A fixed prefix can identify candidate starts
directly; the planner keeps these optimizations separate.

### 2. Make cross-row filtering selectivity-aware

A cheap filter is not automatically cheap: a separate pass doubles input
traffic on rows that survive. The fused plan is implemented and won the
existing benchmarks. A future multi-kernel plan would need:

- a fused per-row filter that immediately verifies a hit; and
- for very selective filters and expensive verifiers, a bitmap/row-ID pass
  followed by compaction and a verification kernel.

Choose between them using sampled literal frequency, row widths, regex cost,
and expected cubin reuse. Measure preprocessing, compaction, and result scatter
inside the end-to-end path.

### 3. Generalize ordered restart removal

Sparse starts and self-looping prefix runs avoid many retries. Capture-free,
assertion-free span/global machines now use the streaming prioritized Glushkov
executor: one pass advances all starts to the selected greedy end and a second
bounded pass recovers the earliest start. Whole-match-only captures are aliased
to that span and can use the same path for capture results. Capture-substitution
replacement, mandatory-prefix searches, lazy quantifiers, zero-width
assertions, and captures whose histories are genuinely observable retain an
ordered DFA, tagged DFA, or Thompson executor. Large alternations that do not
fit the position representation also return to a priority-preserving ordered
DFA.

### 4. Reuse match work in materializing APIs

Replacement, enumeration (`findall` and `extract_all_record`), and split now
use selective bounded span caching. The sizing kernel can retain up to four
matches per row in temporary memory. Emission consumes those spans directly;
rows that exceed the bound are marked and rematched by an overflow-only
kernel. This keeps the direct two-pass implementation as the fallback and
avoids making dense or high-capture workloads pay unbounded staging costs.

Selection is automatic for inputs of at least 131,072 rows. A deterministic
sample of at most 2,048 rows records valid rows, input bytes, bounded match
count, and overflow count. The policy combines those observations with the
compiled executor kind, rejects overflow rates above 1%, rejects average match
counts above 1.25, caps temporary cache storage at 64 MiB, and requires enough
weighted input work to repay the extra cache traffic. Split receives a larger
weight because cached field spans also avoid rebuilding token spans. The
`regex_jit_program_options::span_cache_policy` creation option accepts `AUTO`,
`OFF`, and `FORCE`, making benchmarking and diagnosis explicit and local to each
compiled program.

The cache is API-specific. Enumeration stores capture slots and scatters final
string pairs from them. Replacement stores capture slots plus a per-row match
count, then executes the baked replacement steps from cached captures. Split
stores bounded field spans; reverse limited split still rematches overflowing
rows because it needs the final matches. All cache, overflow, count, and span
buffers use the temporary memory resource. Empty strings retain a non-null
sentinel pointer, matching the direct NVVM pair writer.

Representative RTX A6000 end-to-end measurements at 262,144 rows and 256-byte
maximum/target width showed automatic caching reducing `findall` from 114.56 ms
to 59.86 ms, `extract_all_record` from 136.13 ms to 70.77 ms, and an expensive
replacement from 246.92 ms to 145.34 ms. Split improved from 3.40 ms to 2.60 ms
and from 1.97 ms to 1.60 ms for the two measured late-match expressions. A
fast tagged-deterministic replacement regressed when caching was forced, so the
automatic profitability threshold leaves that workload on the direct path.

### 5. Reduce deterministic-loop instruction count

Inline byte-class ranges are implemented. Remaining comparisons include:

- replacing tiny DFA transition-table loads with bounded selects or immediate
  logic when the state/class product is small;
- specialized all-ASCII loops when column metadata or sampling justifies it;
- processing multiple input characters per iteration in general deterministic
  loops; exact-literal and prefix-seek paths already use packed/wide loads;
- loop unrolling only when it does not worsen register pressure;
- DFA minimization for existential boolean machines;
- hot-state numbering and transition-row layout; and
- smaller transition encodings when state/class counts allow them.

The complete-corpus profiles indicate this is more likely to help complex DFA
cases than cache-policy forcing.

### 6. Schedule skewed rows more deliberately

Start with coarse length bins rather than a full sort. Build a row-index
permutation from the already-available offsets, launch a small number of bins,
and write results to original row indices. Enable it only when length variance,
mean width, and reuse predict that reduced idle lanes repay histogram,
permutation, and launch costs. The late-failure count profile's 6.87 active
threads per warp is a concrete target, but normal timing must include the bin
construction.

For extreme long tails, compare bins with a persistent queue or multi-lane
segmentation of long rows. Segment boundaries need overlap or carried automaton
state; ordered captures make this harder than exact KMP search.

### 7. Treat pivoting as a cached column representation

Do not transpose every cuDF input unconditionally. A pivoted or tiled view is
most plausible when the same stable column is scanned by many patterns. Cache
the transformed representation, use coarse length classes to limit padding,
and include transform construction and cache lifetime in the API design.

### 8. Add multi-pattern filtering only for a real workload

If consumers need many regex predicates over the same column, an
Aho-Corasick filter over exact patterns or extracted mandatory literals can
produce candidate `(row, pattern)` pairs. Specialized regex functions can then
verify only those candidates. Keep the verifier separate from the literal
automaton to avoid the state product of a combined full-regex DFA.

## Measurement checklist

For every proposed change, compare the current executor and candidate plans
across:

- API: contains, matches, find, count, extract, replace, and split;
- row count and total input bytes;
- mean, maximum, and coefficient of variation of row length;
- match and filter selectivity, including adversarial near-misses;
- ASCII and non-ASCII proportions;
- regex states, alphabet classes, assertions, captures, and output density;
- cold compilation, first launch, and warm execution;
- one-shot versus repeated use, including preprocessing amortization; and
- complete output-column construction, not matcher-kernel time alone.

Nsight Compute should track active threads per warp, branch efficiency,
eligible/active warps, achieved occupancy, registers, local-memory traffic,
long-scoreboard stalls, instruction count, L1/L2 hit rates, and DRAM throughput.
Nsight Systems should confirm whether time belongs to compilation, module load,
allocation/scan, launches, or device execution. Profiler replay changes
absolute latency, so presentation and README performance tables should continue
to use normal NVBench timings and use profiler data only to explain them.
