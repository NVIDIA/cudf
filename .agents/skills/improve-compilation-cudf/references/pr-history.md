# cuDF compilation and binary-size PR reference

## Scope and evidence

Researched on **2026-09-22** using live GitHub metadata/descriptions and local git history.
Screened merged PR histories and searched repository-wide for `build time`, `compile time`,
`compilation`, `instantiation(s)`, `binary size`, and `library size`; partitioned capped
queries by date and inspected the two tracking issues.
The tables retain directly relevant changes plus explicitly marked supporting context;
unrelated compiler fixes, features, and routine dependency updates were excluded.
This is a broad historical reference, not a claim that keyword searches can discover every
unmentioned build-side effect. Refresh before claiming an opportunity is still open.

All PRs in the technique tables were returned as **merged**. Measurements below are
**reported in the PRs**, not rerun for this documentation task. They use different hardware,
CUDA versions, architecture sets, and baselines and must not be combined. Descriptions
were screened across the histories; representative core diffs were inspected for dispatch,
explicit-instantiation ownership, comparator indirection, AST metadata, and co-location.
This is not a fresh correctness audit of every listed patch or every review discussion.

Navigate by technique below or search a PR number from SKILL.md. Read the linked patch and
current source before using a precedent; several original files/APIs have since moved or
been removed. Each PR appears once under its main technique; some combine several mechanisms.

## Tracking reports and proposal status

- [#21973 — Compilation optimization report](https://github.com/NVIDIA/cudf/issues/21973)
  motivates several focused follow-up PRs. It explicitly calls for isolated
  revalidation. Its prototype used an 11-kind shared-memory dispatcher; merged #23330
  describes nine supported kinds. Derive the set from current callers, not from either count.
- [#21974 — Improve compilation](https://github.com/NVIDIA/cudf/pull/21974) was **closed without
  merging** when checked. Its experiments and the report are idea sources, not a merged
  implementation. Reported failed directions include constrained global-memory dispatch,
  materialized reduction/tdigest inputs, and per-element type-erased reduction dispatch.
- [#23419 — Deduplicate equivalent CUDA kernels](https://github.com/NVIDIA/cudf/issues/23419)
  connects compile cost with fatbin size and Spark native-library extraction. Its initial
  single-SM prototypes and full-matrix projections are distinct from the subsequently
  merged #23706 and #23733 measurements. The discussion links related cuVS research.
- [#23385 — Nullable global-memory aggregation build time](https://github.com/NVIDIA/cudf/pull/23385)
  was **closed without merging** when checked. Do not list it among
  merged optimization results or infer that its experiment should be resumed.
- [#17835 — Explicit MurmurHash instantiations](https://github.com/NVIDIA/cudf/pull/17835)
  and [#19299 — Test build time](https://github.com/NVIDIA/cudf/pull/19299) were also
  closed without merging; local branches/commit subjects do not establish a merged result.
- [#22496 — LTO IR infrastructure](https://github.com/NVIDIA/cudf/issues/22496) is an **issue**,
  not a PR; see #21625, #22654, and #22680 for related merged implementation work.

## Restrict, factor, or normalize dispatch

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#23282 — Remove redundant type dispatch from column contains](https://github.com/NVIDIA/cudf/pull/23282) | Remove a dispatch whose C++ type is unused; retain the dictionary-specific normalization path. Reported CUDA compile: 67.9s → 23.9s. |
| [#23330 — Reduce compute_shared_memory_aggs build time](https://github.com/NVIDIA/cudf/pull/23330) | Constrain shared-memory aggregation dispatch to nine supported kinds and exclude dictionary/nested value instantiations already rejected by callers. Compile 640s → 127s; objects 1.1 MB → 626 KB. |
| [#16884 — Improve aggregation device functors](https://github.com/NVIDIA/cudf/pull/16884) | Remove aggregate_row template parameters that every caller supplied identically; separate device aggregators into a focused header. |
| [#17726 — Refactor distinct join to use primitive row operators when proper](https://github.com/NVIDIA/cudf/pull/17726) | Primitive row operators offer a bounded fast path instead of the proposed broader three-way specialization; explicitly motivated by avoiding build-time growth. |
| [#18896 — Apply primitive row operators into hash join](https://github.com/NVIDIA/cudf/pull/18896) | Apply primitive operators to ordinary hash join; runtime-oriented companion. Do not infer a build or size win from its runtime results. |
| [#22010 — Remove redundant aggregation identity logic in shared memory groupby](https://github.com/NVIDIA/cudf/pull/22010) | Remove redundant shared-memory aggregation identity logic; consult with #23330 when tracing the current supported aggregation path. |
| [#6457 — [REVIEW] Replace index type-dispatch call with indexalator in cudf::gather](https://github.com/NVIDIA/cudf/pull/6457) | Normalize gather indices with indexalator instead of dispatching the whole gather algorithm for each index type. Compile 200s → 40s; object 10.8 MB → 4.2 MB. |
| [#6461 — [REVIEW] Replace index type-dispatch call with indexalator in cudf::scatter](https://github.com/NVIDIA/cudf/pull/6461) | Apply indexalator to scatter; compile 110s → 45s and object 8.0 MB → 5.5 MB. |
| [#6471 — [REVIEW] Replace index type-dispatch call with indexalator in cudf::strings::substring](https://github.com/NVIDIA/cudf/pull/6471) | Normalize substring position indices; compile 70s → 46s. |
| [#6727 — [REVIEW] Remove 2nd type-dispatcher call from cudf::reduce for simple operations](https://github.com/NVIDIA/cudf/pull/6727) | Remove the second reduction type dispatch by choosing an intermediate/output type and casting only where necessary. Preserve requested output semantics; reported library 88 MB → 50 MB. |
| [#7242 — Refactor dictionary support for reductions any/all](https://github.com/NVIDIA/cudf/pull/7242) | Refactor dictionary any/all reduction instantiations after type-support expansion made compilation expensive. Follow current dictionary semantics rather than copying old restrictions. |
| [#14206 — Enable indexalator for device code](https://github.com/NVIDIA/cudf/pull/14206) | Enable indexalator in device code; supporting infrastructure, not a measured build optimization on its own. |
| [#20927 — Implement more flexible runtime to compile-time dispatching](https://github.com/NVIDIA/cudf/pull/20927) | Flexible runtime-to-template dispatch utility. Cleaner dispatch spelling does not inherently shrink the instantiated combination set. |
| [#10756 — Refactor binaryop/compiled/util.cpp](https://github.com/NVIDIA/cudf/pull/10756) | Factor four-way binary-op dispatch into staged dispatches and derive common types through pairwise operations. Compile 2m52.48s → 57.91s; peak RSS 14.6 GB → 2.4 GB. |
| [#11489 — Move SparkMurmurHash3_32 functor.](https://github.com/NVIDIA/cudf/pull/11489) | Move SparkMurmurHash3_32 to its only consumer and disallow unused nested specializations; nested rows are handled by another layer. |
| [#17753 — Avoid instantiating bloom filter query function for nested and bool types](https://github.com/NVIDIA/cudf/pull/17753) | Exclude unsupported nested/bool bloom-filter instantiations; fixes a compiler bug and may improve compilation. |
| [#7419 — Simplify type dispatch with `device_storage_dispatch`](https://github.com/NVIDIA/cudf/pull/7419) | Device_storage_dispatch simplifies dispatch. Historical whole-build timings were nearly unchanged; refactoring alone is not a demonstrated major win. |
| [#13805 — Reduce `lists::contains` dispatches for scalars](https://github.com/NVIDIA/cudf/pull/13805) | Reduce lists::contains scalar dispatches; operation-specific dispatch simplification. |

## Consolidate runtime policies and nullable iterators

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#6835 — [REVIEW] Move template param to member var to improve compile of hash/groupby.cu](https://github.com/NVIDIA/cudf/pull/6835) | Make skip_rows_with_nulls a member flag instead of duplicating the large aggregate_row graph. Reported compile 16 min → 9 min. |
| [#7516 — Reduce compile time/size for scan.cu](https://github.com/NVIDIA/cudf/pull/7516) | Allow null_replace_accessor to handle non-nullable input; collapse scan calls. Reported compile time and object size roughly halved. |
| [#8914 — Move template parameter to function parameter in cudf::detail::left_semi_anti_join](https://github.com/NVIDIA/cudf/pull/8914) | Move join_kind to a runtime argument, move a helper out of line, and use precompiled gather. Different mechanisms contribute to the result. |
| [#9324 — Use optional-iterator for copy-if-else kernel](https://github.com/NVIDIA/cudf/pull/9324) | Optional iterators collapse four nullable input combinations for copy-if-else into one kernel family; preserve coverage of the no-null path. |
| [#9623 — Allow runtime has_nulls parameter for row operators](https://github.com/NVIDIA/cudf/pull/9623) | Introduce runtime has_nulls support for row operators. Use where measured runtime cost is acceptable. |
| [#11482 — Refactor group_nunique.cu to use nullate::DYNAMIC for reduce-by-key functor](https://github.com/NVIDIA/cudf/pull/11482) | Use nullate::DYNAMIC for group_nunique equality/iterator, reducing reduce_by_key variants; reported nearly 2x compile improvement. |
| [#11622 — Rework contains_scalar to check nulls at runtime](https://github.com/NVIDIA/cudf/pull/11622) | Move contains_scalar null handling to runtime to reduce generated code. |
| [#21312 — Move has_nulls template parameter to runtime in rolling window](https://github.com/NVIDIA/cudf/pull/21312) | Move rolling has_nulls to runtime. Reported about 20% improvement in compile time and object size with unchanged measured rolling runtime. |

## Own instantiations, reuse compiled overloads, and deduplicate kernels

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#23323 — Remove redundant template instantiations from table contains](https://github.com/NVIDIA/cudf/pull/23323) | Delete redundant explicit-instantiation TUs because the caller already instantiates the required implementations. Removed 1,455s and 665s compiles; affected objects 7.98 MB → 3.96 MB. |
| [#23331 — Use explicit filter join kernel instantiations](https://github.com/NVIDIA/cudf/pull/23331) | The complementary ownership choice: include the declaration-only host-wrapper header and reuse existing explicit definitions. TU 73.2s → 18.3s; linked libcudf down 1.44 MB. |
| [#23420 — Remove redundant gather map view in sort groupby helper](https://github.com/NVIDIA/cudf/pull/23420) | Pass the resident gather-map device vector directly to the precompiled span overload; avoid an unnecessary temporary column_view. |
| [#9299 — Use gather.hpp when gather-map exists in device memory](https://github.com/NVIDIA/cudf/pull/9299) | Use declaration-only gather.hpp and the precompiled gather-map overload when data already resides on device. Modern span follow-up: #23420. |
| [#23706 — Reduce libcudf binary size by trimming instantiations](https://github.com/NVIDIA/cudf/pull/23706) | Centralize strings-offset helpers and co-locate scalar/segmented variance and std wrappers. Reported linked reduction 36.78 MiB (3.48%); affected uncached compiles 2935.291s → 2186.562s. Size A/B preceded the final rebase. |
| [#23733 — Reduce segmented sort template instantiations](https://github.com/NVIDIA/cudf/pull/23733) | Co-locate stable/non-stable segmented sort to eliminate equivalent kernel emission. Reported 18.39 MiB reduction; affected uncached compiles 374.784s → 191.793s. Baseline already contained preceding size work. |
| [#16603 — Remove CUDA whole compilation ODR violations](https://github.com/NVIDIA/cudf/pull/16603) | Repair CUDA whole-compilation ODR violations with a host wrapper, keeping launch and compiled kernel definition together. |
| [#14726 — Ensure that all CUDA kernels in cudf have hidden visibility.](https://github.com/NVIDIA/cudf/pull/14726) | Kernel visibility/internal-linkage requirements for static CUDA runtime correctness; retain visibility while consolidating ownership. |
| [#18131 — Optimized compilation of CUDFTESTUTIL's interface sources](https://github.com/NVIDIA/cudf/pull/18131) | Compile cudftestutil interface sources once in a private object library instead of once per consumer. |
| [#19518 — Add primitive row dispatch support for semi/anti join and cudf::contains](https://github.com/NVIDIA/cudf/pull/19518) | Corrected successor to #19361 after revert #19503: NaN handling and exact cooperative-group template sizes matter when moving explicit instantiations. |

## Split translation units for build parallelism

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#23322 — Reduce distinct_helpers build time](https://github.com/NVIDIA/cudf/pull/23322) | Split by row shape, NaN comparison, and keep policy. Longest TU 662s → 103s, but summed compiles 662s → 665s. Materialize nested KEEP_ANY hashes; ordered nested paths retain inline hashing. |
| [#21804 — Split hash join definitions to reduce build time](https://github.com/NVIDIA/cudf/pull/21804) | Split hash-join operations and instantiate only defined member functions; keep row-operator headers out of common declarations. CUDA 13.1/compute_120: longest TU 1,993s → 262s. Recheck full-build wall time separately. |
| [#17089 — Split hash-based groupby into multiple smaller files to reduce build time](https://github.com/NVIDIA/cudf/pull/17089) | Split hash groupby into smaller TUs with explicit instantiations. Early precedent for ownership and source-list updates; no quantified savings in the description. |
| [#17053 — Move `flatten_single_pass_aggs` to its own TU](https://github.com/NVIDIA/cudf/pull/17053) | Move flatten_single_pass_aggs to its own TU without algorithm changes. |
| [#5207 — [REVIEW] Break up backref_re.cu into multiple source files to improve compile time](https://github.com/NVIDIA/cudf/pull/5207) | Split back-reference regex specializations into separate TUs; parallel compile example. |
| [#6245 — [REVIEW] Split up replace.cu into multiple source files](https://github.com/NVIDIA/cudf/pull/6245) | Separate independent replace/null/nan implementations; code-motion precedent. |
| [#6822 — [REVIEW] Split out cudf::distinct_count from drop_duplicates.cu](https://github.com/NVIDIA/cudf/pull/6822) | Separate distinct_count from drop_duplicates to reduce TU concentration. |
| [#8183 — Split up scan.cu to improve compile time](https://github.com/NVIDIA/cudf/pull/8183) | Split scan compilation across sources. |
| [#8168 — Split up hashing.cu to improve compile time](https://github.com/NVIDIA/cudf/pull/8168) | Split hashing compilation across sources. |
| [#9351 — Move rank scan implementations from scan_inclusive.cu to rank_scan.cu](https://github.com/NVIDIA/cudf/pull/9351) | Move rank scan implementations out of scan_inclusive. |
| [#10671 — Split up mixed-join kernels source files](https://github.com/NVIDIA/cudf/pull/10671) | Split mixed-join kernel sources; read alongside the later whole-compilation repair #16603. |
| [#10831 — Split up search.cu to improve compile time](https://github.com/NVIDIA/cudf/pull/10831) | Split search.cu to reduce long individual compilation. |
| [#13169 — Split up unique_count.cu to improve build time](https://github.com/NVIDIA/cudf/pull/13169) | Split unique_count.cu for build parallelism. |
| [#13382 — Split up experimental_row_operator_tests.cu to improve its compile time](https://github.com/NVIDIA/cudf/pull/13382) | Split experimental row-operator tests to improve compilation. |
| [#14358 — Split up scan_inclusive.cu to improve its compile time](https://github.com/NVIDIA/cudf/pull/14358) | Further split scan_inclusive.cu; current hotspots can outgrow earlier splits. |
| [#14826 — Fix debug build by splitting row_operator_tests_utilities.cu](https://github.com/NVIDIA/cudf/pull/14826) | Split row-operator test utilities to repair Debug build. |
| [#15054 — Split out strings/replace.cu and rework its gtests](https://github.com/NVIDIA/cudf/pull/15054) | Split strings replacement implementation and rework tests; related source-organization example. |
| [#18948 — Rework cudf::sorted_order implementation for faster compile](https://github.com/NVIDIA/cudf/pull/18948) | Separate fixed-width fast sorting and use CUB iterators where supported; retain required radix-sort temporary buffers. |
| [#9788 — Improve build time of libcudf iterator tests](https://github.com/NVIDIA/cudf/pull/9788) | Reduce expensive iterator-test compilation; test builds deserve their own hotspot accounting. |
| [#8167 — Split iterator tests to improve parallel compile times](https://github.com/NVIDIA/cudf/pull/8167) | Split iterator tests for parallel compilation. |
| [#14663 — Split parquet test into multiple files](https://github.com/NVIDIA/cudf/pull/14663) | Split Parquet tests; test-TU parallelism precedent. |

## Materialize work, precompute metadata, and control inlining

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#23285 — Reduce sort-based groupby helper build time](https://github.com/NVIDIA/cudf/pull/23285) | Separate nested comparators, remove type-specific gather instantiations, and use two-stage group offsets. Longest TU 491s → 65s; objects 7.69 MB → 1.95 MB. |
| [#23320 — Reduce filtered join build time](https://github.com/NVIDIA/cudf/pull/23320) | Separate primitive/flat/nested filtered-join paths; precompute hashes only for nested paths. Longest TU 432s → 91s; affected objects down 15%; account for one temporary 32-bit hash per row. |
| [#23343 — Reduce range_rolling build time](https://github.com/NVIDIA/cudf/pull/23343) | Split range windows by kind and use a lightweight bounds-materialization kernel. Longest TU 972s → 82s; summed object size 16.5 MB → 5.0 MB; reported rolling benchmarks within 1%. |
| [#23448 — Reduce streaming groupby insert_first build time](https://github.com/NVIDIA/cudf/pull/23448) | Device-resident comparator plus indirect noinline wrapper prevents row-operator expansion into CUB. CUDA 13.1/all RAPIDS architectures: longest TU 2,899s → 158s; combined objects 7.51 MB → 1.55 MB. Preserve stream/lifetime and CUB policy. |
| [#23532 — Optimize sort-merge join data passes](https://github.com/NVIDIA/cudf/pull/23532) | Keep expensive row comparison out of CUB selection by materializing byte flags; reported register usage 120–255 → 48. Runtime-oriented follow-up with no isolated compile-time claim in the description. |
| [#23012 — Rewrite mixed inner/left/full join with post-filtering](https://github.com/NVIDIA/cudf/pull/23012) | Mixed joins implemented through post-filtering; architectural context for later filtered-kernel work, not a generic compile-only rewrite. |
| [#21793 — Improve build time using transform instead of tabulate](https://github.com/NVIDIA/cudf/pull/21793) | Replace selected thrust::tabulate calls with transform over counting_iterator. Explicitly not a blanket recommendation; measured with CUDA 12.9 and 13.1. |
| [#12900 — Rework some code logic to reduce iterator and comparator inlining to improve compile time](https://github.com/NVIDIA/cudf/pull/12900) | Break up expensive nested iterator/comparator logic into kernels. Broad noinline was rejected because some compiles worsened and some large-row cases were 20% slower. |
| [#21197 — Add noinline declaration to secondary type-dispatching row-operators in Debug build](https://github.com/NVIDIA/cudf/pull/21197) | Debug-only noinline for secondary row-operator dispatch to avoid extreme compilation times. |
| [#22675 — Workaround nvcc compiler hangs in libcudf debug build](https://github.com/NVIDIA/cudf/pull/22675) | Targeted Debug noinline workaround for compiler hangs caused by large inlined graphs. |
| [#22699 — Add noinline to n_table_comparator::operator()](https://github.com/NVIDIA/cudf/pull/22699) | Make one n_table_comparator noinline unconditional after Release benchmarks showed no issue; do not generalize to all functors. |
| [#5426 — [REVIEW] Refactor strings code to minimize calls to regex](https://github.com/NVIDIA/cudf/pull/5426) | Reduce expansion of regex logic through strings construction; early precedent for separating expensive per-row work from scan machinery. |
| [#9530 — Force inlining to improve AST performance](https://github.com/NVIDIA/cudf/pull/9530) | Counterexample: AST force-inlining improved runtime about 2x with negligible compilation increase; inlining tradeoffs depend on architecture and context. |
| [#17234 — Precompute AST arity](https://github.com/NVIDIA/cudf/pull/17234) | Precompute AST operator arity on the host instead of dispatching on device. Patch also aligns uploaded metadata. Review discussion corrected cache-hit timings; its later all-architecture table showed several CUDA compiles getting slower despite slightly smaller objects. Not a demonstrated general compile speedup. |
| [#9816 — Move the binary_ops common dispatcher logic to be executed on the CPU](https://github.com/NVIDIA/cudf/pull/9816) | Move binary-op common dispatch to CPU; related to separating device work from host-known metadata. |
| [#6512 — Refactor rolling.cu to reduce compile time](https://github.com/NVIDIA/cudf/pull/6512) | Normalize timestamps, materialize bounds, and split rolling sources. Preserve chrono representation and measure extra traffic. |
| [#19670 — Cache hash values to improve hash-based groupby performance with wide/complex table keys](https://github.com/NVIDIA/cudf/pull/19670) | Cache hashes for wide/complex groupby keys; supporting precedent for selective hash materialization. |

## Narrow headers and dependencies; compile host-only code separately

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#20166 — Split row operator header](https://github.com/NVIDIA/cudf/pull/20166) | Split the large row-operator header into equality, hashing, lexicographic, and preprocessed-table headers; update consumers to include only what they use. |
| [#22418 — Remove anonymous namespaces from cudf headers](https://github.com/NVIDIA/cudf/pull/22418) | Remove anonymous namespaces from headers to avoid per-TU internal identities/copies and ODR hazards; no measured size claim in the description. |
| [#18925 — Add #pragma once to prevent redundant includes and speed up compilation](https://github.com/NVIDIA/cudf/pull/18925) | Add missing pragma-once guards to avoid repeated inclusion. |
| [#18170 — Refactor join internals: separate hash_join declaration and cleanup](https://github.com/NVIDIA/cudf/pull/18170) | Separate hash_join declaration and reorganize implementation; structural predecessor to later ownership/split work. |
| [#23245 — Document the include-what-you-use convention in developer and review guidelines](https://github.com/NVIDIA/cudf/pull/23245) | Document direct includes and removal of unused includes in the developer/review guides; supports header-cost work rather than a measured optimization itself. |
| [#20360 — Get rid of the hashing helper header](https://github.com/NVIDIA/cudf/pull/20360) | Remove the hashing helper header; related dependency cleanup. |
| [#15007 — Clean up detail sequence header inclusion](https://github.com/NVIDIA/cudf/pull/15007) | Clean up detail sequence includes; related header dependency work. |
| [#19682 — Split up rolling.cuh into separate headers](https://github.com/NVIDIA/cudf/pull/19682) | Separate rolling UDF/JIT and device-operator headers; reduce unrelated inclusion. |
| [#21387 — Split up algorithm.cuh into reduce.cuh and copy_if.cuh](https://github.com/NVIDIA/cudf/pull/21387) | Split algorithm.cuh into focused reduce/copy_if/accumulate headers. |
| [#20491 — Rework internal json headers to allow converting gtests files from .cu to .cpp](https://github.com/NVIDIA/cudf/pull/20491) | Separate host JSON declarations from CUDA definitions so tests can compile as .cpp. |
| [#8238 — Split out non-device code from fixed_point_tests.cu](https://github.com/NVIDIA/cudf/pull/8238) | Move non-device fixed-point tests out of .cu compilation. |
| [#8112 — Move scalar function definitions from scalar.hpp to scalar.cpp](https://github.com/NVIDIA/cudf/pull/8112) | Move scalar definitions from header into scalar.cpp; reduce header-instantiated work. |
| [#7159 — Refactor cudf::string_view host and device code](https://github.com/NVIDIA/cudf/pull/7159) | Separate host/device-compatible string_view code from device-only implementation. |
| [#8930 — Move AST evaluator into a separate header](https://github.com/NVIDIA/cudf/pull/8930) | Separate AST evaluation from compute_column machinery to narrow recompilation dependencies. |
| [#8815 — Refactor conditional joins](https://github.com/NVIDIA/cudf/pull/8815) | Refactor conditional joins into focused shared files, remove excess headers, and improve parallel/incremental compilation. |
| [#17078 — Add IWYU to CI](https://github.com/NVIDIA/cudf/pull/17078) | Add IWYU CI analysis for C++ includes. Original implementation covered .cpp, not .cu; verify current coverage before relying on it. |
| [#17170 — Remove includes suggested by include-what-you-use](https://github.com/NVIDIA/cudf/pull/17170) | Apply IWYU include-removal suggestions. |
| [#23708 — Remove Arrow C++ dependency from C++ tests](https://github.com/NVIDIA/cudf/pull/23708) | Remove Arrow C++ from core C++ tests while retaining direct C interface coverage. Reported removal of 223 Arrow compile edges; dependency-build savings, not libcudf kernel deduplication. |
| [#8902 — Move `structs_column_tests.cu` to `.cpp`.](https://github.com/NVIDIA/cudf/pull/8902) | Compile host-only struct-column tests as .cpp. |

## Remove obsolete helpers and deprecated APIs

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#17426 — Remove the unused detail `int_fastdiv.h` header](https://github.com/NVIDIA/cudf/pull/17426) | Remove unused internal int_fastdiv.h; distinguish internal dead code from public compatibility commitments. |
| [#17396 — Remove unused type aliases](https://github.com/NVIDIA/cudf/pull/17396) | Remove unused type aliases; cleanup precedent, without a measured binary-size result. |
| [#17056 — Remove unused hash helper functions](https://github.com/NVIDIA/cudf/pull/17056) | Remove unused hash helpers; search all consumers before removing declarations/definitions. |
| [#18218 — Remove unused round_up_pow2 utility](https://github.com/NVIDIA/cudf/pull/18218) | Remove unused round_up_pow2 utility; dead-code cleanup, not evidence of removed emitted kernels. |
| [#23475 — Remove deprecated `sum_with_overflow` APIs](https://github.com/NVIDIA/cudf/pull/23475) | Remove deprecated sum_with_overflow APIs; removal follows the API lifecycle, not merely absence of internal callers. |
| [#23476 — Remove deprecated key_remapping and filter_join_indices APIs](https://github.com/NVIDIA/cudf/pull/23476) | Remove deprecated key_remapping/filter_join_indices APIs. Historical #23331 paths may no longer exist on the branch being optimized. |
| [#23520 — Remove obsolete distinct filtered join wrapper](https://github.com/NVIDIA/cudf/pull/23520) | Remove obsolete distinct filtered-join wrapper; related dead implementation cleanup. |
| [#24245 — Remove unused JIT device span](https://github.com/NVIDIA/cudf/pull/24245) | Remove unused JIT device span; later cleanup context, not quantified compile/size evidence. |

## Measure builds and adjust build-system or compiler policy

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#9631 — Add utility to format ninja-log build times](https://github.com/NVIDIA/cudf/pull/9631) | Introduce sort_ninja_log.py to report object compilation durations and sizes. |
| [#9927 — Add build-time publish step to cpu build script](https://github.com/NVIDIA/cudf/pull/9927) | Publish build metrics in the CPU build workflow; observability precedent. |
| [#10038 — Add timing chart for libcudf build metrics report page](https://github.com/NVIDIA/cudf/pull/10038) | Add timing charts to build-metrics reports; inspect concurrency rather than adding overlapping durations. |
| [#10577 — Add patch for thrust-cub 1.16 to fix sort compile times](https://github.com/NVIDIA/cudf/pull/10577) | Patch Thrust/CUB sort compile-time behavior; dependency-version-specific precedent. |
| [#6982 — Disable some pragma unroll statements in thrust sort.h](https://github.com/NVIDIA/cudf/pull/6982) | Disable selected Thrust sort unrolling; historical compiler/dependency workaround, not a current flag prescription. |
| [#23999 — CI: Reduce cached Conda C++ build overhead](https://github.com/NVIDIA/cudf/pull/23999) | Reduce cached Conda CI overhead and parallelize examples; separate workflow savings from uncached source compilation. |
| [#23988 — CI: Stage wheel builds by package type](https://github.com/NVIDIA/cudf/pull/23988) | Stage related wheel packages in shared containers; packaging/workflow context. |
| [#23825 — Improve cudf-spark-jni build workflow](https://github.com/NVIDIA/cudf/pull/23825) | Improve Spark JNI build workflow and shared distributed-cache setup; infrastructure context only. Cache-based CI throughput is not uncached compiler evidence. |
| [#7583 — Reduce cudf library size](https://github.com/NVIDIA/cudf/pull/7583) | Fatbin compression and Release assert policy reduce library representation size; not evidence of template deduplication. |
| [#18755 — CUDA 12.9 use updated compression flags](https://github.com/NVIDIA/cudf/pull/18755) | Update CUDA 12.9 compression flags to maintain binary sizes; use current rapids-cmake configuration. |

## Reduce runtime JIT work and embedded LTO payloads

| PR | Mechanism, evidence, or limit |
| --- | --- |
| [#24041 — test: batch JIT expression checks](https://github.com/NVIDIA/cudf/pull/24041) | Batch compatible AST JIT test expressions to reduce NVRTC work while preserving coverage; runtime test compilation, not AOT build cost. |
| [#24197 — PERF: Avoid linking UDF shim for numeric apply](https://github.com/NVIDIA/cudf/pull/24197) | Skip unused UDF shim linkage for numeric apply; retain it where strings/GroupBy need symbols. Runtime JIT context. |
| [#21457 — [FEA] Enable Pre-compiled Headers for faster JIT](https://github.com/NVIDIA/cudf/pull/21457) | Precompiled headers for JIT; separate NVRTC/JIT latency from ahead-of-time libcudf compilation. |
| [#21625 — [FEA] LTO IR Support (1) - Introduce LibRTCX](https://github.com/NVIDIA/cudf/pull/21625) | LibRTCX infrastructure; architectural alternative rather than a drop-in AOT instantiation fix. |
| [#22654 — [FEA] LTO IR Support (3) -  Replace JITIFY usage with LIBRTCX](https://github.com/NVIDIA/cudf/pull/22654) | Replace Jitify with LibRTCX; broader runtime-compilation context. |
| [#22680 — [FEA] LTO IR Support (4) - Implement LTO Transform Kernels](https://github.com/NVIDIA/cudf/pull/22680) | LTO transform kernels; deferred specialization requires cold-link/cache measurements. |
| [#23803 — [FEA] Use a common base architecture as LTO IR target](https://github.com/NVIDIA/cudf/pull/23803) | Share a base LTO IR architecture while nvJitLink still targets the actual GPU. Reported embedded payload down 83.6%; this does not authorize dropping supported AOT SASS architectures. |

## Refresh the reference

Use the canonical repository name `NVIDIA/cudf`; older links using `rapidsai/cudf` redirect.
Read-only discovery examples:

```sh
gh pr list -R NVIDIA/cudf --state merged --search '"build time"' --limit 1000 \
  --json number,title,body,url,mergedAt
gh search prs --repo NVIDIA/cudf --merged '"binary size"' --limit 1000
gh pr view 23706 -R NVIDIA/cudf --json state,mergedAt,body,files,url
gh pr diff 23706 -R NVIDIA/cudf
```

Repeat keyword queries for related mechanisms; partition by merged date if results hit
GitHub's cap, then deduplicate by PR number. Search synonyms, linked issues, and the
affected symbols/files. Search local merged history as a complementary path; `git log --all`
also includes unmerged branches and must not establish merge status by itself. Check for
reverts, successor PRs, and current call sites. Distinguish direct optimization evidence,
related infrastructure, unmerged proposals, and speculative remaining opportunities.
