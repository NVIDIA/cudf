/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {

// Handle for an enqueued map-locality sample: the pinned staging plus its decode context. The
// staging frees stream-ordered, so it may outlive the resolve without a use-after-free. A zero
// count allocates nothing, so empty handles stay cheap.
struct sampled_map_locality {
  cuda::host_buffer<std::byte> staging;
  cudf::type_id map_type{cudf::type_id::EMPTY};
  size_t elem_size{};

  static sampled_map_locality empty(cuda::stream_ref stream)
  {
    return sampled_map_locality{
      cudf::detail::make_pinned_buffer_async<std::byte>(0, stream), cudf::type_id::EMPTY, 0};
  }
};

namespace {

// Default block size of the fused kernel; matches the valid_if_n_kernel/concatenate precedent.
// The explicit block_size overload may select 128/256/512: all are warp multiples, which keeps
// the masked path's dst-mask-word exactly-once invariant and the per-warp valid-count staging
// (BlockSize / warp_size rows) well-formed at every choice.
constexpr size_type kDefaultBlockSize{256};

// Default consecutive descriptors per blockIdx.y tile: each thread loads a row's gather-map
// value once and reuses it across the tile's columns (per-column map re-reads dominated wide
// cells). Measured variable — the driver accepts 1/2/4/8 via the tile_cols parameter.
constexpr size_type kDefaultTileCols{8};

// Default rows per thread of the fused kernel: 1 reproduces the un-coarsened geometry, so the
// explicit-path behavior of every prior round is the R=1 instantiation. The rows_per_thread
// overload accepts {1,2,4}; {2,4} are instantiated only for the latency-bound non-streaming
// narrow tiles (kTileCols 1/2), so 4 is the widest value with a bounded cross product — the
// wide tiles and the streaming tile keep R=1, where coarsening would square the register cost
// for no benefit.
constexpr size_type kDefaultRowsPerThread{1};

// Map-locality routing. Misrouting a random map to the wide tile costs up to ~51% while the
// reverse only forfeits the ~17% coalesced win, so the classifier is strict: coalesced is
// reported only when EVERY sampled window is near-contiguous; anything else takes the safe
// narrow tile.
constexpr size_type kCoalescedTileCols{8};
constexpr size_type kRandomTileCols{1};
constexpr size_type kUnsampledTileCols{2};
// Sampling floors (measured): the row gate keeps the sync off short/wide K-insensitive shapes,
// the work gate keeps it off narrow mid-height shapes; 1048576x8 crosses both and must sample.
constexpr size_type kMinSampledRows{262'144};
constexpr int64_t kMinSampledWork{4'000'000};
constexpr size_type kSampleWindows{16};
constexpr size_type kSampleWindowLen{8};

/**
 * @brief Type-erased per-column parameters consumed by `batch_gather_kernel`.
 *
 * One descriptor per NON-all-null fixed-width source column: a column whose every row is null
 * has its destination pre-set ALL_NULL host-side and carries no descriptor at all, so the
 * kernel never sees a column it would only skip. All pointers are element bases, not byte
 * bases: `src_data` comes from `column_view::data<T>()`, which has the slice offset
 * pre-applied in element units, and `dst_data`/`dst_mask` belong to a freshly allocated
 * destination whose offset is zero. Nothing downstream may re-add an offset to these pointers.
 */
struct alignas(16) column_descriptor {
  void const* src_data;  // first source element; nullptr never (all-null columns: no descriptor)
  void* dst_data;        // first destination element; fresh allocation, offset 0
  bitmask_type const* src_mask;  // source mask BASE pointer; nullptr when no bit can be clear
                                 // (non-nullable or zero null count)
  bitmask_type* dst_mask;        // destination mask; nullptr if dst non-nullable or fast-path
  int64_t mask_offset;           // bit offset into src_mask (== src.offset() for sliced sources)
  int32_t element_size;          // cudf::size_of(src.type()); one of {1, 2, 4, 8, 16}
};

/**
 * @brief Descriptors of a small table, handed to the kernel as one by-value parameter.
 *
 * `__grid_constant__` on the kernel parameter keeps this object in the kernel-parameter space,
 * so no thread gets a private copy and no device allocation or host-to-device copy is needed.
 * `count` mirrors the device path's `num_descriptors`; entries at or past `count` are unused.
 *
 * The size is what bounds the tier: `sizeof(column_descriptor)` is 48 bytes (16-byte aligned),
 * so a block costs `16 + 48 * kCap` bytes — `descs` starts at offset 16 because `count` is
 * padded up to the array's alignment.
 */
template <size_type kCap>
struct descriptor_block {
  size_type count;
  column_descriptor descs[kCap];
};

// The by-value tier's byte arithmetic rests on this layout: 4 pointers, one int64, and one
// int32 total 44 bytes, which the 16-byte alignment rounds to 48. `count` is padded up to
// that alignment, so the tier blocks are 16 + 48 * 16 = 784 and 16 + 48 * 512 = 24'592 bytes.
static_assert(sizeof(column_descriptor) == 48,
              "Descriptor size changed: revisit the by-value tier capacity arithmetic");
static_assert(sizeof(descriptor_block<batch_gather_small_by_value_descriptors>) == 784,
              "Small-tier block size changed: update the tier's byte arithmetic");
static_assert(sizeof(descriptor_block<batch_gather_wide_by_value_descriptors>) == 24'592,
              "Wide-tier block size changed: update the tier's byte arithmetic");

// The block shares the 32'764-byte kernel-parameter limit with the rest of the launch:
// input_indexalator 32 B, two size_type 8 B, two bool 2 B, and one pointer 8 B total 50 B, which
// the block's 16-byte alignment rounds up to 64 B. `kRowsPerThread` and the block size ride the
// kernel's template parameters, not the parameter block, so they add nothing here; both tier
// blocks plus that slack stay inside the budget (the wide tier leaves 8'108 B).
static_assert(sizeof(descriptor_block<batch_gather_small_by_value_descriptors>) + 64 <= 32'764,
              "By-value descriptor block exceeds the kernel-parameter size limit");
static_assert(sizeof(descriptor_block<batch_gather_wide_by_value_descriptors>) + 64 <= 32'764,
              "By-value descriptor block exceeds the kernel-parameter size limit");

// Evict-first cache policy (PTX ISA 74, SM_70+ via the generated cuda::ptx builtins), active
// only in the streaming regime: the sampled coalesced verdict marks the dst/src traffic
// don't-cache so L2 keeps the reuse-critical gather map. Random/unsampled verdicts, test-forced
// geometries, host compilation, and pre-sm_70 all take plain default-cached accesses.
template <bool kStreaming, typename T>
__device__ inline T streaming_load(char const* src)
{
  if constexpr (kStreaming) {
#if defined(__CUDA_ARCH__)
    if constexpr (__CUDA_ARCH__ >= 700) {
      return cuda::ptx::ld_L1_evict_first(cuda::ptx::space_global_t{},
                                          reinterpret_cast<T const*>(src));
    }
#endif
  }
  return *reinterpret_cast<T const*>(src);
}

template <bool kStreaming, typename T>
__device__ inline void streaming_store(char* dst, T value)
{
  if constexpr (kStreaming) {
#if defined(__CUDA_ARCH__)
    if constexpr (__CUDA_ARCH__ >= 700) {
      cuda::ptx::st_L1_evict_first(cuda::ptx::space_global_t{}, reinterpret_cast<T*>(dst), value);
      return;
    }
#endif
  }
  *reinterpret_cast<T*>(dst) = value;
}

// INVARIANT: element_size * row is computed in 64-bit below; it can exceed 2^31 for DECIMAL128.
// 16-byte copies stay aligned because device allocations are >=256B aligned and offsets are
// whole elements. The src loads are evict-first only when kStreaming (the map load stays
// default-cached in every regime: it is the reuse-critical stream and the wide-tile win depends
// on its L2 residency). The staging destination takes a default-cached plain store; the store
// half is store_element, so a coarsened thread can issue every source load before its first
// destination store. The 16-byte arm loads through uint4: in the streaming regime that is the
// CCCL-generated `ld.global.L1::evict_first.b128` form (PTX ISA 8.3, SM_70+, gated by
// `__cccl_ptx_isa >= 830`, above cudf's nvcc 12.9 floor) — one 128-bit access where the
// two-hinted-8-byte fallback would otherwise be needed; elsewhere the compiler emits the plain
// assignment as a single 128-bit vector access.
template <bool kStreaming>
__device__ void load_element(void* dst, char const* src, int32_t element_size)
{
  switch (element_size) {
    case 1:
      streaming_store<false, int8_t>(static_cast<char*>(dst),
                                     streaming_load<kStreaming, int8_t>(src));
      break;
    case 2:
      streaming_store<false, int16_t>(static_cast<char*>(dst),
                                      streaming_load<kStreaming, int16_t>(src));
      break;
    case 4:
      streaming_store<false, int32_t>(static_cast<char*>(dst),
                                      streaming_load<kStreaming, int32_t>(src));
      break;
    case 8:
      streaming_store<false, int64_t>(static_cast<char*>(dst),
                                      streaming_load<kStreaming, int64_t>(src));
      break;
    case 16:
      // One 128-bit load; the paired 128-bit store rides store_element's 16-byte arm.
      streaming_store<false, uint4>(static_cast<char*>(dst),
                                    streaming_load<kStreaming, uint4>(src));
      break;
    default: CUDF_UNREACHABLE("Unexpected element size");
  }
}

// The store half of load_element: a default-cached plain move out of the staging slot. Splitting
// load from store lets a coarsened thread issue every source load before its first store. The
// 16-byte arm is one 128-bit store; in the streaming regime the paired load half above already
// ran the evict-first b128 form, so the whole 16-byte move stays one hinted load plus one store
// (the store is unhinted by design — see the load_element invariant above).
template <bool kStreaming>
__device__ void store_element(char* dst, void const* src, int32_t element_size)
{
  switch (element_size) {
    case 1: streaming_store<kStreaming, int8_t>(dst, *static_cast<int8_t const*>(src)); break;
    case 2: streaming_store<kStreaming, int16_t>(dst, *static_cast<int16_t const*>(src)); break;
    case 4: streaming_store<kStreaming, int32_t>(dst, *static_cast<int32_t const*>(src)); break;
    case 8: streaming_store<kStreaming, int64_t>(dst, *static_cast<int64_t const*>(src)); break;
    case 16: streaming_store<kStreaming, uint4>(dst, *static_cast<uint4 const*>(src)); break;
    default: CUDF_UNREACHABLE("Unexpected element size");
  }
}

// Spreads the R low bits of x into bits {0, stride, 2*stride, ...}: bit q of the input lands at
// position q * stride. Used by assemble_mask_word to interleave R warp ballots into destination
// mask words (verified bit-exact against the per-bit definition for every input).
template <int32_t stride>
__device__ constexpr uint32_t spread_bits(uint32_t x)
{
  if constexpr (stride == 1) {
    return x;
  } else if constexpr (stride == 2) {
    // x &= 0xFFFF keeps the R=2 slice of the 32-bit ballots below bit 16.
    x &= 0xFFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
  } else {
    static_assert(stride == 4, "Unsupported ballot stride");
    // Stride 4 is stride 2 applied twice: first 8 -> 16 bits, then 16 -> 32 bits.
    x &= 0xFF;
    return spread_bits<2>(spread_bits<2>(x));
  }
}

// Builds the destination mask words for the sub-window j of a warp that processed R rows per
// lane. Word j covers destination rows [window_base + 32j, window_base + 32j + 32), whose bit m
// belongs to sub-lane s = m % R at lane bit (32/R)*j + m/R of ballot s. The Morton-style
// composition below realizes that mapping: each ballot contributes its slice of bits at
// positions ≡ s (mod R). Called only by the writing warp's lane 0.
template <size_type rows_per_thread>
__device__ constexpr bitmask_type assemble_mask_word(uint32_t const (&ballots)[rows_per_thread],
                                                     int32_t window)
{
  if constexpr (rows_per_thread == 1) {
    return static_cast<bitmask_type>(ballots[0]);
  } else if constexpr (rows_per_thread == 2) {
    return static_cast<bitmask_type>(spread_bits<2>(ballots[0] >> (16 * window)) |
                                     (spread_bits<2>(ballots[1] >> (16 * window)) << 1));
  } else {
    static_assert(rows_per_thread == 4, "Unsupported rows per thread");
    uint32_t word = 0;
    for (int32_t s = 0; s < rows_per_thread; ++s) {
      word |= spread_bits<4>(ballots[s] >> (8 * window)) << s;
    }
    return static_cast<bitmask_type>(word);
  }
}

// Returns the source column's element base pointer through its storage type (fixed-point
// dispatches to int32/int64/__int128), so the slice offset is applied exactly once, host-side.
// The dispatcher instantiates operator() for every type in the dispatch list, including the
// pass-through types (string_view/list_view/struct_view/dictionary32) that never carry data<T>;
// they are unreachable here because only fixed-width columns enter the descriptor path.
struct data_ptr_functor {
  template <typename T>
  void const* operator()(column_view const& col) const
  {
    if constexpr (cudf::is_rep_layout_compatible<T>()) {
      return static_cast<void const*>(col.data<T>());
    } else {
      CUDF_UNREACHABLE("Non-layout-compatible type on the descriptor path");
    }
  }
};

// Adapter giving the indexalator the same call shape as `direct_map_reader`: the kernels read
// the map uniformly as `reader(row)`, so only the launcher decides which reader a call gets.
// Device-only, not CUDF_HOST_DEVICE: the callee `input_indexalator::operator[]` is __device__
// only, and a __host__ __device__ wrapper around it fails the cross-execution-space check
// (direct_map_reader can carry the dual marker only because a raw-pointer load has none).
struct indexalator_reader {
  input_indexalator it;
  __device__ size_type operator()(size_type row) const { return it[row]; }
};

// Reader for maps whose stored bytes already are their logical indices: one typed load per
// row replaces the indexalator's per-access type dispatch. The widening is plain
// `static_cast<size_type>` — the identical conversion `input_indexalator::normalize_type`
// performs (INT8/16 sign-extend, UINT8/16 zero-fill), so at every representational extreme
// (INT32 min/max, UINT32 max, INT64 2^31/INT64_MAX) both readers produce the same value,
// including the wrap: a negative INT64 value truncates to a negative size_type and wraps by
// the same `idx += num_source_rows` addition as the indexalator path. `MapType` is
// instantiated only for int32_t (INT32 and UINT32 maps alias it: reinterpreting the same 4
// bytes is value-identical because both conversions truncate to the low 32 bits) and int64_t
// (likewise for INT64/UINT64). The pointer must come from `direct_map_data`, which applies the
// slice offset exactly like the indexalator's element base.
template <typename MapType>
struct direct_map_reader {
  using value_type = MapType;
  MapType const* base;
  CUDF_HOST_DEVICE size_type operator()(size_type row) const
  {
    return static_cast<size_type>(base[row]);
  }
};

/**
 * @brief Runs one descriptor tile's row loop, reading the tile's descriptors from `tile_descs`.
 *
 * Shared by all three launch variants so the row and mask machinery exists once: `tile_descs`
 * addresses the shared-memory stage on the device path and the kernel-parameter block on the
 * by-value path. The map source is `MapReader`: `input_indexalator` reads any map type through
 * per-access type dispatch, while `direct_map_reader<MapType>` reads the qualifying wide maps
 * (INT32/UINT32/INT64/UINT64) through one typed load per row. `__forceinline__` keeps the
 * by-value variant's descriptor reads hoistable out of the row loop: only pointers cross this
 * boundary, never a descriptor block by value, so inlining carries the `__grid_constant__`
 * no-copy guarantee through and the tile base stays provably loop-invariant.
 *
 * `tile_begin` is the absolute descriptor position of `tile_descs[0]`; positions at or past
 * `num_descriptors` are absent from the tile and skipped.
 *
 * INVARIANT: every destination row r in [0, num_rows) is written by exactly one thread, per
 * column.
 *
 * Row coarsening (kRowsPerThread > 1): each thread owns the consecutive rows
 * `chunk = (blockIdx.x * BlockSize + threadIdx.x) * kRowsPerThread + [0, kRowsPerThread)`; the
 * launch grid covers every chunk exactly once, so R divides the row stride
 * and a warp still covers a contiguous row block. Threads of a warp span kRowsPerThread * 32
 * consecutive rows: warp w of block b anchors at the chunk base
 * (blockIdx.x * BlockSize + w * (BlockSize / warp_size)) * kRowsPerThread, a multiple of
 * warp_size * kRowsPerThread. Each warp pass therefore covers exactly kRowsPerThread
 * whole 32-row destination windows, and the map reads stay warp-contiguous (coalesced).
 *
 * INVARIANT (kMasked == true): each column's dst_mask word w is written exactly once, by lane 0
 * of the warp covering rows [32w, 32w + 32): this requires BlockSize to be a multiple of
 * warp_size (the row stride is then a multiple of 32), which every supported block size
 * {128, 256, 512} satisfies. Trailing
 * bits of the last word are 0. With coarsening, the covering warp's pass covers rows
 * [warp_chunk_base, warp_chunk_base + 32 * kRowsPerThread) = the kRowsPerThread consecutive
 * windows [warp_chunk_base + 32j, ...): the warp takes kRowsPerThread ballots, ballot s bit l =
 * validity of row warp_chunk_base + l * kRowsPerThread + s, and assemble_mask_word interleaves
 * them into the window words. BlockSize being a multiple of warp_size keeps the warp-chunk base
 * 32R-aligned (a multiple of 32 * kRowsPerThread), so the kRowsPerThread ballot rounds tile the
 * word space exactly once. Trailing bits stay 0 because each row's map read, copy, and ballot
 * gate on `row < num_rows`, which also keeps a tail chunk straddling num_rows from reading the
 * map or storing past the destination column. Per-column bookkeeping keeps this true
 * independently of the tile a column belongs to: a column with its own kernel-written mask takes
 * its own ballots (a reused `inb_ballot` is bit-identical to the per-column ballot whenever
 * `src_mask == nullptr`, because then `valid == in_bounds`), and its count accumulates into its
 * own shared-memory slot.
 */
template <size_type BlockSize,
          size_type kTileCols,
          size_type kRowsPerThread,
          bool kStreaming,
          bool kMasked,
          typename MapReader>
__device__ __forceinline__ void gather_tile_rows(MapReader gather_map,
                                                 column_descriptor const* tile_descs,
                                                 size_type tile_begin,
                                                 size_type num_descriptors,
                                                 size_type num_rows,
                                                 size_type num_source_rows,
                                                 bool check_bounds,
                                                 bool negative_allowed,
                                                 size_type* valid_counts)
{
  // The masked instantiation tallies per-column valid bits and carries the row-loop predicate
  // in one active-lane ballot mask; the unmasked instantiation has neither.
  // Lane 0 stages each mask-bearing column's count DIRECTLY into the warp's own shared-memory
  // slot, at the word-store site inside the column loop where the per-column ballots are in
  // scope. The zeroing below is load-bearing, not cosmetic: a warp whose lane-0 chunk base
  // already sits at or past num_rows never enters the row loop and never stages, so without
  // it its slots hold garbage that the readback sums (observed as negative null counts).
  // This replaces the per-thread kTileCols-entry register array that used to be staged at
  // the end.
  [[maybe_unused]] uint32_t inb_ballot[kRowsPerThread]{};
  [[maybe_unused]] auto active = 0xFFFF'FFFFu;
  [[maybe_unused]] size_type* warp_valid_smem{nullptr};
  [[maybe_unused]] auto warp_id = static_cast<size_type>(-1);
  [[maybe_unused]] auto lane_id = static_cast<size_type>(-1);
  auto const thread_stride      = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto chunk = (static_cast<int64_t>(blockIdx.x) * BlockSize + static_cast<int64_t>(threadIdx.x)) *
               kRowsPerThread;
  auto const chunk_stride = thread_stride * kRowsPerThread;
  // The covering warp's window base for the current pass, also read by the tile-exit count
  // accumulation: declared outside the loop so both share it.
  auto window_base = chunk - static_cast<int64_t>(threadIdx.x % warp_size) * kRowsPerThread;
  if constexpr (kMasked) {
    // Per-warp staging of the per-column valid counts: warps_per_block * kTileCols *
    // sizeof(size_type) bytes, disjoint from tile_descs (distinct static __shared__ objects
    // never overlap, and no dynamic shared memory is used).
    __shared__ size_type warp_valid_smem_storage[BlockSize / warp_size][kTileCols];
    warp_valid_smem = &warp_valid_smem_storage[0][0];
    warp_id         = threadIdx.x / warp_size;
    lane_id         = threadIdx.x % warp_size;
    // Zero this warp's own slots BEFORE the row loop: a warp whose chunk never runs (or a
    // dst_mask-free column in a partial tile) never stages its slot, and the readback sums
    // every warp's slot for each mask-bearing column — an unstaged slot must read 0.
    // Lane 0 alone zeroes: the same lane performs the in-loop staging, so program order
    // orders the zero before the stage within this lane; an all-lanes zero loop would race
    // the staged value (no intra-warp same-address ordering post-Volta — racecheck hazard).
    // The staging site is lane-0-only, so the other lanes' zero-writes are unordered with it.
    if (lane_id == 0) {
      for (int32_t k = 0; k < kTileCols; ++k) {
        warp_valid_smem[warp_id * kTileCols + k] = 0;
      }
    }
    active = __ballot_sync(active, chunk < num_rows);
  }
  if (chunk < num_rows) {
    // One map load per row (wrap + table-level bounds check) serves every column in the tile.
    // Deliberately default-cached: the map is the reuse-critical stream and the wide-tile win
    // depends on its L2 residency (no eviction hint here). The rows are consecutive, so the map
    // reads stay fully coalesced for every kRowsPerThread. A row at or past num_rows is
    // unconditionally skipped: with R > 1 the last chunk can straddle num_rows, and reading
    // that row's map entry (or storing it) would run past the column's allocation.
    window_base = chunk - static_cast<int64_t>(threadIdx.x % warp_size) * kRowsPerThread;
    size_type idx[kRowsPerThread];
    bool row_live[kRowsPerThread];
    bool in_bounds[kRowsPerThread];
#pragma unroll
    for (int32_t s = 0; s < static_cast<int32_t>(kRowsPerThread); ++s) {
      auto const row = static_cast<size_type>(chunk) + s;
      row_live[s]    = row < num_rows;
      // A dead row of a straddling chunk must not read the map either: its entry lies past the
      // map column's allocation. The dummy value is unused — row_live gates every consumer.
      idx[s] = row_live[s] ? gather_map(static_cast<size_type>(row)) : size_type{0};
      if (negative_allowed && idx[s] < 0) { idx[s] += num_source_rows; }
      in_bounds[s] = row_live[s] && (!check_bounds || (idx[s] >= 0 && idx[s] < num_source_rows));
    }
    if constexpr (kMasked) {
      // One ballot per row slot for the whole tile: every lane participates unconditionally
      // (the mask `active` carries the warp's live rows), so the collective-participation
      // contract is unchanged; when a column's `src_mask == nullptr` its per-row validity is
      // exactly `in_bounds[s]`, so the per-column ballots would all repeat this value.
#pragma unroll
      for (int32_t s = 0; s < static_cast<int32_t>(kRowsPerThread); ++s) {
        inb_ballot[s] = __ballot_sync(active, in_bounds[s]);
      }
    }
#pragma unroll
    for (int32_t k = 0; k < kTileCols; ++k) {
      if (tile_begin + k < num_descriptors) {
        auto const d = tile_descs[k];
        // Staging slot for the store phase: all source loads issue before any store so a
        // coarsened thread keeps up to kRowsPerThread loads outstanding (the MLP goal).
        uint64_t payload[kRowsPerThread][sizeof(__int128) / sizeof(uint64_t)];
        uint32_t ballots[kRowsPerThread]{};
#pragma unroll
        for (int32_t s = 0; s < static_cast<int32_t>(kRowsPerThread); ++s) {
          auto const dst =
            static_cast<char*>(d.dst_data) + static_cast<std::size_t>(chunk + s) * d.element_size;
          bool valid = in_bounds[s];
          if constexpr (kMasked) {
            if (valid && d.src_mask != nullptr) {
              valid = bit_is_set(d.src_mask, static_cast<size_type>(d.mask_offset + idx[s]));
            }
          }
          // Data at out-of-bounds/null rows may be left unwritten (fixed-width undefined-value
          // policy); a dead row of a straddling chunk must neither load nor store.
          if (row_live[s] && valid) {
            load_element<kStreaming>(payload[s],
                                     static_cast<char const*>(d.src_data) +
                                       static_cast<std::size_t>(idx[s]) * d.element_size,
                                     d.element_size);
          }
          if constexpr (kMasked) {
            // With a null-free source the validity equals the bounds predicate, so this column's
            // ballot would repeat `inb_ballot[s]` bit-for-bit; it is skipped and the shared value
            // reused by the word assembly below. With a source mask the real ballot is taken
            // (same mask, so participation is unchanged). A block-uniform nullptr dst_mask skips
            // both.
            if (d.dst_mask != nullptr && d.src_mask != nullptr) {
              ballots[s] = __ballot_sync(active, row_live[s] && valid);
            }
          }
          if (row_live[s] && valid) { store_element<kStreaming>(dst, payload[s], d.element_size); }
        }
        if constexpr (kMasked) {
          if (d.dst_mask != nullptr) {
            if (lane_id == 0) {
              // The warp's count for THIS column, accumulated here where the correct ballot
              // array is in scope: a source-masked column counts its own `ballots` (validity
              // includes the src_mask check); a null-free-source column's ballot is
              // bit-identical to the shared `inb_ballot`, so its count comes from there.
              // Ballot s bit l = validity of row warp_chunk_base + l * kRowsPerThread + s,
              // so summing popcount over the R SLOTS counts exactly this warp's valid rows
              // in the windows the word stores below cover. The gate is slot existence
              // (window_base + s < num_rows), NOT the word-window index: slot s's rows span
              // all the windows (assemble_mask_word splits its bits across the words), so a
              // window-based gate would skip whole slots. A fully dead slot contributes 0
              // anyway (its rows are row_live/in_bounds false, so its ballot bits are 0),
              // which also makes the gate safe to keep or drop.
              uint32_t column_valid{0};
#pragma unroll
              for (int32_t s = 0; s < static_cast<int32_t>(kRowsPerThread); ++s) {
                if (window_base + s < num_rows) {
                  column_valid += cuda::std::popcount(
                    static_cast<uint32_t>(d.src_mask != nullptr ? ballots[s] : inb_ballot[s]));
                }
              }
              // Staged straight into this warp's own smem slot (each warp owns its k slots),
              // replacing the per-column block reduction the register-array form needed.
              warp_valid_smem[warp_id * kTileCols + k] = column_valid;
            }
#pragma unroll
            for (int32_t j = 0; j < static_cast<int32_t>(kRowsPerThread); ++j) {
              auto const wrow = window_base + 32 * j;
              if (wrow < num_rows) {
                auto const word = assemble_mask_word<kRowsPerThread>(
                  d.src_mask != nullptr ? ballots : inb_ballot, j);
                d.dst_mask[word_index(static_cast<size_type>(wrow))] = word;
              }
            }
          }
        }
      }
    }
    // The re-entry test is provably dead: grid.x = ceil_div(num_rows, BlockSize *
    // kRowsPerThread) never caps, so the stride is >= num_rows and a chunk already at or past
    // num_rows stays there — no second pass can run. A warp whose chunk starts out of range
    // still enters the loop (its ballot bit is clear), so the loop shape exists only to keep
    // the per-row code nested inside the shared `active` ballot mask.
    chunk += chunk_stride;
    if constexpr (kMasked) { active = __ballot_sync(active, chunk < num_rows); }
  }
  if constexpr (kMasked) {
    if (valid_counts != nullptr) {
      constexpr auto warps_per_block{BlockSize / warp_size};
      // ONE barrier makes every warp's in-loop staged count visible to all warps (the staging
      // happened inside the column loop above, where the correct ballot array is in scope;
      // the entry zeroing covered the slots the stage never reaches — dead warps' rows and
      // dst_mask-free columns). A null-free-source column's count comes from the shared
      // `inb_ballot`; a source-masked column counts its own `ballots`; both are bit-exact
      // tallies of the same valid rows the word stores wrote.
      __syncthreads();
#pragma unroll
      for (int32_t k = 0; k < kTileCols; ++k) {
        auto const col = tile_begin + k;
        if (col < num_descriptors && tile_descs[k].dst_mask != nullptr &&
            static_cast<size_type>(threadIdx.x) == k) {
          size_type block_valid{0};
          for (size_type w = 0; w < warps_per_block; ++w) {
            block_valid += warp_valid_smem[w * kTileCols + k];
          }
          if (block_valid > 0) {
            cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{valid_counts[col]};
            ref.fetch_add(block_valid, cuda::std::memory_order_relaxed);
          }
        }
      }
      // The readback above consumes this tile's staged slots; the NEXT tile iteration's
      // staging overwrites the same slots with no other barrier between, so a fast warp could
      // restage while a slow thread is still reading here (a racecheck-visible WAR). This
      // second barrier orders every thread's readback before any next-tile staging. Masked
      // tiles now pay two barriers — still far below the per-column reductions this replaced.
      __syncthreads();
    }
  }
}

/**
 * @brief Gathers every column described by `descriptors` in one launch (device path).
 *
 * The grid is 2D: x covers the row chunks exactly once (one pass; a capped x would break the
 * single-pass contract), each y block owns a tile of
 * `kTileCols` consecutive descriptors and y-strides over tiles past the 65535 gridDim.y cap. A
 * row's gather-map value is loaded once and reused across the tile's columns; the tile's
 * descriptors are staged in shared memory so the row loop pays no per-column global reads.
 * `kTileCols == 1` degenerates to one column per y block (the pre-tiling geometry) with the
 * same single-launch contract.
 *
 * `kMasked` selects the mask machinery: `true` reads `src_mask`, ballots validity into
 * `dst_mask` words, and accumulates `valid_counts`; `false` is launched only when no descriptor
 * carries either mask, so it drops the ballots, mask-word stores, source-mask reads, and
 * valid-bit tallies entirely. The row and mask invariants are stated on `gather_tile_rows`.
 * `MapReader` selects the map source: `input_indexalator` or `direct_map_reader<MapType>`.
 *
 * The masked path reduces each tile's per-column valid counts with TWO block barriers: lane 0
 * of every warp stages its counts directly in the warp's shared-memory slots at the word
 * store (the slots are zeroed at the tile's entry — a warp whose chunk base sits past
 * num_rows never stages, so an unzeroed slot would leak garbage into the readback), one
 * barrier publishes them, and thread `k` sums column `k` across warps and updates
 * `valid_counts`; a second barrier after the readback orders it before the next tile's
 * staging (the slots are reused across tiles, so the readback-to-restage edge is a WAR the
 * first barrier does not cover) — two barriers per column would serialize eight separate
 * block reductions. The end-of-tile barrier runs only when a later y iteration will restage
 * `tile_descs`; its predicate is exactly the y-stride loop-continuation condition, which is
 * block-uniform, so the block never splits across barrier instances.
 *
 * Row coarsening (kRowsPerThread > 1): each thread owns the consecutive rows
 * `chunk = (blockIdx.x * BlockSize + threadIdx.x) * kRowsPerThread + [0, kRowsPerThread)`; the
 * launch grid covers every chunk exactly once, so R divides the row stride
 * and a warp still covers a contiguous row block. Threads of a warp span kRowsPerThread * 32
 * consecutive rows: warp w of block b anchors at the chunk base
 * (blockIdx.x * BlockSize + w * (BlockSize / warp_size)) * kRowsPerThread, a multiple of
 * warp_size * kRowsPerThread. Each warp pass therefore covers exactly kRowsPerThread
 * whole 32-row destination windows, and the map reads stay warp-contiguous (coalesced).
 *
 * INVARIANT (kMasked == true): each column's dst_mask word w is written exactly once, by lane 0
 * of the warp covering rows [32w, 32w + 32): this requires BlockSize to be a multiple of
 * warp_size (the row stride is then a multiple of 32), which every supported block size
 * {128, 256, 512} satisfies. Trailing bits of the last word are 0. With coarsening, the covering
 * warp's pass covers rows [warp_chunk_base, warp_chunk_base + 32 *
 * kRowsPerThread) = the kRowsPerThread consecutive windows [warp_chunk_base + 32j, ...): the
 * warp takes kRowsPerThread ballots, ballot s bit l = validity of row warp_chunk_base +
 * l * kRowsPerThread + s, and assemble_mask_word interleaves them into the window words.
 * BlockSize being a multiple of warp_size keeps the warp-chunk base 32R-aligned (a multiple of
 * 32 * kRowsPerThread), so the kRowsPerThread ballot rounds tile the word space exactly once.
 * Trailing bits stay 0 because each row's map read, copy, and ballot gate on `row < num_rows`,
 * which also keeps a tail chunk straddling num_rows from reading the map or storing past the
 * destination column. Per-column bookkeeping keeps this true independently of the tile a
 * column belongs to: a column with its own kernel-written mask takes its own ballots (a reused
 * `inb_ballot` is bit-identical to the per-column ballot whenever `src_mask == nullptr`, because
 * then `valid == in_bounds`), and its count accumulates into its own shared-memory slot.
 */
template <size_type BlockSize,
          size_type kTileCols,
          size_type kRowsPerThread,
          bool kStreaming,
          bool kMasked,
          typename MapReader>
CUDF_KERNEL __launch_bounds__(BlockSize) void batch_gather_kernel(
  MapReader gather_map,
  column_descriptor const* descriptors,
  size_type num_descriptors,
  size_type num_rows,
  size_type num_source_rows,
  bool check_bounds,
  bool negative_allowed,
  size_type* valid_counts)
{
  __shared__ column_descriptor tile_descs[kTileCols];
  auto const num_tiles = util::div_rounding_up_safe(num_descriptors, kTileCols);
  for (size_type tile = static_cast<size_type>(blockIdx.y); tile < num_tiles;
       tile += static_cast<size_type>(gridDim.y)) {
    auto const tile_begin = tile * kTileCols;
    // Stage the tile's descriptors in shared memory; the range is block-uniform and the reads
    // below are warp broadcasts.
    if (threadIdx.x < kTileCols &&
        tile_begin + static_cast<size_type>(threadIdx.x) < num_descriptors) {
      tile_descs[threadIdx.x] = descriptors[tile_begin + static_cast<size_type>(threadIdx.x)];
    }
    __syncthreads();
    gather_tile_rows<BlockSize, kTileCols, kRowsPerThread, kStreaming, kMasked>(gather_map,
                                                                                tile_descs,
                                                                                tile_begin,
                                                                                num_descriptors,
                                                                                num_rows,
                                                                                num_source_rows,
                                                                                check_bounds,
                                                                                negative_allowed,
                                                                                valid_counts);
    // All threads are finished reading tile_descs; the next tile iteration rewrites it. The
    // predicate is exactly the y-stride loop-continuation condition (tile + gridDim.y <
    // num_tiles, rewritten to avoid the addition overflowing size_type), and tile, gridDim.y,
    // and num_tiles are block-uniform, so the whole block either takes this barrier or exits
    // the loop together — no thread can wait at an instance another thread skipped.
    if (static_cast<size_type>(gridDim.y) < num_tiles - tile) { __syncthreads(); }
  }
}

/**
 * @brief Gathers every column of a by-value `block` in one launch (parameter-space path).
 *
 * Same grid geometry, row loop, and mask machinery as `batch_gather_kernel`; the difference is
 * the descriptor source. The tile is read straight out of the kernel-parameter block, so there
 * is no shared-memory stage and no barrier around the row loop, and the launch needs neither a
 * descriptor device allocation nor a host-to-device copy.
 *
 * `__grid_constant__` is required, not cosmetic: without it the compiler gives every thread a
 * private copy of the whole block in local memory.
 *
 * One y pass suffices — `kCap` spans at most `kCap / kTileCols` tiles, far below the 65535
 * gridDim.y cap — so `blockIdx.y` names its tile directly and the y-stride loop is omitted.
 * The end-of-tile fence is likewise unnecessary: no later iteration restages anything.
 */
template <size_type BlockSize,
          size_type kTileCols,
          size_type kRowsPerThread,
          bool kStreaming,
          bool kMasked,
          size_type kCap,
          typename MapReader>
CUDF_KERNEL __launch_bounds__(BlockSize) void batch_gather_block_kernel(
  __grid_constant__ descriptor_block<kCap> const block,
  MapReader gather_map,
  size_type num_rows,
  size_type num_source_rows,
  bool check_bounds,
  bool negative_allowed,
  size_type* valid_counts)
{
  // `gather_tile_rows` indexes tile-local positions, so it receives the block's descriptor array
  // offset to this block's tile; the y grid is exactly the tile count, hence no stride loop.
  auto const tile_begin = static_cast<size_type>(blockIdx.y) * kTileCols;
  gather_tile_rows<BlockSize, kTileCols, kRowsPerThread, kStreaming, kMasked>(
    gather_map,
    block.descs + tile_begin,
    tile_begin,
    block.count,
    num_rows,
    num_source_rows,
    check_bounds,
    negative_allowed,
    valid_counts);
}

/**
 * @brief Gathers every column described by `descriptors` in one launch, reading the map through
 * `direct_map_reader<MapType>` (the direct-load fast path for wide integer maps).
 *
 * Same grid geometry, descriptor staging, row loop, and mask machinery as the indexalator
 * `batch_gather_kernel` — expressed through the same `gather_tile_rows` helper; only the map
 * source differs (one typed load per row instead of the indexalator's per-access type
 * dispatch). Launched only for maps the `map_supports_direct_load` predicate accepts, so the
 * typed pointer is valid and the widening matches the indexalator's at every representational
 * extreme (see `direct_map_reader`).
 */
template <size_type BlockSize,
          size_type kTileCols,
          size_type kRowsPerThread,
          bool kStreaming,
          bool kMasked,
          typename MapType>
CUDF_KERNEL __launch_bounds__(BlockSize) void batch_gather_direct_map_kernel(
  MapType const* map_values,
  column_descriptor const* descriptors,
  size_type num_descriptors,
  size_type num_rows,
  size_type num_source_rows,
  bool check_bounds,
  bool negative_allowed,
  size_type* valid_counts)
{
  __shared__ column_descriptor tile_descs[kTileCols];
  auto const num_tiles = util::div_rounding_up_safe(num_descriptors, kTileCols);
  for (size_type tile = static_cast<size_type>(blockIdx.y); tile < num_tiles;
       tile += static_cast<size_type>(gridDim.y)) {
    auto const tile_begin = tile * kTileCols;
    // Stage the tile's descriptors in shared memory; the range is block-uniform and the reads
    // below are warp broadcasts.
    if (threadIdx.x < kTileCols &&
        tile_begin + static_cast<size_type>(threadIdx.x) < num_descriptors) {
      tile_descs[threadIdx.x] = descriptors[tile_begin + static_cast<size_type>(threadIdx.x)];
    }
    __syncthreads();
    gather_tile_rows<BlockSize, kTileCols, kRowsPerThread, kStreaming, kMasked>(
      direct_map_reader<MapType>{map_values},
      tile_descs,
      tile_begin,
      num_descriptors,
      num_rows,
      num_source_rows,
      check_bounds,
      negative_allowed,
      valid_counts);
    // All threads are finished reading tile_descs; the next tile iteration rewrites it. The
    // predicate is exactly the y-stride loop-continuation condition (tile + gridDim.y <
    // num_tiles, rewritten to avoid the addition overflowing size_type), and tile, gridDim.y,
    // and num_tiles are block-uniform, so the whole block either takes this barrier or exits
    // the loop together — no thread can wait at an instance another thread skipped.
    if (static_cast<size_type>(gridDim.y) < num_tiles - tile) { __syncthreads(); }
  }
}

// Host-side predicate for the direct-map fast path, over host-visible type info only: the map
// is a plain fixed-width integer whose in-memory representation IS the logical value, with an
// instantiated element width (4/8 bytes). All eight integer index types pass the
// representation test — the indexalator performs no null check (the map is null-free by the
// entry expects) and its normalization is exactly the `static_cast<size_type>` widening — but
// INT8/INT16/UINT8/UINT16 maps keep the indexalator: their loads are already narrow and the
// extra variants buy nothing. `data<T>()` carries the slice offset in element units, so
// sliced maps need no special case.
bool map_supports_direct_load(column_view const& gather_map)
{
  switch (gather_map.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: break;
    default: return false;
  }
  auto const elem_size = static_cast<size_type>(cudf::size_of(gather_map.type()));
  return elem_size == 4 || elem_size == 8;
}

// Element pointer for a qualifying map: dispatches through the map's storage type, which for
// the qualifying ids is the plain integer type itself; `data<T>()` has the slice offset
// pre-applied in element units, matching the indexalator's element base exactly.
template <typename MapType>
MapType const* direct_map_data(column_view const& gather_map)
{
  return static_cast<MapType const*>(cudf::type_dispatcher<dispatch_storage_type>(
    gather_map.type(), data_ptr_functor{}, gather_map));
}

// Enqueues the map-locality sample copies on the stream (a pitched D2H into pinned staging,
// plus one endpoint-window copy when the window grid is not uniform — no kernel launch, no
// sync). Returns a handle with empty staging for unclassifiable maps.
sampled_map_locality enqueue_map_locality_sample(column_view const& gather_map,
                                                 cuda::stream_ref stream)
{
  switch (gather_map.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: break;
    default:
      return sampled_map_locality::empty(
        stream);  // Not an index type: unclassifiable, safe default.
  }
  auto const num_rows    = gather_map.size();
  auto const elem_size   = static_cast<size_t>(cudf::size_of(gather_map.type()));
  auto const window_size = static_cast<size_t>(kSampleWindowLen) * elem_size;
  auto const last_start  = static_cast<int64_t>(num_rows) - static_cast<int64_t>(kSampleWindowLen);
  // Old grid: window w starts at floor(last_start * w / (kSampleWindows - 1)) rows, with the
  // last window at last_start (ending at num_rows - 1). That grid is uniform only when
  // (kSampleWindows - 1) divides last_start; in the exact case one pitched copy with stride
  // last_start / (kSampleWindows - 1) reproduces every window INCLUDING the endpoint, and
  // otherwise the pitched copy carries windows 0..(kSampleWindows - 2) — each per-window
  // shift from the old grid stays below kSampleWindows - 1 rows, which only feeds a routing
  // heuristic — and the endpoint window keeps its own submission.
  auto const pitch_rows = last_start / static_cast<int64_t>(kSampleWindows - 1);
  bool const exact_grid = pitch_rows * static_cast<int64_t>(kSampleWindows - 1) == last_start;
  // Uninitialized pinned staging: every byte is supplied by the D2H copies below, so the
  // zero-init + alloc sync of the initializing vector factory buys nothing here. Deallocation
  // rides the buffer's stored stream (same stream), so it stays stream-ordered with no new sync.
  auto staging = cudf::detail::make_pinned_buffer_async<std::byte>(
    static_cast<size_t>(kSampleWindows) * window_size, stream);
  CUDF_CUDA_TRY(
    cudaMemcpy2DAsync(staging.data(),           // contiguous rows: window-major
                      window_size,              // dst pitch == one window
                      gather_map.data<char>(),  // window 0 starts at row 0
                      static_cast<size_t>(pitch_rows) * elem_size,
                      window_size,
                      static_cast<size_t>(exact_grid ? kSampleWindows : kSampleWindows - 1),
                      cudaMemcpyDefault,
                      stream.get()));
  if (not exact_grid) {
    // The endpoint window (rows last_start..num_rows - 1) is off the uniform grid.
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      staging.data() + static_cast<size_t>(kSampleWindows - 1) * window_size,
      gather_map.data<char>() + static_cast<size_t>(last_start) * elem_size,
      window_size,
      stream));
  }
  return sampled_map_locality{std::move(staging), gather_map.type().id(), elem_size};
}

// Classifies the staged windows. Requires the stream to already be synchronized: the caller's
// pre-launch synchronization doubles as the sample resolve.
size_type classify_map_locality(sampled_map_locality const& sampled)
{
  if (sampled.staging.empty()) { return kRandomTileCols; }
  auto const elem_size    = sampled.elem_size;
  auto const read_element = [&](std::byte const* p) -> int64_t {
    switch (sampled.map_type) {
      case cudf::type_id::INT8: {
        int8_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::INT16: {
        int16_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::INT32: {
        int32_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::INT64: {
        int64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
      }
      case cudf::type_id::UINT8: {
        uint8_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::UINT16: {
        uint16_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::UINT32: {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::UINT64: {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      default: CUDF_UNREACHABLE("Map type excluded at enqueue time");
    }
  };
  // Streaming needs genuine sequential locality, not merely local contiguity. Two checks per
  // sampled window, both on the staged data:
  // 1. The window's kSampleWindowLen values are a permutation of consecutive integers: a
  //    fanout map (map[i] = i/2, spread kSampleWindowLen/2) or any map with duplicates fails
  //    even though its spread stays small, and those rows are reuse-heavy.
  // 2. The window minima are strictly monotonic in one direction (ascending maps rise,
  //    descending maps fall): a periodic map such as map[i] = i % K repeats its per-window
  //    minimum and is reuse-heavy even when a window happens to be a permutation.
  // A map failing either keeps its routing decision and takes default-cached accesses; this
  // gates only the eviction hint, never the geometry.
  int minima_direction = 0;  // +1 rising, -1 falling; fixed by the first window pair.
  int64_t prev_min     = 0;
  for (size_t w = 0; w < kSampleWindows; ++w) {
    auto const* window = sampled.staging.data() + w * kSampleWindowLen * elem_size;
    auto const first   = read_element(window);
    auto min_value     = first;
    auto max_value     = first;
    for (size_type i = 1; i < kSampleWindowLen; ++i) {
      auto const v = read_element(window + static_cast<size_t>(i) * elem_size);
      min_value    = std::min(min_value, v);
      max_value    = std::max(max_value, v);
    }
    if (max_value - min_value != static_cast<int64_t>(kSampleWindowLen - 1)) {
      return kRandomTileCols;
    }
    uint32_t seen = 0;
    for (size_type i = 0; i < kSampleWindowLen; ++i) {
      seen |=
        uint32_t{1} << (read_element(window + static_cast<size_t>(i) * elem_size) - min_value);
    }
    if (seen != (uint32_t{1} << kSampleWindowLen) - 1) { return kRandomTileCols; }
    if (w > 0) {
      if (min_value == prev_min) { return kRandomTileCols; }  // flat run: periodic/broadcast
      auto const step = (min_value > prev_min) ? 1 : -1;
      if (minima_direction == 0) {
        minima_direction = step;
      } else if (step != minima_direction) {
        return kRandomTileCols;
      }
    }
    prev_min = min_value;
  }
  return kCoalescedTileCols;
}

// Standalone resolve for callers that need the tile width before the launch body: synchronize
// the stream (the sample copies plus whatever was enqueued before them — never a kernel of
// ours) and classify the staged windows.
size_type resolve_map_locality_tile_cols(sampled_map_locality const& sampled,
                                         cuda::stream_ref stream)
{
  cudf::detail::sync_stream(stream);
  return classify_map_locality(sampled);
}

// Tile width for regimes where sampling is skipped; nullopt means "sample the map". Zero/one
// fused columns keep K1; otherwise the unsampled choice is K2 while the two-column working set
// plus the map fit half of L2 (a failed L2 query reports 0 bytes and takes the safe K1).
std::optional<size_type> unsampled_tile_width(size_type num_map_rows,
                                              size_type num_src_rows,
                                              size_type num_fused_cols,
                                              size_type max_fused_element_size,
                                              size_type map_element_size)
{
  if (num_fused_cols < 2) { return kRandomTileCols; }
  auto const work = static_cast<int64_t>(num_map_rows) * static_cast<int64_t>(num_fused_cols);
  if (num_map_rows >= kMinSampledRows && work >= kMinSampledWork) { return std::nullopt; }
  static int const l2_bytes = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, device) != cudaSuccess) { return 0; }
    return value;
  }();
  auto const footprint =
    static_cast<int64_t>(num_src_rows) * static_cast<int64_t>(max_fused_element_size) * 2 +
    static_cast<int64_t>(num_map_rows) * static_cast<int64_t>(map_element_size);
  if (footprint * 2 > static_cast<int64_t>(l2_bytes)) { return kRandomTileCols; }
  return kUnsampledTileCols;
}

}  // namespace

size_type select_gather_map_tile_cols(table_view const& source_table,
                                      column_view const& gather_map,
                                      size_type num_fused_cols,
                                      size_type max_fused_element_size,
                                      cuda::stream_ref stream)
{
  auto const gate = unsampled_tile_width(gather_map.size(),
                                         source_table.num_rows(),
                                         num_fused_cols,
                                         max_fused_element_size,
                                         static_cast<size_type>(cudf::size_of(gather_map.type())));
  if (gate.has_value()) { return *gate; }
  auto const sampled = enqueue_map_locality_sample(gather_map, stream);
  return resolve_map_locality_tile_cols(sampled, stream);
}

// Shared body of the routed and explicit entry points; the routed entry enqueues the sampler
// before its allocation work and leaves the tile width unresolved until the body's pre-launch
// synchronization.
// `explicit_tile_cols` nullopt means "resolve from the sample"; block_size must be one of
// {128, 256, 512} and applies to every kernel instantiation this body can launch.
std::unique_ptr<table> batch_gather_impl(table_view const& source_table,
                                         column_view const& gather_map,
                                         out_of_bounds_policy bounds_policy,
                                         negative_index_policy neg_indices,
                                         cuda::stream_ref stream,
                                         memory_resources mr,
                                         std::optional<size_type> explicit_tile_cols,
                                         sampled_map_locality const& sampled,
                                         std::optional<size_type> explicit_rows_per_thread,
                                         size_type block_size);

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr)
{
  // Routed entry: one host-side type pass gathers the fused-column count and the widest fused
  // element for the gate; when sampling is warranted its copies are ENQUEUED here and RESOLVED
  // inside the body after the destination allocations, so they hide behind that work and the
  // body's single pre-launch synchronization.
  size_type num_fused_cols      = 0;
  size_type max_fused_elem_size = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    auto const& col = source_table.column(i);
    if (cudf::is_fixed_width(col.type())) {
      ++num_fused_cols;
      max_fused_elem_size =
        std::max(max_fused_elem_size, static_cast<size_type>(cudf::size_of(col.type())));
    }
  }
  if (num_fused_cols == 0 or gather_map.has_nulls()) {
    // Nothing to fuse (or the delegate rejects the map): no sampler, no measurable tile width.
    return batch_gather_impl(source_table,
                             gather_map,
                             bounds_policy,
                             neg_indices,
                             stream,
                             mr,
                             std::optional<size_type>{kDefaultTileCols},
                             sampled_map_locality::empty(stream),
                             std::nullopt,
                             kDefaultBlockSize);
  }
  auto const tile_cols =
    unsampled_tile_width(gather_map.size(),
                         source_table.num_rows(),
                         num_fused_cols,
                         max_fused_elem_size,
                         static_cast<size_type>(cudf::size_of(gather_map.type())));
  if (tile_cols.has_value()) {
    return batch_gather_impl(source_table,
                             gather_map,
                             bounds_policy,
                             neg_indices,
                             stream,
                             mr,
                             std::optional<size_type>{*tile_cols},
                             sampled_map_locality::empty(stream),
                             std::nullopt,
                             kDefaultBlockSize);
  }
  auto const sampled = enqueue_map_locality_sample(gather_map, stream);
  // The routed entry takes no block-size parameter: it always runs kDefaultBlockSize. The
  // measurement axis exists only on the explicit (tile_cols, rows_per_thread, block_size)
  // overload.
  return batch_gather_impl(source_table,
                           gather_map,
                           bounds_policy,
                           neg_indices,
                           stream,
                           mr,
                           std::nullopt,
                           sampled,
                           std::nullopt,
                           kDefaultBlockSize);
}

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr,
                                    size_type tile_cols,
                                    size_type block_size)
{
  return batch_gather_impl(source_table,
                           gather_map,
                           bounds_policy,
                           neg_indices,
                           stream,
                           mr,
                           std::optional<size_type>{tile_cols},
                           sampled_map_locality::empty(stream),
                           std::nullopt,
                           block_size);
}

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr,
                                    size_type tile_cols,
                                    size_type rows_per_thread,
                                    size_type block_size)
{
  CUDF_EXPECTS(rows_per_thread == 1 or rows_per_thread == 2 or rows_per_thread == 4,
               "Unsupported batch_gather rows per thread",
               std::invalid_argument);
  return batch_gather_impl(source_table,
                           gather_map,
                           bounds_policy,
                           neg_indices,
                           stream,
                           mr,
                           std::optional<size_type>{tile_cols},
                           sampled_map_locality::empty(stream),
                           std::optional<size_type>{rows_per_thread},
                           block_size);
}

std::unique_ptr<table> batch_gather_impl(table_view const& source_table,
                                         column_view const& gather_map,
                                         out_of_bounds_policy bounds_policy,
                                         negative_index_policy neg_indices,
                                         cuda::stream_ref stream,
                                         memory_resources mr,
                                         std::optional<size_type> explicit_tile_cols,
                                         sampled_map_locality const& sampled,
                                         std::optional<size_type> explicit_rows_per_thread,
                                         size_type block_size)
{
  cudf::scoped_range rng{"batch_gather"};

  CUDF_EXPECTS(not gather_map.has_nulls(), "gather_map contains nulls", std::invalid_argument);

  // INVARIANT: gather_map.size() is the row count of every destination column.
  auto const num_rows     = gather_map.size();
  auto const num_src_rows = source_table.num_rows();
  auto const output_mr    = mr.get_output_mr();
  auto const temp_mr      = mr.get_temporary_mr();

  if (source_table.num_columns() == 0) {
    // Zero-column tables preserve their row count (stock gather parity).
    return std::make_unique<table>(std::vector<std::unique_ptr<column>>{}, num_rows);
  }

  std::vector<size_type> fused_indices;
  std::vector<size_type> fallback_indices;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    (cudf::is_fixed_width(source_table.column(i).type()) ? fused_indices : fallback_indices)
      .push_back(i);
  }

  std::vector<std::unique_ptr<column>> destination(source_table.num_columns());

  auto run_fallback = [&](
                        std::vector<size_type> const& indices, auto map_it_begin, auto map_it_end) {
    std::vector<std::unique_ptr<column>> fallback_columns;
    fallback_columns.reserve(indices.size());
    for (auto const idx : indices) {
      auto const& src = source_table.column(idx);
      fallback_columns.push_back(
        cudf::type_dispatcher<dispatch_storage_type>(src.type(),
                                                     column_gatherer{},
                                                     src,
                                                     map_it_begin,
                                                     map_it_end,
                                                     bounds_policy == out_of_bounds_policy::NULLIFY,
                                                     stream,
                                                     mr));
    }

    // Null-mask phase over the fallback subset, mirroring the stock gather's WHOLE-TABLE
    // conditions: a null-bearing fused column must push the fallback
    // through gather_bitmask exactly as stock does. The APPLICATION stays subset-scoped — the
    // fused columns' masks are kernel-written per the mask-policy table, which matches stock's
    // per-column outcome (gather_bitmask is per-column self-guarding,
    // and set_all_valid_null_masks is a no-op for non-nullable inputs).
    auto fallback_views = std::vector<cudf::column_view>{};
    fallback_views.reserve(indices.size());
    for (auto const idx : indices) {
      fallback_views.push_back(source_table.column(idx));
    }
    auto const fallback_table = cudf::table_view{fallback_views};
    if (bounds_policy == out_of_bounds_policy::NULLIFY or
        cudf::has_nested_nullable_columns(source_table)) {
      if (bounds_policy == out_of_bounds_policy::NULLIFY or cudf::has_nested_nulls(source_table)) {
        gather_bitmask(fallback_table,
                       map_it_begin,
                       fallback_columns,
                       bounds_policy == out_of_bounds_policy::NULLIFY
                         ? gather_bitmask_op::NULLIFY
                         : gather_bitmask_op::DONT_CHECK,
                       stream,
                       mr);
      } else {
        for (size_t i = 0; i < fallback_columns.size(); ++i) {
          cudf::detail::set_all_valid_null_masks(
            fallback_table.column(i), *fallback_columns[i], stream, output_mr);
        }
      }
    }

    for (size_t i = 0; i < indices.size(); ++i) {
      destination[indices[i]] = std::move(fallback_columns[i]);
    }
  };

  // A zero-output call produces no data and no per-row mask work, so the whole staging/launch/
  // readback pipeline is skipped; the destination columns are built host-side in exactly the
  // shapes stock gather produces for an empty map:
  // - fixed-width: `allocate_like(src, 0, NEVER)` (unallocated mask), matching the stock
  //   per-column gatherer plus its zero-range `thrust::gather`;
  // - other types: their gatherers early-return the same thing (`empty_like` for list/struct,
  //   `make_empty_column` for strings/dictionary), so delegating them reproduces stock exactly,
  //   including the unsupported-type `CUDF_FAIL`.
  // Every destination has zero rows and null count 0. A non-empty map is never routed here
  // even when NULLIFY would nullify every gathered row: proving that needs the map values
  // (device data), so those calls run the full path below. With a zero-size map the mask
  // phase degenerates to nothing — `gather_bitmask` returns on its `mask_size == 0` guard
  // and the 0-word masks it attaches make the columns non-nullable.
  if (num_rows == 0) {
    for (auto const idx : fused_indices) {
      destination[idx] = cudf::detail::allocate_like(
        source_table.column(idx), 0, cudf::mask_allocation_policy::NEVER, stream, output_mr);
    }
    // Built unconditionally, like stock's detail::gather does before any per-column work: a
    // non-index map type must throw on a zero-row call the same way it does on a non-empty one.
    auto const map_begin = indexalator_factory::make_input_iterator(gather_map);
    if (not fallback_indices.empty()) { run_fallback(fallback_indices, map_begin, map_begin); }
    return std::make_unique<table>(std::move(destination), num_rows);
  }

  // Fused part: destination columns for the fixed-width subset, with per-column mask policy.
  // Constant masks (ALL_VALID/ALL_NULL) are allocated UNINITIALIZED and filled by the two
  // batched_memset calls after this loop — one cudaMemsetAsync per FILL VALUE instead of per
  // column (~3 µs host per submission). The batched_memset API takes ONE value for the whole
  // span batch, so the two states need two calls.
  std::vector<column_descriptor> descriptors;
  // {descriptor position, fused-column position} pairs with an in-kernel destination mask: the
  // descriptor position indexes the launch-side counts array, the fused position the
  // destination column the count belongs to (they diverge once all-null columns are skipped).
  std::vector<std::pair<size_type, size_type>> kernel_masked_positions;
  std::vector<cudf::device_span<cudf::bitmask_type>> valid_fills;  // ALL_VALID constant masks
  std::vector<cudf::device_span<cudf::bitmask_type>> null_fills;   // ALL_NULL constant masks
  auto const mask_size_words =
    static_cast<std::size_t>(cudf::bitmask_allocation_size_words(num_rows));
  if (not fused_indices.empty()) {
    descriptors.reserve(fused_indices.size());
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      auto dst        = cudf::detail::allocate_like(
        src, num_rows, cudf::mask_allocation_policy::NEVER, stream, output_mr);
      bitmask_type* dst_mask = nullptr;
      // Pre-declared so the all-null early-exit branch can move `dst` before the shared
      // descriptor-path statements run; assigned with the real view below for every branch.
      cudf::mutable_column_view dst_view{};
      bool const all_null = src.size() > 0 and src.null_count() == src.size();
      if (all_null) {
        // Every row is null regardless of the map: pre-set ALL_NULL (the fill itself is batched
        // below) and carry NO descriptor — the kernel would only skip this column, so it takes
        // no tile slot and no per-tile work. The counts are already final here.
        dst->set_null_mask(
          cuda::device_buffer<cudf::bitmask_type>(
            stream,
            output_mr,
            static_cast<std::size_t>(cudf::bitmask_allocation_size_words(num_rows)),
            cuda::no_init),
          num_rows);
        dst_view = dst->mutable_view();
        null_fills.push_back(
          cudf::device_span<cudf::bitmask_type>{dst_view.null_mask(), mask_size_words});
        destination[fused_indices[pos]] = std::move(dst);
        continue;
      } else if (bounds_policy == out_of_bounds_policy::NULLIFY or src.null_count() > 0) {
        auto mask = cudf::detail::create_null_mask(
          num_rows, cudf::mask_state::UNINITIALIZED, stream, output_mr);
        dst_mask = mask.data();
        dst->set_null_mask(std::move(mask), 0);
        kernel_masked_positions.push_back(
          {static_cast<size_type>(descriptors.size()), fused_indices[pos]});
      } else if (src.nullable()) {
        // Nullable source with zero nulls: ALL_VALID fast path, no kernel mask work — the
        // all-ones fill is batched below.
        dst->set_null_mask(
          cuda::device_buffer<cudf::bitmask_type>(
            stream,
            output_mr,
            static_cast<std::size_t>(cudf::bitmask_allocation_size_words(num_rows)),
            cuda::no_init),
          0);
      }
      auto const src_data =
        cudf::type_dispatcher<dispatch_storage_type>(src.type(), data_ptr_functor{}, src);
      // A cached zero null count means no source bit is clear, so the mask cannot change any
      // row's validity: drop it here to keep the call eligible for the mask-free instantiation.
      // The destination mask policy above keys off `dst_mask` only and is unaffected.
      bitmask_type const* src_mask = src.null_mask();
      if (src.null_count() == 0) { src_mask = nullptr; }
      dst_view = dst->mutable_view();
      if (src.nullable()) {
        valid_fills.push_back(
          cudf::device_span<cudf::bitmask_type>{dst_view.null_mask(), mask_size_words});
      }
      descriptors.push_back(column_descriptor{src_data,
                                              dst_view.data<std::byte>(),
                                              src_mask,
                                              dst_mask,
                                              static_cast<int64_t>(src.offset()),
                                              static_cast<int32_t>(cudf::size_of(src.type()))});
      destination[fused_indices[pos]] = std::move(dst);
    }
  }

  // The spans alias the destination columns' mask buffers: built AFTER the loop so no slot can
  // dangle a moved-from unique_ptr, and valid through the memsets (the buffers live in the
  // returned table). UNINITIALIZED means the padding words are also written by the memset —
  // the same bytes create_null_mask's per-call cudaMemsetAsync would have initialized. Enqueued
  // BEFORE the gather kernel: the kernel never writes these dst masks (dst_mask is nullptr for
  // both fill classes), so the stream order keeps the race out regardless of scheduling.
  // The span arrays stage in pinned memory (the parquet batched_memset precedent) so the
  // batched H2D reads device-accessible source bytes: one staging vector, valid fills in the
  // leading slots and null fills appended after.
  if (not valid_fills.empty() or not null_fills.empty()) {
    auto fill_staging =
      cudf::detail::make_empty_pinned_vector<cudf::device_span<cudf::bitmask_type>>(
        valid_fills.size() + null_fills.size(), stream);
    std::copy(valid_fills.begin(), valid_fills.end(), std::back_inserter(fill_staging));
    std::copy(null_fills.begin(), null_fills.end(), std::back_inserter(fill_staging));
    auto const fill_spans =
      cudf::host_span<cudf::device_span<cudf::bitmask_type> const>{fill_staging};
    if (not valid_fills.empty()) {
      cudf::detail::batched_memset<cudf::bitmask_type>(
        fill_spans.first(valid_fills.size()),
        std::numeric_limits<cudf::bitmask_type>::max(),
        stream);
    }
    if (not null_fills.empty()) {
      cudf::detail::batched_memset<cudf::bitmask_type>(
        fill_spans.last(null_fills.size()), cudf::bitmask_type{0}, stream);
    }
  }

  if (not descriptors.empty() and num_rows > 0) {
    // THE single pre-launch synchronization: it resolves the enqueued locality sample (when the
    // routed entry took one) and drains the stream, which is also what makes the pinned staging
    // below safe to write from the host.
    cudf::detail::sync_stream(stream);
    auto const effective_tile_cols =
      explicit_tile_cols.has_value() ? *explicit_tile_cols : classify_map_locality(sampled);
    auto const num_descs = static_cast<size_type>(descriptors.size());
    // Small tables ride in the kernel parameters instead of the device array. The by-value
    // tiers instantiate the tile widths the router can return (1, 2, 8) only, so the
    // explicit-only geometry 4 keeps the device array — the same rule the launch switch below
    // applies.
    auto const by_value_tier =
      explicit_tile_cols != size_type{4} ? batch_gather_descriptors_by_value_tier(num_descs) : 0;

    // The mask-free instantiation is valid only when NO descriptor writes a destination mask or
    // reads a source mask. `kernel_masked_positions` covers the destination side only, so the
    // source side is checked here as well; both are host descriptor fields, not device state.
    auto const any_kernel_masks =
      not kernel_masked_positions.empty() or
      std::any_of(descriptors.begin(), descriptors.end(), [](column_descriptor const& d) {
        return d.src_mask != nullptr;
      });

    // Device path staging: pinned host staging (stream-ordered free, so the H2D needs no host
    // sync and the copy's source stays valid past this scope) plus the device buffer the kernel
    // reads. Built only for the device path. The staging bytes — descriptors and, when folded,
    // the zeroed counts tail — are written from the HOST before the launch, into pinned-pool
    // memory handed out by the ASYNC factory: the pool can return a block whose previous
    // stream-ordered copy is still in flight, so the host writes are safe only AFTER the
    // pre-launch sync above. The receive-only buffers below and the sampler staging have no
    // host writes at all: their writers are stream-ordered copies, and their readers wait on
    // the stream.
    std::optional<cuda::host_buffer<std::byte>> staging;
    std::optional<rmm::device_uvector<std::byte>> d_descriptors;
    // Masked device-array calls fold the kernel's per-descriptor valid counts into the staging
    // tail: one allocation and one H2D carry descriptors + zeroed counts, replacing the
    // separate zeroed `valid_counts` uvector and its `cudaMemsetAsync`. The buffer is BYTE-typed
    // because the tail (4 B per count) is not a whole number of 48 B descriptors; the tail
    // offset num_descs * 48 is a multiple of 4, so the size_type tail is aligned. The tail is
    // zeroed HOST-side (the async factory hands out uninitialized memory, so the bytes must be
    // written here), so the device counts arrive zeroed in the same H2D. The tier paths keep
    // the separate buffer — their descriptors ride the read-only `__grid_constant__` kernel
    // parameter, which cannot carry a mutable counts output.
    size_type const counts_size  = any_kernel_masks ? num_descs : 0;
    std::size_t const head_bytes = descriptors.size() * sizeof(column_descriptor);
    std::size_t const total_bytes =
      head_bytes + static_cast<std::size_t>(counts_size) * sizeof(size_type);
    if (by_value_tier == 0) {
      staging.emplace(cudf::detail::make_pinned_buffer_async<std::byte>(total_bytes, stream));
      std::memcpy(staging->data(), descriptors.data(), head_bytes);
      if (any_kernel_masks) {
        // The kernel accumulates with relaxed device-scope atomics, so the counters start at
        // zero.
        std::memset(staging->data() + head_bytes,
                    0,
                    static_cast<std::size_t>(counts_size) * sizeof(size_type));
      }
      d_descriptors.emplace(total_bytes, stream, temp_mr);
      // Ordered upload: the batched copy's overlap-with-compute hint lets the driver run it
      // concurrently with the consumer kernel, which would read pre-staging descriptor bytes.
      CUDF_CUDA_TRY(
        cudf::detail::memcpy_async(d_descriptors->data(), staging->data(), total_bytes, stream));
    }

    rmm::device_uvector<size_type> valid_counts(0, stream, temp_mr);
    if (by_value_tier != 0 and any_kernel_masks) {
      // Tier path: descriptors ride the kernel parameter block, so the counts need their own
      // zeroed device buffer (unchanged cost — no staging tail exists here to fold into).
      valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        descriptors.size(), stream, temp_mr);
    }
    // The counts the masked kernels accumulate into: the staging tail on the device path
    // (zeroed bytes rode the H2D), the separate buffer on the tier path. Read only when the
    // masked instantiation is launched (counts_ptr is nullptr otherwise).
    size_type* device_counts_ptr =
      by_value_tier == 0 ? reinterpret_cast<size_type*>(
                             reinterpret_cast<std::byte*>(d_descriptors->data()) + head_bytes)
                         : valid_counts.data();

    // One launch per call; the tile width, row coarsening, block size, descriptor tier, and map
    // source pick the kernel instantiation. x dimension: the row blocks exactly cover the
    // chunks, so the device row loop runs one pass — never cap it (e.g. at the y 65535 limit)
    // or the single-pass guarantee breaks. y dimension: one block per column tile (the device
    // kernel's y-stride loop covers tiles past the 65535 cap; the by-value block always fits
    // below it).
    // Plain wide integer maps read the map through one typed load per row (the direct-load
    // fast path); every other map type keeps the indexalator. The direct selection is
    // consistent across tiers: by-value, device, and direct all receive the same reader.
    auto const direct_map = map_supports_direct_load(gather_map);
    // Always built from the factory: the indexalator reader is ALSO the map source for
    // non-covered shapes when `direct_map` is true (e.g. explicit K4, K8-non-streaming), and a
    // default-constructed `input_indexalator` has a null base — launching with it faults. The
    // direct branch passes a `direct_map_reader` instead; the factory call is host-side and free.
    auto const map_begin   = indexalator_factory::make_input_iterator(gather_map);
    auto const map_reader  = indexalator_reader{map_begin};
    auto const launch_tile = [&](auto tile_constant,
                                 auto streaming_constant,
                                 auto rows_per_thread_constant,
                                 auto block_constant,
                                 int by_value_tier) {
      constexpr size_type kTileCols      = decltype(tile_constant)::value;
      constexpr bool kStreaming          = decltype(streaming_constant)::value;
      constexpr size_type kRowsPerThread = decltype(rows_per_thread_constant)::value;
      constexpr size_type kBlockSize     = decltype(block_constant)::value;
      dim3 const grid{static_cast<unsigned>(cudf::util::div_rounding_up_safe(
                        static_cast<int64_t>(num_rows),
                        static_cast<int64_t>(kBlockSize) * static_cast<int64_t>(kRowsPerThread))),
                      static_cast<unsigned>(std::min<size_type>(
                        cudf::util::div_rounding_up_safe(num_descs, kTileCols), 65535)),
                      1};
      // The mask-free choice comes from host descriptor fields, so it selects the
      // instantiation directly; the unmasked one never receives a `valid_counts` buffer.
      auto const launch = [&](auto masked_constant) {
        constexpr bool kMasked = decltype(masked_constant)::value;
        auto const counts_ptr  = kMasked ? device_counts_ptr : nullptr;
        // The map reader rides the kernel template: the direct-load fast path covers the
        // round-9 subset — tile 1 non-streaming and tile 8 streaming (the hot narrow-table
        // shape), each masked/mask-free — every other reachable geometry keeps the
        // indexalator. int32_t and int64_t cover all four qualifying ids through the
        // truncation alias. The constexpr gates keep call sites (and kernel instantiations)
        // out of the non-covered shapes; the runtime `direct_map` guard is the routing
        // counterpart so a covered-but-narrow map still takes the indexalator. R > 1 exists
        // only on the non-streaming narrow tiles; the covered shapes never stream at R > 1,
        // so the direct branch shares those instantiations' kRowsPerThread.
        auto const launch_indexalator_shape = [&](size_type* counts_ptr) {
          if (by_value_tier != 0) {
            // Zero-initialized so every byte the launch copies is defined; the device reads
            // only the first `count` entries.
            auto const fill_and_launch = [&](auto tier_cap) {
              constexpr size_type kCap = decltype(tier_cap)::value;
              auto block               = descriptor_block<kCap>{};
              block.count              = num_descs;
              std::copy(descriptors.begin(), descriptors.end(), block.descs);
              batch_gather_block_kernel<kBlockSize,
                                        kTileCols,
                                        kRowsPerThread,
                                        kStreaming,
                                        kMasked,
                                        kCap><<<grid, kBlockSize, 0, stream.get()>>>(
                block,
                map_reader,
                num_rows,
                num_src_rows,
                bounds_policy == out_of_bounds_policy::NULLIFY,
                neg_indices == negative_index_policy::ALLOWED,
                counts_ptr);
            };
            if (by_value_tier == 1) {
              fill_and_launch(
                std::integral_constant<size_type, batch_gather_small_by_value_descriptors>{});
            } else {
              fill_and_launch(
                std::integral_constant<size_type, batch_gather_wide_by_value_descriptors>{});
            }
          } else {
            batch_gather_kernel<kBlockSize, kTileCols, kRowsPerThread, kStreaming, kMasked>
              <<<grid, kBlockSize, 0, stream.get()>>>(
                map_reader,
                reinterpret_cast<column_descriptor const*>(d_descriptors->data()),
                num_descs,
                num_rows,
                num_src_rows,
                bounds_policy == out_of_bounds_policy::NULLIFY,
                neg_indices == negative_index_policy::ALLOWED,
                counts_ptr);
          }
          CUDF_CUDA_TRY(cudaGetLastError());
        };
        if constexpr ((kTileCols == 1 and not kStreaming) or
                      (kTileCols == kCoalescedTileCols and kStreaming)) {
          if (direct_map) {
            auto const launch_map = [&](auto map_reader) {
              if (by_value_tier != 0) {
                // Zero-initialized so every byte the launch copies is defined; the device
                // reads only the first `count` entries.
                auto const fill_and_launch = [&](auto tier_cap) {
                  constexpr size_type kCap = decltype(tier_cap)::value;
                  auto block               = descriptor_block<kCap>{};
                  block.count              = num_descs;
                  std::copy(descriptors.begin(), descriptors.end(), block.descs);
                  batch_gather_block_kernel<kBlockSize,
                                            kTileCols,
                                            kRowsPerThread,
                                            kStreaming,
                                            kMasked,
                                            kCap><<<grid, kBlockSize, 0, stream.get()>>>(
                    block,
                    map_reader,
                    num_rows,
                    num_src_rows,
                    bounds_policy == out_of_bounds_policy::NULLIFY,
                    neg_indices == negative_index_policy::ALLOWED,
                    counts_ptr);
                };
                if (by_value_tier == 1) {
                  fill_and_launch(
                    std::integral_constant<size_type, batch_gather_small_by_value_descriptors>{});
                } else {
                  fill_and_launch(
                    std::integral_constant<size_type, batch_gather_wide_by_value_descriptors>{});
                }
              } else {
                batch_gather_direct_map_kernel<kBlockSize,
                                               kTileCols,
                                               kRowsPerThread,
                                               kStreaming,
                                               kMasked,
                                               typename decltype(map_reader)::value_type>
                  <<<grid, kBlockSize, 0, stream.get()>>>(
                    map_reader.base,
                    reinterpret_cast<column_descriptor const*>(d_descriptors->data()),
                    num_descs,
                    num_rows,
                    num_src_rows,
                    bounds_policy == out_of_bounds_policy::NULLIFY,
                    neg_indices == negative_index_policy::ALLOWED,
                    counts_ptr);
              }
              CUDF_CUDA_TRY(cudaGetLastError());
            };
            switch (gather_map.type().id()) {
              case cudf::type_id::INT32:  // falls through
              case cudf::type_id::UINT32:
                launch_map(direct_map_reader<int32_t>{direct_map_data<int32_t>(gather_map)});
                break;
              case cudf::type_id::INT64:  // falls through
              case cudf::type_id::UINT64:
                launch_map(direct_map_reader<int64_t>{direct_map_data<int64_t>(gather_map)});
                break;
              default: CUDF_UNREACHABLE("Map type excluded by the direct-load predicate");
            }
          } else {
            launch_indexalator_shape(counts_ptr);
          }
        } else {
          launch_indexalator_shape(counts_ptr);
        }
      };
      if (any_kernel_masks) {
        launch(std::true_type{});
      } else {
        launch(std::false_type{});
      }
    };
    // Evict-first streaming hints ride the classifier verdict only: the sampled coalesced
    // route (which resolves to kCoalescedTileCols and nothing else). Unsampled regimes,
    // random verdicts, and test-forced geometries keep the default-cached accesses.
    auto const streaming =
      not explicit_tile_cols.has_value() and effective_tile_cols == kCoalescedTileCols;
    // Tile 4 is explicit-only and has no by-value instantiation, so it always takes the device
    // array. `by_value_tier` already excludes the explicit 4 (see the tier comment above), which
    // makes this arm unreachable from the by-value dispatch. Row coarsening instantiates
    // {2,4} only for the latency-bound narrow tiles (kTileCols 1/2, non-streaming); the wide
    // tiles (4/8) and the streaming tile already keep enough loads in flight, and R > 1 there
    // squares the register cost for no coarsening benefit — R=1 is the un-coarsened geometry.
    // Block size is a measurement axis, not a routing decision: the routed path always runs
    // kDefaultBlockSize, and only the explicit entry can override it. Every supported size is
    // a warp multiple, so the mask-word exactly-once invariant above holds for all of them.
    auto const dispatch = [&](int by_value_tier) {
      auto const rows_per_thread   = explicit_rows_per_thread.value_or(kDefaultRowsPerThread);
      auto const launch_with_block = [&](auto tile_constant, auto streaming_constant) {
        constexpr bool kTileStreaming = decltype(streaming_constant)::value;
        // The R axis exists only on the latency-bound non-streaming narrow tiles (kTileCols
        // 1/2): the wide tiles and the streaming tile already keep enough loads in flight, and
        // R > 1 there squares the register cost for no coarsening benefit, so those shapes pin
        // the R = 1 instantiation whatever the caller asked (see the header's rows_per_thread
        // contract). The block-size switch is the outer axis and feeds every geometry it can
        // reach.
        auto const launch_rows = [&](auto rows_per_thread_constant) {
          switch (block_size) {
            case 128:
              launch_tile(tile_constant,
                          streaming_constant,
                          rows_per_thread_constant,
                          std::integral_constant<size_type, 128>{},
                          by_value_tier);
              break;
            case 256:
              launch_tile(tile_constant,
                          streaming_constant,
                          rows_per_thread_constant,
                          std::integral_constant<size_type, 256>{},
                          by_value_tier);
              break;
            case 512:
              launch_tile(tile_constant,
                          streaming_constant,
                          rows_per_thread_constant,
                          std::integral_constant<size_type, 512>{},
                          by_value_tier);
              break;
            default: CUDF_FAIL("Unsupported batch_gather block size", std::invalid_argument);
          }
        };
        if constexpr (kTileStreaming or decltype(tile_constant)::value > 2) {
          launch_rows(std::integral_constant<size_type, kDefaultRowsPerThread>{});
        } else {
          switch (rows_per_thread) {
            case 2: launch_rows(std::integral_constant<size_type, 2>{}); break;
            case 4: launch_rows(std::integral_constant<size_type, 4>{}); break;
            default: launch_rows(std::integral_constant<size_type, 1>{}); break;
          }
        }
      };
      if (streaming) {
        launch_with_block(std::integral_constant<size_type, kCoalescedTileCols>{},
                          std::true_type{});
      } else {
        switch (effective_tile_cols) {
          case 1:
            launch_with_block(std::integral_constant<size_type, 1>{}, std::false_type{});
            break;
          case 2:
            launch_with_block(std::integral_constant<size_type, 2>{}, std::false_type{});
            break;
          case 4:
            launch_with_block(std::integral_constant<size_type, 4>{}, std::false_type{});
            break;
          case 8:
            launch_with_block(std::integral_constant<size_type, 8>{}, std::false_type{});
            break;
          default: CUDF_FAIL("Unsupported batch_gather tile width", std::invalid_argument);
        }
      }
    };

    dispatch(by_value_tier);

    // Receive owner for the kernel's per-column valid counts: the launch enqueues its D2H right
    // after the kernel (the D2H reads a buffer the same stream's kernel finished writing), so
    // the fallback work below runs while the copy is in flight instead of behind a drained
    // stream. Uninitialized pinned memory: every byte arrives from the D2H, so no zero-init and
    // no allocation sync; the free is stream-ordered on the same stream.
    std::optional<cuda::host_buffer<size_type>> h_valid_counts_owner;
    if (any_kernel_masks) {
      h_valid_counts_owner.emplace(cudf::detail::make_pinned_buffer_async<size_type>(
        static_cast<std::size_t>(descriptors.size()), stream));
      cudf::detail::cuda_memcpy_async(
        host_span<size_type>{
          h_valid_counts_owner->data(), static_cast<std::size_t>(descriptors.size()), true},
        device_span<size_type const>{device_counts_ptr,
                                     static_cast<std::size_t>(descriptors.size())},
        stream);
    }

    // The one sync where the counts are needed: covers the kernel and this D2H (same stream).
    // The fallback block AFTER this scope enqueues its kernels while the copy is still in
    // flight — masked mixed calls no longer drain the GPU before submitting fallback kernels —
    // and it sets its own null counts on its columns, so nothing further is owed here. The
    // counts MUST be set before the function returns — the table contract reports them.
    if (any_kernel_masks) {
      cudf::detail::sync_stream(stream);
      for (auto const [desc_pos, fused_pos] : kernel_masked_positions) {
        destination[fused_pos]->set_null_count(
          num_rows - h_valid_counts_owner->data()[static_cast<std::size_t>(desc_pos)]);
      }
    }
  }

  if (not fallback_indices.empty()) {
    auto map_begin = indexalator_factory::make_input_iterator(gather_map);
    auto map_end   = map_begin + gather_map.size();
    if (neg_indices == negative_index_policy::ALLOWED) {
      auto const idx_converter = cuda::proclaim_return_type<size_type>(
        [num_src_rows] __device__(size_type in) { return in < 0 ? in + num_src_rows : in; });
      run_fallback(fallback_indices,
                   cuda::transform_iterator{map_begin, idx_converter},
                   cuda::transform_iterator{map_end, idx_converter});
    } else {
      run_fallback(fallback_indices, map_begin, map_end);
    }
  }

  return std::make_unique<table>(std::move(destination), num_rows);
}

}  // namespace detail
}  // namespace cudf
