/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstring>
#include <functional>
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

// A struct column joins the descriptor path when every descendant is a fixed-width leaf; any
// list child at any depth keeps the WHOLE column in the fallback (partial flattening is a
// later stage). A STRING child is no longer disqualifying: top-level strings take the
// string fast path below, and a string inside a struct takes the same fast path via the
// string_descriptor machinery.
bool is_fully_fixed_width_struct(column_view const& col)
{
  for (size_type i = 0; i < col.num_children(); ++i) {
    auto const& child = col.child(i);
    if (child.type().id() == type_id::STRUCT) {
      if (not is_fully_fixed_width_struct(child)) { return false; }
    } else if (not cudf::is_fixed_width(child.type())) {
      return false;
    }
  }
  return true;
}

// Descriptor-path eligibility beyond plain fixed-width. Dictionaries qualify when their index
// child is a signed index type — the adopting factory's contract, which every public factory
// enforces; the child-count/signedness guard only deflects exotic hand-built columns to the
// stock path (which casts unsigned indices through the adopting factory).
bool is_descriptor_path_candidate(column_view const& col)
{
  if (cudf::is_fixed_width(col.type())) { return true; }
  switch (col.type().id()) {
    case type_id::STRUCT: return is_fully_fixed_width_struct(col);
    case type_id::STRING: return true;
    case type_id::DICTIONARY32: {
      // Unsigned index types defer to the stock gatherer, which recasts them signed. No public
      // factory can produce an unsigned-index dictionary — `dictionary::encode` and the
      // adopting factories either require signed indices or recast them signed — so this
      // deferral has no constructible fixture; do not re-add a test case for it.
      if (col.num_children() != 2) { return false; }
      auto const& indices = col.child(0);
      return cudf::is_index_type(indices.type()) and cudf::is_signed(indices.type());
    }
    default: return false;
  }
}

// Host-side count of what the descriptor path produces for one fused column: fixed-width
// columns contribute one data descriptor, fully-fixed-width structs contribute one descriptor
// per fixed-width leaf plus one mask-only descriptor per struct node that may need one, and
// dictionaries contribute one data descriptor over their indices. Mirrors the build walk in
// descriptor_path_builder; the kernel's num_descriptors is this same count.
size_type fused_column_descriptor_count(column_view const& col)
{
  if (col.type().id() == type_id::STRING) { return 0; }  // strings ride the N3 fast path
  if (col.type().id() == type_id::STRUCT) {
    size_type count = 1;  // mask-only descriptor for this node
    for (size_type i = 0; i < col.num_children(); ++i) {
      auto const& child = col.child(i);
      count += (child.type().id() == type_id::STRUCT) ? fused_column_descriptor_count(child)
                                                      : size_type{1};
    }
    return count;
  }
  return 1;  // fixed-width leaf or dictionary index payload
}

size_type fused_descriptor_count(table_view const& source_table)
{
  size_type count = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    if (is_descriptor_path_candidate(source_table.column(i))) {
      count += fused_column_descriptor_count(source_table.column(i));
    }
  }
  return count;
}

// Largest element size among the descriptor path's DATA descriptors; mask-only struct nodes
// carry no payload (element_size 0), so they never raise the working-set footprint.
size_type max_fused_data_element_size(column_view const& col)
{
  size_type max_size = 0;
  if (not is_descriptor_path_candidate(col) or col.type().id() == type_id::STRING) {
    return max_size;  // strings carry no fixed-width descriptor payload
  }
  if (col.type().id() == type_id::STRUCT) {
    for (size_type c = 0; c < col.num_children(); ++c) {
      max_size = std::max(max_size, max_fused_data_element_size(col.child(c)));
    }
  } else {
    // Fixed-width leaf or dictionary: data rides the leaf/indices element size.
    auto const& data_col = (col.type().id() == type_id::DICTIONARY32)
                             ? col.child(0)  // annotated indices; signed per the candidate check
                             : col;
    max_size             = static_cast<size_type>(cudf::size_of(data_col.type()));
  }
  return max_size;
}

size_type max_fused_data_element_size(table_view const& source_table)
{
  size_type max_size = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    max_size = std::max(max_size, max_fused_data_element_size(source_table.column(i)));
  }
  return max_size;
}

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
 * One descriptor per fixed-width LEAF, mask-only STRUCT node, or dictionary index payload — one
 * per NON-all-null fixed-width source column for the flat case: a column whose every row is null
 * has its destination pre-set ALL_NULL host-side and carries no descriptor at all, so the kernel
 * never sees a column it would only skip. All pointers are element bases, not byte bases: a
 * leaf's `src_data` comes from `column_view::data<T>()`, which has that view's own offset
 * pre-applied in element units, plus the accumulated offsets of the struct ancestors the leaf
 * sits under (applied exactly once here); `dst_data`/`dst_mask` belong to a freshly allocated
 * destination whose offset is zero. Nothing downstream may re-add an offset to these pointers.
 *
 * `src_data`/`dst_data` are nullptr for mask-only STRUCT node descriptors (`skip_data` true,
 * nothing to copy). `src_mask` is a mask BASE pointer (nullptr if the source is non-nullable);
 * `mask_offset` is the bit offset of the first row of the viewed range within that
 * base. Descriptors are pushed only for kernel-visible work; an all-null destination (mask
 * pre-set ALL_NULL) gets none — see `plan_destination_mask`'s `skip_data`.
 */
struct alignas(16) column_descriptor {
  void const* src_data;          // first source element; nullptr for mask-only struct nodes
  void* dst_data;                // first destination element; nullptr for mask-only struct nodes
  bitmask_type const* src_mask;  // source mask BASE pointer; nullptr if source non-nullable
  bitmask_type* dst_mask;        // destination mask; nullptr if dst non-nullable or no mask
  int64_t mask_offset;           // bit offset into src_mask (sum of the viewed range's offsets)
  int32_t element_size;          // cudf::size_of(src.type()); one of {0, 1, 2, 4, 8, 16}
  bool skip_data;                // mask-only node: no data copy (all-null leaves get no descriptor)
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
// dispatches to int32/int64/__int128), so the view's own offset is applied exactly once,
// host-side. The dispatcher instantiates operator() for every type in the dispatch list,
// including the pass-through types (string_view/list_view/struct_view/dictionary32) that never
// carry data<T>; they are unreachable here because only fixed-width storage enters the
// descriptor path (plain fixed-width columns, struct leaves, dictionary indices).
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
          // A mask-only descriptor (struct node, skip_data) carries null data pointers and
          // element_size 0: its rows contribute validity only, and any load/store through the
          // null pointers or the zero-size arithmetic would be UB.
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
          // Data at out-of-bounds/null rows may be left unwritten (fixed-width undefined-value
          // policy); a dead row of a straddling chunk must neither load nor store.
          if (row_live[s] && valid && !d.skip_data) {
            load_element<kStreaming>(payload[s],
                                     static_cast<char const*>(d.src_data) +
                                       static_cast<std::size_t>(idx[s]) * d.element_size,
                                     d.element_size);
          }
          if (row_live[s] && valid && !d.skip_data) {
            store_element<kStreaming>(dst, payload[s], d.element_size);
          }
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

// ---------------------------------------------------------------------------
// String fast path (N3).
//
// Stock per-string-column cost (`strings::detail::gather`): a sizes scan that
// synchronizes on its last-element readback once PER COLUMN, then a chars
// kernel, then whole-table mask work — multiplied by the column count. This
// path batches every string column of the call:
//
//   1. ONE virtual keyed exclusive scan (`thrust::exclusive_scan_by_key`, the
//      `contiguous_split.cu` pattern) over the flat (column, row) domain.
//      Every gathered size is computed on the fly from the map, the source
//      mask, and the source offsets — the sizes are never materialized — and
//      the scan writes each destination offsets value into a flat int64
//      scratch. The scratch precedes the offsets children because a gathered
//      column's byte total is unbounded by its source (the map may repeat
//      rows), so the child widths are unknowable until the totals are read.
//      The scan's tabulate output also records each column's total.
//   2. ONE host totals readback: a pinned async D2H enqueued right after the
//      scan and synchronized just before the payload launch — the sampler's
//      enqueue-early/resolve-late pattern, and the only synchronization this
//      path adds. The totals size each column's chars buffer and select its
//      offsets child width (the same selection `create_offsets_child_column`
//      makes, including the column size limit), and every live column's
//      offsets child + chars buffer is carved from ONE call-wide arena
//      (`carve_memory_resource`): a single upstream allocate/free pair serves
//      all destination string allocations, while each column keeps owning its
//      slices through the shared resource copy.
//   3. ONE batched payload kernel for ALL string columns, phased over
//      disjoint outputs (no ordering between phases): characters with
//      warp-cooperative uint4 spans per row, except rows above the block-copy
//      gate which the whole block copies,
//      destination masks written in-kernel for kernel-masked columns
//      (constant masks are pre-set by `plan_destination_mask`), the offsets
//      children finalized from the scratch (narrowing to the child's width),
//      and per-column valid counts accumulated atomically for the
//      post-kernel readback. On tall calls (>= `kBatchMemcpyRowThreshold`)
//      each all-short column's chars runs through one
//      `cub::DeviceMemcpy::Batched` over the routed columns' rows (per-column
//      routing via `chars_via_batched_memcpy`); the routed columns also leave
//      the kernel entirely — their offsets/masks/counts are written by
//      `string_routed_metadata_kernel` — and the launch grid covers only the
//      unrouted cohort's rows.
//
// Trap compliance: chars are addressed from the unsliced base with the sliced
// offsets (the offsets iterator carries the view offset); null rows have zero
// bytes and distinct validity from empty strings; gathered sizes are scanned
// in 64 bits.
// ---------------------------------------------------------------------------

// Row-length gate (bytes) between the warp-cooperative and the block-cooperative copy. A
// 32-byte row spans at most two 16-byte vectors, so one warp's uint4 span copies it without
// idle lanes; a longer row serialises across the warp's lanes and the whole block copies it.
// The 32-byte boundary matches the short/long strategy split of `strings::detail::gather`.
constexpr int64_t kBlockRowCopyThreshold{32};

// Row-count gate (rows) above which the CUB batched-memcpy engine runs instead of the payload
// kernel's chars phase. Mirrors `strings::detail::gather`'s engine split: stock measured
// `cub::DeviceMemcpy::Batched` as the winning short-string strategy at >= 524288 rows on H100,
// and the tall-cell profile shows the payload kernel's per-group barrier walk costing the same
// regime ~7x its CUB counterpart. Below the gate the payload kernel keeps its 65536-row wins.
constexpr int64_t kBatchMemcpyRowThreshold{524'288};

// Row-count gate (rows) between the N3 scan+payload string fast path and the stock per-column
// string path (run_fallback -> column_gatherer<string_view> -> strings::detail::gather). The
// payload's group-walk cost scales with the call's row count: measured A/B on tall cells shows
// every string shape at >= 262144 rows losing to the stock path (up to ~1.9x) while 65536 rows
// wins ~1.7x, and the loss persists even when every column's chars route to CUB — so the
// discriminator is the call's row count, not bytes-per-row. The composed A/B of the fused
// short-string engine confirmed the band exclusion stays: the engine wins the band cell when
// admitted (262144x100 −69.5%, engine-vs-stock falsifier-confirmed) but the band route costs
// the tall regime — above kBatchMemcpyRowThreshold the routed-metadata path (tall-str1)
// replaces the group walk and is the measured tall win (round-10, +26.7% headline) the
// engine cannot serve. The band exclusion trades the band win for the tall-str1 route.
constexpr size_type kStringFallbackRowThreshold{262'144};

// Preset-mask staging bands pay off only above this many preset destination masks in one call:
// a band memset costs ~24 µs (nsys-captured band-fill attribution) while a per-mask memset
// submission costs ~7 µs (submission-latency-bound — the device bytes are negligible at both
// the 8 KiB and 128 KiB mask extents), so the fill-method crossover sits at ~4 preset masks
// with one band populated to ~7 with both. 8 keeps a margin on the drip side while the
// measured win shapes (~60 preset masks) clear it by ~9x. Counted in preset MASKS, not
// destination columns: a wide table with few nullable columns would otherwise pay the fixed
// band cost for almost no replaced drip.
constexpr size_type kMaskBandPresetGate{8};

// Work item for one string column on the N3 path: source addressing plus the
// destination buffers the payload kernel writes. The src fields are complete
// when the item is registered (the scan reads only those); the dst fields are
// filled after the totals readback, before the payload launch. The
// user-provided host-only default ctor keeps the implicit one (which nvcc
// would otherwise attribute __host__ __device__ on this kernel-parameter
// struct) from calling the host-only constexpr offsetalator ctors in device
// context.
struct string_work_item {
  string_work_item() {}
  char const* src_chars;            // raw chars base; the offsets carry addressing
  bitmask_type const* src_mask;     // source mask BASE pointer; nullptr if non-nullable
  int64_t mask_bit_offset;          // bit offset of the viewed range within src_mask
  size_type src_rows;               // viewed source row count (bounds checks)
  input_offsetalator src_offsets;   // source offsets with the view offset applied
  int64_t const* dst_scratch;       // this column's range of the flat scratch
  char* dst_chars;                  // destination chars buffer base
  output_offsetalator dst_offsets;  // destination offsets child (narrowing stores)
  bitmask_type* dst_mask;           // destination mask; nullptr when pre-set
  size_type* valid_count;           // atomic valid-row counter; nullptr when pre-set
  // True when this column's chars ride cub::DeviceMemcpy::Batched (tall column whose
  // average gathered row is at or below the block-copy gate): the kernel's chars phase must
  // skip the column, masks/offsets stay kernel-owned.
  bool chars_via_batched_memcpy{false};
};

static_assert(std::is_trivially_copyable_v<string_work_item>,
              "string_work_item is staged through pinned memory to the device");

// Resolves destination row -> gathered source row, or -1 when the row carries
// no payload. Under DONT_CHECK the bounds test is skipped (stock parity: the
// source is then read at the raw index, undefined for out-of-bounds maps).
__device__ int64_t resolve_gathered_row(input_indexalator gather_map,
                                        size_type row,
                                        size_type num_source_rows,
                                        bool check_bounds,
                                        bool negative_allowed)
{
  auto idx = gather_map[row];
  if (negative_allowed && idx < 0) { idx += num_source_rows; }
  bool const in_bounds = !check_bounds || (idx >= 0 && idx < num_source_rows);
  return in_bounds ? static_cast<int64_t>(idx) : int64_t{-1};
}

// Payload validity of a resolved row: the source column's own mask only.
// Ancestor validity is never consulted (stock never ANDs ancestor masks).
__device__ bool string_row_valid(string_work_item const& item, int64_t src_row)
{
  return item.src_mask == nullptr ||
         bit_is_set(item.src_mask, static_cast<size_type>(item.mask_bit_offset + src_row));
}

// Layout of the flat string workspaces: the scan and the offsets phase index the int64
// scratch with `rows_per_col = num_rows + 1` slots per column, while the chars phase indexes
// flat ROWS with `num_rows` rows per column — two divisors. Every grid cell holds at most
// 41.9M flat rows, so every flat index fits uint32_t (the host enforces the bound); the
// precomputed 32-bit divisors replace the per-element 64-bit //% at each site.
struct string_flat_layout {
  cuda::fast_mod_div<uint32_t> rows_per_col;  // scan/offsets scratch space
  cuda::fast_mod_div<uint32_t> num_rows;      // chars-phase flat-row space
};

// Scan output: the exclusive prefix into the flat scratch; each column's tail
// slot doubles as the per-column total, copied out for the batched readback.
struct string_scan_out_fn {
  int64_t* scratch;
  int64_t* col_totals;
  string_flat_layout layout;

  __device__ void operator()(int64_t flat, int64_t value) const
  {
    scratch[flat] = value;
    auto const r  = static_cast<uint32_t>(flat) % layout.rows_per_col;
    if (r == layout.rows_per_col - 1u) {
      col_totals[static_cast<uint32_t>(flat) / layout.rows_per_col] = value;
    }
  }
};

// One row published by a warp for the block-cooperative copy; `size == 0` is the no-work marker.
struct block_row_copy {
  char const* src;
  char* dst;
  int64_t size;
};

// Char-parallel copy of one all-short block group's rows: every live byte of the group's span
// is addressed by exactly one thread. A thread owns the byte at `base + tid + k*blockDim.x`
// for k >= 0 until the span ends and finds its owning row by a binary search over the staged
// boundaries (thrust::upper_bound + cuda::std::prev, the same device idiom as stock's
// char-parallel kernel), then resolves that destination row through the gather map exactly
// like every other chars path. `group_offsets` is READ-ONLY here — the caller's common
// boundary stage staged it (and its first warp already published the offsets values from
// the same boundaries). Dead rows contribute no bytes (their scratch deltas are 0) but stage
// as zero-width boundaries. The caller fans only groups whose rows all belong to ONE column
// and none is published: a per-column scratch slice restarts at each column boundary, so a
// cross-column window would mix two byte spaces, and the block slot loop would race the fan
// on published rows' bytes.
__device__ void copy_group_chars_char_parallel(string_work_item const& item,
                                               string_flat_layout layout,
                                               int64_t group_start,
                                               int64_t group_end,
                                               int64_t const* group_offsets,
                                               int64_t warps_per_block,
                                               input_indexalator gather_map,
                                               size_type num_source_rows,
                                               bool check_bounds,
                                               bool negative_allowed)
{
  auto const lane = static_cast<int64_t>(threadIdx.x);
  // The window's boundaries are ALREADY staged in `group_offsets` by the caller's common
  // boundary stage: slot w = the dst offset of flat row group_start + w (the +0 boundary of
  // that row), the tail slot closing the window's last row with its +1 scratch entry, slots
  // past the flat end INT64_MAX (an unreachable boundary). Both barriers that separated the
  // staging from the byte copy and from the next group's rewrite live in the caller.
  auto const base = group_offsets[0];
  auto const end  = group_offsets[warps_per_block];
  if (end <= base) { return; }  // no live bytes in the window
  for (int64_t b = base + lane; b < end; b += static_cast<int64_t>(blockDim.x)) {
    auto const it = cuda::std::prev(
      thrust::upper_bound(thrust::seq, group_offsets, group_offsets + warps_per_block, b));
    auto const dst_row = static_cast<size_type>(
      static_cast<uint32_t>(group_start + (it - group_offsets)) % layout.num_rows);
    auto const src =
      resolve_gathered_row(gather_map, dst_row, num_source_rows, check_bounds, negative_allowed);
    item.dst_chars[b] = item.src_chars[item.src_offsets[src] + (b - *it)];
  }
}

// Copies one gathered string row with `stride` threads starting at `tid`: the same alignment
// arithmetic, uint4 interior and scalar edges for both copy strategies. The warp path calls it
// with `(lane, warp_size)`, the block-cooperative path with `(threadIdx.x, BlockSize)`.
__device__ void copy_string_span(
  char* dst_ptr, char const* src_ptr, int64_t size, int64_t tid, int64_t stride)
{
  if (size <= 0) { return; }
  // The stock boundary arithmetic, on the per-row dst base (the base is
  // 16-aligned only at row 0 of the buffer; every row realigns).
  auto const alignment_offset =
    static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(dst_ptr) % 16);
  auto const out_start_aligned =
    (int64_t{4} + alignment_offset + int64_t{15}) / int64_t{16} * int64_t{16} - alignment_offset;
  auto const out_end_aligned =
    (size - int64_t{4} + alignment_offset) / int64_t{16} * int64_t{16} - alignment_offset;
  if (out_end_aligned > out_start_aligned) {
    for (int64_t p = out_start_aligned + tid * int64_t{16}; p < out_end_aligned;
         p += stride * int64_t{16}) {
      *reinterpret_cast<uint4*>(dst_ptr + p) = strings::detail::load_uint4(src_ptr + p);
    }
    if (tid < out_start_aligned) { dst_ptr[tid] = src_ptr[tid]; }
    auto const tail = out_end_aligned + tid;
    if (tail < size) { dst_ptr[tail] = src_ptr[tail]; }
  } else {
    for (int64_t p = tid; p < size; p += stride) {
      dst_ptr[p] = src_ptr[p];
    }
  }
}

// Batched payload kernel for the UNROUTED string columns (one launch per call; the routed
// cohort's metadata rides string_routed_metadata_kernel).
//
// Grid-stride phases over disjoint outputs, no ordering required between
// them:
//   chars   — every (column, row) pair with a live source takes one warp. The
//             warp re-resolves the row per lane (one map read each — the map
//             line is shared, and the resolve is 3 registers), giving every
//             lane the (src, dst, size) tuple without any shuffle; the copy
//             then follows the stock string-parallel strategy exactly (uint4
//             interior between the alignment-safe bounds, scalar edges). A row
//             longer than `kBlockRowCopyThreshold` is published to shared
//             memory and copied by the whole block instead, so it does not
//             serialise across one warp's lanes. A group with NO published row
//             (every live row at or under the gate) is instead copied
//             char-parallel by the whole block: one thread per output byte,
//             the row found by a binary search over the group's staged
//             boundaries — the stock char-parallel strategy applied to the
//             already-scanned offsets, which removes the rows-per-warp
//             starvation on tall short-row tables. EVERY group first runs a
//             common boundary stage that stages the group's row boundaries and
//             (from the same scratch reads) publishes the offsets values. The
//             launch grid covers ONLY the unrouted cohort's flat rows: a
//             column routed to `cub::DeviceMemcpy::Batched`
//             (`chars_via_batched_memcpy`) contributes no group visits at all
//             — CUB owns its bytes, and `string_routed_metadata_kernel` owns
//             its offsets/masks/counts.
//   masks   — one thread per (column, 32-row window): the thread re-resolves
//             each row of its window and packs the gathered validity into one
//             destination mask word, popping the count into the column's
//             atomic valid counter. Routed columns are excluded entirely
//             (their words/counts ride the routed kernel); pre-set masks
//             (ALL_NULL/ALL_VALID) skip the phase (`dst_mask` nullptr).
//   offsets — for unrouted columns, the group loop's boundary stage publishes every flat
//             row's offsets value (the group's first warp, adjacent lanes on adjacent flat
//             rows, from the staged shared-memory boundaries) plus each column's N+1
//             sentinel; the per-entry flat copy loop of the original design is deleted,
//             removing the second full scratch traversal and its per-entry 64-bit div/mod.
//             Routed columns' offsets children are written by the routed kernel instead.
template <size_type BlockSize>
CUDF_KERNEL void string_payload_kernel(string_work_item const* items,
                                       size_type const* unrouted,  // unrouted cohort -> items
                                       int64_t num_unrouted,
                                       string_flat_layout layout,  // uint32 rows_per_col, num_rows
                                       size_type num_source_rows,
                                       bool check_bounds,
                                       bool negative_allowed,
                                       input_indexalator gather_map)
{
  // The chars phase coordinates a block group: one slot per warp, a common boundary stage
  // that also publishes the offsets values, then one of three copy strategies per group.
  static_assert(BlockSize % warp_size == 0);
  auto const global_tid  = static_cast<int64_t>(blockIdx.x) * BlockSize + threadIdx.x;
  auto const grid_stride = static_cast<int64_t>(gridDim.x) * BlockSize;
  auto const lane        = static_cast<size_type>(threadIdx.x % warp_size);

  // ---- chars: one warp per row at or below the block-copy gate; the whole block copies a
  // longer row; an all-short group is copied char-parallel by the whole block ----
  //
  // Rows are handed out per block group: the block's warps take consecutive flat slots and the
  // group bound is block-uniform, so every thread reaches the barriers below.
  auto const warps_per_block = static_cast<int64_t>(BlockSize / warp_size);
  auto const warp_in_block   = static_cast<int64_t>(threadIdx.x) / static_cast<int64_t>(warp_size);
  auto const thread_in_block = static_cast<int64_t>(threadIdx.x);
  __shared__ block_row_copy block_rows[BlockSize / warp_size];
  // The fan window: the group's row boundaries plus the tail sentinel closing the last row.
  __shared__ int64_t group_offsets[BlockSize / warp_size + 1];
  __shared__ size_type pub_col;       // warp 0's publish: the window's column
  __shared__ size_type pub_base_row;  // ...and the first flat row's position inside it

  auto const num_rows   = layout.num_rows;
  auto const total_rows = num_unrouted * static_cast<int64_t>(num_rows);
  for (int64_t group = static_cast<int64_t>(blockIdx.x); group * warps_per_block < total_rows;
       group += static_cast<int64_t>(gridDim.x)) {
    auto const flat = group * warps_per_block + warp_in_block;

    string_work_item const* item_ptr = nullptr;
    size_type row                    = 0;
    char const* src_ptr              = nullptr;
    char* dst_ptr                    = nullptr;
    int64_t size                     = 0;
    if (flat < total_rows) {
      // The flat domain is the UNROUTED cohort's (column, row) space: the cohort table maps
      // the local column index back to the items slot. Every visited column is kernel-owned
      // by construction (the host built the cohort), so no per-item route flag is consulted.
      item_ptr = &items[unrouted[static_cast<uint32_t>(flat) / num_rows]];
      row      = static_cast<size_type>(static_cast<uint32_t>(flat) % num_rows);
      auto const src =
        resolve_gathered_row(gather_map, row, num_source_rows, check_bounds, negative_allowed);
      if (src >= 0 and string_row_valid(*item_ptr, src)) {
        auto const s = static_cast<size_type>(src);
        size         = item_ptr->src_offsets[s + 1] - item_ptr->src_offsets[s];
        if (size > 0) {
          src_ptr = item_ptr->src_chars + item_ptr->src_offsets[s];
          dst_ptr = item_ptr->dst_chars + item_ptr->dst_scratch[row];
        }
      }
    }

    // Publish this warp's row for the block-cooperative copy; an empty slot is the no-work
    // marker and also the all-short vote input (the vote reads the slots below).
    auto const publishable = size > kBlockRowCopyThreshold;
    if (lane == 0) {
      block_rows[warp_in_block] =
        publishable ? block_row_copy{src_ptr, dst_ptr, size} : block_row_copy{};
    }
    auto const group_start = group * warps_per_block;
    auto const group_end   = cuda::std::min(group_start + warps_per_block, total_rows);
    // Common boundary stage, shared by every chars strategy: each warp's lane 0 stages the
    // +0 boundary of its OWN flat row (a dead or empty row stages its zero-delta boundary
    // like any other); a warp past the flat end stages INT64_MAX, the unreachable boundary
    // that keeps the fan's binary search inside the live slots even when a partial final
    // group straddles a column boundary (stale values from the previous group are NOT
    // monotone across it). Slot 0 of the whole domain holds flat row 0's +0 scratch entry.
    if (flat < total_rows) {
      if (lane == 0) { group_offsets[warp_in_block] = item_ptr->dst_scratch[row]; }
    } else if (lane == 0) {
      group_offsets[warp_in_block] = cuda::std::numeric_limits<int64_t>::max();
    }
    // Stage the TAIL boundary: the +1 scratch entry of the window's last live row closes its
    // span for the binary search. (The +0 boundary of the NEXT flat row is staged by the next
    // group's own warp, so the tail must come from this group's rows.) Every executed group
    // has a nonempty window, so this always runs.
    if (thread_in_block == 0) {
      auto const last_row = static_cast<size_type>(static_cast<uint32_t>(group_end - 1) % num_rows);
      group_offsets[warps_per_block] =
        items[unrouted[static_cast<size_type>(static_cast<uint32_t>(group_end - 1) / num_rows)]]
          .dst_scratch[last_row + 1];
    }
    // Hoist the window's column geometry for warp 0's offsets publish: ONE div/mod pair per
    // group (thread 0, group-uniform) replaces the per-lane pair below. Warp 0 cannot
    // broadcast by shuffle — its lanes diverge on the lane_flat bound — so the values ride
    // the boundary-stage shared memory instead.
    if (thread_in_block == 0) {
      pub_col = unrouted[static_cast<size_type>(static_cast<uint32_t>(group_start) / num_rows)];
      pub_base_row = static_cast<size_type>(static_cast<uint32_t>(group_start) % num_rows);
    }
    __syncthreads();

    // Offsets publish FROM the staged boundaries: the group's first warp stores one offsets
    // value per active lane, adjacent lanes taking adjacent flat rows — for a single-column
    // group that is 8 consecutive INT32 entries of one child (the per-warp-lane-0 pattern
    // was rejected: one 4-byte transaction per warp). This replaces the deleted flat offsets
    // loop, whose per-entry 64-bit div/mod re-read every scratch entry a second time from
    // global memory; here the values are already resident in shared memory. One writer per
    // entry: each flat row belongs to exactly one group and one lane, covering dead/empty
    // rows and long/mixed groups alike (their windows stage real boundaries too). A row that
    // ENDS its column (+1 == num_rows) also publishes that column's N+1 tail sentinel —
    // its +1 boundary is the NEXT column's row-0 boundary, staged by the next group, so the
    // sentinel is read from the scratch (one global load per column per call). Row 0's value
    // IS the row-0 sentinel: scratch[0] is both the +0 boundary of flat row 0 and the child's
    // first entry, written by the ordinary row publish above.
    if (warp_in_block == 0) {
      auto const lane_flat = group_start + static_cast<int64_t>(lane);
      // warp_size (32) exceeds the 8-row window: bound by group_end, or lanes 8+ read
      // out-of-bounds shared slots and publish rows other blocks own.
      if (lane_flat < group_end) {
        if (pub_base_row + static_cast<int64_t>(warp_size) <= num_rows) {
          // Uniform window: the only div/mod pair per group is the hoisted one above; the
          // sentinel ride (column-ending lane reads scratch[row+1]) is unchanged.
          auto const& pub_item              = items[pub_col];
          auto const pub_row                = pub_base_row + static_cast<int64_t>(lane);
          *(pub_item.dst_offsets + pub_row) = group_offsets[lane];
          if (pub_row + 1 == num_rows) {
            *(pub_item.dst_offsets + num_rows) = pub_item.dst_scratch[pub_row + 1];
          }
        } else {
          // Cross-column window (num_rows < warp_size): rows span two unrouted columns, so
          // the general per-lane div/mod path applies (mapped through the cohort table).
          auto const& pub_item =
            items[unrouted[static_cast<size_type>(static_cast<uint32_t>(lane_flat) / num_rows)]];
          auto const pub_row = static_cast<size_type>(static_cast<uint32_t>(lane_flat) % num_rows);
          *(pub_item.dst_offsets + pub_row) = group_offsets[lane];
          if (pub_row + 1 == num_rows) {
            *(pub_item.dst_offsets + num_rows) = pub_item.dst_scratch[pub_row + 1];
          }
        }
      }
    }

    // A fan group must be single-column (a per-column scratch slice restarts at each column
    // boundary, so a cross-column window is not one byte space) — the check is block-uniform
    // because num_rows and warps_per_block are.
    auto const single_column = (static_cast<uint32_t>(group_start) / num_rows) ==
                               (static_cast<uint32_t>(group_end - 1) / num_rows);
    // The fan writes every live byte of the group's span by binary search over the staged
    // boundaries; every cohort column owns its chars on the kernel path, so the vote needs
    // no route predicate.
    auto const any_long = thread_in_block < warps_per_block
                            ? block_rows[thread_in_block].size > kBlockRowCopyThreshold
                            : false;
    // __syncthreads_or is barrier-safe here: every thread participates (the predicate is
    // computed unconditionally), only the VALUE differs.
    if (__syncthreads_or(any_long or not single_column)) {
      // Block-cooperative rows, one at a time; rows above the gate are rare, so the serial
      // walk over the block's slots costs nothing on short-row data.
      for (int64_t slot = 0; slot < warps_per_block; ++slot) {
        auto const& row_copy = block_rows[slot];
        if (row_copy.size == 0) { continue; }
        copy_string_span(row_copy.dst,
                         row_copy.src,
                         row_copy.size,
                         thread_in_block,
                         static_cast<int64_t>(BlockSize));
      }
      // The short rows of a mixed group keep the warp copy (the fan did not run).
      if (not publishable) {
        copy_string_span(
          dst_ptr, src_ptr, size, static_cast<int64_t>(lane), static_cast<int64_t>(warp_size));
      }
    } else {
      // The group is all-short, single-column, and kernel-owned: fan every live byte across
      // all BlockSize threads over the boundaries staged above. Offsets were already published.
      copy_group_chars_char_parallel(
        items[unrouted[static_cast<size_type>(static_cast<uint32_t>(group_start) / num_rows)]],
        layout,
        group_start,
        group_end,
        group_offsets,
        warps_per_block,
        gather_map,
        num_source_rows,
        check_bounds,
        negative_allowed);
    }
    __syncthreads();  // the next group rewrites the slots
  }

  // ---- masks + valid counts: one thread per (column, 32-row window) ----
  // Routed columns are outside the cohort domain: string_routed_metadata_kernel owns their
  // masks and counts entirely.
  auto const words_per_col =
    util::div_rounding_up_safe(static_cast<int64_t>(num_rows), static_cast<int64_t>(warp_size));
  auto const total_windows = num_unrouted * words_per_col;
  for (int64_t w = global_tid; w < total_windows; w += grid_stride) {
    auto const& item = items[unrouted[static_cast<size_type>(w / words_per_col)]];
    if (item.dst_mask == nullptr) { continue; }
    auto const word = static_cast<size_type>(w % words_per_col);
    auto const row0 = static_cast<int64_t>(word) * warp_size;
    auto const limit =
      cuda::std::min<int64_t>(int64_t{warp_size}, static_cast<int64_t>(num_rows) - row0);
    bitmask_type word_bits = 0;
    for (int64_t s = 0; s < limit; ++s) {
      auto const src = resolve_gathered_row(gather_map,
                                            static_cast<size_type>(row0 + s),
                                            num_source_rows,
                                            check_bounds,
                                            negative_allowed);
      if (src >= 0 and string_row_valid(item, src)) { word_bits |= bitmask_type{1} << s; }
    }
    item.dst_mask[word] = word_bits;
    if (item.valid_count != nullptr) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*item.valid_count};
      ref.fetch_add(cuda::std::popcount(word_bits), cuda::std::memory_order_relaxed);
    }
  }

  // ---- offsets children: unrouted columns fully covered by the group loop's boundary
  // stage; routed columns by string_routed_metadata_kernel ----
  //
  // Every unrouted flat row's offsets value was published by its group's first warp —
  // adjacent lanes taking adjacent flat rows — from the staged boundaries in shared memory
  // (covering dead/empty rows and long/mixed groups alike), and every column's N+1 sentinel
  // rode the lane whose row ends its column (its row-0 sentinel is the row-0 publish
  // itself). No separate pass remains: deleting the old per-entry flat loop removed the
  // second full scratch traversal and its per-entry 64-bit div/mod.
}

// Row-parallel metadata for the columns routed to cub::DeviceMemcpy::Batched (the group walk
// above does no metadata work for them). One block owns ONE column's BlockSize-row tile per
// iteration; the scan has already written the final exclusive prefix into `dst_scratch`, so
// the kernel only transports it:
//   offsets — every live lane narrows one scratch entry (row or the N+1 sentinel) into the
//             child's width directly from global memory. Tiles never straddle a column
//             boundary (the loop is indexed per column tile), so the tile holding the last
//             row has the unique N+1 writer.
//   masks   — each warp ballots its 32 gathered validities into one word and writes it; the
//             ballot runs unconditionally so every warp lane participates, and warps fully
//             past the column's words store nothing (their ballot is all-zero anyway). No
//             partial OR into uninitialized storage.
//   counts  — each warp pops its word into shared memory, the CTA sums the block total over
//             one barrier, and thread 0 adds it to the column's atomic counter once per
//             block-tile (replacing the per-word device atomics of the generic mask phase).
// A second barrier closes each iteration so the next tile's shared-count writes cannot race
// the just-finished reduction. Offsets publication needs no barrier: the stores are
// independent global writes, stream-ordered before any consumer of the child.
template <size_type BlockSize>
CUDF_KERNEL void string_routed_metadata_kernel(string_work_item const* routed_items,
                                               size_type num_routed,
                                               size_type num_rows,
                                               size_type num_source_rows,
                                               bool check_bounds,
                                               bool negative_allowed,
                                               input_indexalator gather_map)
{
  static_assert(BlockSize % warp_size == 0);
  auto const lane            = static_cast<size_type>(threadIdx.x % warp_size);
  auto const warp            = static_cast<size_type>(threadIdx.x / warp_size);
  auto const warps_per_block = static_cast<size_type>(BlockSize / warp_size);
  __shared__ size_type warp_counts[BlockSize / warp_size];

  auto const tiles_per_col = static_cast<int64_t>(
    util::div_rounding_up_safe(static_cast<int64_t>(num_rows), static_cast<int64_t>(BlockSize)));
  auto const total_tiles = static_cast<int64_t>(num_routed) * tiles_per_col;
  for (int64_t t = static_cast<int64_t>(blockIdx.x); t < total_tiles;
       t += static_cast<int64_t>(gridDim.x)) {
    auto const col       = static_cast<size_type>(t / tiles_per_col);
    auto const col_row0  = (t % tiles_per_col) * static_cast<int64_t>(BlockSize);
    auto const tile_rows = static_cast<int64_t>(
      cuda::std::min<int64_t>(BlockSize, static_cast<int64_t>(num_rows) - col_row0));
    auto const& item = routed_items[col];
    auto const row   = col_row0 + static_cast<int64_t>(threadIdx.x);
    auto const live  = static_cast<int64_t>(threadIdx.x) < tile_rows;

    // offsets: one narrowed store per lane; the lane holding the last row owns the sentinel.
    if (live) {
      *(item.dst_offsets + row) = item.dst_scratch[row];
      if (row + 1 == num_rows) { *(item.dst_offsets + num_rows) = item.dst_scratch[num_rows]; }
    }

    // masks + counts: warp ballots then one CTA reduction. A warp fully past the tile
    // contributes no mask bits and a zero count (its dead lanes all vote false).
    auto const word_row0 = col_row0 + static_cast<int64_t>(warp) * warp_size;
    auto const src       = live ? resolve_gathered_row(gather_map,
                                                 static_cast<size_type>(row),
                                                 num_source_rows,
                                                 check_bounds,
                                                 negative_allowed)
                                : int64_t{-1};
    // Dead rows (src < 0) are invalid — the guard must precede the mask read or the bit
    // index would underflow on NULLIFY calls.
    auto const valid =
      src >= 0 and (item.src_mask == nullptr or
                    bit_is_set(item.src_mask, static_cast<size_type>(item.mask_bit_offset + src)));
    auto const word_bits = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) {
      // Dead-lane ballot bits are already 0, so the packed word is exact for tail words too
      // (rows past num_rows stay clear).
      if (item.dst_mask != nullptr and word_row0 < static_cast<int64_t>(num_rows)) {
        item.dst_mask[static_cast<size_type>(word_row0 / warp_size)] = word_bits;
      }
      warp_counts[warp] = cuda::std::popcount(word_bits);
    }
    __syncthreads();
    size_type block_valid = 0;
    for (size_type w = 0; w < warps_per_block; ++w) {
      block_valid += warp_counts[w];
    }
    if (threadIdx.x == 0 and item.valid_count != nullptr) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*item.valid_count};
      ref.fetch_add(block_valid, cuda::std::memory_order_relaxed);
    }
    __syncthreads();  // the next tile rewrites warp_counts; the reads above must finish first
  }
}

// Tile width for regimes where sampling is skipped; nullopt means "sample the map". Zero/one
// fused columns keep K1; otherwise the unsampled choice is K2 while the two-column working set
// plus the map fit half of L2 (a failed L2 query reports 0 bytes and takes the safe K1).
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

// These carve-layout helpers sit at `cudf::detail` scope, NOT the anonymous namespace:
// their concept-conformance members are odr-used only inside type-erased vtables built in
// librmm, and at internal linkage nvcc flags them "#177-D declared but never referenced"
// (same reason `column_descriptor` sits at this scope).

// Byte needs and slice positions of one destination strings column: the offsets child
// (whose width `create_offsets_child_column` selects — 4 or 8 bytes per element) plus the
// chars buffer. Every carve requests its bytes rounded up to the carve granularity and
// starts 256-aligned, so the running total tracks the bump allocator's position exactly and
// each slice start is 256-aligned. The offsets child and chars slice of one column must be
// carved in THAT order, and columns in `live` order, matching `carve_strings_buffers`.
struct strings_destination_layout {
  static constexpr std::size_t slice_alignment{256};  // CUDA's allocation granularity

  std::size_t total_bytes{0};

  struct slice_offsets {
    std::size_t offsets;
    std::size_t chars;
  };

  // Reserves one column's slices: offsets child (INT32 or INT64 per `chars_bytes`'s
  // threshold, mirroring `create_offsets_child_column`'s width selection) then chars;
  // `chars_bytes == 0` still round-ups the chars region so the reservation mirrors the bump
  // sequence (a zero-byte chars buffer is still constructed).
  slice_offsets carve(int64_t chars_bytes, size_type num_rows)
  {
    auto const threshold     = cudf::strings::detail::get_offset64_threshold();
    auto const offsets_elem  = chars_bytes < threshold ? sizeof(int32_t) : sizeof(int64_t);
    auto const offsets_bytes = static_cast<std::size_t>(num_rows + 1) * offsets_elem;
    slice_offsets offsets;
    offsets.offsets = cudf::util::round_up_safe(total_bytes, slice_alignment);
    total_bytes     = offsets.offsets + offsets_bytes;
    offsets.chars   = cudf::util::round_up_safe(total_bytes, slice_alignment);
    total_bytes = cudf::util::round_up_safe(offsets.chars + static_cast<std::size_t>(chars_bytes),
                                            slice_alignment);
    return offsets;
  }
};

// Stream-ordered resource handing out successive aligned slices of ONE pre-allocated device
// buffer, so a call's destination columns (string offsets/chars through
// `strings_destination_layout`, fixed-width masks/data through `fixed_width_destination_layout`)
// draw from a single upstream allocate/free pair instead of one pair per buffer. Allocation
// bumps a pointer; deallocation is a no-op — the arena block is freed stream-ordered when the
// resource itself dies. Every destination column keeps its slices alive independently: each
// buffer stores a value copy of the `cuda::mr::shared_resource` wrapper created for the call,
// and all copies share the one payload (bump pointer included), so the block outlives the last
// column. The upstream reference is captured as an owning `any_resource` deep copy, so a
// payload that outlives the call still deallocates through a live resource object, matching
// how stock columns capture their allocation resource. Because allocation never frees, every
// slice must be carved exactly once, in the reserving layout's order (strings: offsets child,
// then chars, per column, columns in live order; fixed-width: mask before data per column,
// struct node mask before its children, in `fused_indices` pre-order — see
// `fixed_width_destination_layout`).
class carve_memory_resource {
 public:
  static constexpr std::size_t slice_alignment{strings_destination_layout::slice_alignment};

  carve_memory_resource(rmm::device_async_resource_ref upstream,
                        cuda::stream_ref stream,
                        std::size_t total_bytes)
    : _upstream{upstream},  // deep copy: the payload may outlive the call's resource refs
      _stream{stream},
      _size{total_bytes},
      _arena{total_bytes > 0 ? _upstream.allocate(stream, total_bytes, slice_alignment) : nullptr}
  {
  }

  // A zero-sized carve never allocated (`_arena == nullptr`); deallocating it would still
  // trip instrumented adaptors' counters (e.g. statistics_resource_adaptor), so skip it.
  ~carve_memory_resource()
  {
    if (_size > 0) { _upstream.deallocate(_stream, _arena, _size, slice_alignment); }
  }

  carve_memory_resource(carve_memory_resource const&)            = delete;
  carve_memory_resource& operator=(carve_memory_resource const&) = delete;

  bool operator==(carve_memory_resource const& other) const noexcept
  {
    return this == std::addressof(other);
  }
  bool operator!=(carve_memory_resource const& other) const noexcept { return !(*this == other); }

  // Tag introspection only: CCCL queries this property in unevaluated concept checks, so the
  // function body never runs (precedent: `host_memory.cpp`'s pool resource).
  [[maybe_unused]] friend void get_property(carve_memory_resource const&,
                                            cuda::mr::device_accessible) noexcept
  {
  }

  // Buffer constructions request the full carve granularity (via the allocation-alignment
  // property or the explicit alignment argument), so the bump sequence matches the layout
  // reservation exactly and every slice keeps 256-byte alignment for its widest element.
  [[nodiscard]] void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    CUDF_EXPECTS(alignment <= slice_alignment, "Alignment exceeds the carve granularity");
    auto const offset = cudf::util::round_up_safe(_used, alignment);
    CUDF_EXPECTS(offset + bytes <= _size, "Destination arena exhausted");
    _used = offset + bytes;
    return static_cast<std::byte*>(_arena) + offset;
  }

  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}

  [[nodiscard]] void* allocate(cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) noexcept {}

  // The arena's base pointer, for a call-site-wide fill over the whole block (the preset-mask
  // bands' single memset — padding gaps inside the block are filled too, which no column
  // observes). Null on a zero-sized carve (nothing was allocated); callers guard on size.
  void* arena_base() const noexcept { return _arena; }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> _upstream;  // owns a deep copy
  cuda::stream_ref _stream;  // the arena block's stream-ordered free stream
  std::size_t _size;
  void* _arena;
  std::size_t _used{0};
};

// The buffers below type-erase through `cuda::mr::resource_with` — the concept surface is the
// contract, checked here so an interface drift fails at compile time, not at the erase point.
static_assert(cuda::mr::resource_with<carve_memory_resource, cuda::mr::device_accessible>,
              "carve_memory_resource must satisfy the stream-ordered device resource concepts");

// Byte needs of one destination fixed-width column: the Arrow-padded mask words (the sizing
// `create_null_mask` itself uses) plus the data payload. The reservation pass and the carve
// sites MUST visit columns in the same pre-order (top-level columns in `fused_indices` order;
// within a column: mask before data, struct node mask before its children recurse, dictionary
// mask before its indices) and consume the same round-up steps, so the summed total equals the
// bump pointer's final position exactly — a divergence trips the carve resource's loud
// `arena exhausted` expect rather than corrupting memory.
struct fixed_width_destination_layout {
  static constexpr std::size_t slice_alignment{256};  // CUDA's allocation granularity

  std::size_t total_bytes{0};

  // Reserves one column's mask region (`num_rows` words at the Arrow 64-word padding, zero
  // when the column carries no mask) and data region (`num_rows` fixed-width elements); each
  // region is rounded up to the carve granularity, mirroring the bump steps the constructions
  // will take.
  void carve(size_type num_rows, std::size_t data_bytes, bool with_mask)
  {
    auto const mask_words =
      with_mask ? cudf::bitmask_allocation_size_words(num_rows) : std::size_t{0};
    total_bytes =
      cudf::util::round_up_safe(total_bytes + mask_words * sizeof(bitmask_type), slice_alignment);
    total_bytes = cudf::util::round_up_safe(total_bytes + data_bytes, slice_alignment);
  }
};

// Preset-mask fill reservation for ONE of the two fill patterns (ALL_VALID 0xff / ALL_NULL
// 0x00). During the build walk each preset destination mask is instead carved from a pattern's
// staging arena; after the walk ONE memset per non-empty arena covers every preset fill in that
// pattern, replacing the per-mask cudaMemsetAsync submissions (the measured ~60-mask drip-feed
// class). The staging arenas are created alongside the fw carve arena — slices outside it, so
// the fw reservation mirror above is untouched. The slice sizing (`dst_mask_plan::mask_bytes`)
// MUST track `fixed_width_destination_layout::carve`'s mask sizing exactly
// (`bitmask_allocation_size_words`) so the fills stay byte-identical to per-mask memsets,
// padding words included.
struct preset_mask_fill_layout {
  static constexpr std::size_t slice_alignment{256};  // CUDA's allocation granularity

  std::size_t total_bytes{0};

  // Reserves one preset mask's padded word extent. `num_rows` is the DESTINATION row count
  // (sliced sources included), matching the carve sites' sizing.
  void carve(size_type num_rows)
  {
    total_bytes = cudf::util::round_up_safe(
      total_bytes + cudf::bitmask_allocation_size_words(num_rows) * sizeof(bitmask_type),
      slice_alignment);
  }
};

// Builds ONE destination strings column's carved buffers: the offsets child (a full column
// whose data buffer is a carve slice) and the chars buffer (a carve slice). Both store a value
// copy of the shared carve resource, so the column keeps its slices alive independently and
// the arena block is freed stream-ordered when the last shared copy dies. The offsets child is
// carved FIRST (matching `strings_destination_layout::carve`'s order), then chars. A zero-byte
// chars buffer constructs to a null data pointer exactly as stock does (RMM skips zero-size
// allocations) — pass `chars_bytes == 0` and `chars` is the empty-chars form. The destination
// mask is NOT carved: string masks keep the stock regime (allocated in `build_string`), since
// the strings carve arena only exists after the scan readback that sizes the chars.
struct carved_strings_buffers {
  std::unique_ptr<column> offsets;  // offsets child column, data carved (INT32 or INT64)
  rmm::device_buffer chars;         // chars slice; empty (null data) when chars_bytes == 0
};

carved_strings_buffers carve_strings_buffers(
  size_type num_rows,
  int64_t chars_bytes,
  cuda::mr::shared_resource<carve_memory_resource> const& carve_mr,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref output_mr)
{
  auto const threshold = cudf::strings::detail::get_offset64_threshold();
  if (!cudf::strings::detail::is_large_strings_enabled()) {
    // Stock `create_offsets_child_column` parity: the same overflow contract.
    CUDF_EXPECTS(
      chars_bytes < threshold, "Size of output exceeds the column size limit", std::overflow_error);
  }
  auto const offsets_type =
    chars_bytes < threshold ? data_type{type_id::INT32} : data_type{type_id::INT64};
  auto const offsets_bytes = (static_cast<std::size_t>(num_rows) + 1) *
                             (chars_bytes < threshold ? sizeof(int32_t) : sizeof(int64_t));

  // Construction order = reservation order: offsets child first, then chars. Each buffer
  // stores a value copy of the one shared carve resource and asks it for its slice (the
  // resource bumps once per buffer at the full carve granularity), so the carve sequence
  // matches the reservation exactly. `rmm::device_buffer`'s alignment-argument form carries
  // the carve granularity; `column`'s generic ctor forwards it into `_data`, so the child
  // keeps genuine `rmm::device_buffer` ownership of its slice.
  auto offsets_col = std::make_unique<column>(
    offsets_type,
    static_cast<size_type>(num_rows) + 1,
    rmm::device_buffer{offsets_bytes, carve_memory_resource::slice_alignment, stream, carve_mr},
    cudf::detail::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, output_mr),
    0,
    std::vector<std::unique_ptr<column>>{});
  auto chars = rmm::device_buffer{static_cast<std::size_t>(chars_bytes),
                                  carve_memory_resource::slice_alignment,
                                  stream,
                                  carve_mr};

  return carved_strings_buffers{std::move(offsets_col), std::move(chars)};
}

// One device staging upload drawn from the staging carve arena: a fresh `rmm::device_uvector`
// allocation against the arena's bump allocator (its deallocation is a no-op — the single
// upstream pair frees stream-ordered when the arena dies). Sizing the arena is a closed-form
// computation at worst case, so the caller's only obligation is that the reservation covers
// every carve it makes; any shortfall trips the carve resource's loud `arena exhausted` expect.
template <typename T>
rmm::device_uvector<std::remove_cv_t<T>> carve_device_vector(
  host_span<T> source_data,
  cuda::stream_ref stream,
  cuda::mr::shared_resource<carve_memory_resource> const& carve_mr)
{
  rmm::device_uvector<std::remove_cv_t<T>> ret(source_data.size(), stream, carve_mr);
  if (source_data.size() > 0) {
    cudf::detail::cuda_memcpy_async<std::remove_cv_t<T>>(ret, source_data, stream);
  }
  return ret;
}

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
  // Routed entry: one host-side type pass gathers the flattened descriptor count (what the
  // kernel will actually receive — struct leaves and mask-only nodes, dictionary indices) and
  // the widest fused data element for the gate; when sampling is warranted its copies are
  // ENQUEUED here and RESOLVED inside the body after the destination allocations, so they hide
  // behind that work and the body's single pre-launch synchronization.
  size_type const num_fused_cols      = fused_descriptor_count(source_table);
  size_type const max_fused_elem_size = max_fused_data_element_size(source_table);
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

namespace {

// Destination mask plan shared by every node on the descriptor path. A destination mask is
// needed iff the source is all-null (ALL_NULL pre-set), any gathered row can be null (NULLIFY
// policy or a null-bearing source — the fused kernel then writes it and reports the valid
// count), or the source is nullable with zero nulls (ALL_VALID pre-set). The plan carries the
// state and fill semantics only; the mask BUFFER is created by the caller through whichever
// allocation regime the column rides (the carve arenas or stock `create_null_mask`), so
// planning never allocates and the builder's carve reservations can be summed before the
// build walk starts. The policy inputs are host-side view accessors only — the unsliced null
// counts are cached values, never a device readback.
struct dst_mask_plan {
  mask_state state{mask_state::UNALLOCATED};
  // True when the mask must be memset here (ALL_VALID/ALL_NULL); kernel-written
  // (UNINITIALIZED) masks are left for the kernel, which writes every word.
  bool preset{};
  bitmask_type* kernel_mask{};  // non-null iff the fused kernel writes the mask
  size_type null_count{};       // host-visible count; 0 placeholder when kernel-masked
  bool skip_data{};             // all-null source: mask pre-set ALL_NULL, no descriptor pushed
  // Preset masks only: which pattern's staging arena the mask slices from (0 = ALL_VALID,
  // 1 = ALL_NULL) and its padded byte extent, so the carve consumes the pre-walk reservation
  // and the post-walk memset fills the matching pattern. `use_band` = the preset mask rides
  // the band; false (below `kMaskBandPresetGate`) takes the stock per-mask carve + memset.
  std::size_t band{};
  std::size_t mask_bytes{};
  bool use_band{};
};

// One preset-mask fill pattern's staging arena: a carve_memory_resource whose slices are
// filled by a SINGLE post-walk memset (pattern byte 0xff for ALL_VALID, 0x00 for ALL_NULL),
// plus the reservation that sized it. Index into the builder's two-element array with the
// pattern (see `dst_mask_plan::band`). The arena is created AFTER the builder's aggregate
// initialization (`shared_resource` has no default constructor), so — like the fw carve
// arena — it is owned by the enclosing scope: the member is a pointer null until the
// post-walk pass sets it; every use site runs after that point, and destination columns
// hold their own value copies so the wrapper's lifetime is not entangled with the builder's.
struct mask_band {
  preset_mask_fill_layout layout;
  std::unique_ptr<cuda::mr::shared_resource<carve_memory_resource>> mr;
};

// The exact condition under which `plan_destination_mask` hands the fused kernel a
// destination mask (and the builder then records a masked position). Host-side view
// accessors only — cached null counts, never a device readback — so the count estimator
// below evaluates it without planning (planning would allocate a device mask).
bool kernel_writes_dst_mask(column_view const& src, bool nullify)
{
  bool const all_null = src.size() > 0 and src.null_count() == src.size();
  return not all_null and (nullify or src.null_count() > 0);
}

// Exact descriptor/kernel-masked counts the build pass emits for one descriptor-path source,
// mirroring `build_struct`/`build_dictionary`/`emit_fixed_width`: a fixed-width leaf or a
// top-level dictionary emits one data descriptor, a struct node adds one MASK-ONLY descriptor
// iff the node's mask is kernel-written, and struct children recurse. Mirroring the recursion
// (rather than a closed formula) keeps the counts correct by construction when the builder's
// emission rules change; KEEP IN SYNC with the `plan.kernel_mask != nullptr` branch of
// `plan_destination_mask` — both must agree on when the kernel writes a mask.
struct descriptor_counts {
  size_type descriptors;
  size_type kernel_masked;
};

descriptor_counts estimate_descriptor_counts(column_view const& src, bool nullify)
{
  if (src.type().id() == type_id::STRING) { return {0, 0}; }  // strings ride the N3 fast path
  if (src.type().id() == type_id::STRUCT) {
    size_type descriptors = kernel_writes_dst_mask(src, nullify) ? 1 : 0;
    size_type masked      = descriptors;
    for (size_type i = 0; i < src.num_children(); ++i) {
      auto const child = estimate_descriptor_counts(src.child(i), nullify);
      descriptors += child.descriptors;
      masked += child.kernel_masked;
    }
    return {descriptors, masked};
  }
  // Fixed-width leaf or top-level dictionary: one data descriptor carrying the mask iff
  // kernel-written, none when the source is all-null (the builder takes the skip_data path).
  // (A dictionary never sits below a struct on this path: a DICTIONARY32 struct child fails
  // `is_fully_fixed_width_struct`.)
  if (src.size() > 0 and src.null_count() == src.size()) { return {0, 0}; }
  auto const masked = kernel_writes_dst_mask(src, nullify) ? 1 : 0;
  return {1, masked};
}

// The mask plan is evaluated against the DESTINATION row count (`num_rows`,
// i.e. `gather_map.size()`), not `src.size()`: a sliced source has
// `src.size() < num_rows`, so an all-null sliced source must still mark all
// `num_rows` destination rows null. Callers must therefore pass the call's
// `num_rows` even though only `null_count` consumes it.
dst_mask_plan plan_destination_mask(column_view const& src,
                                    bool nullify,
                                    size_type num_rows,
                                    bool bands_enabled)
{
  dst_mask_plan plan;
  bool const all_null = src.size() > 0 and src.null_count() == src.size();
  if (all_null) {
    // Every row is null regardless of the map: pre-set ALL_NULL.
    plan.state      = mask_state::ALL_NULL;
    plan.preset     = true;
    plan.null_count = num_rows;
    plan.skip_data  = true;
    plan.band       = 1;
  } else if (nullify or src.null_count() > 0) {
    // The kernel writes this mask; the caller assigns `kernel_mask` from whichever mask
    // buffer it constructs (carved or stock).
    plan.state = mask_state::UNINITIALIZED;
  } else if (src.nullable()) {
    // Nullable source with zero nulls: ALL_VALID fast path, no kernel mask work.
    plan.state  = mask_state::ALL_VALID;
    plan.preset = true;
  }
  if (plan.preset) {
    plan.mask_bytes = cudf::bitmask_allocation_size_words(num_rows) * sizeof(bitmask_type);
    plan.use_band   = bands_enabled;
  }
  return plan;
}

// Applies `plan`'s fill semantics to a freshly created destination mask. A preset mask fills
// its padded extent (`plan.mask_bytes`) with its pattern byte: banded — carved from its
// pattern's staging arena, filled by the post-walk band memset — when the caller took the band
// path (`bands_enabled`), else the per-mask memset (the stock form). A kernel-written
// UNINITIALIZED mask is left untouched; UNALLOCATED must never reach this helper. The caller
// assigns `plan.kernel_mask` from the buffer before pushing any descriptor.
void fill_destination_mask(dst_mask_plan& plan,
                           cuda::device_buffer<bitmask_type>& mask,
                           bool bands_enabled,
                           cuda::stream_ref stream)
{
  if (plan.state == mask_state::UNALLOCATED) { return; }
  plan.kernel_mask = plan.state == mask_state::UNINITIALIZED ? mask.data() : nullptr;
  if (not plan.preset) { return; }
  if (bands_enabled) { return; }  // the post-walk band memset fills it
  CUDF_CUDA_TRY(cudaMemsetAsync(
    mask.data(), plan.state == mask_state::ALL_VALID ? 0xff : 0x00, plan.mask_bytes, stream.get()));
}

// Builds ONE destination fixed-width column whose mask and data buffers are slices of the
// call-wide carve arena. The mask is created FIRST (matching `fixed_width_destination_layout::
// carve`'s mask-then-data order), then the data slice is carved; a banded preset mask slices
// the matching pattern's staging arena (its fill rides the post-walk band memset), a
// stock-path preset mask carves the fw arena (filled here per-mask), and a kernel-written mask
// slices the fw arena untouched. `plan.kernel_mask` is assigned from the mask buffer so the
// caller can push its descriptor with it. Every slice stores a value copy of the shared
// resource it carves from, so the column keeps its slices alive independently and an arena
// block is freed stream-ordered when the last shared copy dies. A zero-size data slice
// (num_rows == 0) constructs to a null data pointer exactly as stock does (RMM skips zero-size
// allocations).
std::unique_ptr<column> make_carved_fixed_width(
  column_view const& src,
  size_type num_rows,
  dst_mask_plan& plan,
  cuda::mr::shared_resource<carve_memory_resource> const& carve_mr,
  mask_band* bands,
  bool bands_enabled,
  cuda::stream_ref stream)
{
  auto const data_bytes = static_cast<std::size_t>(num_rows) * cudf::size_of(src.type());

  // Construction order = reservation order: mask first, then data. The env property pins the
  // mask buffer's allocation alignment to the carve granularity (the default would be the
  // element's natural 4-byte alignment); `rmm::device_buffer`'s alignment-argument form carries
  // it for the data slice, and `column`'s generic ctor forwards both into their owned members.
  auto const mask_env =
    cuda::std::execution::prop{cuda::allocation_alignment, carve_memory_resource::slice_alignment};
  auto mask_slice =
    plan.use_band
      ? cuda::device_buffer<bitmask_type>{stream,
                                          *bands[plan.band].mr,
                                          plan.mask_bytes / sizeof(bitmask_type),
                                          cuda::no_init,
                                          mask_env}
      : cuda::device_buffer<bitmask_type>{
          stream,
          carve_mr,
          plan.state == mask_state::UNALLOCATED ? 0 : cudf::bitmask_allocation_size_words(num_rows),
          cuda::no_init,
          mask_env};
  fill_destination_mask(plan, mask_slice, bands_enabled, stream);
  auto data_slice =
    rmm::device_buffer{data_bytes, carve_memory_resource::slice_alignment, stream, carve_mr};

  auto dst = std::make_unique<column>(src.type(),
                                      num_rows,
                                      std::move(data_slice),
                                      std::move(mask_slice),
                                      plan.state == mask_state::ALL_NULL ? num_rows : 0,
                                      std::vector<std::unique_ptr<column>>{});
  return dst;
}

// Plans the descriptor-path columns of one gather call and builds their destination columns.
//
// Structs are flattened recursively: every fixed-width leaf under a row-preserving struct chain
// becomes a data descriptor carrying the leaf's OWN gathered mask, and every struct node
// becomes a MASK-ONLY descriptor (skip_data, null data pointers) carrying the node's OWN
// gathered mask — stock never ANDs ancestor validity into descendants, so each node keeps its
// own mask and the output hierarchy is reconstructed with `create_structs_hierarchy`, which
// (unlike `make_structs_column`) does not superimpose parent nulls into children. Dictionary
// columns contribute one data descriptor over their annotated fixed-width indices (source
// address = indices base + parent offset, source mask = the dictionary parent's own mask) and
// are assembled with the adopting `make_dictionary_column` over a deep-copied key catalog:
// full key order, keys unreferenced by the gathered indices retained, no dedup/remap/re-encode.
//
// An all-null source (leaf, struct node, or dictionary parent) still builds its destination
// column (ALL_NULL mask, undefined payload) but emits NO descriptor: the kernel's per-row map
// load and source-mask read would be dead work. A struct whose children are all skipped keeps
// its own MASK-ONLY descriptor as the only remaining work; when nothing is left for the launch
// the caller's `descriptors.empty()` guard skips it.
//
// Offset invariant: for a source path of views v1(top)...vk(leaf), each view's `offset()` is
// applied exactly once — a leaf adds the accumulated ancestor offsets to `data<T>()` (which
// already carries the leaf view's own offset, zero for raw children) and to its mask bit
// offset; a node's own mask descriptor uses the accumulated offset of every view down to and
// including the node. Raw struct children and dictionary index children always view their full
// underlying extent (offset 0), matching `get_sliced_child`/`get_indices_annotated` without
// paying their null-count readback.
struct descriptor_path_builder {
  std::vector<column_descriptor>& descriptors;
  std::vector<size_type>& kernel_masked_positions;  // positions of kernel-masked descriptors
  std::vector<column*>& kernel_masked_columns;      // slot i's count target (columns[i])
  size_type num_rows;
  bool nullify;
  // The call-wide band-path decision (`kMaskBandPresetGate`): banded preset-mask fills vs the
  // stock per-mask memsets. Mirrored into every `dst_mask_plan` the builder plans.
  bool bands_enabled;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref output_mr;
  rmm::device_async_resource_ref temp_mr;
  fixed_width_destination_layout fw_layout;  // reservation, advanced during the build walk
  // One staging arena per preset-mask fill pattern (index 0 = ALL_VALID 0xff, 1 = ALL_NULL
  // 0x00). Slices are handed out during the build walk; each arena's layout was reserved by
  // the pre-walk pass and ONE memset per non-empty arena (issued post-walk, before the launch)
  // fills every preset mask in that pattern.
  mask_band mask_bands[2];
  // Set once the fw arena exists (after the build walk); the fw carve class checks it so a
  // misordered construction that would carve after teardown fails loudly.
  cuda::mr::shared_resource<carve_memory_resource>* fw_carve_mr{};

  // String-keyed dictionary columns defer their whole assembly to `assemble_staged_keys`:
  // one batched D2H copy of every catalog's offsets plus a single synchronization replaces
  // copy_slice's per-column blocking offset reads (two per string-keyed catalog). A staged
  // record holds everything the deferred factory call needs; the destination slot is a pointer
  // into `destination`, which batch_gather_impl never resizes after construction, so the
  // pointers stay valid until the assembly fills them (the builder outlives the assembly).
  struct staged_keys {
    column_view keys;                 // full keys view, including the parent's offset
    std::unique_ptr<column> offsets;  // deep copy of the full offsets child, unadjusted
    std::unique_ptr<column> indices;  // gathered indices; moves into the deferred factory call
    // The parent's destination mask, created and filled in `build_dictionary` (its pointer
    // already rides the pushed descriptor) and moved through here into the deferred factory
    // call. An empty buffer means the plan was UNALLOCATED.
    std::optional<cuda::device_buffer<bitmask_type>> dst_mask;  // when allocated
    size_type null_count{};  // host-visible parent null count (0 placeholder when kernel-masked)
    size_type keys_offset;   // keys view's own offset
    size_type keys_size;     // keys view's own size
    std::unique_ptr<column>* dst_slot{};  // destination[index] slot to fill
    // Position in kernel_masked_columns when the parent mask is kernel-written; the target
    // pointer is patched to the final column after the deferred assembly.
    std::optional<size_t> kernel_slot;
  };
  std::vector<staged_keys> pending_dictionaries;

  // Top-level string columns on the N3 fast path: one record per column, held
  // so the launch block can finish their destinations (allocate chars from the
  // totals, build the offsets children) after the virtual scan's readback. A
  // record owns every device allocation its destination needs; `dst_slot`
  // points into `destination`, which is never resized after construction. On
  // the carve path (`assemble_strings`'s live loop) the offsets child and chars
  // are slices of ONE shared call-wide arena; the buffers' stored copies of the
  // carve resource keep the arena alive for as long as any column needs it.
  struct staged_string {
    column_view src;                  // full source view (offset applied by the offsets iterator)
    std::unique_ptr<column> offsets;  // destination offsets child (INT32 or INT64, allocated late)
    rmm::device_buffer chars;         // destination chars buffer (allocated late)
    std::optional<cuda::device_buffer<bitmask_type>> dst_mask;  // when allocated
    size_type null_count{};  // host-visible count; 0 placeholder when kernel-masked
    std::unique_ptr<column>* dst_slot{};
    // True when the payload kernel writes this column's mask: the kernel then
    // needs the mask pointer and the per-column atomic counter, and assembly
    // takes the count from the deferred string_counts readback (never
    // kernel_masked_columns — the descriptor readback's slots must stay
    // descriptor-only).
    bool kernel_mask_written{false};
  };
  std::vector<staged_string> pending_strings;

  size_type push_descriptor(column_descriptor const& desc)
  {
    auto const position = static_cast<size_type>(descriptors.size());
    descriptors.push_back(desc);
    return position;
  }

  // Marks the descriptor at `position` as kernel-masked and returns the target-patch token.
  // Compact `valid_counts` slots are assigned after the build loop once the caller has
  // stable-partitioned the descriptors masked-first, so push order must stay natural. The
  // target column may arrive later (the owning factory owns it first); `set_target` patches
  // the slot after the column is assembled.
  size_t record_kernel_masked(size_type position)
  {
    kernel_masked_positions.push_back(position);
    kernel_masked_columns.push_back(nullptr);
    return kernel_masked_columns.size() - 1;
  }

  void set_target(size_t slot, column* target) { kernel_masked_columns[slot] = target; }
  // Batches the per-catalog offsets reads into ONE D2H staged copy plus ONE synchronization,
  // then assembles every deferred string-keyed dictionary column from the staged host offsets
  // and fills its destination slot. Called once per build walk, before the fused launch;
  // `output_mr` owns the produced columns' buffers.
  void assemble_staged_keys()
  {
    if (pending_dictionaries.empty()) { return; }

    // One pinned staging of every catalog's offsets child, copied in a single batched D2H
    // and read only after the single synchronization below. Catalog i's elements live at
    // `spans[i]` in the flat staging; per-catalog width is recorded because the offsets
    // width follows the chars size (INT64 only past the offset64 threshold), so a mixed
    // batch is legal. Spans are width-aligned so an 8-byte catalog is never read
    // misaligned when a narrower one lands at an odd 4-byte multiple.
    std::vector<size_t> spans;
    std::vector<bool> wide;
    spans.reserve(pending_dictionaries.size());
    wide.reserve(pending_dictionaries.size());
    size_t total_bytes = 0;
    for (auto const& staged : pending_dictionaries) {
      bool const is_wide = staged.offsets->type().id() == type_id::INT64;
      wide.push_back(is_wide);
      total_bytes = cudf::util::round_up_safe(total_bytes, sizeof(int64_t));
      spans.push_back(total_bytes);
      total_bytes +=
        (static_cast<size_t>(staged.keys_size) + 1) * (is_wide ? sizeof(int64_t) : sizeof(int32_t));
    }

    auto staging = cudf::detail::make_pinned_vector_async<std::byte>(total_bytes, stream);
    for (size_t i = 0; i < pending_dictionaries.size(); ++i) {
      auto const& staged = pending_dictionaries[i];
      auto const bytes   = wide[i] ? sizeof(int64_t) : sizeof(int32_t);
      auto const count   = static_cast<size_t>(staged.keys_size) + 1;
      auto const* src    = staged.offsets->view().head<std::byte>() +
                        static_cast<std::size_t>(staged.keys_offset) * bytes;
      CUDF_CUDA_TRY(
        cudf::detail::memcpy_async(staging.data() + spans[i], src, count * bytes, stream));
    }
    cudf::detail::sync_stream(stream);
    auto const staged_bytes = staging.data();
    // Reads the staged copy at the span-local element index (index 0 is the catalog's
    // element at its own `keys_offset`); absolute child indices never index the staging.
    auto const read_offset = [&](size_t catalog, size_type index) -> int64_t {
      auto const bytes = wide[catalog] ? sizeof(int64_t) : sizeof(int32_t);
      auto const* p    = staged_bytes + spans[catalog] + static_cast<std::size_t>(index) * bytes;
      return wide[catalog] ? static_cast<int64_t>(*(reinterpret_cast<int64_t const*>(p)))
                           : static_cast<int64_t>(*(reinterpret_cast<int32_t const*>(p)));
    };

    for (size_t i = 0; i < pending_dictionaries.size(); ++i) {
      auto& staged            = pending_dictionaries[i];
      auto const chars_offset = staged.keys_offset == 0 ? 0L : read_offset(i, 0);
      auto const keys_bytes   = read_offset(i, staged.keys_size) - chars_offset;

      // The staged offsets copy holds the full underlying child; the copy needs an OWNED
      // column of exactly keys_size + 1 elements starting at keys_offset, rebased so element
      // 0 is zero — the same deep-copy-and-rebase copy_slice performs. `column(column_view)`
      // materializes the slice (honoring the view's offset), then the transform shifts it.
      auto out_offsets = std::make_unique<column>(
        cudf::detail::slice(staged.offsets->view(),
                            {staged.keys_offset, staged.keys_offset + staged.keys_size + 1},
                            stream)
          .front(),
        stream,
        output_mr);
      if (chars_offset > 0) {
        // Rebasing shifts every staged offset down by `chars_offset` so the copy's own
        // offsets start at zero; element 0 (the original left bound) becomes 0 exactly.
        auto const in =
          cudf::detail::offsetalator_factory::make_input_iterator(out_offsets->view(), 0);
        auto const out =
          cudf::detail::offsetalator_factory::make_output_iterator(out_offsets->mutable_view());
        thrust::transform(
          rmm::exec_policy_nosync(stream, temp_mr),
          in,
          in + staged.keys_size + 1,
          out,
          cuda::proclaim_return_type<int64_t>(
            [chars_offset] __device__(int64_t offset) { return offset - chars_offset; }));
      }
      // `keys.head<char>()` is the chars base (a strings column's data buffer IS the chars;
      // `chars_begin` does not apply the parent offset, the offsets carry the byte addressing).
      auto const* chars_src = staged.keys.head<char>() + chars_offset;
      auto chars_buffer =
        rmm::device_buffer{chars_src, static_cast<std::size_t>(keys_bytes), stream, output_mr};

      auto keys_mask = cudf::detail::copy_bitmask(staged.keys, stream, output_mr);
      auto keys_col =
        cudf::make_strings_column(staged.keys_size,
                                  std::move(out_offsets),
                                  std::move(chars_buffer),
                                  staged.keys.null_count(),
                                  cuda::device_buffer<cudf::bitmask_type>{std::move(keys_mask)});
      *staged.dst_slot =
        make_dictionary_column(std::move(keys_col),
                               std::move(staged.indices),
                               cuda::device_buffer<cudf::bitmask_type>{std::move(*staged.dst_mask)},
                               staged.null_count);
      if (staged.kernel_slot.has_value()) {
        kernel_masked_columns[*staged.kernel_slot] = staged.dst_slot->get();
      }
    }
    pending_dictionaries.clear();
  }

  // Fixed-width leaf (a top-level column or a struct descendant): payload plus the leaf's own
  // mask, both carved from the call-wide fw arena. `ancestor_offset` is the accumulated offset
  // of the struct views above the leaf — rows for the mask bit offset, elements for the data
  // base.
  std::unique_ptr<column> emit_fixed_width(column_view const& src, size_type ancestor_offset)
  {
    CUDF_EXPECTS(fw_carve_mr != nullptr, "fw carve arena missing at leaf emission");
    auto plan               = plan_destination_mask(src, nullify, num_rows, bands_enabled);
    auto const element_size = static_cast<int32_t>(cudf::size_of(src.type()));
    // The reservation for this leaf was already added by the pre-walk pass; the arena replay
    // below consumes it (mask first, then data — matching `fixed_width_destination_layout::
    // carve`'s per-column regions).
    auto dst =
      make_carved_fixed_width(src, num_rows, plan, *fw_carve_mr, mask_bands, bands_enabled, stream);
    if (plan.skip_data) {
      // Every destination row is null: pre-set ALL_NULL mask, undefined payload (stock parity),
      // and no descriptor — the kernel would only re-read the source mask per row.
      return dst;
    }
    auto const src_data =
      static_cast<char const*>(
        cudf::type_dispatcher<dispatch_storage_type>(src.type(), data_ptr_functor{}, src)) +
      static_cast<std::size_t>(ancestor_offset) * static_cast<std::size_t>(element_size);
    auto const position =
      push_descriptor(column_descriptor{src_data,
                                        dst->mutable_view().data<std::byte>(),
                                        src.null_mask(),
                                        plan.kernel_mask,
                                        static_cast<int64_t>(src.offset()) + ancestor_offset,
                                        element_size,
                                        false});
    if (plan.kernel_mask != nullptr) { set_target(record_kernel_masked(position), dst.get()); }
    return dst;
  }

  // Recursively flattens a fully-fixed-width struct node: leaves become data descriptors and
  // nested struct nodes recurse; the node's own mask rides a MASK-ONLY descriptor when the
  // kernel must write it. An all-null node emits no descriptor (its mask is pre-set ALL_NULL),
  // and its children are still gathered with their own masks — parent and child validity are
  // independent; when every child is skipped too, the launch keeps only the node's MASK-ONLY
  // descriptor (its mask is the remaining work) unless the node's mask is also pre-set. Returns
  // the assembled destination struct column.
  std::unique_ptr<column> build_struct(column_view const& node, size_type ancestor_offset)
  {
    CUDF_EXPECTS(fw_carve_mr != nullptr, "fw carve arena missing at struct build");
    auto const base_offset = ancestor_offset + node.offset();
    auto plan              = plan_destination_mask(node, nullify, num_rows, bands_enabled);
    // The node's own destination mask is carved BEFORE the children recurse: the fw arena is
    // a single bump allocator replaying the reservation's pre-order (node mask first, then
    // each child's regions), so any other construction order could out-consume the reserved
    // advance and falsely trip `arena exhausted` on a later column. A banded preset mask
    // slices its pattern's staging arena (its band carve mirrors the band reservation, which
    // the walk also issued pre-order); the mask is filled by the post-walk band memset and
    // moves into the column at the end; the MASK-ONLY descriptor (when pushed) takes the
    // carved pointer, which the move preserves.
    auto node_mask_env = cuda::std::execution::prop{cuda::allocation_alignment,
                                                    carve_memory_resource::slice_alignment};
    auto node_mask =
      plan.use_band
        ? cuda::device_buffer<bitmask_type>{stream,
                                            *mask_bands[plan.band].mr,
                                            plan.mask_bytes / sizeof(bitmask_type),
                                            cuda::no_init,
                                            node_mask_env}
        : cuda::device_buffer<bitmask_type>{stream,
                                            *fw_carve_mr,
                                            plan.state == mask_state::UNALLOCATED
                                              ? 0
                                              : cudf::bitmask_allocation_size_words(num_rows),
                                            cuda::no_init,
                                            node_mask_env};
    fill_destination_mask(plan, node_mask, bands_enabled, stream);

    std::vector<std::unique_ptr<column>> children;
    children.reserve(node.num_children());
    for (size_type i = 0; i < node.num_children(); ++i) {
      auto const& child = node.child(i);
      if (child.type().id() == type_id::STRUCT) {
        children.push_back(build_struct(child, base_offset));
      } else {
        children.push_back(emit_fixed_width(child, base_offset));
      }
    }

    size_t mask_slot = 0;
    if (plan.kernel_mask != nullptr) {
      // MASK-ONLY descriptor: no payload; the kernel gathers the node's own mask.
      mask_slot =
        record_kernel_masked(push_descriptor(column_descriptor{nullptr,
                                                               nullptr,
                                                               node.null_mask(),
                                                               plan.kernel_mask,
                                                               static_cast<int64_t>(base_offset),
                                                               0,
                                                               true}));
    }
    // The mask buffer (when allocated) moves into the column; its data pointer captured in the
    // descriptor stays valid across the move. With no mask allocated, pass the same zero-size
    // mask stock's struct assembly uses.
    auto col = create_structs_hierarchy(
      num_rows, std::move(children), plan.null_count, std::move(node_mask), stream, output_mr);
    if (plan.kernel_mask != nullptr) { set_target(mask_slot, col.get()); }
    return col;
  }

  // Dictionary: the annotated indices are fixed-width, so their payload copies through a data
  // descriptor addressed as indices base + parent offset, with the PARENT's own mask as the
  // source mask; the gathered mask lands on the dictionary parent via the adopting factory.
  // The descriptor consumes the same `plan_destination_mask` policy as the fixed-width
  // columns: an all-null dictionary pre-sets its mask ALL_NULL and takes the skip_data
  // fast path (no per-row source-mask reads, indices left unwritten), a nullable
  // zero-null dictionary pre-sets ALL_VALID with no kernel mask work, and only a
  // null-bearing or NULLIFY dictionary carries a source mask / kernel-written mask.
  // Indices are gathered only at valid rows — the kernel gates on `valid`, like stock's
  // gather_if, so index values at null source rows stay uninitialized (every in-tree
  // consumer guards such reads through validity, never raw storage).
  //
  // The whole string-keyed dictionary column is DEFERRED: the stock column copy of a string
  // catalog reads two offset values back to the host with a stream synchronization each, and
  // one catalog per column would serialize the whole build walk. This method pushes the
  // descriptor and records a staged entry; `assemble_staged_keys` performs the batched
  // offsets read — one D2H copy plus one synchronization for every string-keyed catalog in
  // the call — and assembles each destination column. Full key order, keys unreferenced by
  // the gathered indices retained, no dedup/remap/re-encode: identical catalog bytes as the
  // stock path.
  std::unique_ptr<column> build_dictionary(column_view const& src,
                                           std::unique_ptr<column>* dst_slot)
  {
    auto const dict     = dictionary_column_view{src};
    auto const& indices = dict.indices();
    CUDF_EXPECTS(fw_carve_mr != nullptr, "fw carve arena missing at dictionary build");
    auto plan = plan_destination_mask(src, nullify, num_rows, bands_enabled);
    // The indices payload and the parent mask are carved (their reservations were added by
    // the pre-walk pass). The mask must exist before the descriptor is pushed: the kernel
    // writes through the descriptor's pointer, and even the deferred string-keyed assembly
    // runs BEFORE the launch, so creating it here for every path is safe and keeps one
    // upstream pair for the whole class. The carve order mirrors the reservation's per-column
    // mask-then-data: mask first, then the indices data. A banded preset mask slices its
    // pattern's staging arena (filled by the post-walk band memset), a stock-path preset mask
    // or a kernel-written mask slices the fw arena.
    auto parent_mask_env = cuda::std::execution::prop{cuda::allocation_alignment,
                                                      carve_memory_resource::slice_alignment};
    auto parent_mask =
      plan.use_band
        ? cuda::device_buffer<bitmask_type>{stream,
                                            *mask_bands[plan.band].mr,
                                            plan.mask_bytes / sizeof(bitmask_type),
                                            cuda::no_init,
                                            parent_mask_env}
        : cuda::device_buffer<bitmask_type>{stream,
                                            *fw_carve_mr,
                                            plan.state == mask_state::UNALLOCATED
                                              ? 0
                                              : cudf::bitmask_allocation_size_words(num_rows),
                                            cuda::no_init,
                                            parent_mask_env};
    fill_destination_mask(plan, parent_mask, bands_enabled, stream);
    auto indices_dst = rmm::device_buffer{
      static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(cudf::size_of(indices.type())),
      carve_memory_resource::slice_alignment,
      stream,
      *fw_carve_mr};
    auto indices_col = std::make_unique<column>(
      indices.type(),
      num_rows,
      std::move(indices_dst),
      cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, output_mr),
      0,
      std::vector<std::unique_ptr<column>>{});
    if (plan.skip_data) {
      // Every destination row is null: pre-set ALL_NULL parent mask (stock parity), undefined
      // index payload, no descriptor, and keys still owned-copied.
      auto keys_copy = std::make_unique<column>(dict.keys(), stream, output_mr);
      return make_dictionary_column(
        std::move(keys_copy), std::move(indices_col), std::move(parent_mask), plan.null_count);
    }
    auto const element_size = static_cast<int32_t>(cudf::size_of(indices.type()));
    auto const src_data =
      static_cast<char const*>(
        cudf::type_dispatcher<dispatch_storage_type>(indices.type(), data_ptr_functor{}, indices)) +
      static_cast<std::size_t>(src.offset()) * static_cast<std::size_t>(element_size);
    auto const position =
      push_descriptor(column_descriptor{src_data,
                                        indices_col->mutable_view().data<std::byte>(),
                                        src.null_mask(),
                                        plan.kernel_mask,
                                        static_cast<int64_t>(src.offset()),
                                        element_size,
                                        false});
    size_t mask_slot = 0;
    if (plan.kernel_mask != nullptr) { mask_slot = record_kernel_masked(position); }

    if (dict.keys().type().id() == type_id::STRING) {
      // DEFERRED assembly: the stock column copy of a string catalog reads two offset values
      // back with a stream synchronization each, so this path records everything the deferred
      // factory call needs and lets `assemble_staged_keys` build the column after ONE batched
      // offsets read for all string-keyed catalogs in the call. The destination slot stays
      // null until then; the kernel-masked bookkeeping is patched there too. The carved
      // parent mask moves into the staged record (its pointer already rides the descriptor).
      staged_keys staged;
      staged.keys               = dict.keys();
      staged.keys_offset        = dict.keys().offset();
      staged.keys_size          = dict.keys().size();
      staged.null_count         = plan.null_count;
      auto const& offsets_child = dict.keys().child(strings_column_view::offsets_column_index);
      staged.offsets            = std::make_unique<column>(offsets_child, stream, output_mr);
      staged.indices            = std::move(indices_col);
      staged.dst_mask.emplace(std::move(parent_mask));
      staged.dst_slot = dst_slot;
      if (plan.kernel_mask != nullptr) { staged.kernel_slot = mask_slot; }
      pending_dictionaries.push_back(std::move(staged));
      return nullptr;
    }

    // Non-string keys: the stock device-side deep copy has no synchronization, so it runs
    // inline exactly as before. The keys copy is built HERE, not at function entry: the
    // string-keyed path defers assembly and would otherwise pay the copy's two offset
    // readbacks per catalog only to discard it.
    auto keys_copy = std::make_unique<column>(dict.keys(), stream, output_mr);
    auto col       = make_dictionary_column(
      std::move(keys_copy), std::move(indices_col), std::move(parent_mask), plan.null_count);
    if (plan.kernel_mask != nullptr) { set_target(mask_slot, col.get()); }
    return col;
  }

  // Top-level STRING column on the N3 fast path. The destination offsets
  // child and chars buffer are DEFERRED (their sizes are unknowable until the
  // virtual scan's totals are read back); this method only plans the mask,
  // records the deferred entry, and returns nullptr — the launch block fills
  // `*dst_slot` in `assemble_strings` after the payload launch, or pre-emptively
  // for the all-null / zero-row cases.
  //
  // Mask policy mirrors `plan_destination_mask`: all-null source → ALL_NULL
  // pre-set, no payload; nullable-with-zero-nulls → ALL_VALID; kernel-masked →
  // UNINITIALIZED with the kernel writing it and the valid count atomically
  // accumulated. The source view's own offset is NOT baked into src_chars:
  // `src_offsets` (an offsetalator over the FULL offsets child, offset
  // applied) carries the byte addressing, so chars are addressed from the
  // unsliced base through the sliced offsets exactly as stock does.
  std::unique_ptr<column> build_string(column_view const& src, std::unique_ptr<column>* dst_slot)
  {
    auto plan = plan_destination_mask(src, nullify, num_rows, bands_enabled);
    // Strings are NOT carved (chars bytes unknowable until the scan readback); their
    // destination masks keep the stock allocation regime.
    if (plan.skip_data) {
      // All-null source: ALL_NULL mask, undefined payload, no scan entry. The
      // destination must still be a full strings column (full-size offsets
      // child, empty chars) — `allocate_like` is fixed-width-only, so the
      // column is assembled directly, matching the empty-source completion.
      auto offsets_col =
        cudf::strings::detail::create_offsets_child_column(0, num_rows + 1, stream, output_mr);
      // Empty rows: offsets are all zero. The child's data is uninitialized,
      // and the comparator flags a null row as dirty unless its span is zero.
      CUDF_CUDA_TRY(cudaMemsetAsync(offsets_col->mutable_view().data<int32_t>(),
                                    0,
                                    offsets_col->size() * sizeof(int32_t),
                                    stream.get()));
      return make_strings_column(
        num_rows,
        std::move(offsets_col),
        rmm::device_buffer{},
        plan.null_count,
        cudf::detail::create_null_mask(num_rows, mask_state::ALL_NULL, stream, output_mr));
    }
    staged_string staged;
    staged.src        = src;
    staged.null_count = plan.null_count;
    staged.dst_slot   = dst_slot;
    if (plan.state != mask_state::UNALLOCATED) {
      staged.dst_mask.emplace(
        cudf::detail::create_null_mask(num_rows, plan.state, stream, output_mr));
    }
    if (plan.state == mask_state::UNINITIALIZED) {
      // A string's mask is kernel-written but never rides kernel_masked_columns
      // (the descriptor readback's slots): its null count comes from the
      // payload kernel's own string_counts readback at assembly time.
      staged.kernel_mask_written = true;
    }
    pending_strings.push_back(std::move(staged));
    return nullptr;
  }

  // Completes every deferred string column: the virtual keyed exclusive scan
  // over all string columns' gathered sizes, one batched totals readback, the
  // batched payload kernel, then per-column assembly of the destination
  // strings columns. Called by the launch block; `explicit_tile_cols` is
  // resolved under the totals synchronization when the routed entry left it
  // open (the sampler latency hides behind the scan).
  void assemble_strings(column_view const& gather_map,
                        size_type num_source_rows,
                        bool check_bounds,
                        bool negative_allowed,
                        sampled_map_locality const& sampled,
                        std::optional<size_type>& explicit_tile_cols)
  {
    if (pending_strings.empty()) { return; }
    auto const num_strings = static_cast<size_type>(pending_strings.size());

    // The CUB batched-memcpy engine only runs above this row gate; every host-side addition
    // it requires (the routing walk, the flag init, the CUB launch) keys off this one
    // predicate, so below the gate the host path stays shaped like the pre-routing path.
    auto const chars_routing_active = num_rows >= kBatchMemcpyRowThreshold;

    // Live sources only: an empty SOURCE column (possibly no offsets child)
    // carries no payload work, but its destination still needs a full-size
    // offsets child (every row empty). Such columns are excluded from the
    // device items and completed host-side below.
    std::vector<size_type> live;
    live.reserve(num_strings);
    for (size_type i = 0; i < num_strings; ++i) {
      auto const& staged = pending_strings[i];
      if (staged.src.size() > 0 and staged.src.num_children() > 0) { live.push_back(i); }
    }

    std::vector<string_work_item> items(live.size());
    for (size_t k = 0; k < live.size(); ++k) {
      auto const& staged        = pending_strings[live[k]];
      auto const& offsets_child = staged.src.child(cudf::strings_column_view::offsets_column_index);
      items[k].src_chars        = staged.src.head<char>();
      items[k].src_mask         = staged.src.null_mask();
      items[k].mask_bit_offset  = staged.src.offset();
      items[k].src_rows         = staged.src.size();
      items[k].src_offsets =
        cudf::detail::offsetalator_factory::make_input_iterator(offsets_child, staged.src.offset());
    }

    // Staging-arena sizing, BEFORE the arena is created: a carve never frees, so the arena is
    // sized once, up front, at worst case — every upload cohort is bounded by live.size()
    // (the routing walk only SPLITS the live columns between the routed and unrouted lists),
    // so the arena covers the per-class maximum of all five segments and the actual carve
    // sequence always consumes at most the reservation. The bound is the per-segment
    // aligned sum, NOT the raw byte sum: `string_work_item` is 16-aligned (its offsetalator
    // members carry `alignas(16)`) while its 104 B size is not a multiple of 16, so each
    // carve request bumps the arena pointer by up to 8 bytes; `size_type` requests shift it
    // by up to 4. Reserving the aligned segment sizes covers every round-up the sequence
    // can make (the all-routed path d_items + d_payload_items + d_memcpy_live +
    // d_routed_items is the worst case); a shortfall trips the carve resource's loud
    // `arena exhausted` expect rather than corrupting memory. Each segment's bump
    // round-up is bounded by (alignment − 1) over an arbitrary start, so the reservation
    // is the raw byte sum plus that slack per segment slot.
    auto const staging_total =
      3 * (items.size() * sizeof(string_work_item) + (std::alignment_of_v<string_work_item> - 1)) +
      2 * (items.size() * sizeof(size_type) + (std::alignment_of_v<size_type> - 1));
    // The staging arena is created HERE — sized from the exact reservation, before the first
    // carve — and dies with this call scope. Its buffers must NOT outlive the call (the
    // design's own lifetime constraint): every consumer is a launch enqueued on `stream` while
    // the arena is alive, and the shared resource's stream-ordered free is enqueued when the
    // last buffer copy dies at this scope's exit, AFTER every consuming enqueue. The
    // destination carve classes keep the opposite lifetime (call-outliving).
    auto staging_carve_mr =
      cuda::mr::make_shared_resource<carve_memory_resource>(temp_mr, stream, staging_total);

    auto const rows_per_col = static_cast<int64_t>(num_rows) + 1;
    auto const flat_size    = static_cast<int64_t>(live.size()) * rows_per_col;
    // The 32-bit precomputed divisors are valid only while every flat index fits uint32_t;
    // this bound is the enforcement the string_flat_layout contract relies on. The fan-byte
    // dividend adds up to block_size row slots on top of a flat row index, so the bound is
    // tightened by that slack.
    CUDF_EXPECTS(flat_size <= cuda::std::numeric_limits<uint32_t>::max() - block_size,
                 "Flat string index space exceeds uint32_t; reduce the number of string columns "
                 "or rows");
    // Every grid cell holds at most 41.9M flat rows, so the flat index fits uint32_t and the
    // precomputed 32-bit divisors replace the per-element 64-bit //% (string_flat_layout).
    string_flat_layout const scan_layout{
      cuda::fast_mod_div<uint32_t>{static_cast<uint32_t>(rows_per_col)},
      cuda::fast_mod_div<uint32_t>{static_cast<uint32_t>(num_rows)}};
    auto scratch =
      rmm::device_uvector<int64_t>(static_cast<std::size_t>(flat_size), stream, temp_mr);
    auto col_totals =
      cudf::detail::make_zeroed_device_uvector_async<int64_t>(live.size(), stream, temp_mr);

    auto staging_items =
      cudf::detail::make_pinned_vector_async<string_work_item>(items.size(), stream);
    std::copy(items.begin(), items.end(), staging_items.begin());
    auto d_items = carve_device_vector(
      host_span<string_work_item const>{staging_items}, stream, staging_carve_mr);

    auto const map_begin = indexalator_factory::make_input_iterator(gather_map);
    {
      auto const keys = cuda::transform_iterator(
        cuda::counting_iterator<int64_t>{0},
        cuda::proclaim_return_type<int64_t>([scan_layout] __device__(int64_t flat) {
          return static_cast<int64_t>(static_cast<uint32_t>(flat) / scan_layout.rows_per_col);
        }));
      auto const values = cuda::transform_iterator(
        cuda::counting_iterator<int64_t>{0},
        cuda::proclaim_return_type<int64_t>([d_items = d_items.data(),
                                             map_begin,
                                             scan_layout,
                                             num_source_rows,
                                             check_bounds,
                                             negative_allowed] __device__(int64_t flat) -> int64_t {
          auto const col   = static_cast<uint32_t>(flat) / scan_layout.rows_per_col;
          auto const row   = static_cast<uint32_t>(flat) % scan_layout.rows_per_col;
          auto const& item = d_items[col];
          if (row >= scan_layout.rows_per_col - 1u) { return int64_t{0}; }
          auto const src = resolve_gathered_row(map_begin,
                                                static_cast<size_type>(row),
                                                num_source_rows,
                                                check_bounds,
                                                negative_allowed);
          if (src < 0 or not string_row_valid(item, src)) { return int64_t{0}; }
          return item.src_offsets[static_cast<size_type>(src) + 1] -
                 item.src_offsets[static_cast<size_type>(src)];
        }));
      auto const out = cuda::make_tabulate_output_iterator(
        string_scan_out_fn{scratch.data(), col_totals.data(), scan_layout});
      thrust::exclusive_scan_by_key(
        rmm::exec_policy_nosync(stream, temp_mr), keys, keys + flat_size, values, out, int64_t{0});
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // One batched totals D2H; its synchronization also resolves the routed
    // tile width, hiding the sampler latency behind the scan.
    auto h_col_totals = cudf::detail::make_pinned_vector_async<int64_t>(live.size(), stream);
    cudf::detail::cuda_memcpy_async<int64_t>(
      host_span<int64_t>{h_col_totals.data(), static_cast<std::size_t>(live.size())},
      device_span<int64_t const>{col_totals.data(), static_cast<std::size_t>(live.size())},
      stream);
    std::optional<size_type> tile_from_sampler;
    if (not explicit_tile_cols.has_value()) {
      tile_from_sampler = resolve_map_locality_tile_cols(sampled, stream);
    }
    explicit_tile_cols = explicit_tile_cols.has_value() ? explicit_tile_cols : tile_from_sampler;
    // resolve_map_locality_tile_cols already synchronized when it sampled; the
    // explicit sync is needed only on the paths that skipped it (explicit tile
    // width, or an unclassifiable map). Either way ONE synchronization covers
    // sampler decode, the scan, and this totals copy.
    if (tile_from_sampler.has_value() and sampled.staging.empty()) {
      cudf::detail::sync_stream(stream);
    } else if (not tile_from_sampler.has_value()) {
      cudf::detail::sync_stream(stream);
    }

    auto any_kernel_masked = false;
    for (size_t k = 0; k < live.size(); ++k) {
      if (pending_strings[live[k]].kernel_mask_written) { any_kernel_masked = true; }
    }
    // Landing buffer for the deferred counts readback (b3): allocated HERE,
    // right after the totals sync with an idle stream, because every
    // host_vector construction synchronizes on allocate. Allocating it after
    // the payload launch would block the host on the whole payload kernel.
    auto h_string_counts = cudf::detail::make_pinned_vector_async<size_type>(
      any_kernel_masked ? static_cast<std::size_t>(live.size()) : 0u, stream);

    // Destination allocation from the totals: ONE carve arena covers every live column's
    // offsets child and chars buffer — the allocation class b0 attributed ~783 µs/call to —
    // so the per-buffer `cudaMallocFromPoolAsync`/`cudaFreeAsync` storm collapses to a single
    // upstream allocate/free pair. Each column keeps owning its slices: the carve resource,
    // shared through every buffer's stored value copy, is the transferable part. Non-live
    // staged entries (empty SOURCE columns) and the all-null/empty-map completions below stay
    // on the stock `create_offsets_child_column` path: at most one small column per call, not
    // part of the measured allocation storm. String destination masks stay stock too (allocated
    // in `build_string`); the fixed-width destination class draws from its own arena.
    strings_destination_layout dst_layout;
    for (size_t k = 0; k < live.size(); ++k) {
      dst_layout.carve(h_col_totals[k], num_rows);
    }
    auto carve_mr = cuda::mr::make_shared_resource<carve_memory_resource>(
      output_mr, stream, dst_layout.total_bytes);

    std::vector<cudf::device_span<int64_t>> scratch_spans(live.size());
    std::vector<cudf::device_span<char>> chars_spans(live.size());
    for (size_t k = 0; k < live.size(); ++k) {
      auto& staged = pending_strings[live[k]];
      auto carved  = carve_strings_buffers(num_rows, h_col_totals[k], carve_mr, stream, output_mr);
      staged.offsets   = std::move(carved.offsets);
      staged.chars     = std::move(carved.chars);
      scratch_spans[k] = cudf::device_span<int64_t>{
        reinterpret_cast<int64_t*>(scratch.data()) + static_cast<std::ptrdiff_t>(k) * rows_per_col,
        static_cast<std::size_t>(rows_per_col)};
      auto const bytes = h_col_totals[k];
      if (bytes > 0) {
        chars_spans[k] = cudf::device_span<char>{static_cast<char*>(staged.chars.data()),
                                                 static_cast<std::size_t>(bytes)};
      }
    }

    // ---- batched-memcpy engine: per-column routing BEFORE the items upload ----
    // After the totals sync the host knows each column's exact byte count, so the routing
    // predicate is exact: a column whose average gathered row is <= 32 B has its chars
    // served by one cub::DeviceMemcpy::Batched over the routed flat (column, row) domain —
    // stock parity, see `strings::detail::gather`'s engine split. The decision rides the
    // per-item `chars_via_batched_memcpy` flag, so the kernel skips ONLY the routed
    // columns' chars: a tall mixed-length call keeps every unrouted column's chars on the
    // kernel path, and every output region is written by exactly one engine. The items
    // upload below must stay AFTER this block: the flag is part of the staged struct.
    auto any_batched_memcpy = false;
    std::vector<size_type> memcpy_live;      // live indices routed to cub::DeviceMemcpy::Batched
    std::vector<size_type> memcpy_unrouted;  // live indices keeping the kernel chars+metadata
    if (chars_routing_active) {
      for (size_t k = 0; k < live.size(); ++k) {
        auto const bytes = h_col_totals[k];
        if (bytes > 0 and bytes / num_rows <= static_cast<int64_t>(kBlockRowCopyThreshold)) {
          memcpy_live.push_back(static_cast<size_type>(k));
          items[k].chars_via_batched_memcpy = true;
          any_batched_memcpy                = true;
        } else {
          // Unrouted cohort: this column keeps the kernel path AND its metadata rides the
          // payload kernel (the routed kernel covers only the routed columns).
          memcpy_unrouted.push_back(static_cast<size_type>(k));
        }
      }
    } else {
      // Below the routing gate every live column is unrouted by definition.
      for (size_t k = 0; k < live.size(); ++k) {
        memcpy_unrouted.push_back(static_cast<size_type>(k));
      }
    }
    // ---- fused short-string engine admission: BELOW the routing gate, the payload kernel's
    // group walk (3 CTA barriers per 8-row group) is replaced by the warp-tile engine when
    // every live column's average gathered row is short enough for it — the exact premise the
    // group walk was designed for. Calls in the excluded seam band never reach the kernel
    // (those strings take the stock fallback); at ≥524288 rows the routed-metadata path owns
    // string metadata, so the engine serves only below-band short-row calls. Above the
    // gate the payload kernel serves only the UNROUTED cohort (long-row columns), which the
    // engine does NOT admit (subgroup copies serialize on long rows), so the group walk stays
    // authoritative there. The AVERAGE does not bound the maximum: long outlier rows are safe
    // on the engine (they ride the full-warp copy), so the short-average check alone admits.
    auto engine_active = false;
    std::vector<size_type> engine_cols;
    if (not chars_routing_active and not memcpy_unrouted.empty()) {
      auto all_short = true;
      for (size_t k = 0; k < memcpy_unrouted.size(); ++k) {
        auto const& idx  = memcpy_unrouted[k];
        auto const bytes = h_col_totals[idx];
        if (bytes / num_rows > static_cast<int64_t>(kBlockRowCopyThreshold)) {
          all_short = false;
          break;
        }
      }
      if (all_short) {
        engine_cols = std::move(memcpy_unrouted);
        memcpy_unrouted.clear();
        engine_active = true;
      }
    }

    // Wire the payload work items (num_rows > 0 guaranteed for live columns).
    // any_kernel_masked was computed above the landing-buffer allocation.
    rmm::device_uvector<size_type> string_counts(0, stream, temp_mr);
    if (any_kernel_masked) {
      string_counts =
        cudf::detail::make_zeroed_device_uvector_async<size_type>(live.size(), stream, temp_mr);
    }
    for (size_t k = 0; k < live.size(); ++k) {
      auto const slot      = live[k];
      auto& staged         = pending_strings[slot];
      items[k].dst_scratch = scratch_spans[k].data();
      items[k].dst_chars   = chars_spans[k].empty() ? nullptr : chars_spans[k].data();
      items[k].dst_offsets =
        cudf::detail::offsetalator_factory::make_output_iterator(staged.offsets->mutable_view());
      items[k].dst_mask    = staged.kernel_mask_written ? staged.dst_mask->data() : nullptr;
      items[k].valid_count = staged.kernel_mask_written ? string_counts.data() + k : nullptr;
    }
    auto d_payload_items =
      carve_device_vector(host_span<string_work_item const>{items}, stream, staging_carve_mr);

    if (any_batched_memcpy) {
      // (source ptr, destination ptr, byte size) per row over the routed columns' flat
      // (column, row) domain, built as on-the-fly transform iterators exactly like stock's
      // own Batched call (`strings::detail::gather`'s engine): NO staging arrays and no
      // extra pass — each of CUB's three iterators re-runs the resolve lambda instead. The
      // lambda replicates the kernel chars-phase gate verbatim (resolve/validity/size>0),
      // nullptr/0 for dead rows; the routed-column table is `d_memcpy_live` (device copy of
      // the host list), and the buffer count is the flat routed row count.
      auto const d_memcpy_live =
        carve_device_vector(host_span<size_type const>{memcpy_live}, stream, staging_carve_mr);
      // Hoist the member `num_rows` into a local: a `__device__` lambda inside this member
      // function cannot capture `this`, so the unqualified member name would be ill-formed
      // in the capture lists below.
      auto const num_rows_local = num_rows;
      auto const num_buffers    = static_cast<int64_t>(memcpy_live.size()) * num_rows_local;
      auto const d_routed       = d_memcpy_live.data();
      auto const d_items_ptr    = d_payload_items.data();
      auto const src_addrs_it   = cuda::make_transform_iterator(
        cuda::counting_iterator<int64_t>{0},
        cuda::proclaim_return_type<char const*>(
          [d_routed,
           d_items_ptr,
           num_rows_local,
           num_source_rows,
           check_bounds,
           negative_allowed,
           map_begin] __device__(int64_t flat) -> char const* {
            auto const col  = static_cast<size_type>(flat / num_rows_local);
            auto const row  = static_cast<size_type>(flat % num_rows_local);
            auto const item = d_items_ptr[d_routed[col]];
            auto const src =
              resolve_gathered_row(map_begin, row, num_source_rows, check_bounds, negative_allowed);
            if (src < 0 or not string_row_valid(item, src)) {
              return static_cast<char const*>(nullptr);
            }
            auto const s = static_cast<size_type>(src);
            return item.src_offsets[s + 1] - item.src_offsets[s] > 0
                       ? item.src_chars + item.src_offsets[s]
                       : static_cast<char const*>(nullptr);
          }));
      auto const dst_addrs_it = cuda::make_transform_iterator(
        cuda::counting_iterator<int64_t>{0},
        cuda::proclaim_return_type<char*>([d_routed,
                                           d_items_ptr,
                                           num_rows_local,
                                           num_source_rows,
                                           check_bounds,
                                           negative_allowed,
                                           map_begin] __device__(int64_t flat) -> char* {
          auto const col  = static_cast<size_type>(flat / num_rows_local);
          auto const row  = static_cast<size_type>(flat % num_rows_local);
          auto const item = d_items_ptr[d_routed[col]];
          auto const src =
            resolve_gathered_row(map_begin, row, num_source_rows, check_bounds, negative_allowed);
          if (src < 0 or not string_row_valid(item, src)) { return static_cast<char*>(nullptr); }
          auto const s = static_cast<size_type>(src);
          return item.src_offsets[s + 1] - item.src_offsets[s] > 0
                   ? item.dst_chars + item.dst_scratch[row]
                   : static_cast<char*>(nullptr);
        }));
      auto const copy_sizes_it = cuda::make_transform_iterator(
        cuda::counting_iterator<int64_t>{0},
        cuda::proclaim_return_type<int64_t>([d_routed,
                                             d_items_ptr,
                                             num_rows_local,
                                             num_source_rows,
                                             check_bounds,
                                             negative_allowed,
                                             map_begin] __device__(int64_t flat) -> int64_t {
          auto const col  = static_cast<size_type>(flat / num_rows_local);
          auto const row  = static_cast<size_type>(flat % num_rows_local);
          auto const item = d_items_ptr[d_routed[col]];
          auto const src =
            resolve_gathered_row(map_begin, row, num_source_rows, check_bounds, negative_allowed);
          if (src < 0 or not string_row_valid(item, src)) { return int64_t{0}; }
          auto const s     = static_cast<size_type>(src);
          auto const bytes = item.src_offsets[s + 1] - item.src_offsets[s];
          return bytes > 0 ? bytes : int64_t{0};
        }));
      // The env-overload routes CUB's temporary storage to `temp_mr` (not the wrapper's
      // default-resource form), matching `strings::detail::gather`'s own batched call.
      // Explicit template args: the CTAD form trips nvcc 13.3 in TUs that lack an
      // earlier `<thrust/execution_policy.h>` include (include-order sensitive).
      auto const cub_env =
        cuda::std::execution::env<cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>,
                                  cuda::std::execution::prop<cuda::mr::get_memory_resource_t,
                                                             rmm::device_async_resource_ref>>{
          cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>{cuda::get_stream_t{},
                                                                           stream},
          cuda::std::execution::prop<cuda::mr::get_memory_resource_t,
                                     rmm::device_async_resource_ref>{
            cuda::mr::get_memory_resource_t{}, temp_mr}};
      CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
        src_addrs_it, dst_addrs_it, copy_sizes_it, num_buffers, cub_env));
    }

    // ---- the ONE batched payload launch (unrouted cohort only) ----
    // The grid covers only the UNROUTED cohort's flat rows: a routed column contributes no
    // group visits at all (its warps' flat slots are out of the launched domain), so the
    // group walk's barriers/visits disappear for it — the routed kernel owns its metadata.
    auto const unrouted_rows =
      static_cast<int64_t>(memcpy_unrouted.size()) * static_cast<int64_t>(num_rows);
    if (unrouted_rows > 0) {
      auto const d_unrouted =
        carve_device_vector(host_span<size_type const>{memcpy_unrouted}, stream, staging_carve_mr);
      auto const grid = static_cast<unsigned>(std::min<int64_t>(
        util::div_rounding_up_safe(unrouted_rows, static_cast<int64_t>(block_size)),
        static_cast<int64_t>(65535)));
      string_payload_kernel<block_size>
        <<<grid, block_size, 0, stream.get()>>>(d_payload_items.data(),
                                                d_unrouted.data(),
                                                static_cast<int64_t>(memcpy_unrouted.size()),
                                                scan_layout,
                                                num_source_rows,
                                                check_bounds,
                                                negative_allowed,
                                                map_begin);
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // ---- routed metadata: one launch over the routed cohort's per-column tiles ----
    // The routed items reference the SAME dst buffers as the payload items (distinct output
    // buffers stay owned by the columns; the count slot indexing rides the same live slot
    // index), so the two kernels write disjoint metadata domains. Empty (unrouted) cohort is
    // skipped: no launch, no allocation.
    if (not memcpy_live.empty()) {
      std::vector<string_work_item> routed_items(memcpy_live.size());
      for (size_t r = 0; r < memcpy_live.size(); ++r) {
        routed_items[r] = items[memcpy_live[r]];
      }
      auto d_routed_items = carve_device_vector(
        host_span<string_work_item const>{routed_items}, stream, staging_carve_mr);
      auto const routed_rows = static_cast<int64_t>(memcpy_live.size()) * num_rows;
      auto const meta_grid   = static_cast<unsigned>(
        std::min<int64_t>(util::div_rounding_up_safe(routed_rows, static_cast<int64_t>(block_size)),
                          static_cast<int64_t>(65535)));
      string_routed_metadata_kernel<block_size>
        <<<meta_grid, block_size, 0, stream.get()>>>(d_routed_items.data(),
                                                     static_cast<size_type>(memcpy_live.size()),
                                                     num_rows,
                                                     num_source_rows,
                                                     check_bounds,
                                                     negative_allowed,
                                                     map_begin);
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // Deferred counts readback (b3): the D2H is only enqueued here — the
    // blocking sync moved below the assembly loops, so the host builds the
    // destination columns while the payload kernel is still running. The
    // counts' sole consumer is the per-column null count, applied after the
    // sync via set_null_count (the same post-hoc patch the descriptor path
    // uses); every other assembly input comes from the totals readback.
    // The destination must be the pinned landing buffer allocated above: a
    // pageable destination would make this copy complete synchronously at
    // enqueue, re-blocking the host on the payload kernel.
    if (any_kernel_masked) {
      cudf::detail::cuda_memcpy_async<size_type>(
        host_span<size_type>{h_string_counts.data(), static_cast<std::size_t>(live.size())},
        device_span<size_type const>{string_counts.data(), static_cast<std::size_t>(live.size())},
        stream);
    }

    // Assemble the destination columns: chars, offsets, mask, null count.
    // Non-live staged entries (empty SOURCE columns) still get a full-size
    // offsets child and no chars: every gathered row is empty. Their masks ride
    // the same plan (all-null was handled in build_string; NULLIFY/all-valid
    // arrive here kernel-masked or pre-set).
    for (size_t k = 0; k < live.size(); ++k) {
      auto& staged = pending_strings[live[k]];
      // The kernel-written mask is already correct on device; the host-visible
      // null count is patched in after the deferred readback's sync below.
      size_type null_count = staged.kernel_mask_written ? 0 : staged.null_count;
      auto null_mask       = staged.dst_mask.has_value()
                               ? cuda::device_buffer<bitmask_type>{std::move(*staged.dst_mask)}
                               : cuda::device_buffer<bitmask_type>{stream, output_mr};
      auto col             = make_strings_column(num_rows,
                                     std::move(staged.offsets),
                                     std::move(staged.chars),
                                     null_count,
                                     std::move(null_mask));
      // The count is patched in below (deferred readback); nothing to patch
      // into kernel_masked_columns (descriptor slots stay descriptor-only).
      *staged.dst_slot = std::move(col);
    }
    // The deferred counts readback is only now synchronized: everything above
    // enqueued without waiting on the payload kernel. The null count is the
    // counts' sole host consumer, so the post-hoc set_null_count patch (the
    // pattern the descriptor path already uses for fixed-width columns)
    // applies it after the fact.
    if (any_kernel_masked) {
      cudf::detail::sync_stream(stream);
      for (size_t k = 0; k < live.size(); ++k) {
        if (not pending_strings[live[k]].kernel_mask_written) { continue; }
        pending_strings[live[k]].dst_slot->get()->set_null_count(num_rows - h_string_counts[k]);
      }
    }
    // Empty-source completions: offsets only, no kernel slot (no mask work is
    // possible on a zero-row source beyond ALL_VALID/ALL_NULL pre-sets, and a
    // kernel-written mask on an empty source still needs the kernel — but the
    // scan never saw this column, so it completes host-side with its planned
    // mask).
    for (auto& staged : pending_strings) {
      if (staged.dst_slot == nullptr or *staged.dst_slot != nullptr) { continue; }
      auto offsets_col =
        cudf::strings::detail::create_offsets_child_column(0, num_rows + 1, stream, output_mr);
      // Empty rows: offsets are all zero. The child's data is uninitialized,
      // and the comparator flags a null row as dirty unless its span is zero.
      CUDF_CUDA_TRY(cudaMemsetAsync(offsets_col->mutable_view().data<int32_t>(),
                                    0,
                                    offsets_col->size() * sizeof(int32_t),
                                    stream.get()));
      auto null_mask   = staged.dst_mask.has_value()
                           ? cuda::device_buffer<bitmask_type>{std::move(*staged.dst_mask)}
                           : cuda::device_buffer<bitmask_type>{stream, output_mr};
      auto col         = make_strings_column(num_rows,
                                     std::move(offsets_col),
                                     rmm::device_buffer{},
                                     staged.null_count,
                                     std::move(null_mask));
      *staged.dst_slot = std::move(col);
    }
    pending_strings.clear();
  }
};

}  // namespace

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
  // Same walk estimates the build pass's descriptor and kernel-masked counts so the three
  // descriptor-path vectors are reserved once, up front: a deeply nested struct otherwise
  // reallocates them (with element copies) on every push. The estimate mirrors the builder's
  // emission recursion exactly — same per-node conditions, same arithmetic — so the capacity
  // matches the build output with no slack; a pessimistic guess would silently reintroduce
  // the reallocations it exists to avoid.
  size_type estimated_descriptors   = 0;
  size_type estimated_kernel_masked = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    auto const& src = source_table.column(i);
    // Tall string columns ride the stock per-column fallback unless the batched-memcpy
    // routing engine will be active for the call: the routed metadata path (tall-str1)
    // removes the group-walk metadata work that made the fused path lose tall cells, but it
    // only engages above `kBatchMemcpyRowThreshold` (the host routing walk keys off
    // `chars_routing_active`). The seam band between the two thresholds is EXCLUDED: the
    // engine wins the band cell when admitted (262144x100 −69.5%, engine-vs-stock) but the
    // band route costs the tall regime — the band exclusion trades that one-cell win for
    // keeping ≥524288-row strings on the routed-metadata path (tall-str1), the measured tall
    // win (round-10, +26.7% headline) a wholesale exclusion would forfeit. Gathered row sizes
    // are unknowable before the scan, so no stronger per-column proxy is possible here; the
    // exact post-totals routing still splits per column. Below the band `assemble_strings`
    // picks the engine for short-row shapes; an all-long-row below-band table falls to the
    // payload kernel's group walk (the pre-engine fused behavior) — its loss on that shape is
    // the known admission risk the falsifier's regression checks watch.
    if (is_descriptor_path_candidate(src) and
        not(src.type().id() == type_id::STRING and num_rows >= kStringFallbackRowThreshold and
            num_rows < kBatchMemcpyRowThreshold)) {
      fused_indices.push_back(i);
      if (not(num_rows == 0 and not cudf::is_fixed_width(src.type()))) {
        // Mirror the build's skip: an empty map routes non-fixed-width columns to empty_like
        // without emitting descriptors, so those must not inflate the estimate.
        auto const counts =
          estimate_descriptor_counts(src, bounds_policy == out_of_bounds_policy::NULLIFY);
        estimated_descriptors += counts.descriptors;
        estimated_kernel_masked += counts.kernel_masked;
      }
    } else {
      fallback_indices.push_back(i);
    }
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
  // - struct/dictionary: the stock per-type gatherers early-return `empty_like` /
  //   `make_empty_column(DICTIONARY32)`, mirrored here so the shapes stay stock-exact.
  // Every destination has zero rows and null count 0. A non-empty map is never routed here
  // even when NULLIFY would nullify every gathered row: proving that needs the map values
  // (device data), so those calls run the full path below. With a zero-size map the mask
  // phase degenerates to nothing — `gather_bitmask` returns on its `mask_size == 0` guard
  // and the 0-word masks it attaches make the columns non-nullable.
  if (num_rows == 0) {
    for (auto const idx : fused_indices) {
      auto const& src = source_table.column(idx);
      if (src.type().id() == type_id::STRUCT) {
        destination[idx] = cudf::empty_like(src);
      } else if (src.type().id() == type_id::DICTIONARY32) {
        destination[idx] = cudf::make_empty_column(type_id::DICTIONARY32);
      } else {
        destination[idx] = cudf::detail::allocate_like(
          src, 0, cudf::mask_allocation_policy::NEVER, stream, output_mr);
      }
    }
    // Built unconditionally, like stock's detail::gather does before any per-column work: a
    // non-index map type must throw on a zero-row call the same way it does on a non-empty one.
    auto const map_begin = indexalator_factory::make_input_iterator(gather_map);
    if (not fallback_indices.empty()) { run_fallback(fallback_indices, map_begin, map_begin); }
    return std::make_unique<table>(std::move(destination), num_rows);
  }

  // Fused part: destination columns for the descriptor-path subset, with per-node mask policy.
  // Plain fixed-width columns emit one data descriptor; fully-fixed-width structs flatten into
  // leaf descriptors plus mask-only node descriptors (reassembled after the launch); dictionary
  // columns emit one data descriptor over their annotated indices. All-null sources emit no
  // descriptor (their masks are pre-set ALL_NULL); if nothing remains, the launch is skipped.
  // The band-path decision, made once per call BEFORE any band arena exists: the band fill
  // trades a fixed ~24 µs-per-band memset for per-mask ~7 µs submissions, so with fewer than
  // `kMaskBandPresetGate` preset masks the per-mask drip wins and every preset mask must take
  // the stock carve + per-mask memset (a populated band memset ≈ 24 µs vs ~7 µs per mask
  // submission: 8 masks ≈ 56 µs > 2 × 24 µs). The
  // count is computed from the same `plan_destination_mask` classification the build walk
  // uses (struct children recursed), so the walk and this decision cannot diverge.
  bool const bands_enabled = [&] {
    if (fused_indices.empty()) { return false; }
    size_type preset_masks                                      = 0;
    std::function<void(column_view const&, bool)> count_presets = [&](column_view const& src,
                                                                      bool nullify_flag) {
      if (src.type().id() == type_id::STRING) { return; }
      auto const plan = plan_destination_mask(src, nullify_flag, num_rows, false);
      if (plan.preset) { ++preset_masks; }
      if (src.type().id() == type_id::STRUCT) {
        for (size_type i = 0; i < src.num_children(); ++i) {
          count_presets(src.child(i), nullify_flag);
        }
      }
    };
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      if (num_rows == 0 and not cudf::is_fixed_width(src.type())) { continue; }
      count_presets(src, bounds_policy == out_of_bounds_policy::NULLIFY);
    }
    return preset_masks > kMaskBandPresetGate;
  }();

  std::vector<column_descriptor> descriptors;
  std::vector<size_type> kernel_masked_positions;  // positions of kernel-masked descriptors
  std::vector<column*> kernel_masked_columns;      // slot i's count target (columns[i])
  descriptor_path_builder builder{descriptors,
                                  kernel_masked_positions,
                                  kernel_masked_columns,
                                  num_rows,
                                  bounds_policy == out_of_bounds_policy::NULLIFY,
                                  bands_enabled,
                                  stream,
                                  output_mr,
                                  temp_mr};
  if (not fused_indices.empty()) {
    // Pre-walk reservation for the fixed-width carve: sum every fw destination's mask+data
    // bytes (dry planning — no allocation) so ONE arena covers the whole class with a single
    // upstream allocate/free pair. The reservation visits columns in the SAME pre-order the
    // build walk will carve them (see `fixed_width_destination_layout`), so the total equals
    // the bump pointer's final position exactly; a divergence trips the carve resource's loud
    // `arena exhausted` expect. The fw arena is created BEFORE the build walk because the fw
    // columns construct hierarchically during it (leaves, struct nodes, dictionaries);
    // strings stay outside it (chars bytes unknowable until the post-walk scan readback) and
    // keep their own carve in `assemble_strings`. The mirror must include every buffer the
    // build walk creates: an all-null destination still owns a full-size (uninitialized) data
    // buffer and a pre-set ALL_NULL mask.
    // std::function, not `auto`: the reservation mirrors the build walk's STRUCT recursion,
    // so the lambda must call itself from its own initializer (an `auto` initializer cannot
    // self-reference). One host-side allocation per call, outside the carve path.
    std::function<void(column_view const&, bool)> fw_reserve = [&](column_view const& src,
                                                                   bool nullify_flag) {
      if (src.type().id() == type_id::STRING) { return; }
      // A banded preset mask's reservation rides its pattern's band layout, not the fw arena
      // (the build walk carves banded preset masks from the band, kernel-written masks from
      // the arena — `plan_destination_mask` computes the same classification here as at build
      // time, so the two mirrors stay in lockstep). A stock-path preset mask (below the band
      // gate) carves the fw arena with the rest, matching `plan.use_band` at build time.
      auto const plan = plan_destination_mask(src, nullify_flag, num_rows, bands_enabled);
      if (src.type().id() == type_id::STRUCT) {
        if (plan.preset and plan.use_band) { builder.mask_bands[plan.band].layout.carve(num_rows); }
        if (plan.state != mask_state::UNALLOCATED and not plan.use_band) {
          builder.fw_layout.carve(num_rows, 0, true);
        }
        for (size_type i = 0; i < src.num_children(); ++i) {
          fw_reserve(src.child(i), nullify_flag);
        }
        return;
      }
      if (src.type().id() == type_id::DICTIONARY32) {
        auto const indices    = dictionary_column_view{src}.indices();
        auto const data_bytes = static_cast<std::size_t>(num_rows) *
                                static_cast<std::size_t>(cudf::size_of(indices.type()));
        if (plan.use_band) {
          // The parent mask rides the band; the gathered indices payload still carves the
          // fw arena at build time (`build_dictionary` carves it on every path), so its
          // region must be reserved here too or the replay out-consumes the layout.
          builder.mask_bands[plan.band].layout.carve(num_rows);
          builder.fw_layout.carve(num_rows, data_bytes, false);
        } else if (plan.state != mask_state::UNALLOCATED) {
          builder.fw_layout.carve(num_rows, data_bytes, true);
        } else {
          builder.fw_layout.carve(num_rows, data_bytes, false);
        }
        return;
      }
      auto const data_bytes =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(cudf::size_of(src.type()));
      if (plan.use_band) {
        builder.mask_bands[plan.band].layout.carve(num_rows);
        builder.fw_layout.carve(num_rows, data_bytes, false);
      } else {
        builder.fw_layout.carve(num_rows, data_bytes, plan.state != mask_state::UNALLOCATED);
      }
    };
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      if (num_rows == 0 and not cudf::is_fixed_width(src.type())) { continue; }
      fw_reserve(src, bounds_policy == out_of_bounds_policy::NULLIFY);
    }
    auto fw_carve_mr = cuda::mr::make_shared_resource<carve_memory_resource>(
      output_mr, stream, builder.fw_layout.total_bytes);
    builder.fw_carve_mr = &fw_carve_mr;
    for (auto& band : builder.mask_bands) {
      band.mr = std::make_unique<cuda::mr::shared_resource<carve_memory_resource>>(
        cuda::mr::make_shared_resource<carve_memory_resource>(
          output_mr, stream, band.layout.total_bytes));
    }

    descriptors.reserve(estimated_descriptors);
    kernel_masked_positions.reserve(estimated_kernel_masked);
    kernel_masked_columns.reserve(estimated_kernel_masked);
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      if (num_rows == 0 and not cudf::is_fixed_width(src.type())) {
        // Empty map: stock's per-type gatherers return exactly these shapes for these types.
        destination[fused_indices[pos]] =
          (src.type().id() == type_id::STRUCT)
            ? cudf::empty_like(src)
            : (src.type().id() == type_id::STRING ? cudf::make_empty_column(type_id::STRING)
                                                  : cudf::make_empty_column(type_id::DICTIONARY32));
        continue;
      }
      switch (src.type().id()) {
        case type_id::STRUCT: destination[fused_indices[pos]] = builder.build_struct(src, 0); break;
        case type_id::STRING:
          // The N3 fast path defers the whole column: the scan launch block
          // fills the slot in `assemble_strings`. Only an empty map (handled
          // above) or an all-null source (skip_data, built inline) returns a
          // column here.
          destination[fused_indices[pos]] =
            builder.build_string(src, &destination[fused_indices[pos]]);
          break;
        case type_id::DICTIONARY32:
          // String-keyed catalogs defer their assembly and fill the slot in
          // `assemble_staged_keys`; every other path builds the column here.
          destination[fused_indices[pos]] =
            builder.build_dictionary(src, &destination[fused_indices[pos]]);
          break;
        default: destination[fused_indices[pos]] = builder.emit_fixed_width(src, 0); break;
      }
    }
    // One memset per non-empty band replaces the per-mask fills: every banded preset mask
    // carved from that band gets the pattern byte in a single submission, stream-ordered
    // before any consumer (the kernel launch and the deferred assemblies both follow on the
    // same stream). Below the band gate the per-mask fills already ran during the walk.
    if (bands_enabled) {
      for (size_t band = 0; band < 2; ++band) {
        auto const& b = builder.mask_bands[band];
        if (b.layout.total_bytes == 0) { continue; }
        auto const pattern = band == 0 ? 0xff : 0x00;
        CUDF_CUDA_TRY(
          cudaMemsetAsync(b.mr->get().arena_base(), pattern, b.layout.total_bytes, stream.get()));
      }
    }
    // Zero rows never reach `build_dictionary` (the empty-map branch above returns early for
    // every non-fixed-width type), so `pending_dictionaries` is empty here; this call is a
    // defensive no-op that keeps the deferred contract satisfied on every path that skips
    // the launch block below.
    if (num_rows == 0) { builder.assemble_staged_keys(); }
  }

  auto const any_kernel_masks = not kernel_masked_positions.empty();
  if (any_kernel_masks) {
    // Masked descriptors must occupy the leading positions [0, M) so a masked descriptor's
    // position doubles as its `valid_counts` slot. Descriptors are position-independent (all
    // addressing is via their pointers), so the stable partition is semantically free; visit
    // order is preserved within each run. This must precede the staging copy below.
    std::stable_partition(descriptors.begin(), descriptors.end(), [](column_descriptor const& d) {
      return d.dst_mask != nullptr;
    });
    for (size_t i = 0; i < kernel_masked_positions.size(); ++i) {
      kernel_masked_positions[i] = static_cast<size_type>(i);
    }
  }

  // String fast path (N3): when string columns are present they ride the
  // batched scan+payload machinery, which also resolves the routed tile width
  // under its totals synchronization. The descriptor launch below reuses the
  // resolved width via `explicit_tile_cols` being set. When there are no
  // descriptors at all (a pure-strings call), the SAME assemble_strings call
  // performs everything — the launch block below is skipped.
  auto mutable_tile_cols = explicit_tile_cols;
  if (not builder.pending_strings.empty() and num_rows > 0) {
    builder.assemble_strings(gather_map,
                             num_src_rows,
                             bounds_policy == out_of_bounds_policy::NULLIFY,
                             neg_indices == negative_index_policy::ALLOWED,
                             sampled,
                             mutable_tile_cols);
    explicit_tile_cols = mutable_tile_cols;
  } else if (not builder.pending_strings.empty()) {
    // Zero rows: every deferred string column becomes an empty strings column
    // (stock parity with gather_strings on an empty map).
    for (auto& staged : builder.pending_strings) {
      *staged.dst_slot = cudf::make_empty_column(type_id::STRING);
    }
    builder.pending_strings.clear();
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
    // parameter, which cannot carry a mutable counts output. The device landing buffer is
    // carved from a descriptor-staging arena (ONE upstream pair instead of one malloc/free per
    // call); its scope is this launch block, and `batch_gather_kernel` — the buffer's only
    // consumer — is enqueued inside it before the stream-ordered free.
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
      auto desc_staging_carve_mr = cuda::mr::make_shared_resource<carve_memory_resource>(
        temp_mr, stream, total_bytes);
      d_descriptors.emplace(total_bytes, stream, desc_staging_carve_mr.get());
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
      for (size_t i = 0; i < kernel_masked_positions.size(); ++i) {
        kernel_masked_columns[i]->set_null_count(
          num_rows - h_valid_counts_owner->data()[kernel_masked_positions[i]]);
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
