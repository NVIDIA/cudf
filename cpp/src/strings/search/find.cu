/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * a warp-parallel function is used.
 *
 * Note that this value is shared by find, rfind, and contains functions.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Find function handles a string per thread
 */
template <typename TargetIterator, bool forward = true>
struct finder_fn {
  column_device_view const d_strings;
  TargetIterator const d_targets;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty() && (start > 0)) { return -1; }
    if (stop >= 0 && start > stop) { return -1; }
    auto const d_target = d_targets[idx];

    auto const count = (stop < 0) ? stop : (stop - start);
    return forward ? d_str.find(d_target, start, count) : d_str.rfind(d_target, start, count);
  }
};

/**
 * @brief Special logic handles an empty target for find/rfind
 *
 * where length = number of characters in the input string
 * if forward = true:
 *   return start iff (start <= length), otherwise return -1
 * if forward = false:
 *   return stop iff (0 <= stop <= length), otherwise return length
 */
template <bool forward = true>
struct empty_target_fn {
  column_device_view const d_strings;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto d_str = d_strings.element<string_view>(idx);

    // common case shortcut
    if (forward && start == 0) { return 0; }

    auto const length = d_str.length();
    if (start > length) { return -1; }
    if constexpr (forward) { return start; }

    return (stop < 0) || (stop > length) ? length : stop;
  }
};

/**
 * @brief String per warp function for find/rfind
 */
template <typename TargetIterator, bool forward = true>
CUDF_KERNEL void finder_warp_parallel_fn(column_device_view const d_strings,
                                         TargetIterator const d_targets,
                                         size_type const start,
                                         size_type const stop,
                                         size_type* d_results)
{
  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }

  auto const d_str    = d_strings.element<string_view>(str_idx);
  auto const d_target = d_targets[str_idx];

  auto const [begin, left_over] = bytes_to_character_position(d_str, start);
  auto const start_char_pos     = start - left_over;  // keep track of character position

  auto const end = [d_str, start, stop, begin = begin] {
    if (stop < 0) { return d_str.size_bytes(); }
    if (stop <= start) { return begin; }
    // we count from `begin` instead of recounting from the beginning of the string
    return begin + cuda::std::get<0>(bytes_to_character_position(
                     string_view(d_str.data() + begin, d_str.size_bytes() - begin), stop - start));
  }();

  // each thread compares the target with the thread's individual starting byte
  size_type position = forward ? cuda::std::numeric_limits<size_type>::max() : -1;
  for (auto itr = begin + lane_idx; itr + d_target.size_bytes() <= end;
       itr += cudf::detail::warp_size) {
    if (d_target.compare(d_str.data() + itr, d_target.size_bytes()) == 0) {
      position = itr;
      if (forward) break;
    }
  }

  // find stores the minimum position while rfind stores the maximum position
  auto const result = forward ? cg::reduce(warp, position, cg::less<size_type>())
                              : cg::reduce(warp, position, cg::greater<size_type>());

  if (lane_idx == 0) {
    // the final result needs to be fixed up convert max() to -1
    // and a byte position to a character position
    d_results[str_idx] =
      ((result < cuda::std::numeric_limits<size_type>::max()) && (result >= begin))
        ? start_char_pos + characters_in_string(d_str.data() + begin, result - begin)
        : -1;
  }
}

template <typename TargetIterator, bool forward = true>
void find_utility(strings_column_view const& input,
                  TargetIterator const& target_itr,
                  column& output,
                  size_type start,
                  size_type stop,
                  cuda::stream_ref stream)
{
  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = output.mutable_view().data<size_type>();
  if ((input.chars_size(stream) / (input.size() - input.null_count())) > AVG_CHAR_BYTES_THRESHOLD) {
    // warp-per-string runs faster for longer strings (but not shorter ones)
    constexpr auto block_size             = 256;
    constexpr thread_index_type warp_size = cudf::detail::warp_size;
    cudf::detail::grid_1d grid{input.size() * warp_size, block_size};
    finder_warp_parallel_fn<TargetIterator, forward>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        *d_strings, target_itr, start, stop, d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    // string-per-thread function
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_results,
                      finder_fn<TargetIterator, forward>{*d_strings, target_itr, start, stop});
  }
}

template <bool forward = true>
std::unique_ptr<column> find_fn(strings_column_view const& input,
                                string_scalar const& target,
                                size_type start,
                                size_type stop,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  CUDF_EXPECTS(stop <= 0 or start <= stop, "Parameter start must be less than stop.");

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  auto d_target = string_view(target.data(), target.size());

  // special logic for empty target results
  if (d_target.empty()) {
    auto d_strings = column_device_view::create(input.parent(), stream);
    auto d_results = results->mutable_view().data<size_type>();
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      d_results,
                      empty_target_fn<forward>{*d_strings, start, stop});
    return results;
  }

  // find-utility function fills in the results column
  auto target_itr      = cuda::make_constant_iterator(d_target);
  using TargetIterator = decltype(target_itr);
  find_utility<TargetIterator, forward>(input, target_itr, *results, start, stop, stream);
  results->set_null_count(input.null_count());
  return results;
}
}  // namespace

std::unique_ptr<column> find(strings_column_view const& input,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  return find_fn<true>(input, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& input,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  return find_fn<false>(input, target, start, stop, stream, mr);
}

template <bool forward = true>
std::unique_ptr<column> find(strings_column_view const& input,
                             strings_column_view const& target,
                             size_type start,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  CUDF_EXPECTS(input.size() == target.size(), "input and target columns must be the same size");

  // create output column
  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, input.size(), rmm::device_buffer{}, 0, stream, mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  // call find utility with target iterator
  auto d_targets  = column_device_view::create(target.parent(), stream);
  auto target_itr = cudf::detail::make_null_replacement_iterator<string_view>(
    *d_targets, string_view{}, target.has_nulls());
  find_utility<decltype(target_itr), forward>(input, target_itr, *results, start, -1, stream);

  // AND the bitmasks from input and target
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view({input.parent(), target.parent()}), stream, mr);
  results->set_null_mask(std::move(null_mask), null_count);
  return results;
}

}  // namespace detail

// external APIs

std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find(strings, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rfind(strings, target, start, stop, stream, mr);
}

std::unique_ptr<column> find(strings_column_view const& input,
                             strings_column_view const& target,
                             size_type start,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find<true>(input, target, start, stream, mr);
}

namespace detail {
namespace {

/**
 * @brief Logarithmic Radix Binning (LRB) scheduler for contains(column, scalar).
 *
 * Background: the previous implementation picked one execution shape for the whole column from
 * the *average* row width (thread-per-row when avg <= 64 B, else warp-per-row). Any real column
 * with a spread of row lengths pays for that: with thread-per-row, a warp runs at the pace of its
 * longest row (31 lanes idle); with warp-per-row, a 20-byte row still occupies a full warp and a
 * 20 MB row still gets only one warp. PR #21917 splits rows at one fixed length (96 B) into two
 * shapes. LRB generalizes that idea: rows are binned by ceil(log2(bytes)), so all rows in a bin
 * are within 2x of each other, and every bin is executed by a thread group sized for it.
 *
 * Green et al., "Logarithmic Radix Binning and Vectorized Triangle Counting", HPEC 2018,
 * doi:10.1109/HPEC.2018.8547581. Their recipe: (1) bin = word_width - clz(work), (2) histogram
 * the bins, (3) prefix-sum the (tiny) histogram, (4) scatter items into bin order with atomics,
 * (5) process each bin with a suitable parallel granularity. The steps here map 1:1.
 *
 * Pipeline (3 kernels, no host<->device synchronization, O(rows) extra memory for the permutation):
 *   lrb_histogram_kernel   thread-per-row. Rows that are null or shorter than the target get
 *                          result=false immediately; rows of at most LRB_DIRECT_MAX_BYTES are
 *                          searched right here thread-per-row. Neither kind is ever binned
 *                          (bin 0). Longer rows are counted in a per-block shared histogram.
 *   lrb_scatter_kernel     recomputes the 32-entry prefix sum per block, ranks rows within the
 *                          block per bin (warp-aggregated shared atomics) and appends them to
 *                          their bin with one global atomic per (block, bin). Rows from the same
 *                          block stay adjacent inside a bin, which keeps character reads local.
 *   lrb_search_kernel      persistent grid-stride kernel. Each warp maps its virtual task id to
 *                          a bin using the same 32-entry scan (kept in registers via warp
 *                          shuffles). A task is 32/G rows searched with G threads each, G in
 *                          {1,2,4,8,16,32} by bin, or, for rows longer than LRB_CHUNK_BYTES, one
 *                          chunk of one row, so every task carries bounded work and a few huge
 *                          rows spread over the whole GPU instead of serializing on one warp.
 *
 * @see LRB_SHIFT, LRB_CHUNK_BYTES, LRB_DIRECT_MAX_BYTES
 */
constexpr int LRB_NUM_BINS   = 32;
constexpr int LRB_BLOCK_SIZE = 256;

/**
 * @brief Threads cooperating on one row = 1 << clamp(bin - LRB_SHIFT, 0, 5).
 *
 * bin b holds rows with 2^(b-1) < bytes <= 2^b. With LRB_SHIFT = 4:
 *   bytes <= 16: 1 thread   17..32: 2   33..64: 4   65..128: 8   129..256: 16   > 256: 32 (a warp)
 * so each lane scans at most ~16 starting positions per row before the group moves on.
 */
constexpr int LRB_SHIFT = 4;

/**
 * @brief Rows longer than this are scheduled as several warp tasks, one per chunk of this many
 * bytes, so every task carries a bounded amount of work regardless of row length.
 *
 * A warp scans a 4 KiB chunk in 32 steps of 128 bytes. A 1 MiB row becomes 256 independent tasks
 * that spread across the GPU, and a column with many 16 KiB rows produces 4 tasks each; neither
 * case depends on how many long rows there are. Chunks of one row never conflict: the scatter pass
 * writes `false` for every binned row and chunk tasks only ever write `true`.
 */
constexpr int LRB_CHUNK_BYTES = 4096;

constexpr int LRB_CHUNK_SHIFT = 12;  // log2(LRB_CHUNK_BYTES)
static_assert((1 << LRB_CHUNK_SHIFT) == LRB_CHUNK_BYTES);

/// log2 of the number of chunk tasks per row of `bin`: rows in bin b have at most 2^b bytes.
__host__ __device__ constexpr int lrb_chunk_shift(int bin)
{
  return bin > LRB_CHUNK_SHIFT ? bin - LRB_CHUNK_SHIFT : 0;
}

/**
 * @brief Rows of at most this many bytes are searched thread-per-row directly inside the
 * histogram pass and are never binned or scattered.
 *
 * For short rows the fixed cost of the permutation (two extra passes over the offsets plus
 * writing and reading one index per row) exceeds the imbalance it removes, because a warp of
 * thread-per-row lanes can only wait on at most this many bytes. Set to 0 for pure LRB.
 */
constexpr size_type LRB_DIRECT_MAX_BYTES = 64;

/**
 * @brief Bin index of a row. Bin 0 = resolved in the histogram pass (null, shorter than the
 * target, or at most LRB_DIRECT_MAX_BYTES bytes).
 *
 * @param size_bytes Row length in bytes (any value if is_null)
 * @param target_bytes Target length in bytes (> 0)
 * @param is_null Whether the row is null
 * @return Bin in [0, 31]
 */
__device__ __forceinline__ int lrb_bin(size_type size_bytes, size_type target_bytes, bool is_null)
{
  if (is_null || size_bytes < target_bytes || size_bytes <= LRB_DIRECT_MAX_BYTES) { return 0; }
  // ceil(log2(bytes)): bin b holds 2^(b-1) < bytes <= 2^b, so a row of exactly 2^b bytes sits at
  // the top of bin b and the bin's chunk count (2^b / LRB_CHUNK_BYTES) is exact for it
  return cuda::std::max(1, 32 - __clz(static_cast<unsigned>(size_bytes - 1)));
}

__host__ __device__ constexpr int lrb_threads_per_row(int bin)
{
  auto const shift = bin - LRB_SHIFT;
  return 1 << (shift < 0 ? 0 : (shift > 5 ? 5 : shift));
}

/**
 * @brief Pass 1: per-bin row counts; rows in bin 0 are resolved (false) here.
 */
CUDF_KERNEL void lrb_histogram_kernel(column_device_view const d_strings,
                                      string_view const d_target,
                                      bool* d_results,
                                      size_type* d_hist)
{
  auto const target_bytes = d_target.size_bytes();
  __shared__ size_type s_hist[LRB_NUM_BINS];
  if (threadIdx.x < LRB_NUM_BINS) { s_hist[threadIdx.x] = 0; }
  __syncthreads();

  auto const num_rows = static_cast<thread_index_type>(d_strings.size());
  auto const stride   = cudf::detail::grid_1d::grid_stride();
  auto const lane     = threadIdx.x % cudf::detail::warp_size;
  for (auto idx = cudf::detail::grid_1d::global_thread_id(); idx < num_rows; idx += stride) {
    auto const row     = static_cast<size_type>(idx);
    auto const is_null = d_strings.is_null(row);
    auto const bytes   = is_null ? 0 : d_strings.element<string_view>(row).size_bytes();
    auto const bin     = lrb_bin(bytes, target_bytes, is_null);
    if (bin == 0) {
      // resolved here: false unless it is a short valid row that actually contains the target
      auto found = false;
      if (!is_null && bytes >= target_bytes) {
        auto const d_str = d_strings.element<string_view>(row);
        for (size_type i = 0; !found && (i <= bytes - target_bytes); ++i) {
          found = d_target.compare(d_str.data() + i, target_bytes) == 0;
        }
      }
      d_results[row] = found;
    }
    // one shared atomic per (warp, distinct bin) instead of one per lane; nothing at all when
    // the whole warp was resolved directly (the common case for short-string columns)
    auto const active = __activemask();
    if (__any_sync(active, bin != 0)) {
      auto const peers = __match_any_sync(active, bin);
      if (bin != 0 && lane == static_cast<unsigned>(__ffs(peers) - 1)) {
        atomicAdd(&s_hist[bin], __popc(peers));
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < LRB_NUM_BINS && s_hist[threadIdx.x] != 0) {
    atomicAdd(&d_hist[threadIdx.x], s_hist[threadIdx.x]);
  }
}

/**
 * @brief Pass 2: append each row index to its bin's segment of `d_sorted`.
 *
 * @param d_results Set to false for every binned row (the search pass only writes true)
 * @param d_hist Per-bin counts from pass 1 (read) followed by per-bin cursors (read/write),
 *               i.e. an array of 2 * LRB_NUM_BINS
 * @param d_sorted Output permutation grouped by bin, in bin order
 */
CUDF_KERNEL void lrb_scatter_kernel(column_device_view const d_strings,
                                    size_type const target_bytes,
                                    bool* d_results,
                                    size_type* d_hist,
                                    size_type* d_sorted)
{
  namespace cg = cooperative_groups;
  __shared__ size_type s_start[LRB_NUM_BINS];
  __shared__ size_type s_count[LRB_NUM_BINS];
  __shared__ size_type s_base[LRB_NUM_BINS];

  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane = warp.thread_rank();
  __shared__ size_type s_binned;
  if (warp.meta_group_rank() == 0) {
    auto const count = d_hist[lane];
    auto const incl  = cg::inclusive_scan(warp, count);
    s_start[lane]    = incl - count;
    if (lane == cudf::detail::warp_size - 1) { s_binned = incl; }
  }
  __syncthreads();
  if (s_binned == 0) { return; }  // every row was resolved in the histogram pass

  auto const num_rows = static_cast<thread_index_type>(d_strings.size());
  auto const stride   = cudf::detail::grid_1d::grid_stride();
  auto* d_cursor      = d_hist + LRB_NUM_BINS;
  for (auto base = static_cast<thread_index_type>(blockIdx.x) * blockDim.x; base < num_rows;
       base += stride) {
    if (threadIdx.x < LRB_NUM_BINS) { s_count[threadIdx.x] = 0; }
    __syncthreads();

    auto const idx = base + threadIdx.x;
    auto bin       = 0;
    if (idx < num_rows) {
      auto const row     = static_cast<size_type>(idx);
      auto const is_null = d_strings.is_null(row);
      auto const bytes   = is_null ? 0 : d_strings.element<string_view>(row).size_bytes();
      bin                = lrb_bin(bytes, target_bytes, is_null);
    }
    // rank within (block, bin): warp-aggregated shared atomic, then lane order within the warp
    auto const peers  = __match_any_sync(0xffffffffu, bin);
    auto const leader = __ffs(peers) - 1;
    size_type wbase   = 0;
    if (bin != 0 && lane == static_cast<unsigned>(leader)) {
      wbase = atomicAdd(&s_count[bin], __popc(peers));
    }
    wbase           = __shfl_sync(0xffffffffu, wbase, leader);
    auto const rank = wbase + __popc(peers & ((1u << lane) - 1u));
    __syncthreads();

    if (threadIdx.x < LRB_NUM_BINS && s_count[threadIdx.x] != 0) {
      s_base[threadIdx.x] = atomicAdd(&d_cursor[threadIdx.x], s_count[threadIdx.x]);
    }
    __syncthreads();

    if (bin != 0) {
      d_sorted[s_start[bin] + s_base[bin] + rank] = static_cast<size_type>(idx);
      d_results[idx]                              = false;  // chunk tasks only ever set true
    }
  }
}

/**
 * @brief Search one row with a tile of G threads (G == 1 is plain thread-per-row).
 *
 * All rows scheduled on one warp belong to the same bin, so `row` is either valid for every lane
 * of the tile or -1 for every lane of the tile; the early return is tile-uniform. `chunk` selects
 * which LRB_CHUNK_BYTES window of starting positions this task owns (0 for rows scheduled whole);
 * `row_is_chunked` says whether other tasks share this row, in which case a result already
 * written by one of them ends this task immediately.
 */
template <int G, typename Tile>
__device__ __forceinline__ void lrb_search_row_tile(Tile const& tile,
                                                    column_device_view const& d_strings,
                                                    string_view const d_target,
                                                    bool* d_results,
                                                    size_type const row,
                                                    int const chunk,
                                                    bool const row_is_chunked)
{
  if (row < 0) { return; }
  // A row split into several chunk tasks is resolved as soon as any chunk finds the target:
  // skip the rest of its chunks. The load is device-scope so it is not served from a stale L1.
  if (row_is_chunked && cuda::atomic_ref<bool, cuda::thread_scope_device>(d_results[row])
                          .load(cuda::memory_order_relaxed)) {
    return;
  }

  auto const d_str        = d_strings.element<string_view>(row);
  auto const bytes        = d_str.size_bytes();
  auto const target_bytes = d_target.size_bytes();
  // starting positions owned by this task: [begin, end). A match may extend past `end`.
  auto const begin = static_cast<size_type>(int64_t{chunk} * LRB_CHUNK_BYTES);
  auto const end   = static_cast<size_type>(
    cuda::std::min<int64_t>((chunk + 1) * LRB_CHUNK_BYTES, int64_t{bytes} - target_bytes + 1));
  auto found = false;

  if constexpr (G == 1) {
    for (auto i = begin; !found && (i < end); ++i) {
      found = d_target.compare(d_str.data() + i, target_bytes) == 0;
    }
  } else {
    // Lanes cover a window of 4*G consecutive starting positions per iteration, 4 per lane.
    // The work of a row is the position of its *first* match, not its length, so the whole
    // tile stops at the iteration in which any lane matches (one vote per iteration). The loop
    // bound is tile-uniform because every lane of the tile searches the same row.
    auto constexpr bytes_per_lane = 4;
    auto constexpr window         = G * bytes_per_lane;
    auto const lane_offset        = static_cast<size_type>(tile.thread_rank()) * bytes_per_lane;
    for (auto base = begin; base < end; base += window) {
      auto const pos = base + lane_offset;
      for (auto j = 0; !found && (j < bytes_per_lane); ++j) {
        found = ((pos + j) < end) && (d_target.compare(d_str.data() + pos + j, target_bytes) == 0);
      }
      found = tile.any(found);  // tile-uniform from here on
      if (found) { break; }
    }
  }

  if (found && tile.thread_rank() == 0) { d_results[row] = true; }
}

template <int G>
__device__ __forceinline__ void lrb_search_row(
  cooperative_groups::thread_block_tile<cudf::detail::warp_size> const& warp,
  column_device_view const& d_strings,
  string_view const d_target,
  bool* d_results,
  size_type const row,
  int const chunk,
  bool const row_is_chunked)
{
  if constexpr (G == cudf::detail::warp_size) {
    lrb_search_row_tile<G>(warp, d_strings, d_target, d_results, row, chunk, row_is_chunked);
  } else {
    lrb_search_row_tile<G>(cooperative_groups::tiled_partition<G>(warp),
                           d_strings,
                           d_target,
                           d_results,
                           row,
                           chunk,
                           row_is_chunked);
  }
}

/**
 * @brief Pass 3: persistent warp search over every binned row.
 *
 * A task is 32/G consecutive rows of one bin for the sub-warp tiers, or one LRB_CHUNK_BYTES window
 * of one row for rows longer than a chunk. Task counts are 64-bit because chunk tasks can exceed
 * the row count.
 */
CUDF_KERNEL void lrb_search_kernel(column_device_view const d_strings,
                                   string_view const d_target,
                                   bool* d_results,
                                   size_type const* d_hist,
                                   size_type const* d_sorted)
{
  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane = static_cast<int>(warp.thread_rank());

  // Schedule table, one bin per lane: task counts and their prefix sums live in registers.
  auto const bin_count     = d_hist[lane];
  auto const rows_per_task = cudf::detail::warp_size / lrb_threads_per_row(lane);
  auto const chunk_shift   = lrb_chunk_shift(lane);  // chunk tasks per row = 1 << chunk_shift
  auto const bin_tasks =
    static_cast<int64_t>(cudf::util::div_rounding_up_safe(bin_count, rows_per_task)) << chunk_shift;
  auto const tasks_incl  = cg::inclusive_scan(warp, bin_tasks);
  auto const tasks_excl  = tasks_incl - bin_tasks;
  auto const bin_start   = cg::exclusive_scan(warp, bin_count);
  auto const total_tasks = warp.shfl(tasks_incl, cudf::detail::warp_size - 1);

  auto const warp_id   = cudf::detail::grid_1d::global_thread_id() / cudf::detail::warp_size;
  auto const num_warps = cudf::detail::grid_1d::grid_stride() / cudf::detail::warp_size;

  for (int64_t task = warp_id; task < total_tasks; task += num_warps) {
    // the bin owning this task is the first lane whose inclusive task count exceeds it
    auto const bin = __ffs(warp.ballot(task < tasks_incl)) - 1;
    // per-lane candidates computed for the lane's own bin, then broadcast from lane `bin`
    auto const my_local = task - tasks_excl;
    auto const my_first =
      bin_start + static_cast<size_type>(my_local >> chunk_shift) * rows_per_task;
    auto const my_chunk = my_local & ((int64_t{1} << chunk_shift) - 1);
    auto const first    = warp.shfl(my_first, bin);
    auto const chunk    = static_cast<int>(warp.shfl(my_chunk, bin));
    auto const end      = warp.shfl(bin_start + bin_count, bin);
    auto const chunked  = lrb_chunk_shift(bin) > 0;  // rows of this bin span several tasks
    auto const G        = lrb_threads_per_row(bin);
    auto const slot     = first + lane / G;
    auto const row      = (slot < end) ? d_sorted[slot] : -1;

    switch (G) {  // warp-uniform
      case 1: lrb_search_row<1>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      case 2: lrb_search_row<2>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      case 4: lrb_search_row<4>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      case 8: lrb_search_row<8>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      case 16: lrb_search_row<16>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      default: lrb_search_row<32>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
    }
  }
}

std::unique_ptr<column> contains_lrb(strings_column_view const& input,
                                     string_scalar const& target,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  if (num_rows == 0) { return make_empty_column(type_id::BOOL8); }
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  auto const d_target = string_view(target.data(), target.size());

  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     num_rows,
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  if (d_target.empty()) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 d_results,
                 d_results + num_rows,
                 true);
    results->set_null_count(input.null_count());
    return results;
  }

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const tmp_mr    = cudf::get_current_device_resource_ref();
  // [0, 32): bin counts; [32, 64): scatter cursors
  rmm::device_uvector<size_type> hist(2 * LRB_NUM_BINS, stream, tmp_mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(hist.data(), 0, hist.size() * sizeof(size_type), stream.get()));
  rmm::device_uvector<size_type> sorted(num_rows, stream, tmp_mr);

  auto const num_sms     = static_cast<int64_t>(cudf::detail::num_multiprocessors());
  auto const row_blocks  = cudf::util::div_rounding_up_safe<int64_t>(num_rows, LRB_BLOCK_SIZE);
  auto const stride_grid = static_cast<int>(std::min<int64_t>(row_blocks, num_sms * 8));

  // one row per thread (like thrust::transform) rather than grid-stride: the direct search in
  // this pass is the whole job for short-string columns and benefits from maximal parallelism
  lrb_histogram_kernel<<<static_cast<int>(row_blocks), LRB_BLOCK_SIZE, 0, stream.get()>>>(
    *d_strings, d_target, d_results, hist.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  lrb_scatter_kernel<<<stride_grid, LRB_BLOCK_SIZE, 0, stream.get()>>>(
    *d_strings, d_target.size_bytes(), d_results, hist.data(), sorted.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // persistent launch filling the GPU; the task count (rows plus chunks of long rows) is only
  // known on the device, and idle warps cost a few microseconds at most
  int max_blocks_per_sm = 0;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks_per_sm, lrb_search_kernel, LRB_BLOCK_SIZE, 0));
  auto const search_grid = static_cast<int>(std::max<int64_t>(1, max_blocks_per_sm * num_sms));
  lrb_search_kernel<<<search_grid, LRB_BLOCK_SIZE, 0, stream.get()>>>(
    *d_strings, d_target, d_results, hist.data(), sorted.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  results->set_null_count(input.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating the presence of
 * a given target string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for target.
 * @param target UTF-8 encoded string to check in strings column.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    string_scalar const& target,
                                    BoolFunction pfn,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  if (target.size() == 0)  // empty target string returns true
  {
    auto const true_scalar =
      make_fixed_width_scalar<bool>(true, stream, cudf::get_current_device_resource_ref());
    auto results = make_column_from_scalar(*true_scalar, strings.size(), stream, mr);
    results->set_null_mask(cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                           strings.null_count());
    return results;
  }

  auto d_target       = string_view(target.data(), target.size());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    d_results,
                    [d_strings, pfn, d_target] __device__(size_type idx) {
                      return !d_strings.is_null(idx) &&
                             bool{pfn(d_strings.element<string_view>(idx), d_target)};
                    });
  results->set_null_count(strings.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating the presence of
 * a string targets[i] in strings[i].
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for `targets[i]`.
 * @param targets Column of strings to be checked in `strings[i]``.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    BoolFunction pfn,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(targets.size() == strings.size(),
               "strings and targets column must be the same size");

  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{strings.size()},
    d_results,
    [d_strings, pfn, d_targets] __device__(size_type idx) {
      // empty target string returns true
      if (d_targets.is_valid(idx) && d_targets.element<string_view>(idx).length() == 0) {
        return true;
      } else if (!d_strings.is_null(idx) && !d_targets.is_null(idx)) {
        return bool{pfn(d_strings.element<string_view>(idx), d_targets.element<string_view>(idx))};
      } else {
        return false;
      }
    });
  results->set_null_count(strings.null_count());
  return results;
}
}  // namespace

std::unique_ptr<column> contains(strings_column_view const& input,
                                 string_scalar const& target,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  // Logarithmic radix binning adapts the thread-group size per row, so no column-wide
  // average-width heuristic (and no device-to-host read of the offsets) is needed.
  return contains_lrb(input, target, stream, mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    for (size_type i = 0; i <= (d_string.size_bytes() - d_target.size_bytes()); ++i) {
      if (d_target.compare(d_string.data() + i, d_target.size_bytes()) == 0) { return true; }
    }
    return false;
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return contains_fn(strings, target, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return contains_fn(strings, target, pfn, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return contains_fn(strings, targets, pfn, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 string_scalar const& target,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, target, stream, mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, targets, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, target, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, targets, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, target, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
