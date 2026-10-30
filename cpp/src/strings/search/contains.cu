/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/stream>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Logarithmic Radix Binning (LRB) scheduler for contains(column, scalar).
 *
 * A single execution shape for a whole column fits rows of varying length poorly: with
 * thread-per-row, a warp runs at the pace of its longest row while the other lanes idle, and with
 * warp-per-row, a short row occupies a full warp while a very long row still gets only one warp.
 * LRB bins rows by ceil(log2(bytes)), so all rows in a bin are within 2x of each other, and each
 * bin is searched by a thread group sized for it.
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
 *                          {8,16,32} by bin, or, for rows longer than LRB_CHUNK_BYTES, one
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
 * bin b holds rows with 2^(b-1) < bytes <= 2^b. Only rows longer than LRB_DIRECT_MAX_BYTES reach
 * the search pass, so with LRB_SHIFT = 4:
 *   65..128 bytes: 8 threads   129..256: 16   > 256: 32 (a warp)
 * and each lane scans at most ~16 starting positions per row before the group moves on.
 */
constexpr int LRB_SHIFT = 4;

constexpr int LRB_CHUNK_SHIFT = 12;  // log2(LRB_CHUNK_BYTES)

/**
 * @brief Rows longer than this are scheduled as several warp tasks, one per chunk of this many
 * bytes, so every task carries a bounded amount of work regardless of row length.
 *
 * A warp scans a 4 KiB chunk in 16 steps of 256 bytes. A 1 MiB row becomes 256 independent tasks
 * that spread across the GPU, and a column with many 16 KiB rows produces 4 tasks each; neither
 * case depends on how many long rows there are. Chunks of one row never conflict: the scatter pass
 * writes `false` for every binned row and chunk tasks only ever write `true`.
 */
constexpr int LRB_CHUNK_BYTES = 1 << LRB_CHUNK_SHIFT;  // 4 KiB

/**
 * @brief Rows of at most this many bytes are searched thread-per-row directly inside the
 * histogram pass and are never binned or scattered.
 *
 * For short rows the fixed cost of the permutation (two extra passes over the offsets plus
 * writing and reading one index per row) exceeds the imbalance it removes, because a warp of
 * thread-per-row lanes can only wait on at most this many bytes.
 */
constexpr size_type LRB_DIRECT_MAX_BYTES = 64;

/**
 * @brief Bin index of a row. Bin 0 = resolved in the histogram pass (null, shorter than the
 * target, or at most LRB_DIRECT_MAX_BYTES bytes).
 *
 * @param size_bytes Row length in bytes (any value if is_null)
 * @param target_bytes Target length in bytes (> 0)
 * @param is_null Whether the row is null
 * @return 0, or a bin in [7, 31] for rows longer than LRB_DIRECT_MAX_BYTES
 */
__device__ __forceinline__ int lrb_bin(size_type size_bytes, size_type target_bytes, bool is_null)
{
  if (is_null || size_bytes < target_bytes || size_bytes <= LRB_DIRECT_MAX_BYTES) { return 0; }
  // ceil(log2(bytes)): bin b holds 2^(b-1) < bytes <= 2^b, so a row of exactly 2^b bytes sits at
  // the top of bin b and the bin's chunk count (2^b / LRB_CHUNK_BYTES) is exact for it
  return cuda::std::bit_width(static_cast<unsigned>(size_bytes - 1));
}

/**
 * @brief Threads that cooperate on one row of `bin`, i.e. 1 << clamp(bin - LRB_SHIFT, 0, 5).
 *
 * Bins below 7 are never populated, but the search kernel evaluates this for every lane of its
 * bin table, so the lower clamp keeps the shift valid for them.
 *
 * @param bin Bin index
 * @return 8, 16 or 32 for the bins that reach the search pass
 */
CUDF_HOST_DEVICE constexpr int lrb_threads_per_row(int bin)
{
  auto const shift = bin - LRB_SHIFT;
  return 1 << (shift < 0 ? 0 : (shift > 5 ? 5 : shift));
}

// The search kernel only instantiates tiles of 8, 16 and 32 threads: the shortest rows it sees
// are LRB_DIRECT_MAX_BYTES + 1 = 65 bytes, which fall in bin 7.
static_assert(LRB_DIRECT_MAX_BYTES == 64 && lrb_threads_per_row(7) == 8,
              "lrb_search_kernel must dispatch every tile size lrb_threads_per_row() returns");

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
  for (auto idx = cudf::detail::grid_1d::global_thread_id(); idx < num_rows; idx += stride) {
    auto const row     = static_cast<size_type>(idx);
    auto const is_null = d_strings.is_null(row);
    auto const bytes   = is_null ? 0 : d_strings.element<string_view>(row).size_bytes();
    auto const bin     = lrb_bin(bytes, target_bytes, is_null);
    if (bin == 0) {
      // resolved here: false unless it is a short valid row that actually contains the target
      auto found = false;
      if (!is_null && bytes >= target_bytes) {
        auto const d_str      = d_strings.element<string_view>(row);
        auto const first_char = d_target.data()[0];
        for (size_type i = 0; !found && (i <= bytes - target_bytes); ++i) {
          found =
            d_str.data()[i] == first_char && d_target.compare(d_str.data() + i, target_bytes) == 0;
        }
      }
      d_results[row] = found;
    }
    // one shared atomic per (warp, distinct bin) instead of one per lane; nothing at all when
    // the whole warp was resolved directly (the common case for short-string columns)
    auto const active = cooperative_groups::coalesced_threads();
    if (active.any(bin != 0)) {
      auto const peers = active.match_any(bin);  // lanes of `active` with the same bin
      if (bin != 0 &&
          active.thread_rank() == static_cast<unsigned>(cuda::std::countr_zero(peers))) {
        cuda::atomic_ref<size_type, cuda::thread_scope_block>{s_hist[bin]}.fetch_add(
          cuda::std::popcount(peers), cuda::memory_order_relaxed);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < LRB_NUM_BINS && s_hist[threadIdx.x] != 0) {
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{d_hist[threadIdx.x]}.fetch_add(
      s_hist[threadIdx.x], cuda::memory_order_relaxed);
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

  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane  = warp.thread_rank();
  auto const count = d_hist[lane];
  // every warp reads the same counts, so this exit is uniform across the block
  if (!warp.any(count != 0)) { return; }  // every row was resolved in the histogram pass
  if (warp.meta_group_rank() == 0) {
    auto const incl = cg::inclusive_scan(warp, count);
    s_start[lane]   = incl - count;
  }
  __syncthreads();

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
    auto const peers  = warp.match_any(bin);
    auto const leader = cuda::std::countr_zero(peers);
    size_type wbase   = 0;
    if (bin != 0 && lane == static_cast<unsigned>(leader)) {
      wbase = cuda::atomic_ref<size_type, cuda::thread_scope_block>{s_count[bin]}.fetch_add(
        cuda::std::popcount(peers), cuda::memory_order_relaxed);
    }
    wbase           = warp.shfl(wbase, leader);
    auto const rank = wbase + cuda::std::popcount(peers & ((1u << lane) - 1u));
    __syncthreads();

    if (threadIdx.x < LRB_NUM_BINS && s_count[threadIdx.x] != 0) {
      s_base[threadIdx.x] =
        cuda::atomic_ref<size_type, cuda::thread_scope_device>{d_cursor[threadIdx.x]}.fetch_add(
          s_count[threadIdx.x], cuda::memory_order_relaxed);
    }
    __syncthreads();

    if (bin != 0) {
      d_sorted[s_start[bin] + s_base[bin] + rank] = static_cast<size_type>(idx);
      d_results[idx]                              = false;  // chunk tasks only ever set true
    }
  }
}

/**
 * @brief Search one row with a tile of G threads.
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
  auto result = cuda::atomic_ref<bool, cuda::thread_scope_device>(d_results[row]);
  // A row split into several chunk tasks is resolved as soon as any chunk finds the target:
  // skip the rest of its chunks. The vote makes the exit uniform across the tile even if its
  // lanes observe the result at different times.
  if (row_is_chunked && tile.any(result.load(cuda::memory_order_relaxed))) { return; }

  auto const d_str        = d_strings.element<string_view>(row);
  auto const target_bytes = d_target.size_bytes();
  // Starting positions owned by this task are [begin, begin + count); a match may extend past
  // them. Positions are relative to `begin` so the counters stay small: a task never owns more
  // than LRB_CHUNK_BYTES positions, even in the last chunk of a row near 2 GB.
  auto const begin      = int64_t{chunk} * LRB_CHUNK_BYTES;
  auto const count      = static_cast<size_type>(cuda::std::min<int64_t>(
    LRB_CHUNK_BYTES, int64_t{d_str.size_bytes()} - target_bytes + 1 - begin));
  auto const d_chunk    = d_str.data() + begin;
  auto const first_char = d_target.data()[0];

  // Lanes cover a window of 8*G consecutive starting positions per iteration, 8 per lane.
  // The work of a row is the position of its *first* match, not its length, so the whole
  // tile stops at the iteration in which any lane matches (one vote per iteration). The loop
  // bound is tile-uniform because every lane of the tile searches the same row.
  auto constexpr bytes_per_lane = 8;
  auto constexpr window         = G * bytes_per_lane;
  auto const lane_offset        = static_cast<size_type>(tile.thread_rank()) * bytes_per_lane;
  auto found                    = false;
  for (size_type base = 0; base < count; base += window) {
    auto const pos = base + lane_offset;
    for (auto j = 0; !found && (j < bytes_per_lane); ++j) {
      // checking the first byte in a register skips the full compare at most positions
      found = ((pos + j) < count) && d_chunk[pos + j] == first_char &&
              (d_target.compare(d_chunk + pos + j, target_bytes) == 0);
    }
    found = tile.any(found);  // tile-uniform from here on
    if (found) { break; }
  }

  if (found && tile.thread_rank() == 0) { result.store(true, cuda::memory_order_relaxed); }
}

/**
 * @brief Dispatch lrb_search_row_tile() with a tile of G lanes carved from `warp`.
 *
 * The warp itself is the tile when G == 32; otherwise the warp is partitioned into 32/G tiles that
 * each take one row.
 *
 * @tparam G Threads per row for this bin
 * @param warp The calling warp
 * @param d_strings Column of input strings
 * @param d_target String to search for
 * @param d_results Output flags, one per row
 * @param row Row for this lane's tile, or -1 when the tile has no row
 * @param chunk Index of the LRB_CHUNK_BYTES window this task owns
 * @param row_is_chunked Whether other tasks share this row
 */
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
  auto const bin_count = d_hist[lane];
  if (!warp.any(bin_count != 0)) { return; }  // every row was resolved in the histogram pass
  auto const rows_per_task = cudf::detail::warp_size / lrb_threads_per_row(lane);
  // log2 of the chunk tasks per row of this lane's bin: rows in bin b have at most 2^b bytes
  auto const chunk_shift = cuda::std::max(lane - LRB_CHUNK_SHIFT, 0);
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
    auto const bin = cuda::std::countr_zero(warp.ballot(task < tasks_incl));
    // per-lane candidates computed for the lane's own bin, then broadcast from lane `bin`
    auto const my_local = task - tasks_excl;
    auto const my_first =
      bin_start + static_cast<size_type>(my_local >> chunk_shift) * rows_per_task;
    auto const my_chunk = my_local & ((int64_t{1} << chunk_shift) - 1);
    auto const first    = warp.shfl(my_first, bin);
    auto const chunk    = static_cast<int>(warp.shfl(my_chunk, bin));
    auto const end      = warp.shfl(bin_start + bin_count, bin);
    auto const chunked  = bin > LRB_CHUNK_SHIFT;  // rows of this bin span several tasks
    auto const G        = lrb_threads_per_row(bin);
    auto const slot     = first + lane / G;
    auto const row      = (slot < end) ? d_sorted[slot] : -1;

    switch (G) {  // warp-uniform
      case 8: lrb_search_row<8>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      case 16: lrb_search_row<16>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
      default: lrb_search_row<32>(warp, d_strings, d_target, d_results, row, chunk, chunked); break;
    }
  }
}

}  // namespace

/**
 * @brief contains(column, scalar) scheduled with logarithmic radix binning.
 *
 * Runs the histogram, scatter and search passes described above. Rows of at most
 * LRB_DIRECT_MAX_BYTES are resolved in the histogram pass; longer rows are binned by
 * ceil(log2(bytes)) and searched by thread groups sized for their bin, with rows longer than
 * LRB_CHUNK_BYTES split into one task per chunk.
 *
 * @param input Strings column to search
 * @param target Scalar string to look for in each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return BOOL8 column, true where the row contains `target`; nulls follow the input
 */
std::unique_ptr<column> contains(strings_column_view const& input,
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
  auto hist = cuda::device_buffer<size_type>(stream, tmp_mr, 2 * LRB_NUM_BINS, cuda::no_init);
  CUDF_CUDA_TRY(cudaMemsetAsync(hist.data(), 0, hist.size() * sizeof(size_type), stream.get()));
  auto sorted = cuda::device_buffer<size_type>(stream, tmp_mr, num_rows, cuda::no_init);

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

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) { return make_empty_column(type_id::BOOL8); }

  CUDF_EXPECTS(targets.size() == strings.size(),
               "strings and targets column must be the same size");

  auto const d_strings = column_device_view::create(strings.parent(), stream);
  auto const d_targets = column_device_view::create(targets.parent(), stream);

  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{strings.size()},
    results->mutable_view().data<bool>(),
    [d_strings = *d_strings, d_targets = *d_targets] __device__(size_type idx) {
      // empty target string returns true
      if (d_targets.is_valid(idx) && d_targets.element<string_view>(idx).length() == 0) {
        return true;
      }
      if (d_strings.is_null(idx) || d_targets.is_null(idx)) { return false; }
      auto const d_str    = d_strings.element<string_view>(idx);
      auto const d_target = d_targets.element<string_view>(idx);
      for (size_type i = 0; i <= (d_str.size_bytes() - d_target.size_bytes()); ++i) {
        if (d_target.compare(d_str.data() + i, d_target.size_bytes()) == 0) { return true; }
      }
      return false;
    });
  results->set_null_count(strings.null_count());
  return results;
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

}  // namespace strings
}  // namespace cudf
