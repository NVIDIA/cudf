/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "mark_join.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/join/mark_join.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/device/device_transform.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/static_multiset_ref.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/sequence.h>

namespace cg = cooperative_groups;

namespace cudf {
namespace detail {
namespace {

std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");

  auto const temp_mr = cudf::get_current_device_resource_ref();
  if (nullable_columns.size() > 1) {
    auto row_bitmask =
      cudf::detail::bitmask_and(table_view{nullable_columns}, stream, temp_mr).first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream, temp_mr}, nullable_columns.front().null_mask());
}

struct row_is_null {
  bitmask_type const* _row_bitmask;
  __device__ bool operator()(size_type i) const noexcept
  {
    return !cudf::bit_is_set(_row_bitmask, i);
  }
};

static constexpr int32_t mark_block_size = 1024;

using slot_type = mark_key_type;

static_assert(sizeof(slot_type) == sizeof(uint64_t));

__device__ inline bool slot_is_empty(slot_type const& slot, slot_type const& sentinel)
{
  return *reinterpret_cast<uint64_t const*>(&slot) == *reinterpret_cast<uint64_t const*>(&sentinel);
}

template <typename FilterType>
cuco::extent<std::size_t> get_bloom_filter_blocks(std::size_t filter_bytes)
{
  std::size_t num_sub_filters =
    filter_bytes / (sizeof(typename FilterType::word_type) * FilterType::words_per_block);
  return cuda::std::max<std::size_t>(num_sub_filters, 1);
}

template <typename IteratorType, typename FilterType, typename ElementType>
class mark_join_prefilter_operator {
 public:
  mark_join_prefilter_operator(IteratorType iterator,
                               FilterType filter,
                               cudf::size_type num_elements,
                               bitmask_type const* row_bitmask)
    : _num_elements{num_elements}, _iterator{iterator}, _filter{filter}, _row_bitmask{row_bitmask}
  {
  }

  __device__ __forceinline__ bool predicate(ElementType const& slot) const
  {
    auto const row_index = static_cast<size_type>(slot.second);
    return (_row_bitmask == nullptr || cudf::bit_is_set(_row_bitmask, row_index)) &&
           _filter.contains(slot.first);
  }

  __device__ constexpr __forceinline__ ElementType accept(ElementType const& slot) const
  {
    return slot;
  }

  __device__ __forceinline__ auto get_bucket(cudf::size_type index) const
  {
    return _iterator + index;
  }

  __device__ constexpr __forceinline__ cudf::size_type num_buckets() const { return _num_elements; }

 private:
  cudf::size_type const _num_elements;
  IteratorType const _iterator;
  FilterType const _filter;
  bitmask_type const* _row_bitmask;
};

template <int32_t block_size, typename InputType, typename OutputType, typename TaskOperator>
CUDF_KERNEL __launch_bounds__(block_size) void compact_if_kernel(OutputType* out,
                                                                 cudf::size_type* global_offset,
                                                                 TaskOperator op)
{
  constexpr uint32_t warp_size = cudf::detail::warp_size;
  auto const block             = cg::this_thread_block();
  auto const warp              = cg::tiled_partition<warp_size>(block);
  auto const tid               = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride            = cudf::detail::grid_1d::grid_stride<block_size>();
  auto const loop_bound =
    cudf::util::round_up_unsafe(static_cast<thread_index_type>(op.num_buckets()), stride);

  constexpr int buffer_capacity_factor = 4;
  constexpr int buffer_capacity        = block_size * buffer_capacity_factor;
  constexpr int warp_buffer_capacity   = warp_size * buffer_capacity_factor;
  int const warp_buffer_offset         = warp_buffer_capacity * warp.meta_group_rank();
  uint32_t buffer_offset               = 0;

  __shared__ OutputType buffer[buffer_capacity];
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_off{*global_offset};

  for (thread_index_type i = tid; i < loop_bound; i += stride) {
    InputType slot{};
    bool do_fill = false;
    if (i < op.num_buckets()) {
      auto bucket = op.get_bucket(i);
      slot        = *bucket;
      do_fill     = op.predicate(slot);
    }
    bool work_todo = warp.any(do_fill);
    while (work_todo) {
      uint32_t offset = 0;
      if (do_fill) {
        auto active_group = cg::coalesced_threads();
        offset            = buffer_offset + active_group.thread_rank();
        if (offset < warp_buffer_capacity) {
          buffer[offset + warp_buffer_offset] = op.accept(slot);
          do_fill                             = false;
        }
      }
      offset        = cg::reduce(warp, offset, cg::greater<uint32_t>{});
      buffer_offset = offset + 1;
      work_todo     = (offset >= warp_buffer_capacity);
      warp.sync();

      if (work_todo) {
        buffer_offset            = 0;
        auto const output_offset = cg::invoke_one_broadcast(warp, [&]() {
          return global_off.fetch_add(warp_buffer_capacity, cuda::memory_order_relaxed);
        });
#pragma unroll
        for (int k = warp.thread_rank(); k < warp_buffer_capacity; k += warp_size) {
          out[output_offset + k] = buffer[k + warp_buffer_offset];
        }
      }
      warp.sync();
    }
  }

  if (buffer_offset > 0) {
    auto const output_offset = cg::invoke_one_broadcast(
      warp, [&]() { return global_off.fetch_add(buffer_offset, cuda::memory_order_relaxed); });
    for (uint32_t k = warp.thread_rank(); k < buffer_offset; k += warp.num_threads()) {
      out[output_offset + k] = buffer[k + warp_buffer_offset];
    }
  }
}

template <int32_t block_size, typename Comparator>
CUDF_KERNEL __launch_bounds__(block_size) void mark_probe_kernel(
  storage_ref_type storage,
  masked_probing_scheme probing_scheme,
  Comparator comparator,
  slot_type empty_sentinel,
  right_key_type const* __restrict__ right_rows,
  cudf::size_type num_rows,
  cudf::size_type* global_mark_counter,
  bitmask_type const* right_row_bitmask)
{
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  cudf::size_type mark_counter = 0;
  __shared__ cuda::atomic<cudf::size_type, cuda::thread_scope_block> block_mark_counter;
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_counter{*global_mark_counter};
  cg::invoke_one(block, [&]() { block_mark_counter.store(0, cuda::memory_order_relaxed); });
  block.sync();

  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();

  for (thread_index_type i = tid; i < num_rows; i += stride) {
    bool const is_active = (right_row_bitmask == nullptr || cudf::bit_is_set(right_row_bitmask, i));

    if (is_active) {
      auto const query  = right_rows[i];
      auto probing_iter = probing_scheme.template make_iterator<storage_ref_type::bucket_size>(
        query, storage.extent());

      bool found_empty = false;
      while (!found_empty) {
        auto const bucket_idx  = *probing_iter;
        auto* mutable_slot_p   = &storage.data()[bucket_idx];
        auto const entry_value = *mutable_slot_p;

        if (slot_is_empty(entry_value, empty_sentinel)) {
          found_empty = true;
        } else {
          auto const right_hash = unset_mark(query.first);
          auto const entry_hash = unset_mark(entry_value.first);
          if (right_hash == entry_hash && comparator(query, entry_value)) {
            auto expected = entry_value.first;
            if (!is_marked(expected)) {
              auto const desired = set_mark(expected);
              cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> key_ref{
                mutable_slot_p->first};
              if (key_ref.compare_exchange_strong(expected, desired, cuda::memory_order_relaxed)) {
                ++mark_counter;
              }
            }
          }
        }
        ++probing_iter;
      }
    }
  }

  auto const warp_sum = cg::reduce(warp, mark_counter, cg::plus<cudf::size_type>{});
  cg::invoke_one(warp,
                 [&]() { block_mark_counter.fetch_add(warp_sum, cuda::memory_order_relaxed); });
  block.sync();

  cg::invoke_one(block, [&]() {
    global_counter.fetch_add(block_mark_counter.load(cuda::memory_order_relaxed),
                             cuda::memory_order_relaxed);
  });
}

template <int32_t block_size, bool is_anti_join>
CUDF_KERNEL __launch_bounds__(block_size) void mark_retrieve_kernel(
  storage_ref_type storage,
  slot_type empty_sentinel,
  cudf::size_type* __restrict__ output,
  cudf::size_type* global_offset,
  cudf::thread_index_type num_buckets)
{
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  constexpr int buffer_capacity_factor = 4;
  constexpr int warp_buffer_capacity   = cudf::detail::warp_size * buffer_capacity_factor;
  constexpr int buffer_capacity        = block_size * buffer_capacity_factor;
  int const warp_buffer_offset         = warp_buffer_capacity * warp.meta_group_rank();
  uint32_t left_buffer_offset          = 0;

  // Each warp owns a fixed slice of shared memory and batches row indices there before
  // reserving a contiguous range in the global output.
  __shared__ cudf::size_type left_buffer[buffer_capacity];
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_off{*global_offset};

  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  auto const loop_bound =
    cudf::util::round_up_unsafe(static_cast<thread_index_type>(num_buckets), stride);

  for (thread_index_type i = tid; i < loop_bound; i += stride) {
    bool has_match = false;
    cudf::size_type row_idx{};

    if (i < num_buckets) {
      auto const entry_value = storage.data()[i];
      bool const is_filled   = !slot_is_empty(entry_value, empty_sentinel);
      if (is_filled) {
        bool const marked = is_marked(entry_value.first);
        // A marked entry has at least one probe-side match. Semi joins emit marked
        // left rows, while anti joins emit the entries that were never marked.
        if constexpr (is_anti_join) {
          has_match = !marked;
        } else {
          has_match = marked;
        }
        if (has_match) { row_idx = static_cast<cudf::size_type>(entry_value.second); }
      }
    }

    // Compact the active lanes into the warp-local buffer. coalesced_threads()
    // gives each matching lane a dense rank among the currently active lanes so
    // the warp writes a contiguous chunk into shared memory.
    //
    // If any lane would overflow the warp-local buffer, leave that lane pending,
    // flush the full buffer to global memory, and retry the pending lanes.
    bool pending_writes = warp.any(has_match);
    while (pending_writes) {
      uint32_t offset = 0;
      if (has_match) {
        auto active_group = cg::coalesced_threads();
        offset            = left_buffer_offset + active_group.thread_rank();
        if (offset < static_cast<uint32_t>(warp_buffer_capacity)) {
          left_buffer[offset + warp_buffer_offset] = row_idx;
          has_match                                = false;
        }
      }
      offset             = cg::reduce(warp, offset, cg::greater<uint32_t>{});
      left_buffer_offset = offset + 1;
      pending_writes     = (offset >= static_cast<uint32_t>(warp_buffer_capacity));
      warp.sync();

      if (pending_writes) {
        left_buffer_offset = 0;
        // Reserve one contiguous output range for the whole warp and flush the
        // full warp-local buffer there.
        auto const output_offset = cg::invoke_one_broadcast(warp, [&]() {
          return global_off.fetch_add(warp_buffer_capacity, cuda::memory_order_relaxed);
        });
#pragma unroll
        for (int k = warp.thread_rank(); k < warp_buffer_capacity; k += cudf::detail::warp_size) {
          output[output_offset + k] = left_buffer[k + warp_buffer_offset];
        }
      }
      warp.sync();
    }
  }
  block.sync();

  // Drain the partially filled warp-local buffer left after the final scan iteration.
  if (left_buffer_offset > 0) {
    auto const output_offset = cg::invoke_one_broadcast(
      warp, [&]() { return global_off.fetch_add(left_buffer_offset, cuda::memory_order_relaxed); });
    for (uint32_t k = warp.thread_rank(); k < left_buffer_offset; k += warp.num_threads()) {
      output[output_offset + k] = left_buffer[k + warp_buffer_offset];
    }
  }
}

template <int32_t block_size>
CUDF_KERNEL __launch_bounds__(block_size) void clear_marks_kernel(
  storage_ref_type storage, slot_type empty_sentinel, cudf::thread_index_type num_buckets)
{
  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (thread_index_type i = tid; i < num_buckets; i += stride) {
    auto const entry_value = storage.data()[i];
    if (!slot_is_empty(entry_value, empty_sentinel)) {
      if (is_marked(entry_value.first)) {
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> key_ref{
          storage.data()[i].first};
        key_ref.store(unset_mark(entry_value.first), cuda::memory_order_relaxed);
      }
    }
  }
}

static std::size_t compute_mark_join_capacity(cudf::table_view tbl, double load_factor)
{
  return static_cast<std::size_t>(
    cuco::make_valid_extent<masked_probing_scheme, mark_storage_type, std::size_t>(tbl.num_rows(),
                                                                                   load_factor));
}

}  // namespace

void mark_join::clear_marks(rmm::cuda_stream_view stream)
{
  auto const storage_ref = _bucket_storage.ref();
  auto const num_buckets = static_cast<cudf::thread_index_type>(storage_ref.num_buckets());
  if (num_buckets == 0) return;

  auto const grid_size = cudf::util::div_rounding_up_unsafe(num_buckets, mark_block_size);
  clear_marks_kernel<mark_block_size><<<grid_size, mark_block_size, 0, stream.value()>>>(
    storage_ref, static_cast<slot_type>(masked_empty_sentinel), num_buckets);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Comparator>
cudf::size_type mark_join::mark_probe_without_prefilter(storage_ref_type storage_ref,
                                                        Comparator comparator,
                                                        right_key_type const* right_rows,
                                                        cudf::size_type num_right_rows,
                                                        bitmask_type const* right_row_bitmask,
                                                        rmm::cuda_stream_view stream)
{
  cudf::detail::device_scalar<cudf::size_type> d_mark_counter(
    0, stream, cudf::get_current_device_resource_ref());

  if (num_right_rows > 0) {
    int grid_size = 0;
    CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &grid_size, mark_probe_kernel<mark_block_size, Comparator>, mark_block_size, 0));
    int num_sms = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    grid_size *= num_sms;

    mark_probe_kernel<mark_block_size><<<grid_size, mark_block_size, 0, stream.value()>>>(
      storage_ref,
      masked_probing_scheme{},
      comparator,
      static_cast<slot_type>(masked_empty_sentinel),
      right_rows,
      num_right_rows,
      d_mark_counter.data(),
      right_row_bitmask);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return d_mark_counter.value(stream);
}

template <typename Comparator>
cudf::size_type mark_join::mark_probe_with_prefilter(storage_ref_type storage_ref,
                                                     Comparator comparator,
                                                     right_key_type const* right_rows,
                                                     cudf::size_type num_right_rows,
                                                     bitmask_type const* right_row_bitmask,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(_bloom_filter != nullptr, "Prefilter-enabled mark_join is missing bloom filter.");

  auto filtered_right_rows = rmm::device_uvector<right_key_type>(num_right_rows, stream, mr);
  cudf::detail::device_scalar<cudf::size_type> d_filtered_count(0, stream, mr);

  auto const filter_ref = _bloom_filter->ref();
  using prefilter_operator_type =
    mark_join_prefilter_operator<right_key_type const*, decltype(filter_ref), right_key_type>;
  auto const prefilter_op =
    prefilter_operator_type{right_rows, filter_ref, num_right_rows, right_row_bitmask};

  int filter_grid_size = 0;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &filter_grid_size,
    compact_if_kernel<mark_block_size, right_key_type, right_key_type, prefilter_operator_type>,
    mark_block_size,
    0));
  int num_sms = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  filter_grid_size *= num_sms;

  compact_if_kernel<mark_block_size, right_key_type, right_key_type, prefilter_operator_type>
    <<<filter_grid_size, mark_block_size, 0, stream.value()>>>(
      filtered_right_rows.data(), d_filtered_count.data(), prefilter_op);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const filtered_count = d_filtered_count.value(stream);
  return mark_probe_without_prefilter(
    storage_ref, comparator, filtered_right_rows.data(), filtered_count, nullptr, stream);
}

template <typename Comparator>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::mark_probe_and_retrieve(
  cudf::table_view const& right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_right,
  join_kind kind,
  Comparator comparator,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const temp_mr          = cudf::get_current_device_resource_ref();
  auto materialize_right_rows = [&](auto const& key_fn) {
    rmm::device_uvector<right_key_type> right_rows(right.num_rows(), stream, temp_mr);
    cub::DeviceTransform::Transform(cuda::counting_iterator<size_type>{0},
                                    right_rows.begin(),
                                    right.num_rows(),
                                    cuda::proclaim_return_type<right_key_type>(key_fn),
                                    stream.value());
    return right_rows;
  };

  rmm::device_uvector<right_key_type> right_rows(0, stream, temp_mr);
  if (is_primitive_row_op_compatible(_left)) {
    auto const d_right_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, preprocessed_right};
    right_rows =
      materialize_right_rows(masked_key_fn<rhs_index_type, primitive_row_hasher>{d_right_hasher});
  } else {
    auto const d_right_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_right}.device_hasher(nullate::YES{});
    right_rows = materialize_right_rows(masked_key_fn<rhs_index_type, row_hasher>{d_right_hasher});
  }

  auto const storage_ref = _bucket_storage.ref();
  auto const num_buckets = static_cast<cudf::thread_index_type>(storage_ref.num_buckets());

  auto const right_has_nulls = has_nested_nulls(right);
  rmm::device_buffer right_bitmask_buffer(0, stream, temp_mr);
  bitmask_type const* right_bitmask_ptr = nullptr;
  if (right_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
    auto bitmask_buffer_and_ptr = build_row_bitmask(right, stream);
    right_bitmask_buffer        = std::move(bitmask_buffer_and_ptr.first);
    right_bitmask_ptr           = bitmask_buffer_and_ptr.second;
  }

  auto const marked_count =
    (_prefilter == cudf::join_prefilter::YES)
      ? mark_probe_with_prefilter(storage_ref,
                                  comparator,
                                  right_rows.data(),
                                  right.num_rows(),
                                  right_bitmask_ptr,
                                  stream,
                                  mr)
      : mark_probe_without_prefilter(
          storage_ref, comparator, right_rows.data(), right.num_rows(), right_bitmask_ptr, stream);

  auto const null_left_rows  = static_cast<size_type>(_left.num_rows()) - num_left_inserted();
  auto const is_anti         = (kind == join_kind::LEFT_ANTI_JOIN);
  auto const unmatched_valid = num_left_inserted() - marked_count;
  auto const null_contribution =
    (is_anti && _nulls_equal == null_equality::UNEQUAL) ? null_left_rows : size_type{0};
  auto const result_count =
    (kind == join_kind::LEFT_SEMI_JOIN) ? marked_count : (unmatched_valid + null_contribution);

  auto result = rmm::device_uvector<size_type>(result_count, stream, mr);
  if (result_count == 0) {
    return std::make_unique<rmm::device_uvector<size_type>>(std::move(result));
  }

  cudf::detail::device_scalar<cudf::size_type> d_scan_offset(0, stream, temp_mr);

  {
    int grid_size = 0;
    if (kind == join_kind::LEFT_SEMI_JOIN) {
      CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size, mark_retrieve_kernel<mark_block_size, false>, mark_block_size, 0));
    } else {
      CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size, mark_retrieve_kernel<mark_block_size, true>, mark_block_size, 0));
    }
    int num_sms = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    grid_size *= num_sms;

    if (kind == join_kind::LEFT_SEMI_JOIN) {
      mark_retrieve_kernel<mark_block_size, false>
        <<<grid_size, mark_block_size, 0, stream.value()>>>(
          storage_ref,
          static_cast<slot_type>(masked_empty_sentinel),
          result.data(),
          d_scan_offset.data(),
          num_buckets);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      mark_retrieve_kernel<mark_block_size, true>
        <<<grid_size, mark_block_size, 0, stream.value()>>>(
          storage_ref,
          static_cast<slot_type>(masked_empty_sentinel),
          result.data(),
          d_scan_offset.data(),
          num_buckets);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  }

  if (null_contribution > 0) {
    auto const bitmask_buffer_and_ptr = build_row_bitmask(_left, stream);
    auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
    thrust::copy_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator{_left.num_rows()},
                    result.begin() + unmatched_valid,
                    row_is_null{row_bitmask_ptr});
  }

  return std::make_unique<rmm::device_uvector<size_type>>(std::move(result));
}

mark_join::mark_join(cudf::table_view const& left,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     cudf::join_prefilter prefilter,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _has_nested_columns{cudf::has_nested_columns(left)},
    _left{left},
    _nulls_equal{compare_nulls},
    _prefilter{prefilter},
    _preprocessed_left{cudf::detail::row::equality::preprocessed_table::create(left, stream)},
    _bucket_storage{
      cuco::extent<std::size_t>{compute_mark_join_capacity(left, checked_load_factor(load_factor))},
      rmm::mr::polymorphic_allocator<char>{mr},
      stream.value()}
{
  cudf::scoped_range range{"mark_join::mark_join"};
  if (_left.num_rows() == 0) return;
  _bucket_storage.initialize(masked_empty_sentinel, stream);
  if (_prefilter == cudf::join_prefilter::YES) {
    _bloom_filter = std::make_unique<bloom_filter_type>(
      get_bloom_filter_blocks<bloom_filter_type>(_left.num_rows()),
      cuco::cuda_thread_scope<cuda::thread_scope_device>{},
      bloom_filter_policy_type{},
      bloom_filter_allocator_type{std::move(mr)},
      stream.value());
  }

  // Any mismatch in nullate between right and left row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of right nulls and left nulls. However, since we do not
  // know if the right has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // left and right row operators use consistent null handling.
  auto const has_null_left_keys =
    cudf::has_nested_nulls(_left) && _nulls_equal == null_equality::UNEQUAL;

  auto const temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_buffer row_bitmask_buffer(0, stream, temp_mr);
  bitmask_type const* row_bitmask_ptr = nullptr;
  if (has_null_left_keys) {
    auto bitmask_buffer_and_ptr = build_row_bitmask(_left, stream);
    row_bitmask_buffer          = std::move(bitmask_buffer_and_ptr.first);
    row_bitmask_ptr             = bitmask_buffer_and_ptr.second;
  }

  auto do_bloom_filter_insert = [&](auto const& hashes) {
    if (_bloom_filter == nullptr) return;
    if (has_null_left_keys) {
      _bloom_filter->add_if_async(hashes.begin(),
                                  hashes.end(),
                                  cuda::counting_iterator{size_type{0}},
                                  row_is_valid{row_bitmask_ptr},
                                  stream.value());
    } else {
      _bloom_filter->add_async(hashes.begin(), hashes.end(), stream.value());
    }
  };

  auto do_insert = [&](auto const& left_iter, auto const& insert_ref) {
    auto const grid_size =
      cuco::detail::grid_size(_left.num_rows(), masked_probing_scheme::cg_size);
    if (has_null_left_keys) {
      cuco::detail::open_addressing_ns::insert_if_n<masked_probing_scheme::cg_size,
                                                    cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          left_iter,
          _left.num_rows(),
          cuda::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<masked_probing_scheme::cg_size,
                                                    cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          left_iter,
          _left.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          insert_ref);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  };

  auto do_build_and_filter = [&](auto const& d_left_hasher, auto const& d_left_comparator) {
    if (_prefilter == cudf::join_prefilter::YES) {
      rmm::device_uvector<hash_value_type> left_hashes(_left.num_rows(), stream, temp_mr);
      cub::DeviceTransform::Transform(
        cuda::counting_iterator{size_type{0}},
        left_hashes.begin(),
        _left.num_rows(),
        cuda::proclaim_return_type<hash_value_type>(masked_hash_value_fn{d_left_hasher}),
        stream.value());
      auto const left_iter = cudf::detail::make_counting_transform_iterator(
        size_type{0}, hash_pair_fn<lhs_index_type>{left_hashes.data()});
      cuco::static_multiset_ref set_ref{masked_empty_sentinel,
                                        insertion_adapter{d_left_comparator},
                                        masked_probing_scheme{},
                                        cuco::thread_scope_device,
                                        _bucket_storage.ref()};
      do_insert(left_iter, set_ref.rebind_operators(cuco::insert));
      do_bloom_filter_insert(left_hashes);
    } else {
      auto const left_iter = cudf::detail::make_counting_transform_iterator(
        size_type{0},
        masked_key_fn<lhs_index_type, std::decay_t<decltype(d_left_hasher)>>{d_left_hasher});
      cuco::static_multiset_ref set_ref{masked_empty_sentinel,
                                        insertion_adapter{d_left_comparator},
                                        masked_probing_scheme{},
                                        cuco::thread_scope_device,
                                        _bucket_storage.ref()};
      do_insert(left_iter, set_ref.rebind_operators(cuco::insert));
    }
  };

  if (is_primitive_row_op_compatible(left)) {
    auto const d_left_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_left, _preprocessed_left, compare_nulls};
    auto const d_left_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, _preprocessed_left};
    do_build_and_filter(d_left_hasher, d_left_comparator);
  } else if (_has_nested_columns) {
    auto const d_left_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_left}.equal_to<true>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    auto const d_left_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_left}.device_hasher(nullate::YES{});
    do_build_and_filter(d_left_hasher, d_left_comparator);
  } else {
    auto const d_left_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_left}.equal_to<false>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    auto const d_left_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_left}.device_hasher(nullate::YES{});
    do_build_and_filter(d_left_hasher, d_left_comparator);
  }

  if (has_null_left_keys) {
    _num_left_inserted = cudf::detail::count_set_bits(row_bitmask_ptr, 0, _left.num_rows(), stream);
  } else {
    _num_left_inserted = _left.num_rows();
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_anti_join(
  cudf::table_view const& right,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  clear_marks(stream);

  auto const preprocessed_right = [&right, stream] {
    cudf::scoped_range range{"mark_join::semi_anti_join::preprocessed_right"};
    return cudf::detail::row::equality::preprocessed_table::create(right, stream);
  }();

  if (is_primitive_row_op_compatible(_left)) {
    auto const d_left_right_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_left, preprocessed_right, _nulls_equal};

    return mark_probe_and_retrieve(
      right, preprocessed_right, kind, masked_comparator_fn{d_left_right_comparator}, stream, mr);
  } else {
    auto const d_left_right_comparator =
      cudf::detail::row::equality::two_table_comparator{_preprocessed_left, preprocessed_right};

    if (_has_nested_columns) {
      auto d_left_right_nan_comparator = d_left_right_comparator.equal_to<true>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      return mark_probe_and_retrieve(right,
                                     preprocessed_right,
                                     kind,
                                     masked_comparator_fn{d_left_right_nan_comparator},
                                     stream,
                                     mr);
    } else {
      auto d_left_right_nan_comparator = d_left_right_comparator.equal_to<false>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      return mark_probe_and_retrieve(right,
                                     preprocessed_right,
                                     kind,
                                     masked_comparator_fn{d_left_right_nan_comparator},
                                     stream,
                                     mr);
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_join(
  cudf::table_view const& right, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (_left.num_rows() == 0 || right.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  return semi_anti_join(right, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::anti_join(
  cudf::table_view const& right, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (_left.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if (right.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(_left.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     result->begin(),
                     result->end());
    return result;
  }
  return semi_anti_join(right, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

// Public API

mark_join::~mark_join() = default;

mark_join::mark_join(cudf::table_view const& left,
                     cudf::null_equality compare_nulls,
                     cudf::join_prefilter prefilter,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<detail::mark_join>(
      left, compare_nulls, detail::CUCO_DESIRED_LOAD_FACTOR, prefilter, stream, std::move(mr))}
{
}

mark_join::mark_join(cudf::table_view const& left,
                     double load_factor,
                     cudf::null_equality compare_nulls,
                     cudf::join_prefilter prefilter,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<detail::mark_join>(
      left, compare_nulls, load_factor, prefilter, stream, std::move(mr))}
{
}

std::unique_ptr<rmm::device_uvector<size_type>> mark_join::semi_join(
  cudf::table_view const& right,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"mark_join::semi_join"};
  return _impl->semi_join(right, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mark_join::anti_join(
  cudf::table_view const& right,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"mark_join::anti_join"};
  return _impl->anti_join(right, stream, mr);
}

}  // namespace cudf
