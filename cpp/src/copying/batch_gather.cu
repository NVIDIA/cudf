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
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/stream>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cudf {
namespace detail {
namespace {

// Block size of the fused kernel; matches the valid_if_n_kernel/concatenate precedent.
constexpr size_type block_size{256};

/**
 * @brief Type-erased per-column parameters consumed by `batch_gather_kernel`.
 *
 * One descriptor per fixed-width source column. All pointers are element bases, not byte bases:
 * `src_data` comes from `column_view::data<T>()`, which has the slice offset pre-applied in
 * element units, and `dst_data`/`dst_mask` belong to a freshly allocated destination whose
 * offset is zero. Nothing downstream may re-add an offset to these pointers.
 */
struct alignas(16) column_descriptor {
  void const* src_data;          // first source element; nullptr never (skip_data covers all-null)
  void* dst_data;                // first destination element; fresh allocation, offset 0
  bitmask_type const* src_mask;  // source mask BASE pointer; nullptr if source non-nullable or skip
  bitmask_type* dst_mask;        // destination mask; nullptr if dst non-nullable or fast-path
  int64_t mask_offset;           // bit offset into src_mask (== src.offset() for sliced sources)
  int32_t element_size;          // cudf::size_of(src.type()); one of {1, 2, 4, 8, 16}
  bool skip_data;                // all-null fast path: dst mask pre-set ALL_NULL, data undefined
};

// INVARIANT: element_size * row is computed in 64-bit below; it can exceed 2^31 for DECIMAL128.
// 16-byte copies stay aligned because device allocations are >=256B aligned and offsets are
// whole elements.
__device__ void copy_element(char* dst, char const* src, int32_t element_size)
{
  switch (element_size) {
    case 1: *dst = *src; break;
    case 2: *reinterpret_cast<int16_t*>(dst) = *reinterpret_cast<int16_t const*>(src); break;
    case 4: *reinterpret_cast<int32_t*>(dst) = *reinterpret_cast<int32_t const*>(src); break;
    case 8: *reinterpret_cast<int64_t*>(dst) = *reinterpret_cast<int64_t const*>(src); break;
    case 16: {
      auto* d8       = reinterpret_cast<int64_t*>(dst);
      auto const* s8 = reinterpret_cast<int64_t const*>(src);
      d8[0]          = s8[0];
      d8[1]          = s8[1];
      break;
    }
    default: CUDF_UNREACHABLE("Unexpected element size");
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

/**
 * @brief Gathers every column described by `descriptors` in one launch.
 *
 * The grid is 2D: x covers row blocks (grid-stride over rows), y covers columns (y-stride past
 * the 65535 gridDim.y cap). Each block works a single column at a time, so the descriptor and the
 * element-size switch are block-uniform.
 *
 * INVARIANT: every destination row r in [0, num_rows) is written by exactly one thread.
 * INVARIANT: dst_mask word w is written exactly once, by lane 0 of the warp covering rows
 * [32w, 32w + 32): this requires BLOCK_SIZE to be a multiple of warp_size (the row stride is then
 * a multiple of 32), which the driver guarantees. Trailing bits of the last word are 0.
 */
template <int32_t BLOCK_SIZE>
CUDF_KERNEL __launch_bounds__(BLOCK_SIZE) void batch_gather_kernel(
  input_indexalator gather_map,
  column_descriptor const* descriptors,
  size_type num_descriptors,
  size_type num_rows,
  size_type num_source_rows,
  bool check_bounds,
  bool negative_allowed,
  size_type* valid_counts)
{
  for (size_type col = static_cast<size_type>(blockIdx.y); col < num_descriptors;
       col += static_cast<size_type>(gridDim.y)) {
    auto const d               = descriptors[col];
    size_type warp_valid_count = 0;
    auto const row_stride      = static_cast<int64_t>(blockDim.x) * gridDim.x;
    auto row                   = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    auto active                = __ballot_sync(0xFFFF'FFFFu, row < num_rows);
    while (row < num_rows) {
      auto idx = gather_map[static_cast<size_type>(row)];
      if (negative_allowed && idx < 0) { idx += num_source_rows; }
      bool valid = true;
      if (check_bounds) { valid = (idx >= 0) & (idx < num_source_rows); }
      if (valid && d.src_mask != nullptr) {
        valid = bit_is_set(d.src_mask, static_cast<size_type>(d.mask_offset + idx));
      }
      // Data at out-of-bounds/null rows may be left unwritten (fixed-width undefined-value policy).
      if (valid && !d.skip_data) {
        copy_element(
          static_cast<char*>(d.dst_data) + static_cast<std::size_t>(row) * d.element_size,
          static_cast<char const*>(d.src_data) + static_cast<std::size_t>(idx) * d.element_size,
          d.element_size);
      }
      if (d.dst_mask != nullptr) {
        auto const word = __ballot_sync(active, valid);
        if (threadIdx.x % warp_size == 0) {
          d.dst_mask[word_index(static_cast<size_type>(row))] = word;
          warp_valid_count += cuda::std::popcount(word);
        }
      }
      row += row_stride;
      active = __ballot_sync(active, row < num_rows);
    }
    if (d.dst_mask != nullptr) {
      auto const block_valid = single_lane_block_sum_reduce<BLOCK_SIZE, 0>(warp_valid_count);
      if (threadIdx.x == 0 && block_valid > 0) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{valid_counts[col]};
        ref.fetch_add(block_valid, cuda::std::memory_order_relaxed);
      }
    }
  }
}

}  // namespace

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr)
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

    // Null-mask phase over the fallback subset, mirroring the stock gather's conditions.
    auto fallback_views = std::vector<cudf::column_view>{};
    fallback_views.reserve(indices.size());
    for (auto const idx : indices) {
      fallback_views.push_back(source_table.column(idx));
    }
    auto const fallback_table = cudf::table_view{fallback_views};
    if (bounds_policy == out_of_bounds_policy::NULLIFY or
        cudf::has_nested_nullable_columns(fallback_table)) {
      if (bounds_policy == out_of_bounds_policy::NULLIFY or
          cudf::has_nested_nulls(fallback_table)) {
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

  // Fused part: destination columns for the fixed-width subset, with per-column mask policy.
  std::vector<column_descriptor> descriptors;
  std::vector<size_type> kernel_masked_positions;  // descriptor positions with in-kernel masks
  if (not fused_indices.empty()) {
    descriptors.reserve(fused_indices.size());
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      auto dst        = cudf::detail::allocate_like(
        src, num_rows, cudf::mask_allocation_policy::NEVER, stream, output_mr);
      bitmask_type* dst_mask = nullptr;
      bool skip_data         = false;
      bool const all_null    = src.size() > 0 and src.null_count() == src.size();
      if (all_null) {
        // Every row is null regardless of the map: pre-set ALL_NULL and skip the data copy.
        dst->set_null_mask(
          cudf::detail::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, output_mr),
          num_rows);
        skip_data = true;
      } else if (bounds_policy == out_of_bounds_policy::NULLIFY or src.null_count() > 0) {
        auto mask = cudf::detail::create_null_mask(
          num_rows, cudf::mask_state::UNINITIALIZED, stream, output_mr);
        dst_mask = mask.data();
        dst->set_null_mask(std::move(mask), 0);
        kernel_masked_positions.push_back(pos);
      } else if (src.nullable()) {
        // Nullable source with zero nulls: ALL_VALID fast path, no kernel mask work.
        dst->set_null_mask(
          cudf::detail::create_null_mask(num_rows, cudf::mask_state::ALL_VALID, stream, output_mr),
          0);
      }
      auto const src_data =
        cudf::type_dispatcher<dispatch_storage_type>(src.type(), data_ptr_functor{}, src);
      descriptors.push_back(column_descriptor{src_data,
                                              dst->mutable_view().data<std::byte>(),
                                              skip_data ? nullptr : src.null_mask(),
                                              dst_mask,
                                              static_cast<int64_t>(src.offset()),
                                              static_cast<int32_t>(cudf::size_of(src.type())),
                                              skip_data});
      destination[fused_indices[pos]] = std::move(dst);
    }
  }

  if (not descriptors.empty() and num_rows > 0) {
    // Pinned staging: stream-ordered deallocation, so the H2D needs no host sync.
    auto staging =
      cudf::detail::make_pinned_vector_async<column_descriptor>(descriptors.size(), stream);
    std::copy(descriptors.begin(), descriptors.end(), staging.begin());
    auto d_descriptors =
      rmm::device_uvector<column_descriptor>(descriptors.size(), stream, temp_mr);
    cudf::detail::cuda_memcpy_async(device_span<column_descriptor>{d_descriptors},
                                    host_span<column_descriptor const>{staging},
                                    stream);

    auto const any_kernel_masks = not kernel_masked_positions.empty();
    rmm::device_uvector<size_type> valid_counts(0, stream, temp_mr);
    if (any_kernel_masks) {
      valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        descriptors.size(), stream, temp_mr);
    }

    dim3 const grid{static_cast<unsigned>(cudf::util::div_rounding_up_safe(num_rows, block_size)),
                    static_cast<unsigned>(std::min<size_type>(descriptors.size(), 65535)),
                    1};
    batch_gather_kernel<block_size>
      <<<grid, block_size, 0, stream.get()>>>(indexalator_factory::make_input_iterator(gather_map),
                                              d_descriptors.data(),
                                              static_cast<size_type>(descriptors.size()),
                                              num_rows,
                                              num_src_rows,
                                              bounds_policy == out_of_bounds_policy::NULLIFY,
                                              neg_indices == negative_index_policy::ALLOWED,
                                              any_kernel_masks ? valid_counts.data() : nullptr);
    CUDF_CUDA_TRY(cudaGetLastError());

    if (any_kernel_masks) {
      auto const h_valid_counts = cudf::detail::make_host_vector(valid_counts, stream);
      for (auto const pos : kernel_masked_positions) {
        destination[fused_indices[pos]]->set_null_count(num_rows - h_valid_counts[pos]);
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
