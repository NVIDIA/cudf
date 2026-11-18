/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/execution_env.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_partition.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/buffer>
#include <cuda/execution.determinism.h>
#include <cuda/execution.output_ordering.h>
#include <cuda/execution.require.h>
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/stream>

#include <memory>

namespace cudf {
namespace detail {
namespace {
[[nodiscard]] bool is_fast_path(column_view const& column)
{
  return cudf::is_fixed_width(column.type());
}

/**
 * @brief Returns the selection key for `values[row]`
 *
 * Floating-point values are canonicalized to follow cudf's ordering: every NaN compares equal to
 * every other NaN and greater than all other values, and -0.0 compares equal to +0.0.
 */
template <typename T>
struct key_at_row {
  T const* values;

  __device__ T operator()(size_type row) const
  {
    auto const value = values[row];
    if constexpr (cudf::is_floating_point<T>()) {
      if (cuda::std::isnan(value)) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
      return value == T{0} ? T{0} : value;  // -0.0 becomes +0.0
    } else {
      return value;
    }
  }
};

struct row_is_valid {
  bitmask_type const* null_mask;
  size_type offset;

  __device__ bool operator()(size_type row) const
  {
    return cudf::bit_is_set(null_mask, offset + row);
  }
};

/// Selects the top rows among `valid_rows` if set, otherwise among every row of `input`
struct dispatch_top_k_fn {
  column_view input;
  cuda::std::optional<device_span<size_type const>> valid_rows;
  order topk_order;
  device_span<size_type> output;
  cuda::stream_ref stream;
  cudf::memory_resources mr;

  template <typename KeysIterator, typename ValuesIterator>
  void select(KeysIterator keys_in, ValuesIterator vals_in, size_type num_items) const
  {
    auto const requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                       cuda::execution::tie_break::unspecified,
                                                       cuda::execution::output_ordering::unsorted);
    auto const env          = cudf::detail::make_env(stream, mr, requirements);
    auto const keys_out     = cuda::make_discard_iterator();
    auto const num_out      = static_cast<size_type>(output.size());
    if (topk_order == order::ASCENDING) {
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        keys_in, keys_out, vals_in, output.data(), num_items, num_out, env));
    } else {
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        keys_in, keys_out, vals_in, output.data(), num_items, num_out, env));
    }
  }

  template <typename T>
  void select_rows() const
  {
    auto const values = input.begin<T>();
    if (not valid_rows.has_value()) {
      auto const rows = cuda::counting_iterator<size_type>{0};
      if constexpr (cudf::is_floating_point<T>()) {
        select(cuda::transform_iterator(rows, key_at_row<T>{values}), rows, input.size());
      } else {
        select(values, rows, input.size());
      }
    } else {
      auto const rows = valid_rows->data();
      select(cuda::transform_iterator(rows, key_at_row<T>{values}),
             rows,
             static_cast<size_type>(valid_rows->size()));
    }
  }

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_chrono<T>())
  void operator()() const
  {
    select_rows<T>();
  }

  template <typename T>
    requires(cudf::is_chrono<T>())
  void operator()() const
  {
    select_rows<typename T::rep>();
  }

  template <typename T>
    requires(not cudf::is_fixed_width<T>())
  void operator()() const
  {
    CUDF_UNREACHABLE("unexpected type for top_k fast path");
  }
};

/**
 * @brief Returns the k selected row indices for a fixed-width column
 *
 * Nulls are ordered last in either direction, so they are returned only when `k` exceeds the
 * number of non-null rows, which `null_count()` gives without synchronizing. Nulls are partitioned
 * out rather than encoded as a sentinel key, since integral keys have no spare bit pattern.
 */
[[nodiscard]] rmm::device_uvector<size_type> select_top_k_rows(column_view const& col,
                                                               size_type k,
                                                               order topk_order,
                                                               cuda::stream_ref stream,
                                                               cudf::memory_resources mr)
{
  auto const temp_mr   = mr.get_temporary_mr();
  auto const size      = col.size();
  auto const num_valid = size - col.null_count();

  auto output = rmm::device_uvector<size_type>(k, stream, mr.get_output_mr());

  if (col.null_count() == 0) {
    type_dispatcher<dispatch_storage_type>(
      col.type(), dispatch_top_k_fn{col, cuda::std::nullopt, topk_order, output, stream, mr});
    return output;
  }

  auto const rows     = cuda::counting_iterator<size_type>(0);
  auto const is_valid = row_is_valid{col.null_mask(), col.offset()};

  if (k >= num_valid) {
    // Every non-null row is selected and nulls fill the rest: partition the non-null rows ahead of
    // the nulls and take the first k.
    auto partitioned_rows = cuda::device_buffer<size_type>(stream, temp_mr, size, cuda::no_init);
    CUDF_CUDA_TRY(cub::DevicePartition::If(rows,
                                           partitioned_rows.data(),
                                           cuda::make_discard_iterator(),
                                           size,
                                           is_valid,
                                           cudf::detail::make_env(stream, mr)));
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      output.data(), partitioned_rows.data(), k * sizeof(size_type), stream));
    return output;
  }

  // Select among the compacted non-null rows. Their indices are passed as the selection's values,
  // so the output needs no remapping.
  auto valid_rows = cuda::device_buffer<size_type>(stream, temp_mr, num_valid, cuda::no_init);
  cudf::detail::copy_if_async(
    rows, rows + size, valid_rows.data(), is_valid, stream, mr.as_temporary());

  type_dispatcher<dispatch_storage_type>(
    col.type(),
    dispatch_top_k_fn{col,
                      device_span<size_type const>{valid_rows.data(), valid_rows.size()},
                      topk_order,
                      output,
                      stream,
                      mr});

  return output;
}

}  // namespace

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return empty_like(col); }
  if (k >= col.size()) { return std::make_unique<column>(col, stream, mr); }

  auto const indices = [&]() -> std::unique_ptr<column> {
    auto const temp_mr = mr.get_temporary_mr();
    if (is_fast_path(col)) {
      return std::make_unique<column>(
        select_top_k_rows(col, k, topk_order, stream, mr.as_temporary()),
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
        0);
    }
    auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
    return sorted_order<sort_method::STABLE>(col, topk_order, nulls, stream, temp_mr);
  }();

  auto const k_indices = cudf::detail::split(indices->view(), {k}, stream).front();

  auto const dont_check  = out_of_bounds_policy::DONT_CHECK;
  auto const not_allowed = negative_index_policy::NOT_ALLOWED;
  auto result =
    cudf::detail::gather(cudf::table_view({col}), k_indices, dont_check, not_allowed, stream, mr);
  return std::move(result->release().front());
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return make_empty_column(cudf::type_to_id<size_type>()); }
  if (k >= col.size()) {
    return cudf::detail::sequence(
      col.size(), numeric_scalar<size_type>(0, true, stream, mr.get_temporary_mr()), stream, mr);
  }

  auto const temp_mr = mr.get_temporary_mr();
  if (is_fast_path(col)) {
    return std::make_unique<column>(select_top_k_rows(col, k, topk_order, stream, mr),
                                    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    0);
  }
  auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto indices     = sorted_order<sort_method::STABLE>(col, topk_order, nulls, stream, temp_mr);

  return std::make_unique<column>(
    cudf::detail::split(indices->view(), {k}, stream).front(), stream, mr);
}

}  // namespace detail

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k(col, k, topk_order, stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, topk_order, stream, mr);
}

}  // namespace cudf
