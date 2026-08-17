/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

#include <algorithm>

namespace cudf {
namespace detail {

template <typename DeviceComparatorType>
struct ohe_equality_functor {
  ohe_equality_functor(size_type input_size, DeviceComparatorType d_equal)
    : _input_size(input_size), _d_equal(d_equal)
  {
  }

  auto __device__ operator()(size_type i) const noexcept
  {
    auto const element_index  = cudf::detail::row::lhs_index_type{i % _input_size};
    auto const category_index = cudf::detail::row::rhs_index_type{i / _input_size};
    return _d_equal(element_index, category_index);
  }

 private:
  size_type _input_size;
  DeviceComparatorType _d_equal;
};

std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::have_same_types(input, categories),
               "Mismatch type between input and categories.",
               cudf::data_type_error);

  if (categories.is_empty()) { return {make_empty_column(type_id::BOOL8), table_view{}}; }

  if (input.is_empty()) {
    auto empty_data = make_empty_column(type_id::BOOL8);
    std::vector<column_view> views(categories.size(), empty_data->view());
    return {std::move(empty_data), table_view{views}};
  }

  auto const total_size = input.size() * categories.size();
  auto all_encodings =
    make_numeric_column(data_type{type_id::BOOL8}, total_size, mask_state::UNALLOCATED, stream, mr);

  auto const t_lhs      = table_view{{input}};
  auto const t_rhs      = table_view{{categories}};
  auto const comparator = cudf::detail::row::equality::two_table_comparator{t_lhs, t_rhs, stream};

  auto const comparator_helper = [&](auto const d_equal) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{total_size},
                      all_encodings->mutable_view().begin<bool>(),
                      ohe_equality_functor<decltype(d_equal)>(input.size(), d_equal));
  };

  if (cudf::detail::has_nested_columns(t_lhs) or cudf::detail::has_nested_columns(t_rhs)) {
    auto const d_equal = comparator.equal_to<true>(
      nullate::DYNAMIC{has_nested_nulls(t_lhs) || has_nested_nulls(t_rhs)});
    comparator_helper(d_equal);
  } else {
    auto const d_equal = comparator.equal_to<false>(
      nullate::DYNAMIC{has_nested_nulls(t_lhs) || has_nested_nulls(t_rhs)});
    comparator_helper(d_equal);
  }

  auto const split_iter =
    make_counting_transform_iterator(1, [width = input.size()](auto i) { return i * width; });
  std::vector<size_type> split_indices(split_iter, split_iter + categories.size() - 1);

  auto encodings_view = table_view{detail::split(all_encodings->view(), split_indices, stream)};

  return {std::move(all_encodings), encodings_view};
}

}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::one_hot_encode(input, categories, stream, mr);
}
}  // namespace cudf
