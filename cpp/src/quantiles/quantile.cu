/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "quantiles/quantiles_util.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/transform.h>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {
template <bool exact, typename SortMapIterator>
struct quantile_functor {
  SortMapIterator ordered_indices;
  size_type size;
  std::vector<double> const& q;
  interpolation interp;
  bool retain_types;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input)
    requires(not std::is_arithmetic_v<T> and not cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("quantile does not support non-numeric types");
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input)
    requires(std::is_arithmetic_v<T> or cudf::is_fixed_point<T>())
  {
    using StorageType   = cudf::device_storage_type_t<T>;
    using ExactResult   = std::conditional_t<exact and not cudf::is_fixed_point<T>(), double, T>;
    using StorageResult = cudf::device_storage_type_t<ExactResult>;

    auto const type =
      is_fixed_point(input.type()) ? input.type() : data_type{type_to_id<StorageResult>()};
    auto output = make_fixed_width_column(type, q.size(), mask_state::UNALLOCATED, stream, mr);

    if (output->size() == 0) { return output; }

    if (input.is_empty()) {
      auto mask = cudf::detail::create_null_mask(output->size(), mask_state::ALL_NULL, stream, mr);
      output->set_null_mask(std::move(mask), output->size());
      return output;
    }

    auto d_input  = column_device_view::create(input, stream);
    auto d_output = mutable_column_device_view::create(output->mutable_view(), stream);

    auto q_device =
      cudf::detail::make_device_uvector(q, stream, cudf::get_current_device_resource_ref());

    if (!cudf::is_dictionary(input.type())) {
      auto sorted_data =
        cuda::make_permutation_iterator(input.data<StorageType>(), ordered_indices);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        q_device.begin(),
                        q_device.end(),
                        d_output->template begin<StorageResult>(),
                        cuda::proclaim_return_type<StorageResult>(
                          [sorted_data, interp = interp, size = size] __device__(double q) {
                            return select_quantile_data<StorageResult>(
                              sorted_data, size, q, interp);
                          }));
    } else {
      auto sorted_data = cuda::make_permutation_iterator(
        dictionary::detail::make_dictionary_iterator<T>(*d_input), ordered_indices);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        q_device.begin(),
                        q_device.end(),
                        d_output->template begin<StorageResult>(),
                        cuda::proclaim_return_type<StorageResult>(
                          [sorted_data, interp = interp, size = size] __device__(double q) {
                            return select_quantile_data<StorageResult>(
                              sorted_data, size, q, interp);
                          }));
    }

    if (input.nullable()) {
      auto sorted_validity = cuda::transform_iterator(
        ordered_indices,
        cuda::proclaim_return_type<bool>(
          [input = *d_input] __device__(size_type idx) { return input.is_valid_nocheck(idx); }));

      auto [mask, null_count] = valid_if(
        q_device.begin(),
        q_device.end(),
        [sorted_validity, interp = interp, size = size] __device__(double q) {
          return select_quantile_validity(sorted_validity, size, q, interp);
        },
        stream,
        mr);

      output->set_null_mask(std::move(mask), null_count);
    }

    return output;
  }
};

template <bool exact, typename SortMapIterator>
std::unique_ptr<column> quantile(column_view const& input,
                                 SortMapIterator ordered_indices,
                                 size_type size,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 bool retain_types,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto functor = quantile_functor<exact, SortMapIterator>{
    ordered_indices, size, q, interp, retain_types, stream, mr};

  auto input_type = cudf::is_dictionary(input.type()) && !input.is_empty()
                      ? dictionary_column_view(input).keys().type()
                      : input.type();

  return type_dispatcher(input_type, functor, input);
}

std::unique_ptr<column> quantile(column_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 column_view const& indices,
                                 bool exact,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  if (indices.is_empty()) {
    auto begin = cuda::counting_iterator<size_type>{0};
    if (exact) {
      return quantile<true>(input, begin, input.size(), q, interp, exact, stream, mr);
    } else {
      return quantile<false>(input, begin, input.size(), q, interp, exact, stream, mr);
    }

  } else {
    CUDF_EXPECTS(indices.type() == data_type{type_to_id<size_type>()},
                 "`indices` type must be `INT32`.");
    if (exact) {
      return quantile<true>(
        input, indices.begin<size_type>(), indices.size(), q, interp, exact, stream, mr);
    } else {
      return quantile<false>(
        input, indices.begin<size_type>(), indices.size(), q, interp, exact, stream, mr);
    }
  }
}

}  // namespace detail

std::unique_ptr<column> quantile(column_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 column_view const& ordered_indices,
                                 bool exact,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::quantile(input, q, interp, ordered_indices, exact, stream, mr);
}

}  // namespace cudf
