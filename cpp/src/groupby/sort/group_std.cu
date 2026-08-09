/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename ResultType, typename Iterator>
struct var_transform {
  column_device_view const d_values;
  Iterator values_iter;
  ResultType const* d_means;
  size_type const* d_group_sizes;
  size_type const* d_group_labels;
  size_type ddof;

  __device__ ResultType operator()(size_type i) const
  {
    if (d_values.is_null(i)) return 0.0;

    auto x = static_cast<ResultType>(values_iter[i]);

    size_type group_idx  = d_group_labels[i];
    size_type group_size = d_group_sizes[group_idx];

    // prevent divide by zero error
    if (group_size == 0 or group_size - ddof <= 0) return 0.0;

    ResultType mean = d_means[group_idx];
    return (x - mean) * (x - mean) / (group_size - ddof);
  }
};

template <typename ResultType, typename Iterator>
void reduce_by_key_fn(column_device_view const& values,
                      Iterator values_iter,
                      cudf::device_span<size_type const> group_labels,
                      ResultType const* d_means,
                      size_type const* d_group_sizes,
                      size_type ddof,
                      ResultType* d_result,
                      rmm::cuda_stream_view stream)
{
  auto var_fn = var_transform<ResultType, decltype(values_iter)>{
    values, values_iter, d_means, d_group_sizes, group_labels.data(), ddof};
  auto const itr = cuda::counting_iterator<size_type>{0};
  // Using a temporary buffer for intermediate transform results instead of
  // using the transform-iterator directly in thrust::reduce_by_key
  // improves compile-time significantly.
  auto vars = rmm::device_uvector<ResultType>(values.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    itr,
                    itr + values.size(),
                    vars.begin(),
                    var_fn);

  cudf::detail::reduce_by_key_async(group_labels.begin(),
                                    group_labels.end(),
                                    vars.begin(),
                                    cuda::make_discard_iterator(),
                                    d_result,
                                    cuda::std::plus<ResultType>(),
                                    stream);
}

struct var_functor {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     column_view const& group_means,
                                     column_view const& group_sizes,
                                     cudf::device_span<size_type const> group_labels,
                                     size_type ddof,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_arithmetic_v<T>)
  {
    using ResultType = cudf::detail::target_type_t<T, aggregation::Kind::VARIANCE>;

    std::unique_ptr<column> result = make_numeric_column(data_type(type_to_id<ResultType>()),
                                                         group_sizes.size(),
                                                         mask_state::UNINITIALIZED,
                                                         stream,
                                                         mr);

    auto values_view = column_device_view::create(values, stream);
    auto d_values    = *values_view;

    auto d_means       = group_means.data<ResultType>();
    auto d_group_sizes = group_sizes.data<size_type>();
    auto d_result      = result->mutable_view().data<ResultType>();

    if (!cudf::is_dictionary(values.type())) {
      auto values_iter = d_values.begin<T>();
      reduce_by_key_fn(
        d_values, values_iter, group_labels, d_means, d_group_sizes, ddof, d_result, stream);
    } else {
      auto values_iter = cudf::dictionary::detail::make_dictionary_iterator<T>(*values_view);
      reduce_by_key_fn(
        d_values, values_iter, group_labels, d_means, d_group_sizes, ddof, d_result, stream);
    }

    // set nulls
    auto result_view  = mutable_column_device_view::create(*result, stream);
    auto null_count   = cudf::detail::device_scalar<cudf::size_type>(0, stream, mr);
    auto d_null_count = null_count.data();
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<cudf::size_type>{0},
      group_sizes.size(),
      [d_result = *result_view, d_group_sizes, ddof, d_null_count] __device__(size_type i) {
        size_type group_size = d_group_sizes[i];
        if (group_size == 0 or group_size - ddof <= 0) {
          d_result.set_null(i);
          // Assuming that typical data does not have too many nulls this
          // atomic shouldn't serialize the code too much. The alternatives
          // would be 1) writing a more complex kernel using cub/shmem to
          // increase parallelism, or 2) calling `cudf::count_nulls` after the
          // fact. (1) is more work than it's worth without benchmarking, and
          // this approach should outperform (2) unless large amounts of the
          // data is null.
          cudf::detail::atomic_add_relaxed(d_null_count, size_type{1});
        } else {
          d_result.set_valid(i);
        }
      });

    result->set_null_count(null_count.value(stream));
    return result;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!std::is_arithmetic_v<T>)
  {
    CUDF_FAIL("Only numeric types are supported in std/variance");
  }
};

}  // namespace

std::unique_ptr<column> group_var(column_view const& values,
                                  column_view const& group_means,
                                  column_view const& group_sizes,
                                  cudf::device_span<size_type const> group_labels,
                                  size_type ddof,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(
    values_type, var_functor{}, values, group_means, group_sizes, group_labels, ddof, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
