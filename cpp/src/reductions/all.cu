/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/atomic>
#include <cuda/iterator>
#include <thrust/for_each.h>
#include <thrust/reduce.h>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

/**
 * @brief Compute reduction all() for dictionary columns.
 *
 * This compiles 10x faster than using thrust::reduce or the
 * cudf::simple::reduction::detail::reduce utility.
 * Both of these use the CUB DeviceReduce which aggressively inlines
 * the input iterator logic.
 */
struct all_fn {
  template <typename Iterator>
  struct all_true_fn {
    __device__ void operator()(size_type idx)
    {
      if (*d_result && (iter[idx] != *d_result)) {
        cuda::atomic_ref<int32_t, cuda::thread_scope_device> ref{*d_result};
        ref.fetch_and(0, cuda::std::memory_order_relaxed);
      }
    }
    Iterator iter;
    int32_t* d_result;
  };

  template <typename T>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_arithmetic_v<T>)
  {
    auto const d_dict = cudf::column_device_view::create(input, stream);
    auto const iter   = [&] {
      auto null_iter = op::min{}.template get_null_replacing_element_transformer<bool>();
      auto pair_iter =
        cudf::dictionary::detail::make_dictionary_pair_iterator<T>(*d_dict, input.has_nulls());
      return cuda::transform_iterator(pair_iter, null_iter);
    }();
    auto d_result =
      cudf::detail::device_scalar<int32_t>(1, stream, cudf::get_current_device_resource_ref());
    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<size_type>{0},
                       input.size(),
                       all_true_fn<decltype(iter)>{iter, d_result.data()});
    return std::make_unique<numeric_scalar<bool>>(d_result.value(stream), true, stream, mr);
  }
  template <typename T>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(!std::is_arithmetic_v<T>)
  {
    CUDF_FAIL("Unexpected key type for dictionary in reduction all()");
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> all(column_view const& col,
                                  cudf::data_type const output_dtype,
                                  std::optional<std::reference_wrapper<scalar const>> init,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(output_dtype == cudf::data_type(cudf::type_id::BOOL8),
               "all() operation can be applied with output type `BOOL8` only");

  if (cudf::is_dictionary(col.type())) {
    return cudf::type_dispatcher(
      dictionary_column_view(col).keys().type(), detail::all_fn{}, col, stream, mr);
  }
  using reducer = simple::detail::bool_result_element_dispatcher<op::min>;
  // dispatch for non-dictionary types
  return cudf::type_dispatcher(col.type(), reducer{}, col, init, stream, mr);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
