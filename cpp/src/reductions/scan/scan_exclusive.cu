/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scan.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/stream>
#include <thrust/scan.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Dispatcher for running a scan operation on an input column
 *
 * @tparam Op device binary operator (e.g. min, max, sum)
 */
template <typename Op>
struct scan_dispatcher {
 public:
  /**
   * @brief Creates a new column from input column by applying exclusive scan operation
   *
   * @tparam T type of input column
   *
   * @param input  Input column view
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return Output column with scan results
   */
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     bitmask_type const*,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cuda::std::is_arithmetic_v<T>)
  {
    auto output_column =
      detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view output = output_column->mutable_view();

    auto d_input  = column_device_view::create(input, stream);
    auto identity = Op::template identity<T>();

    auto begin = make_null_replacement_iterator(*d_input, identity, input.has_nulls());

    // CUB 2.0.0 requires that the binary operator returns the same type as the identity.
    auto const binary_op = cudf::detail::cast_functor<T>(Op{});
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           begin,
                           begin + input.size(),
                           output.data<T>(),
                           identity,
                           binary_op);

    CUDF_CHECK_CUDA(stream.get());
    return output_column;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(not cuda::std::is_arithmetic_v<T>)
  {
    CUDF_FAIL("Non-arithmetic types not supported for exclusive scan");
  }
};

}  // namespace

std::unique_ptr<column> scan_exclusive(column_view const& input,
                                       scan_aggregation const& agg,
                                       null_policy null_handling,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto [mask, null_count] = [&] {
    if (null_handling == null_policy::EXCLUDE) {
      return std::make_pair(std::move(detail::copy_bitmask(input, stream, mr)), input.null_count());
    } else if (input.nullable()) {
      return mask_scan(input, scan_type::EXCLUSIVE, stream, mr);
    }
    return std::make_pair(rmm::device_buffer{}, size_type{0});
  }();

  auto output = scan_agg_dispatch<scan_dispatcher>(
    input, agg, static_cast<bitmask_type*>(mask.data()), stream, mr);
  output->set_null_mask(std::move(mask), null_count);

  return output;
}

}  // namespace detail

}  // namespace cudf
