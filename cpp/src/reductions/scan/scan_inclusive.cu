/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reductions/scan/scan.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/scan.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/reduction.hpp>
#include <cudf/strings/detail/scan.hpp>
#include <cudf/structs/detail/scan.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/find.h>
#include <thrust/scan.h>

#include <type_traits>

namespace cudf {
namespace detail {

// logical-and scan of the null mask of the input view
std::pair<rmm::device_buffer, size_type> mask_scan(column_view const& input_view,
                                                   scan_type inclusive,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  rmm::device_buffer mask =
    detail::create_null_mask(input_view.size(), mask_state::UNINITIALIZED, stream, mr);
  auto d_input   = column_device_view::create(input_view, stream);
  auto valid_itr = detail::make_validity_iterator(*d_input);

  auto first_null_position = [&] {
    size_type const first_null =
      thrust::find_if_not(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          valid_itr,
                          valid_itr + input_view.size(),
                          cuda::std::identity{}) -
      valid_itr;
    size_type const exclusive_offset = (inclusive == scan_type::EXCLUSIVE) ? 1 : 0;
    return std::min(input_view.size(), first_null + exclusive_offset);
  }();

  set_null_mask(static_cast<bitmask_type*>(mask.data()), 0, first_null_position, true, stream);
  set_null_mask(
    static_cast<bitmask_type*>(mask.data()), first_null_position, input_view.size(), false, stream);
  return {std::move(mask), input_view.size() - first_null_position};
}

namespace {

template <typename Op, typename T>
struct scan_functor {
  static std::unique_ptr<column> invoke(column_view const& input_view,
                                        bitmask_type const*,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
  {
    auto output_column = detail::allocate_like(
      input_view, input_view.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view result = output_column->mutable_view();

    auto d_input = column_device_view::create(input_view, stream);
    auto const begin =
      make_null_replacement_iterator(*d_input, Op::template identity<T>(), input_view.has_nulls());

    // CUB 2.0.0 requires that the binary operator returns the same type as the identity.
    auto const binary_op = cudf::detail::cast_functor<T>(Op{});
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           begin,
                           begin + input_view.size(),
                           result.data<T>(),
                           binary_op);

    CUDF_CHECK_CUDA(stream.get());
    return output_column;
  }
};

template <typename Op>
  requires(not std::is_same_v<Op, DeviceCount>)
struct scan_functor<Op, cudf::string_view> {
  static std::unique_ptr<column> invoke(column_view const& input_view,
                                        bitmask_type const* mask,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
  {
    return cudf::strings::detail::scan_inclusive<Op>(input_view, mask, stream, mr);
  }
};

template <typename Op>
  requires(not std::is_same_v<Op, DeviceCount>)
struct scan_functor<Op, cudf::struct_view> {
  static std::unique_ptr<column> invoke(column_view const& input,
                                        bitmask_type const*,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
  {
    return cudf::structs::detail::scan_inclusive<Op>(input, stream, mr);
  }
};

template <typename Op, typename T>
  requires(std::is_same_v<Op, DeviceCount>)
struct scan_functor<Op, T> {
  static std::unique_ptr<column> invoke(column_view const& input_view,
                                        bitmask_type const* mask,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
  {
    auto output_column = make_numeric_column(data_type{type_to_id<size_type>()},
                                             input_view.size(),
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
    auto result        = output_column->mutable_view();

    auto const begin = make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_type>([mask] __device__(auto idx) -> size_type {
        return static_cast<size_type>(mask == nullptr || bit_is_set(mask, idx));
      }));

    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           begin,
                           begin + input_view.size(),
                           result.data<size_type>());

    CUDF_CHECK_CUDA(stream.get());
    return output_column;
  }
};

/**
 * @brief Dispatcher for running a Scan operation on an input column
 *
 * @tparam Op device binary operator
 */
template <typename Op>
struct scan_dispatcher {
 private:
  template <typename T>
  static constexpr bool is_supported()
  {
    if constexpr (std::is_same_v<Op, DeviceCount>) { return true; }
    if constexpr (std::is_same_v<T, cudf::struct_view>) {
      return std::is_same_v<Op, DeviceMin> || std::is_same_v<Op, DeviceMax>;
    } else {
      return std::is_invocable_v<Op, T, T> && !cudf::is_dictionary<T>();
    }
  }

 public:
  /**
   * @brief Creates a new column from the input column by applying the scan operation
   *
   * @param input Input column view
   * @param null_handling How null row entries are to be processed
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return
   *
   * @tparam T type of input column
   */
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     bitmask_type const* output_mask,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_supported<T>())
  {
    return scan_functor<Op, T>::invoke(input, output_mask, stream, mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!is_supported<T>())
  {
    CUDF_FAIL("Unsupported type for inclusive scan operation");
  }
};

}  // namespace

std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       scan_aggregation const& agg,
                                       null_policy null_handling,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto [mask, null_count] = [&] {
    if (null_handling == null_policy::EXCLUDE) {
      return std::make_pair(std::move(detail::copy_bitmask(input, stream, mr)), input.null_count());
    } else if (input.nullable()) {
      return mask_scan(input, scan_type::INCLUSIVE, stream, mr);
    }
    return std::make_pair(rmm::device_buffer{}, size_type{0});
  }();

  auto output = scan_agg_dispatch<scan_dispatcher>(
    input, agg, static_cast<bitmask_type*>(mask.data()), stream, mr);
  // Use the null mask produced by the op for EWM
  if (agg.kind != aggregation::EWMA) { output->set_null_mask(std::move(mask), null_count); }

  // If the input is a structs column, we also need to push down nulls from the parent output column
  // into the children columns.
  if (input.type().id() == type_id::STRUCT && output->has_nulls()) {
    auto const num_rows   = output->size();
    auto const null_count = output->null_count();
    auto content          = output->release();

    // Build new children columns.
    auto const null_mask = reinterpret_cast<bitmask_type const*>(content.null_mask->data());
    std::for_each(content.children.begin(),
                  content.children.end(),
                  [null_mask, null_count, stream, mr](auto& child) {
                    child = structs::detail::superimpose_and_sanitize_nulls(
                      null_mask, null_count, std::move(child), stream, mr);
                  });

    // Replace the children columns.
    output = cudf::create_structs_hierarchy(
      num_rows, std::move(content.children), null_count, std::move(*content.null_mask), stream, mr);
  }

  return output;
}
}  // namespace detail
}  // namespace cudf
