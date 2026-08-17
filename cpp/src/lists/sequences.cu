/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <limits>
#include <optional>
#include <stdexcept>

namespace cudf::lists {
namespace detail {
namespace {
template <typename T>
struct tabulator {
  size_type const n_lists;
  size_type const n_elements;

  T const* const starts;
  T const* const steps;
  int32_t const* const offsets;

  template <typename U>
  static T __device__ multiply(U x, size_type times)
    requires(!cudf::is_duration<U>())
  {
    return x * static_cast<T>(times);
  }

  template <typename U>
  static T __device__ multiply(U x, size_type times)
    requires(cudf::is_duration<U>())
  {
    return T{x.count() * times};
  }

  auto __device__ operator()(size_type idx) const
  {
    auto const list_idx_end = thrust::upper_bound(thrust::seq, offsets, offsets + n_lists, idx);
    auto const list_idx     = cuda::std::distance(offsets, list_idx_end) - 1;
    auto const list_offset  = offsets[list_idx];
    auto const list_step    = steps ? steps[list_idx] : T{1};
    return starts[list_idx] + multiply(list_step, idx - list_offset);
  }
};

template <typename T, typename Enable = void>
struct sequences_functor {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported per-list sequence type-agg combination.");
  }
};

struct sequences_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(size_type n_lists,
                                     size_type n_elements,
                                     column_view const& starts,
                                     std::optional<column_view> const& steps,
                                     int32_t const* offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return sequences_functor<T>::invoke(n_lists, n_elements, starts, steps, offsets, stream, mr);
  }
};

template <typename T>
static constexpr bool is_supported()
{
  return (cudf::is_numeric<T>() && !cudf::is_boolean<T>()) || cudf::is_duration<T>();
}

template <typename T>
struct sequences_functor<T, std::enable_if_t<is_supported<T>()>> {
  static std::unique_ptr<column> invoke(size_type n_lists,
                                        size_type n_elements,
                                        column_view const& starts,
                                        std::optional<column_view> const& steps,
                                        int32_t const* offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
  {
    auto result =
      make_fixed_width_column(starts.type(), n_elements, mask_state::UNALLOCATED, stream, mr);
    if (starts.is_empty()) { return result; }

    auto const result_begin = result->mutable_view().template begin<T>();

    // Use pointers instead of column_device_view to access start and step values should be enough.
    // This is because we don't need to check for nulls and only support numeric and duration types.
    auto const starts_begin = starts.template begin<T>();
    auto const steps_begin  = steps ? steps.value().template begin<T>() : nullptr;

    auto const op = tabulator<T>{n_lists, n_elements, starts_begin, steps_begin, offsets};
    thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     result_begin,
                     result_begin + n_elements,
                     op);

    return result;
  }
};

std::unique_ptr<column> sequences(column_view const& starts,
                                  std::optional<column_view> const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!starts.has_nulls() && !sizes.has_nulls(),
               "starts and sizes input columns must not have nulls.");
  CUDF_EXPECTS(starts.size() == sizes.size(),
               "starts and sizes input columns must have the same number of rows.");
  CUDF_EXPECTS(cudf::is_index_type(sizes.type()),
               "Input sizes column must be of integer types.",
               cudf::data_type_error);

  if (steps) {
    auto const& steps_cv = steps.value();
    CUDF_EXPECTS(!steps_cv.has_nulls(), "steps input column must not have nulls.");
    CUDF_EXPECTS(starts.size() == steps_cv.size(),
                 "starts and steps input columns must have the same number of rows.");
    CUDF_EXPECTS(cudf::have_same_types(starts, steps_cv),
                 "starts and steps input columns must have the same type.",
                 cudf::data_type_error);
  }

  auto const n_lists = starts.size();
  if (n_lists == 0) { return cudf::make_empty_lists_column(starts.type()); }

  auto const sizes_input_it = cudf::detail::indexalator_factory::make_input_iterator(sizes);
  // Generate list offsets for the output.
  auto [list_offsets, n_elements] = cudf::detail::make_offsets_child_column(
    sizes_input_it, sizes_input_it + sizes.size(), stream, mr);
  auto const offsets_begin = list_offsets->view().template begin<int32_t>();

  auto child = type_dispatcher(starts.type(),
                               sequences_dispatcher{},
                               n_lists,
                               static_cast<size_type>(n_elements),
                               starts,
                               steps,
                               offsets_begin,
                               stream,
                               mr);

  return make_lists_column(
    n_lists, std::move(list_offsets), std::move(child), 0, rmm::device_buffer(0, stream, mr));
}

}  // anonymous namespace

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  return sequences(starts, std::nullopt, sizes, stream, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  return sequences(starts, std::optional<column_view>{steps}, sizes, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequences(starts, sizes, stream, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequences(starts, steps, sizes, stream, mr);
}

}  // namespace cudf::lists
