/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream_ref>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

namespace cudf {
namespace detail {
namespace {
// This functor only exists here because using a lambda directly in the tabulate() call generates
// the cryptic
// __T289 link error.  This seems to be related to lambda usage within functions using SFINAE.
template <typename T>
struct tabulator {
  cudf::numeric_scalar_device_view<T> const n_init;
  cudf::numeric_scalar_device_view<T> const n_step;

  T __device__ operator()(cudf::size_type i)
  {
    return n_init.value() + (static_cast<T>(i) * n_step.value());
  }
};

template <typename T>
struct const_tabulator {
  cudf::numeric_scalar_device_view<T> const n_init;

  T __device__ operator()(cudf::size_type i) { return n_init.value() + static_cast<T>(i); }
};

/**
 * @brief Functor called by the `type_dispatcher` to generate the sequence specified
 * by init and step.
 */
struct sequence_functor {
  template <typename T>
  std::unique_ptr<column> operator()(size_type size,
                                     scalar const& init,
                                     scalar const& step,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<T>() and not cudf::is_boolean<T>())
  {
    auto result = make_fixed_width_column(init.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto result_device_view = mutable_column_device_view::create(*result, stream);

    auto n_init =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(init)));
    auto n_step =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(step)));

    // not using thrust::sequence because it requires init and step to be passed as
    // constants, not iterators. to do that we would have to retrieve the scalar values off the gpu,
    // which is undesirable from a performance perspective.
    thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     result_device_view->begin<T>(),
                     result_device_view->end<T>(),
                     tabulator<T>{n_init, n_step});

    return result;
  }

  template <typename T>
  std::unique_ptr<column> operator()(size_type size,
                                     scalar const& init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<T>() and not cudf::is_boolean<T>())
  {
    auto result = make_fixed_width_column(init.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto result_device_view = mutable_column_device_view::create(*result, stream);

    auto n_init =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(init)));

    // not using thrust::sequence because it requires init and step to be passed as
    // constants, not iterators. to do that we would have to retrieve the scalar values off the gpu,
    // which is undesirable from a performance perspective.
    thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     result_device_view->begin<T>(),
                     result_device_view->end<T>(),
                     const_tabulator<T>{n_init});

    return result;
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(not cudf::is_numeric<T>() or cudf::is_boolean<T>())
  {
    CUDF_FAIL("Unsupported sequence scalar type", cudf::data_type_error);
  }
};

}  // anonymous namespace

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::have_same_types(init, step),
               "init and step must be of the same type.",
               cudf::data_type_error);
  CUDF_EXPECTS(size >= 0, "size must be >= 0", std::invalid_argument);
  CUDF_EXPECTS(
    is_numeric(init.type()), "Input scalar types must be numeric", std::invalid_argument);
  CUDF_EXPECTS(init.is_valid(stream), "init must be a valid scalar", std::invalid_argument);
  CUDF_EXPECTS(step.is_valid(stream), "step must be a valid scalar", std::invalid_argument);

  return type_dispatcher(init.type(), sequence_functor{}, size, init, step, stream, mr);
}

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(size >= 0, "size must be >= 0", std::invalid_argument);
  CUDF_EXPECTS(is_numeric(init.type()), "init scalar type must be numeric", cudf::data_type_error);
  CUDF_EXPECTS(init.is_valid(stream), "init must be a valid scalar", std::invalid_argument);

  return type_dispatcher(init.type(), sequence_functor{}, size, init, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequence(size, init, step, stream, mr);
}

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequence(size, init, stream, mr);
}

}  // namespace cudf
