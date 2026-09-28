/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace detail {
namespace {

struct replace_nans_functor {
  template <typename T, typename Replacement>
  std::unique_ptr<column> operator()(column_view const& input,
                                     Replacement const& replacement,
                                     bool replacement_nullable,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
  {
    CUDF_EXPECTS(input.type() == replacement.type(),
                 "Input and replacement must be of the same type");

    if (input.is_empty()) { return cudf::make_empty_column(input.type()); }

    auto input_device_view = column_device_view::create(input, stream);
    size_type size         = input.size();

    auto predicate = [dinput = *input_device_view] __device__(auto i) {
      return dinput.is_null(i) or !std::isnan(dinput.element<T>(i));
    };

    auto input_iterator =
      make_optional_iterator<T>(*input_device_view, nullate::DYNAMIC{input.has_nulls()});
    auto replacement_iterator =
      make_optional_iterator<T>(replacement, nullate::DYNAMIC{replacement_nullable});
    return copy_if_else(input.has_nulls() or replacement_nullable,
                        input_iterator,
                        input_iterator + size,
                        replacement_iterator,
                        predicate,
                        input.type(),
                        stream,
                        mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!std::is_floating_point_v<T>)
  {
    CUDF_FAIL("NAN is not supported in a Non-floating point type column");
  }
};

}  // namespace

std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.size() == replacement.size(),
               "Input and replacement must be of the same size");

  return type_dispatcher(input.type(),
                         replace_nans_functor{},
                         input,
                         *column_device_view::create(replacement, stream),
                         replacement.nullable(),
                         stream,
                         mr);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  return type_dispatcher(
    input.type(), replace_nans_functor{}, input, replacement, true, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, stream, mr);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, stream, mr);
}

}  // namespace cudf

namespace {  // anonymous

template <typename T>
struct normalize_nans_and_zeros_lambda {
  cudf::column_device_view in;
  T __device__ operator()(cudf::size_type i)
  {
    auto e = in.element<T>(i);
    if (isnan(e)) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
    if (T{0.0} == e) { return T{0.0}; }
    return e;
  }
};

/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `normalize_nans_and_zeros` with the appropriate data types.
 */
struct normalize_nans_and_zeros_kernel_forwarder {
  // floats and doubles. what we really care about.
  template <typename T>
  void operator()(cudf::column_device_view in,
                  cudf::mutable_column_device_view out,
                  cuda::stream_ref stream)
    requires(std::is_floating_point_v<T>)
  {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{in.size()},
                      out.head<T>(),
                      normalize_nans_and_zeros_lambda<T>{in});
  }

  // if we get in here for anything but a float or double, that's a problem.
  template <typename T, typename... Args>
  void operator()(Args&&...)
    requires(not std::is_floating_point_v<T>)
  {
    CUDF_FAIL("Unexpected non floating-point type.");
  }
};

}  // end anonymous namespace

namespace cudf {
namespace detail {
void normalize_nans_and_zeros(mutable_column_view in_out, cuda::stream_ref stream)
{
  if (in_out.is_empty()) { return; }
  CUDF_EXPECTS(
    in_out.type() == data_type(type_id::FLOAT32) || in_out.type() == data_type(type_id::FLOAT64),
    "Expects float or double input");

  // wrapping the in_out data in a column_view so we can call the same lower level code.
  // that we use for the non in-place version.
  column_view input = in_out;

  // to device. unique_ptr which gets automatically cleaned up when we leave
  auto device_in = column_device_view::create(input, stream);

  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto device_out = mutable_column_device_view::create(in_out, stream);

  // invoke the actual kernel.
  cudf::type_dispatcher(
    input.type(), normalize_nans_and_zeros_kernel_forwarder{}, *device_in, *device_out, stream);
}

std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  // output. copies the input
  auto out = std::make_unique<column>(input, stream, mr);

  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto out_view = out->mutable_view();
  detail::normalize_nans_and_zeros(out_view, stream);
  out->set_null_count(input.null_count());

  return out;
}

}  // namespace detail

/**
 * @brief Makes all Nans and zeroes positive.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_nans_and_zeros(input, stream, mr);
}

/**
 * @brief Makes all Nans and zeroes positive.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] in_out mutable_column_view representing input data. data is processed in-place
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void normalize_nans_and_zeros(mutable_column_view& in_out, cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  detail::normalize_nans_and_zeros(in_out, stream);
}

}  // namespace cudf
