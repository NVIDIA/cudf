/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/udf/managed_udf_string.cuh>
#include <cudf/strings/udf/udf_apis.hpp>
#include <cudf/strings/udf/udf_string.cuh>
#include <cudf/strings/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace udf {

namespace detail {

namespace {

/**
 * @brief Functor wraps string_view objects around udf_string objects
 *
 * No string data is copied.
 */
struct managed_udf_string_to_string_view_transform_fn {
  __device__ cudf::string_view operator()(
    cudf::strings::udf::managed_udf_string const& managed_dstr)
  {
    return managed_dstr.udf_str.data() == nullptr
             ? cudf::string_view{}
             : cudf::string_view{managed_dstr.udf_str.data(), managed_dstr.udf_str.size_bytes()};
  }
};

}  // namespace

/**
 * @copydoc to_string_view_array
 *
 * @param stream CUDA stream used for allocating/copying device memory and launching kernels
 */
std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input,
                                                         cuda::stream_ref stream)
{
  return std::make_unique<rmm::device_buffer>(
    std::move(cudf::strings::create_string_vector_from_column(
                cudf::strings_column_view(input), stream, cudf::get_current_device_resource_ref())
                .release()));
}

/**
 * @copydoc column_from_managed_udf_string_array
 *
 * @param stream CUDA stream used for allocating/copying device memory and launching kernels
 */
std::unique_ptr<cudf::column> column_from_managed_udf_string_array(
  managed_udf_string* managed_strings, cudf::size_type size, cuda::stream_ref stream)
{
  // create string_views of the udf_strings
  auto indices = rmm::device_uvector<cudf::string_view>(size, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    managed_strings,
                    managed_strings + size,
                    indices.data(),
                    managed_udf_string_to_string_view_transform_fn{});

  auto result = cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), stream);
  stream.sync();
  return result;
}

}  // namespace detail

// external APIs

int get_cuda_build_version() { return CUDA_VERSION; }

std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input)
{
  return detail::to_string_view_array(input, cudf::get_default_stream());
}

std::unique_ptr<cudf::column> column_from_managed_udf_string_array(
  managed_udf_string* managed_strings, cudf::size_type size)
{
  return detail::column_from_managed_udf_string_array(
    managed_strings, size, cudf::get_default_stream());
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
