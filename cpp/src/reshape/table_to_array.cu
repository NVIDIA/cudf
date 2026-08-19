/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reshape.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reshape.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace cudf {
namespace detail {
namespace {

template <typename T>
void table_to_array_impl(table_view const& input,
                         device_span<cuda::std::byte> output,
                         rmm::cuda_stream_view stream)
{
  auto const num_columns = input.num_columns();
  auto const num_rows    = input.num_rows();
  auto const item_size   = sizeof(T);
  auto const total_bytes = static_cast<size_t>(num_columns) * num_rows * item_size;

  CUDF_EXPECTS(output.size() >= total_bytes, "Output span is too small", std::invalid_argument);
  CUDF_EXPECTS(cudf::all_have_same_types(input.begin(), input.end()),
               "All columns must have the same data type",
               cudf::data_type_error);
  CUDF_EXPECTS(!cudf::has_nulls(input), "All columns must contain no nulls", std::invalid_argument);

  auto* base_ptr = output.data();

  auto h_srcs = make_host_vector<T const*>(num_columns, stream);
  auto h_dsts = make_host_vector<T*>(num_columns, stream);

  std::transform(input.begin(), input.end(), h_srcs.begin(), [](auto& col) {
    return const_cast<T*>(col.template data<T>());
  });

  for (int i = 0; i < num_columns; ++i) {
    h_dsts[i] = reinterpret_cast<T*>(base_ptr + i * item_size * num_rows);
  }

  auto const mr = cudf::get_current_device_resource_ref();

  auto d_srcs = cudf::detail::make_device_uvector_async(h_srcs, stream, mr);
  auto d_dsts = cudf::detail::make_device_uvector_async(h_dsts, stream, mr);

  cuda::constant_iterator<size_t> sizes(static_cast<size_t>(item_size * num_rows));

  cudf::detail::batched_memcpy_async(
    d_srcs.begin(), d_dsts.begin(), sizes, num_columns, stream.value());
  stream.synchronize();  // ensures h_srcs and h_dsts are not destroyed before the copy is done
}

struct table_to_array_dispatcher {
  table_view const& input;
  device_span<cuda::std::byte> output;
  rmm::cuda_stream_view stream;

  template <typename T, CUDF_ENABLE_IF(is_fixed_width<T>())>
  void operator()() const
  {
    table_to_array_impl<T>(input, output, stream);
  }

  template <typename T, CUDF_ENABLE_IF(!is_fixed_width<T>())>
  void operator()() const
  {
    CUDF_FAIL("Unsupported dtype");
  }
};

}  // namespace

void table_to_array(table_view const& input,
                    device_span<cuda::std::byte> output,
                    rmm::cuda_stream_view stream)
{
  if (input.num_columns() == 0) return;

  auto const dtype = input.column(0).type();

  cudf::type_dispatcher<cudf::dispatch_storage_type>(
    dtype, table_to_array_dispatcher{input, output, stream});
}

}  // namespace detail

void table_to_array(table_view const& input,
                    device_span<cuda::std::byte> output,
                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  cudf::detail::table_to_array(input, output, stream);
}

}  // namespace cudf
