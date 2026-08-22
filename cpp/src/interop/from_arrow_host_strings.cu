/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"
#include "from_arrow_host.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>

#include <numeric>
#include <string>
#include <vector>

namespace cudf {
namespace detail {
namespace {

constexpr int chars_buffer_idx = 2;

std::unique_ptr<column> from_arrow_string(ArrowSchemaView const* schema,
                                          ArrowArray const* input,
                                          std::unique_ptr<rmm::device_buffer>&& mask,
                                          size_type null_count,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto [offsets_column, offset, char_data_length] = get_offsets_column(schema, input, stream, mr);

  rmm::device_buffer chars(char_data_length, stream, mr);
  auto const* chars_data = static_cast<uint8_t const*>(input->buffers[chars_buffer_idx]) + offset;
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(chars.data(), chars_data, chars.size(), stream));

  return make_strings_column(static_cast<size_type>(input->length),
                             std::move(offsets_column),
                             std::move(chars),
                             null_count,
                             std::move(*mask.release()));
}

constexpr int stringview_vector_idx = 1;

std::unique_ptr<column> from_arrow_stringview(ArrowSchemaView const* schema,
                                              ArrowArray const* input,
                                              std::unique_ptr<rmm::device_buffer>&& mask,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  ArrowArrayView view;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&view, schema->schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&view, input, nullptr));

  // first copy stringview array to device
  auto items   = view.buffer_views[stringview_vector_idx].data.as_binary_view;
  auto d_items = rmm::device_uvector<ArrowBinaryView>(input->length, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    d_items.data(), items + input->offset, input->length * sizeof(ArrowBinaryView), stream));

  // then copy variadic buffers to device
  auto variadics     = std::vector<rmm::device_buffer>();
  auto variadic_ptrs = std::vector<char const*>();
  for (auto i = 0L; i < view.n_variadic_buffers; ++i) {
    variadics.emplace_back(view.variadic_buffers[i], view.variadic_buffer_sizes[i], stream);
    variadic_ptrs.push_back(static_cast<char const*>(variadics.back().data()));
  }

  // copy variadic device pointers to device
  auto d_variadic_ptrs = cudf::detail::make_device_uvector_async(
    variadic_ptrs, stream, cudf::get_current_device_resource_ref());
  auto d_ptrs = d_variadic_ptrs.data();
  auto d_mask = static_cast<cudf::bitmask_type*>(mask->data());

  using string_index_pair = cudf::strings::detail::string_index_pair;

  // create indices to string fragments for the make_strings_column gather
  auto d_indices = rmm::device_uvector<string_index_pair>(input->length, stream, mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{static_cast<cudf::size_type>(input->length)},
    d_indices.begin(),
    [d_items = d_items.data(), d_ptrs, d_mask] __device__(auto idx) -> string_index_pair {
      if (d_mask && !bit_is_set(d_mask, idx)) { return string_index_pair{nullptr, 0}; }
      auto const& item = d_items[idx];
      auto const size  = static_cast<cudf::size_type>(item.inlined.size);
      auto const data  = (size <= NANOARROW_BINARY_VIEW_INLINE_SIZE)
                           ? reinterpret_cast<char const*>(item.inlined.data)
                           : d_ptrs[item.ref.buffer_index] + item.ref.offset;
      return {data, size};
    });

  cudf::detail::sync_stream(stream);
  return cudf::make_strings_column(d_indices, stream, mr);
}

}  // namespace

std::unique_ptr<column> string_column_from_arrow_host(ArrowSchemaView const* schema,
                                                      ArrowArray const* input,
                                                      std::unique_ptr<rmm::device_buffer>&& mask,
                                                      size_type null_count,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  return schema->type == NANOARROW_TYPE_STRING_VIEW
           ? from_arrow_stringview(schema, input, std::move(mask), stream, mr)
           : from_arrow_string(schema, input, std::move(mask), null_count, stream, mr);
}

}  // namespace detail
}  // namespace cudf
