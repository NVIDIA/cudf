/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/utility>
#include <cuda/stream>

namespace cudf {

namespace strings::detail {

CUDF_EXPORT std::pair<std::unique_ptr<column>, int64_t> make_offsets_child_column(
  device_span<size_type const> sizes, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  return make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
}

std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<cudf::device_span<string_index_pair const>> const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::pair<string_index_pair const*, size_type>> ranges;
  ranges.reserve(input.size());
  for (auto const& pairs : input) {
    ranges.emplace_back(pairs.data(), static_cast<size_type>(pairs.size()));
  }
  return make_strings_column_batch(ranges, {}, stream, mr);
}

}  // namespace strings::detail

// Create a strings-type column from vector of pointer/size pairs
std::unique_ptr<column> make_strings_column(
  device_span<cuda::std::pair<char const*, size_type> const> strings,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return cudf::strings::detail::make_strings_column(strings.begin(), strings.end(), stream, mr);
}

std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<cudf::device_span<cuda::std::pair<char const*, size_type> const>> const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return cudf::strings::detail::make_strings_column_batch(input, stream, mr);
}

namespace {
struct string_view_to_pair {
  string_view null_placeholder;
  string_view_to_pair(string_view n) : null_placeholder(n) {}
  __device__ cuda::std::pair<char const*, size_type> operator()(string_view const& i)
  {
    return (i.data() == null_placeholder.data())
             ? cuda::std::pair<char const*, size_type>{nullptr, 0}
             : cuda::std::pair<char const*, size_type>{i.data(), i.size_bytes()};
  }
};

}  // namespace

std::unique_ptr<column> make_strings_column(device_span<string_view const> string_views,
                                            string_view null_placeholder,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto it_pair =
    cuda::transform_iterator(string_views.begin(), string_view_to_pair{null_placeholder});
  return cudf::strings::detail::make_strings_column(
    it_pair, it_pair + string_views.size(), stream, mr);
}

std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            std::unique_ptr<column> offsets_column,
                                            rmm::device_buffer&& chars_buffer,
                                            size_type null_count,
                                            cuda::device_buffer<cudf::bitmask_type>&& null_mask)
{
  CUDF_FUNC_RANGE();

  if (null_count > 0) { CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable."); }
  CUDF_EXPECTS(num_strings == offsets_column->size() - 1,
               "Invalid offsets column size for strings column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));

  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  std::move(chars_buffer),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            rmm::device_uvector<size_type>&& offsets,
                                            rmm::device_uvector<char>&& chars,
                                            cuda::device_buffer<cudf::bitmask_type>&& null_mask,
                                            size_type null_count)
{
  CUDF_FUNC_RANGE();

  if (num_strings == 0) { return make_empty_column(type_id::STRING); }

  auto const offsets_size = static_cast<size_type>(offsets.size());

  if (null_count > 0) CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable.");

  CUDF_EXPECTS(num_strings == offsets_size - 1, "Invalid offsets column size for strings column.");

  auto offsets_column = std::make_unique<column>(  //
    data_type{type_id::INT32},
    offsets_size,
    offsets.release(),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
    0);

  auto children = std::vector<std::unique_ptr<column>>();

  children.emplace_back(std::move(offsets_column));

  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  chars.release(),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
