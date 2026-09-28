/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<column> replace_nulls(strings_column_view const& input,
                                      string_scalar const& repl,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS(repl.is_valid(stream), "Parameter repl must be valid.");

  string_view d_repl(repl.data(), repl.size());

  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_strings      = *strings_column;

  // build offsets column
  auto offsets_transformer_itr = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([d_strings, d_repl] __device__(size_type idx) {
      return d_strings.is_null(idx) ? d_repl.size_bytes()
                                    : d_strings.element<string_view>(idx).size_bytes();
    }));
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // build chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto d_chars = chars.data();
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     strings_count,
                     [d_strings, d_repl, d_offsets, d_chars] __device__(size_type idx) {
                       string_view d_str = d_repl;
                       if (!d_strings.is_null(idx)) d_str = d_strings.element<string_view>(idx);
                       memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
                     });

  return make_strings_column(
    strings_count, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
