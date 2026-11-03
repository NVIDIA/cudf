/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Reverse individual characters in each string
 */
struct reverse_characters_fn {
  column_device_view const d_strings;
  cudf::detail::input_offsetalator d_offsets;
  char* d_chars;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str = d_strings.element<string_view>(idx);
    // pointer to the end of the output string
    auto d_output = d_chars + d_offsets[idx] + d_str.size_bytes();
    for (auto const chr : d_str) {          // iterate through each character;
      d_output -= bytes_in_char_utf8(chr);  // position output;
      from_char_utf8(chr, d_output);        // place character into output
    }
  }
};

}  // namespace

std::unique_ptr<column> reverse(strings_column_view const& input,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }

  // copy the column; replace data in the chars column
  auto result          = std::make_unique<column>(input.parent(), stream, mr);
  auto sv              = strings_column_view(result->view());
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(sv.offsets());
  auto d_chars         = result->mutable_view().head<char>();

  auto const d_column = column_device_view::create(input.parent(), stream);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     input.size(),
                     reverse_characters_fn{*d_column, d_offsets, d_chars});

  return result;
}

}  // namespace detail

std::unique_ptr<column> reverse(strings_column_view const& input,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
