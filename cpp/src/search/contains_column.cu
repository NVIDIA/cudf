/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

namespace {

std::unique_ptr<column> contains_dictionary(column_view const& haystack_in,
                                            column_view const& needles_in,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  dictionary_column_view const haystack(haystack_in);
  dictionary_column_view const needles(needles_in);
  // first combine keys so both dictionaries have the same set
  auto needles_matched = dictionary::detail::add_keys(
    needles, haystack.keys(), stream, cudf::get_current_device_resource_ref());
  auto const needles_view = dictionary_column_view(needles_matched->view());
  auto haystack_matched   = dictionary::detail::set_keys(
    haystack, needles_view.keys(), stream, cudf::get_current_device_resource_ref());
  auto const haystack_view = dictionary_column_view(haystack_matched->view());

  // now just use the indices for the contains
  column_view const haystack_indices = haystack_view.get_indices_annotated();
  column_view const needles_indices  = needles_view.get_indices_annotated();
  auto result_v                      = detail::contains(table_view{{haystack_indices}},
                                   table_view{{needles_indices}},
                                   null_equality::EQUAL,
                                   nan_equality::ALL_EQUAL,
                                   stream,
                                   mr);
  return std::make_unique<column>(std::move(result_v),
                                  detail::copy_bitmask(needles_indices, stream, mr),
                                  needles_indices.null_count());
}

}  // namespace

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // Dictionary columns require key normalization; all other types share the type-erased path.
  if (haystack.type().id() == type_id::DICTIONARY32) {
    return contains_dictionary(haystack, needles, stream, mr);
  }
  auto result_v = detail::contains(table_view{{haystack}},
                                   table_view{{needles}},
                                   null_equality::EQUAL,
                                   nan_equality::ALL_EQUAL,
                                   stream,
                                   mr);
  return std::make_unique<column>(
    std::move(result_v), detail::copy_bitmask(needles, stream, mr), needles.null_count());
}

}  // namespace detail

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, stream, mr);
}

}  // namespace cudf
