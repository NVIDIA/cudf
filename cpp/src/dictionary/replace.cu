/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/replace.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cuda/stream_ref>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

/**
 * @brief This utility uses `copy_if_else` to replace null entries using the input bitmask as a
 * predicate.
 *
 * The predicate identifies which column row to copy from and the bitmask specifies which rows
 * are null. Since the `copy_if_else` accepts iterators, we also supply it with pair-iterators
 * created from indexalators and the validity masks.
 *
 * @tparam ReplacementItr must be a pair iterator of (index,valid).
 *
 * @param input lhs for `copy_if_else`
 * @param replacement_iter rhs for `copy_if_else`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Always returns column of type INT32 (size_type)
 */
template <typename ReplacementIter>
std::unique_ptr<column> replace_indices(column_view const& input,
                                        ReplacementIter replacement_iter,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto const input_view = column_device_view::create(input, stream);
  auto const d_input    = *input_view;
  auto predicate        = [d_input] __device__(auto i) { return d_input.is_valid(i); };

  auto input_iterator = cudf::detail::indexalator_factory::make_input_optional_iterator(input);

  return cudf::detail::copy_if_else(true,
                                    input_iterator,
                                    input_iterator + input.size(),
                                    replacement_iter,
                                    predicate,
                                    data_type{type_to_id<size_type>()},
                                    stream,
                                    mr);
}

}  // namespace

/**
 * @copydoc cudf::dictionary::detail::replace_nulls(cudf::column_view const&,cudf::column_view
 * const& cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      dictionary_column_view const& replacement,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::empty_like(input.parent()); }
  if (!input.has_nulls()) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }
  CUDF_EXPECTS(cudf::have_same_types(input.keys(), replacement.keys()),
               "keys must match",
               cudf::data_type_error);
  CUDF_EXPECTS(replacement.size() == input.size(), "column sizes must match");

  // first combine the keys so both input dictionaries have the same set
  auto matched =
    match_dictionaries(std::vector<dictionary_column_view>({input, replacement}), stream, mr);

  // now build the new indices by doing replace-null using the updated input indices
  auto const input_indices =
    dictionary_column_view(matched.front()->view()).get_indices_annotated();
  auto const repl_indices = dictionary_column_view(matched.back()->view()).get_indices_annotated();

  auto new_indices =
    replace_indices(input_indices,
                    cudf::detail::indexalator_factory::make_input_optional_iterator(repl_indices),
                    stream,
                    mr);

  return make_dictionary_column(
    std::move(matched.front()->release().children.back()), std::move(new_indices), stream, mr);
}

/**
 * @copydoc cudf::dictionary::detail::replace_nulls(cudf::column_view const&,cudf::scalar
 * const&, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      scalar const& replacement,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::empty_like(input.parent()); }
  if (!input.has_nulls() || !replacement.is_valid(stream)) {
    return std::make_unique<cudf::column>(input.parent(), stream, mr);
  }
  CUDF_EXPECTS(cudf::have_same_types(input.parent(), replacement),
               "keys must match scalar type",
               cudf::data_type_error);

  // first add the replacement to the keys so only the indices need to be processed
  auto input_matched = dictionary::detail::add_keys(
    input,
    make_column_from_scalar(replacement, 1, stream, cudf::get_current_device_resource_ref())
      ->view(),
    stream,
    mr);
  auto const input_view = dictionary_column_view(input_matched->view());
  auto const scalar_index =
    get_index(input_view, replacement, stream, cudf::get_current_device_resource_ref());

  // now build the new indices by doing replace-null on the updated indices
  auto const input_indices = input_view.get_indices_annotated();
  auto new_indices         = replace_indices(
    input_indices,
    cudf::detail::indexalator_factory::make_input_optional_iterator(*scalar_index, stream),
    stream,
    mr);
  new_indices->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);

  return make_dictionary_column(
    std::move(input_matched->release().children.back()), std::move(new_indices), stream, mr);
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
