/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/strings/detail/copy_range.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cuda/iterator>
#include <cuda/stream_ref>

#include <memory>
#include <stdexcept>

namespace {
template <typename T>
void in_place_copy_range(cudf::column_view const& source,
                         cudf::mutable_column_view& target,
                         cudf::size_type source_begin,
                         cudf::size_type source_end,
                         cudf::size_type target_begin,
                         cuda::stream_ref stream)
{
  auto p_source_device_view = cudf::column_device_view::create(source, stream);
  if (source.has_nulls()) {
    cudf::detail::copy_range(
      cudf::detail::make_null_replacement_iterator<T>(*p_source_device_view, T()) + source_begin,
      cudf::detail::make_validity_iterator(*p_source_device_view) + source_begin,
      target,
      target_begin,
      target_begin + (source_end - source_begin),
      stream);
  } else {
    cudf::detail::copy_range(p_source_device_view->begin<T>() + source_begin,
                             cuda::make_constant_iterator(true),  // dummy
                             target,
                             target_begin,
                             target_begin + (source_end - source_begin),
                             stream);
  }
}

struct in_place_copy_range_dispatch {
  cudf::column_view const& source;
  cudf::mutable_column_view& target;

  template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
  void operator()(cudf::size_type source_begin,
                  cudf::size_type source_end,
                  cudf::size_type target_begin,
                  cuda::stream_ref stream)
  {
    in_place_copy_range<T>(source, target, source_begin, source_end, target_begin, stream);
  }

  template <typename T, typename... Args>
  void operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for in-place copy.");
  }
};

struct out_of_place_copy_range_dispatch {
  cudf::column_view const& source;
  cudf::column_view const& target;

  template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
  std::unique_ptr<cudf::column> operator()(
    cudf::size_type source_begin,
    cudf::size_type source_end,
    cudf::size_type target_begin,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    auto p_ret = std::make_unique<cudf::column>(target, stream, mr);
    if ((!p_ret->nullable()) && source.has_nulls(source_begin, source_end, stream)) {
      p_ret->set_null_mask(
        cudf::detail::create_null_mask(p_ret->size(), cudf::mask_state::ALL_VALID, stream, mr), 0);
    }

    if (source_end != source_begin) {  // otherwise no-op
      auto ret_view = p_ret->mutable_view();
      in_place_copy_range<T>(source, ret_view, source_begin, source_end, target_begin, stream);
      p_ret->set_null_count(ret_view.null_count());
    }

    return p_ret;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_rep_layout_compatible<T>(), std::unique_ptr<cudf::column>>
  operator()(Args...)
  {
    CUDF_FAIL("Unsupported type for out of place copy.", cudf::data_type_error);
  }
};

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<cudf::string_view>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return cudf::strings::detail::copy_range(
    source, target, source_begin, source_end, target_begin, stream, mr);
}

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<cudf::dictionary32>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // check the keys in the source and target
  cudf::dictionary_column_view const dict_source(source);
  cudf::dictionary_column_view const dict_target(target);
  CUDF_EXPECTS(cudf::have_same_types(dict_source.keys(), dict_target.keys()),
               "dictionary keys must be the same type",
               cudf::data_type_error);

  // combine keys so both dictionaries have the same set
  auto target_matched =
    cudf::dictionary::detail::add_keys(dict_target, dict_source.keys(), stream, mr);
  auto const target_view = cudf::dictionary_column_view(target_matched->view());
  auto source_matched    = cudf::dictionary::detail::set_keys(
    dict_source, target_view.keys(), stream, cudf::get_current_device_resource_ref());
  auto const source_view = cudf::dictionary_column_view(source_matched->view());

  // build the new indices by calling in_place_copy_range on just the indices
  auto const source_indices = source_view.get_indices_annotated();
  auto target_contents      = target_matched->release();
  auto target_indices(std::move(target_contents.children.front()));
  cudf::mutable_column_view new_indices(
    target_indices->type(),
    dict_target.size(),
    target_indices->mutable_view().head(),
    static_cast<cudf::bitmask_type*>(target_contents.null_mask->data()),
    dict_target.null_count());
  cudf::type_dispatcher(new_indices.type(),
                        in_place_copy_range_dispatch{source_indices, new_indices},
                        source_begin,
                        source_end,
                        target_begin,
                        stream);
  auto null_count = new_indices.null_count();
  auto indices_column =
    std::make_unique<cudf::column>(new_indices.type(),
                                   new_indices.size(),
                                   std::move(*(target_indices->release().data.release())),
                                   rmm::device_buffer{0, stream, mr},
                                   0);

  // take the keys from the matched column allocated using mr
  auto keys_column(std::move(target_contents.children.back()));

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(target_contents.null_mask.release())),
                                null_count);
}

}  // namespace

namespace cudf {
namespace detail {
void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin,
                         cuda::stream_ref stream)
{
  CUDF_EXPECTS(cudf::is_fixed_width(target.type()),
               "In-place copy_range does not support variable-sized types.",
               cudf::data_type_error);
  CUDF_EXPECTS((source_begin >= 0) && (source_end <= source.size()) &&
                 (source_begin <= source_end) && (target_begin >= 0) &&
                 (target_begin <= target.size() - (source_end - source_begin)),
               "Range is out of bounds.",
               std::out_of_range);
  CUDF_EXPECTS(cudf::have_same_types(target, source), "Data type mismatch.", cudf::data_type_error);
  CUDF_EXPECTS(target.nullable() || not source.has_nulls(),
               "target should be nullable if source has null values.",
               std::invalid_argument);

  if (source_end != source_begin) {  // otherwise no-op
    cudf::type_dispatcher<dispatch_storage_type>(target.type(),
                                                 in_place_copy_range_dispatch{source, target},
                                                 source_begin,
                                                 source_end,
                                                 target_begin,
                                                 stream);
  }
}

std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((source_begin >= 0) && (source_end <= source.size()) &&
                 (source_begin <= source_end) && (target_begin >= 0) &&
                 (target_begin <= target.size() - (source_end - source_begin)),
               "Range is out of bounds.",
               std::out_of_range);
  CUDF_EXPECTS(cudf::have_same_types(target, source), "Data type mismatch.", cudf::data_type_error);

  return cudf::type_dispatcher<dispatch_storage_type>(
    target.type(),
    out_of_place_copy_range_dispatch{source, target},
    source_begin,
    source_end,
    target_begin,
    stream,
    mr);
}

}  // namespace detail

void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin,
                         cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::copy_range_in_place(
    source, target, source_begin, source_end, target_begin, stream);
}

std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_range(source, target, source_begin, source_end, target_begin, stream, mr);
}

}  // namespace cudf
