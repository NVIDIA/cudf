/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "maps_column_view.hpp"

#include <cudf/lists/contains.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf::jni {

namespace {
column_view make_lists(column_view const& lists_child, lists_column_view const& lists_of_structs)
{
  return column_view{data_type{type_id::LIST},
                     lists_of_structs.size(),
                     nullptr,
                     lists_of_structs.null_mask(),
                     lists_of_structs.null_count(),
                     lists_of_structs.offset(),
                     {lists_of_structs.offsets(), lists_child}};
}
}  // namespace

maps_column_view::maps_column_view(lists_column_view const& lists_of_structs,
                                   cuda::stream_ref stream)
  : keys_{make_lists(lists_of_structs.child().child(0), lists_of_structs)},
    values_{make_lists(lists_of_structs.child().child(1), lists_of_structs)}
{
  auto const structs = lists_of_structs.child();
  CUDF_EXPECTS(structs.type().id() == type_id::STRUCT,
               "maps_column_view input must have exactly 1 child (STRUCT) column.");
  CUDF_EXPECTS(structs.num_children() == 2,
               "maps_column_view key-value struct must have exactly 2 children.");
}

template <typename KeyT>
std::unique_ptr<column> get_values_for_impl(maps_column_view const& maps_view,
                                            KeyT const& lookup_keys,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const keys_   = maps_view.keys();
  auto const values_ = maps_view.values();
  CUDF_EXPECTS(lookup_keys.type().id() == keys_.child().type().id(),
               "Lookup keys must have the same type as the keys of the map column.");
  auto key_indices              = cudf::lists::index_of(keys_,
                                           lookup_keys,
                                           lists::duplicate_find_option::FIND_LAST,
                                           stream,
                                           cudf::get_current_device_resource_ref());
  auto constexpr absent_offset  = size_type{-1};
  auto constexpr nullity_offset = std::numeric_limits<size_type>::min();
  thrust::replace(rmm::exec_policy_nosync(stream),
                  key_indices->mutable_view().template begin<size_type>(),
                  key_indices->mutable_view().template end<size_type>(),
                  absent_offset,
                  nullity_offset);
  return cudf::lists::extract_list_element(values_, key_indices->view(), stream, mr);
}

std::unique_ptr<column> maps_column_view::get_values_for(column_view const& lookup_keys,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(lookup_keys.size() == size(),
               "Lookup keys must have the same size as the map column.");

  return get_values_for_impl(*this, lookup_keys, stream, mr);
}

std::unique_ptr<column> maps_column_view::get_values_for(cudf::scalar const& lookup_key,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr) const
{
  return get_values_for_impl(*this, lookup_key, stream, mr);
}

template <typename KeyT>
std::unique_ptr<column> contains_impl(maps_column_view const& maps_view,
                                      KeyT const& lookup_keys,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const keys = maps_view.keys();
  CUDF_EXPECTS(lookup_keys.type().id() == keys.child().type().id(),
               "Lookup keys must have the same type as the keys of the map column.");
  auto const contains =
    cudf::lists::contains(keys, lookup_keys, stream, cudf::get_current_device_resource_ref());
  // Replace nulls with BOOL8{false};
  auto const scalar_false = cudf::numeric_scalar<bool>{false, true, stream};
  return cudf::replace_nulls(contains->view(), scalar_false, stream, mr);
}

std::unique_ptr<column> maps_column_view::contains(column_view const& lookup_keys,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(lookup_keys.size() == size(),
               "Lookup keys must have the same size as the map column.");

  return contains_impl(*this, lookup_keys, stream, mr);
}

std::unique_ptr<column> maps_column_view::contains(cudf::scalar const& lookup_key,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr) const
{
  return contains_impl(*this, lookup_key, stream, mr);
}

}  // namespace cudf::jni
