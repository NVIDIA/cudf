/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/scatter.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/count.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace detail {

/**
 * @brief Convert a scatter map into a gather map.
 *
 * The caller is expected to use the output map on a subsequent gather_bitmask()
 * function using the PASSTHROUGH op since the resulting map may contain index
 * values outside the target's range.
 *
 * First, the gather-map is initialized with an invalid index.
 * The value `numeric_limits::lowest()` is used since it should always be outside the target size.
 * Then, `output[scatter_map[i]] = i` for each `i`.
 *
 * @tparam MapIterator Iterator type of the input scatter map.
 * @param scatter_map_begin Beginning of scatter map.
 * @param scatter_map_end End of the scatter map.
 * @param gather_rows Number of rows in the output map.
 * @param stream Stream used for CUDA kernel calls.
 * @return Output gather map.
 */
template <typename MapIterator>
auto scatter_to_gather(MapIterator scatter_map_begin,
                       MapIterator scatter_map_end,
                       size_type gather_rows,
                       rmm::cuda_stream_view stream)
{
  using MapValueType = cuda::std::iter_value_t<MapIterator>;

  // The gather_map is initialized with `numeric_limits::lowest()` value to identify pass-through
  // entries when calling the gather_bitmask() which applies a pass-through whenever it finds a
  // value outside the range of the target column.
  // We'll use the `numeric_limits::lowest()` value for this since it should always be outside the
  // valid range.
  auto gather_map = rmm::device_uvector<size_type>(gather_rows, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    gather_map.begin(),
    gather_map.end(),
    std::numeric_limits<size_type>::lowest());

  // Convert scatter map to a gather map
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  cuda::counting_iterator<MapValueType>{0},
                  cuda::counting_iterator{
                    static_cast<MapValueType>(std::distance(scatter_map_begin, scatter_map_end))},
                  scatter_map_begin,
                  gather_map.begin());

  return gather_map;
}

/**
 * @brief Create a complement map of `scatter_to_gather` map
 *
 * The purpose of this map is to create an identity-mapping for the rows that are not
 * touched by the `scatter_map`.
 *
 * The output result of this mapping is firstly initialized as an identity-mapping
 * (i.e., `output[i] = i`). Then, for each value `idx` from `scatter_map`, the value `output[idx]`
 * is set to `numeric_limits::lowest()`, which is an invalid, out-of-bound index to identify the
 * pass-through entries when calling the `gather_bitmask()` function.
 *
 */
template <typename MapIterator>
auto scatter_to_gather_complement(MapIterator scatter_map_begin,
                                  MapIterator scatter_map_end,
                                  size_type gather_rows,
                                  rmm::cuda_stream_view stream)
{
  auto gather_map = rmm::device_uvector<size_type>(gather_rows, stream);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   gather_map.begin(),
                   gather_map.end(),
                   0);

  auto const out_of_bounds_begin =
    cuda::make_constant_iterator(std::numeric_limits<size_type>::lowest());
  auto const out_of_bounds_end =
    out_of_bounds_begin + cuda::std::distance(scatter_map_begin, scatter_map_end);
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  out_of_bounds_begin,
                  out_of_bounds_end,
                  scatter_map_begin,
                  gather_map.begin());
  return gather_map;
}

template <typename Element, typename Enable = void>
struct column_scatterer_impl {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported type for scatter.");
  }
};

template <typename Element>
struct column_scatterer_impl<Element, std::enable_if_t<cudf::is_fixed_width<Element>()>> {
  template <typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    auto result      = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // NOTE use source.begin + scatter rows rather than source.end in case the
    // scatter map is smaller than the number of source rows
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    source.begin<Element>(),
                    source.begin<Element>() + cudf::distance(scatter_map_begin, scatter_map_end),
                    scatter_map_begin,
                    result_view.begin<Element>());

    return result;
  }
};

template <>
struct column_scatterer_impl<string_view> {
  template <typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    auto d_column    = column_device_view::create(source, stream);
    auto const begin = d_column->begin<string_view>();
    auto const end   = begin + cudf::distance(scatter_map_begin, scatter_map_end);
    return strings::detail::scatter(begin, end, scatter_map_begin, target, stream, mr);
  }
};

template <>
struct column_scatterer_impl<list_view> {
  template <typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    return cudf::lists::detail::scatter(
      source, scatter_map_begin, scatter_map_end, target, stream, mr);
  }
};

template <>
struct column_scatterer_impl<dictionary32> {
  template <typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source_in,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target_in,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    if (target_in.is_empty())  // empty begets empty
      return make_empty_column(type_id::DICTIONARY32);
    if (source_in.is_empty())  // no input, just make a copy
      return std::make_unique<column>(target_in, stream, mr);

    // check the keys match
    dictionary_column_view const source(source_in);
    dictionary_column_view const target(target_in);
    CUDF_EXPECTS(cudf::have_same_types(source.keys(), target.keys()),
                 "scatter dictionary keys must be the same type",
                 cudf::data_type_error);

    // first combine keys so both dictionaries have the same set
    auto target_matched    = dictionary::detail::add_keys(target, source.keys(), stream, mr);
    auto const target_view = dictionary_column_view(target_matched->view());
    auto source_matched    = dictionary::detail::set_keys(
      source, target_view.keys(), stream, cudf::get_current_device_resource_ref());
    auto const source_view = dictionary_column_view(source_matched->view());

    // now build the new indices by doing a scatter on just the matched indices
    auto source_itr  = indexalator_factory::make_input_iterator(source_view.indices());
    auto new_indices = std::make_unique<column>(target_view.get_indices_annotated(), stream, mr);
    auto target_itr  = indexalator_factory::make_output_iterator(new_indices->mutable_view());
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    source_itr,
                    source_itr + std::distance(scatter_map_begin, scatter_map_end),
                    scatter_map_begin,
                    target_itr);

    // take the keys from the matched column allocated using mr
    std::unique_ptr<column> keys_column(std::move(target_matched->release().children.back()));

    // create column with keys_column and indices_column
    return make_dictionary_column(std::move(keys_column), std::move(new_indices), stream, mr);
  }
};

struct column_scatterer {
  template <typename Element, typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    column_scatterer_impl<Element> scatterer{};
    return scatterer(source, scatter_map_begin, scatter_map_end, target, stream, mr);
  }
};

template <>
struct column_scatterer_impl<struct_view> {
  template <typename MapItRoot>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapItRoot scatter_map_begin,
                                     MapItRoot scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(source.num_children() == target.num_children(),
                 "Scatter source and target are not of the same type.");

    auto const scatter_map_size = std::distance(scatter_map_begin, scatter_map_end);
    if (scatter_map_size == 0) { return std::make_unique<column>(target, stream, mr); }

    structs_column_view const structs_src(source);
    structs_column_view const structs_target(target);
    std::vector<std::unique_ptr<column>> output_struct_members(structs_src.num_children());

    std::transform(structs_src.child_begin(),
                   structs_src.child_end(),
                   structs_target.child_begin(),
                   output_struct_members.begin(),
                   [&scatter_map_begin, &scatter_map_end, stream, mr](auto const& source_col,
                                                                      auto const& target_col) {
                     return type_dispatcher<dispatch_storage_type>(source_col.type(),
                                                                   column_scatterer{},
                                                                   source_col,
                                                                   scatter_map_begin,
                                                                   scatter_map_end,
                                                                   target_col,
                                                                   stream,
                                                                   mr);
                   });

    // We still need to call `gather_bitmask` even when the source's children are not nullable,
    // as if the target's children have null_masks, those null_masks need to be updated after
    // being scattered onto.
    auto const child_nullable = std::any_of(structs_src.child_begin(),
                                            structs_src.child_end(),
                                            [](auto const& col) { return col.nullable(); }) or
                                std::any_of(structs_target.child_begin(),
                                            structs_target.child_end(),
                                            [](auto const& col) { return col.nullable(); });
    if (child_nullable) {
      auto const gather_map =
        scatter_to_gather(scatter_map_begin, scatter_map_end, target.size(), stream);
      gather_bitmask(cudf::table_view{std::vector<cudf::column_view>{structs_src.child_begin(),
                                                                     structs_src.child_end()}},
                     gather_map.begin(),
                     output_struct_members,
                     gather_bitmask_op::PASSTHROUGH,
                     stream,
                     mr);
    }

    // Need to put the result column in a vector to call `gather_bitmask`.
    std::vector<std::unique_ptr<column>> result;
    result.emplace_back(cudf::make_structs_column(target.size(),
                                                  std::move(output_struct_members),
                                                  0,
                                                  rmm::device_buffer{0, stream, mr},
                                                  stream,
                                                  mr));

    // Only gather bitmask from the target column for the rows that have not been scattered onto
    // The bitmask from the source column will be gathered at the top level `scatter()` call.
    if (target.nullable()) {
      auto const gather_map =
        scatter_to_gather_complement(scatter_map_begin, scatter_map_end, target.size(), stream);
      gather_bitmask(table_view{std::vector<cudf::column_view>{target}},
                     gather_map.begin(),
                     result,
                     gather_bitmask_op::PASSTHROUGH,
                     stream,
                     mr);
    }

    return std::move(result.front());
  }
};

/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_begin[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined. This range might have negative values, which will be modified by adding target.size()
 *
 * @throws cudf::logic_error if scatter map index is out of bounds
 * @throws cudf::logic_error if scatter_map.size() > source.num_rows()
 *
 * @param[in] source The input columns containing values to be scattered into the
 * target columns
 * @param[in] scatter_map_begin Beginning of iterator range of integer indices that has been
 *provided.
 * @param[in] scatter_map_end End of iterator range of integer indices that has been provided.
 * source columns to rows in the target columns
 * @param[in] target The set of columns into which values from the source_table
 * are to be scattered
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Result of scattering values from source to target
 */
template <typename MapIterator>
std::unique_ptr<table> scatter(table_view const& source,
                               MapIterator scatter_map_begin,
                               MapIterator scatter_map_end,
                               table_view const& target,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  using MapType = cuda::std::iter_value_t<MapIterator>;

  CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) <= source.num_rows(),
               "scatter map size should be <= to number of rows in source");

  // Transform negative indices to index + target size.
  auto updated_scatter_map_begin =
    thrust::make_transform_iterator(scatter_map_begin, index_converter<MapType>{target.num_rows()});
  auto updated_scatter_map_end =
    thrust::make_transform_iterator(scatter_map_end, index_converter<MapType>{target.num_rows()});
  auto result = std::vector<std::unique_ptr<column>>(target.num_columns());

  std::transform(source.begin(),
                 source.end(),
                 target.begin(),
                 result.begin(),
                 [=](auto const& source_col, auto const& target_col) {
                   return type_dispatcher<dispatch_storage_type>(source_col.type(),
                                                                 column_scatterer{},
                                                                 source_col,
                                                                 updated_scatter_map_begin,
                                                                 updated_scatter_map_end,
                                                                 target_col,
                                                                 stream,
                                                                 mr);
                 });

  // We still need to call `gather_bitmask` even when the source columns are not nullable,
  // as if the target has null_mask, that null_mask needs to be updated after scattering.
  auto const nullable =
    std::any_of(source.begin(), source.end(), [](auto const& col) { return col.nullable(); }) or
    std::any_of(target.begin(), target.end(), [](auto const& col) { return col.nullable(); });
  if (nullable) {
    auto const gather_map = scatter_to_gather(
      updated_scatter_map_begin, updated_scatter_map_end, target.num_rows(), stream);
    gather_bitmask(source, gather_map.begin(), result, gather_bitmask_op::PASSTHROUGH, stream, mr);

    // For struct columns, we need to superimpose the null_mask of the parent over the null_mask of
    // the children.
    std::for_each(result.begin(), result.end(), [=](auto& col) {
      auto const col_view = col->view();
      if (col_view.type().id() == type_id::STRUCT and col_view.nullable()) {
        auto const num_rows   = col_view.size();
        auto const null_count = col_view.null_count();
        auto contents         = col->release();

        // Children null_mask will be superimposed during structs column construction.
        col = cudf::make_structs_column(num_rows,
                                        std::move(contents.children),
                                        null_count,
                                        std::move(*contents.null_mask),
                                        stream,
                                        mr);
      }
    });
  }
  return std::make_unique<table>(std::move(result), target.num_rows());
}
}  // namespace detail
}  // namespace cudf
