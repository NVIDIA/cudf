/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {
struct indices_handler_fn {
  cudf::detail::input_indexalator const d_iterator;
  column_device_view const d_indices;
  size_type oob_index;  // out-of-bounds index identifies nulls
  __device__ size_type operator()(size_type idx) const
  {
    return d_indices.is_null(idx) ? oob_index : d_iterator[idx];
  }
};
}  // namespace

/**
 * @brief Decode a column from a dictionary.
 */
std::unique_ptr<column> decode(dictionary_column_view const& source,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  if (source.is_empty()) return make_empty_column(type_id::EMPTY);

  // annotated indices include the offset, size and bitmask from it's parent
  auto const indices       = source.get_indices_annotated();
  auto const d_indices     = column_device_view::create(indices, stream);
  auto const d_iterator    = cudf::detail::indexalator_factory::make_input_iterator(indices);
  auto const indices_begin = cudf::detail::make_counting_transform_iterator(
    0, indices_handler_fn{d_iterator, *d_indices, source.keys().size()});

  auto table_column = cudf::detail::gather(table_view{{source.keys()}},
                                           indices_begin,
                                           indices_begin + source.size(),
                                           cudf::out_of_bounds_policy::NULLIFY,
                                           stream,
                                           mr)
                        ->release();
  auto output_column = std::unique_ptr<column>(std::move(table_column.front()));

  // apply any nulls to the output column
  output_column->set_null_mask(cudf::detail::copy_bitmask(source.parent(), stream, mr),
                               source.null_count());

  return output_column;
}

}  // namespace detail

std::unique_ptr<column> decode(dictionary_column_view const& source,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode(source, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
