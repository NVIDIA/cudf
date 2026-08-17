/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Generates the new set of offsets that regroups the concatenated-by-column inputs
 * into concatenated-by-rows inputs, and the associated null mask.
 *
 * If we have the following input columns:
 *
 * s1 = [{0, 1}, {2, 3, 4}, {5}, {},           {6, 7}]
 * s2 = [{8},    {9},       {},  {10, 11, 12}, {13, 14, 15, 16}]
 *
 * We can rearrange the child data using a normal concatenate and a gather such that
 * the resulting values are in the correct order. For the above example, the
 * child column would look like:
 *
 * {0, 1, 8, 2, 3, 4, 9, 5, 10, 11, 12, 6, 7, 13, 14, 15}
 *
 * Because we did a regular concatenate (and a subsequent gather to reorder the rows),
 * the top level rows of the list column would look like:
 *
 * (2N rows)
 * [{0, 1}, {8}, {2, 3, 4}, {9}, {5}, {10, 11, 12}, {6, 7}, {13, 14, 15, 16}]
 *
 * What we really want is:
 *
 * (N rows)
 * [{0, 1, 8}, {2, 3, 4, 9}, {5}, {10, 11, 12}, {6, 7, 13, 14, 15, 16}]
 *
 * We can do this by recomputing a new offsets column that does this regrouping.
 *
 */
std::tuple<std::unique_ptr<column>, rmm::device_buffer, size_type>
generate_regrouped_offsets_and_null_mask(table_device_view const& input,
                                         bool build_null_mask,
                                         concatenate_null_policy null_policy,
                                         device_span<size_type const> row_null_counts,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  // outgoing offsets.
  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, input.num_rows() + 1, mask_state::UNALLOCATED, stream, mr);

  auto keys =
    thrust::make_transform_iterator(cuda::counting_iterator<std::size_t>{0},
                                    cuda::proclaim_return_type<size_type>(
                                      [num_columns = input.num_columns()] __device__(
                                        std::size_t i) -> size_type { return i / num_columns; }));

  // generate sizes for the regrouped rows
  auto values = thrust::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<size_type>([input,
                                           row_null_counts = row_null_counts.data(),
                                           null_policy] __device__(std::size_t i) -> size_type {
      auto const col_index = i % input.num_columns();
      auto const row_index = i / input.num_columns();

      // nullify the whole output row
      if (row_null_counts) {
        if ((null_policy == concatenate_null_policy::NULLIFY_OUTPUT_ROW &&
             row_null_counts[row_index] > 0) ||
            (null_policy == concatenate_null_policy::IGNORE &&
             row_null_counts[row_index] == input.num_columns())) {
          return 0;
        }
      }
      auto offsets =
        input.column(col_index).child(lists_column_view::offsets_column_index).data<int32_t>() +
        input.column(col_index).offset();
      return offsets[row_index + 1] - offsets[row_index];
    }));

  cudf::detail::reduce_by_key_async(keys,
                                    keys + (input.num_rows() * input.num_columns()),
                                    values,
                                    cuda::make_discard_iterator(),
                                    offsets->mutable_view().begin<int32_t>(),
                                    cuda::std::plus<size_type>(),
                                    stream);

  // convert to offsets
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         offsets->view().begin<int32_t>(),
                         offsets->view().begin<int32_t>() + input.num_rows() + 1,
                         offsets->mutable_view().begin<int32_t>(),
                         0);

  // generate appropriate null mask
  auto [null_mask, null_count] = [&]() {
    // if the input doesn't contain nulls, no work to do
    if (!build_null_mask) {
      return std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{}, 0};
    }

    // row is null if -all- input rows are null
    if (null_policy == concatenate_null_policy::IGNORE) {
      return cudf::detail::valid_if(
        row_null_counts.begin(),
        row_null_counts.begin() + input.num_rows(),
        [num_columns = input.num_columns()] __device__(size_type null_count) {
          return null_count != num_columns;
        },
        stream,
        mr);
    }

    // row is null if -any- input rows are null
    return cudf::detail::valid_if(
      row_null_counts.begin(),
      row_null_counts.begin() + input.num_rows(),
      [] __device__(size_type null_count) { return null_count == 0; },
      stream,
      mr);
  }();

  return {std::move(offsets), std::move(null_mask), null_count};
}

rmm::device_uvector<size_type> generate_null_counts(table_device_view const& input,
                                                    rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> null_counts(input.num_rows(), stream);

  auto keys =
    thrust::make_transform_iterator(cuda::counting_iterator<std::size_t>{0},
                                    cuda::proclaim_return_type<size_type>(
                                      [num_columns = input.num_columns()] __device__(
                                        std::size_t i) -> size_type { return i / num_columns; }));

  auto null_values = thrust::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<size_type>([input] __device__(std::size_t i) -> size_type {
      auto const col_index = i % input.num_columns();
      auto const row_index = i / input.num_columns();
      auto const& col      = input.column(col_index);
      return col.null_mask() ? (bit_is_set(col.null_mask(), row_index + col.offset()) ? 0 : 1) : 0;
    }));

  cudf::detail::reduce_by_key_async(keys,
                                    keys + (input.num_rows() * input.num_columns()),
                                    null_values,
                                    cuda::make_discard_iterator(),
                                    null_counts.data(),
                                    cuda::std::plus<size_type>(),
                                    stream);

  return null_counts;
}

}  // anonymous namespace

/**
 * @copydoc cudf::lists::concatenate_rows
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.num_columns() > 0, "The input table must have at least one column.");

  auto const entry_type = lists_column_view(*input.begin()).child().type();
  CUDF_EXPECTS(
    std::all_of(input.begin(),
                input.end(),
                [](column_view const& col) { return col.type().id() == cudf::type_id::LIST; }),
    "All columns of the input table must be of list column type.",
    cudf::data_type_error);
  CUDF_EXPECTS(cudf::all_have_same_types(input.begin(), input.end()),
               "The types of entries in the input columns must be the same.",
               cudf::data_type_error);

  auto const num_rows = input.num_rows();
  auto const num_cols = input.num_columns();
  if (num_rows == 0) { return cudf::empty_like(input.column(0)); }
  if (num_cols == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }

  // concatenate the input table into one column.
  std::vector<column_view> cols(input.num_columns());
  std::copy(input.begin(), input.end(), cols.begin());
  auto concat = cudf::detail::concatenate(cols, stream, cudf::get_current_device_resource_ref());

  // whether or not we should be generating a null mask at all
  auto const build_null_mask = concat->has_nulls();

  auto input_dv = table_device_view::create(input, stream);

  // if the output needs a null mask, generate a vector of null counts per row of input, where the
  // count is the number of columns that contain a null for a given row.
  auto row_null_counts = build_null_mask ? generate_null_counts(*input_dv, stream)
                                         : rmm::device_uvector<size_type>{0, stream};

  // if we have nulls, overlay an appropriate null mask onto the
  // concatenated column so that gather() sanitizes out the child data of rows that will ultimately
  // be nullified.
  if (build_null_mask) {
    auto [null_mask, null_count] = [&]() {
      auto iter = cuda::counting_iterator<std::size_t>{0};

      // IGNORE.  Output row is nullified if all input rows are null.
      if (null_policy == concatenate_null_policy::IGNORE) {
        return cudf::detail::valid_if(
          iter,
          iter + (input.num_rows() * input.num_columns()),
          cuda::proclaim_return_type<size_type>(
            [num_rows        = input.num_rows(),
             num_columns     = input.num_columns(),
             row_null_counts = row_null_counts.data()] __device__(std::size_t i) -> size_type {
              auto const row_index = i % num_rows;
              return row_null_counts[row_index] != num_columns;
            }),
          stream,
          cudf::get_current_device_resource_ref());
      }
      // NULLIFY_OUTPUT_ROW.  Output row is nullfied if any input row is null
      return cudf::detail::valid_if(
        iter,
        iter + (input.num_rows() * input.num_columns()),
        cuda::proclaim_return_type<size_type>(
          [num_rows        = input.num_rows(),
           row_null_counts = row_null_counts.data()] __device__(std::size_t i) -> size_type {
            auto const row_index = i % num_rows;
            return row_null_counts[row_index] == 0;
          }),
        stream,
        cudf::get_current_device_resource_ref());
    }();
    concat->set_null_mask(std::move(null_mask), null_count);
  }

  // perform the gather to rearrange the rows in desired child order. this will produce -almost-
  // what we want. the data of the children will be exactly what we want, but will be grouped as if
  // we had concatenated all the rows together instead of concatenating within the rows.  To fix
  // this we can simply swap in a new set of offsets that re-groups them.  bmo
  auto iter = thrust::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<size_type>(
      [num_columns = input.num_columns(),
       num_rows    = input.num_rows()] __device__(std::size_t i) -> size_type {
        auto const src_col_index    = i % num_columns;
        auto const src_row_index    = i / num_columns;
        auto const concat_row_index = (src_col_index * num_rows) + src_row_index;
        return concat_row_index;
      }));
  auto gathered = cudf::detail::gather(table_view({*concat}),
                                       iter,
                                       iter + (input.num_columns() * input.num_rows()),
                                       out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);

  // generate regrouped offsets and null mask
  auto [offsets, null_mask, null_count] = generate_regrouped_offsets_and_null_mask(
    *input_dv, build_null_mask, null_policy, row_null_counts, stream, mr);

  // reassemble the underlying child data with the regrouped offsets and null mask
  column& col   = gathered->get_column(0);
  auto contents = col.release();
  return cudf::make_lists_column(
    input.num_rows(),
    std::move(offsets),
    std::move(contents.children[lists_column_view::child_column_index]),
    null_count,
    std::move(null_mask));
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_rows
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_rows(input, null_policy, stream, mr);
}

}  // namespace lists
}  // namespace cudf
