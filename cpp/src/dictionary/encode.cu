/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

template <typename SetRef>
struct encode_fn {
  SetRef set_ref;
  column_device_view d_input;
  __device__ size_type operator()(size_type idx)
  {
    return d_input.is_valid(idx) ? *cuda::std::get<0>(set_ref.insert_and_find(idx)) : idx;
  }
};

}  // namespace

std::unique_ptr<column> encode(column_view const& input,
                               data_type indices_type,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_signed(indices_type) && is_index_type(indices_type),
               "indices must be type signed integer",
               cudf::data_type_error);
  CUDF_EXPECTS(input.type().id() != type_id::DICTIONARY32,
               "cannot encode a dictionary from a dictionary",
               std::invalid_argument);
  CUDF_EXPECTS(not cudf::is_nested(input.type()),
               "encoding nested types not supported",
               std::invalid_argument);

  auto indices_column = cudf::make_numeric_column(
    indices_type, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (input.is_empty()) {
    return make_dictionary_column(
      make_empty_column(input.type()), std::move(indices_column), rmm::device_buffer{}, 0);
  }

  using encode_probe_t = cuco::linear_probing<
    1,
    cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                               cudf::nullate::DYNAMIC>>;

  auto const has_nulls  = nullate::DYNAMIC{input.has_nulls()};
  auto const tv         = cudf::table_view({input});
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const row_hash   = cudf::detail::row::hash::row_hasher(tv, stream, temp_mr);
  auto const row_equal  = cudf::detail::row::equality::self_comparator(tv, stream, temp_mr);
  auto const comparator = cudf::detail::row::equality::nan_equal_physical_equality_comparator{};
  auto const d_equal    = row_equal.equal_to<false>(has_nulls, null_equality::EQUAL, comparator);
  auto const empty_key  = cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL};
  auto probe            = encode_probe_t{row_hash.device_hasher(has_nulls)};
  auto allocator        = rmm::mr::polymorphic_allocator<char>{temp_mr};
  auto set =
    cuco::static_set{input.size(), 0.5, empty_key, d_equal, probe, {}, {}, allocator, stream.get()};
  auto set_ref    = set.ref(cuco::insert_and_find);
  using set_ref_t = decltype(set_ref);

  // build a static_set of the input values
  // and keep track of the indices of the unique values
  auto d_indices = rmm::device_uvector<size_type>(input.size(), stream, temp_mr);
  auto d_input   = column_device_view::create(input, stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    d_indices.begin(),
                    encode_fn{set_ref, *d_input});

  auto keys_indices = rmm::device_uvector<size_type>(input.size(), stream, temp_mr);
  auto keys_end     = set.retrieve_all(keys_indices.begin(), stream.get());
  keys_indices.resize(cuda::std::distance(keys_indices.begin(), keys_end), stream);

  // sort the keys_indices so we can use lower-bound on them
  thrust::sort(rmm::exec_policy_nosync(stream, temp_mr), keys_indices.begin(), keys_indices.end());

  // use keys_indices to retrieve the keys
  auto const oob_policy   = cudf::out_of_bounds_policy::DONT_CHECK;
  auto const index_policy = cudf::negative_index_policy::NOT_ALLOWED;
  auto keys_column =
    std::move(cudf::detail::gather(tv, keys_indices, oob_policy, index_policy, stream, mr)
                ->release()
                .front());

  // call lower-bound with keys_indices and d_indices to get the output indices_column
  auto d_result =
    cudf::detail::indexalator_factory::make_output_iterator(indices_column->mutable_view());
  thrust::lower_bound(rmm::exec_policy_nosync(stream, temp_mr),
                      keys_indices.begin(),
                      keys_indices.end(),
                      d_indices.begin(),
                      d_indices.end(),
                      d_result);

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                cudf::detail::copy_bitmask(input, stream, mr),
                                input.null_count());
}

/**
 * @copydoc cudf::dictionary::detail::get_indices_type_for_size
 */
data_type get_indices_type_for_size(size_type keys_size)
{
  if (keys_size <= std::numeric_limits<int8_t>::max()) return data_type{type_id::INT8};
  if (keys_size <= std::numeric_limits<int16_t>::max()) return data_type{type_id::INT16};
  return data_type{type_id::INT32};
}

}  // namespace detail

// external API

std::unique_ptr<column> encode(column_view const& input_column,
                               data_type indices_type,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode(input_column, indices_type, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
