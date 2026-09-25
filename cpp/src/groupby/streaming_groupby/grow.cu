/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/output_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>

#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby {

namespace {

/*
 * Files one slot of the old hash set into the new one.  Every stored value is a distinct
 * dense ID, probed at the hash its key was inserted under, so each lands in a slot of its own.
 */
template <typename SetRef>
struct reinsert_fn {
  size_type const* old_slots;
  mutable SetRef set_ref;

  __device__ void operator()(int64_t slot) const
  {
    auto const dense_id = old_slots[slot];
    if (dense_id != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { set_ref.insert(dense_id); }
  }
};

}  // namespace

void streaming_groupby::impl::grow(size_type new_capacity, cuda::stream_ref stream)
{
  CUDF_EXPECTS(new_capacity > _capacity, "Internal error: streaming_groupby can only grow.");

  auto const distinct = _distinct_keys.load(std::memory_order_relaxed);
  auto const mr       = cudf::get_current_device_resource_ref();

  // The aggregation kernels of earlier calls may still be writing the results table.
  for (auto const& aggregation : _inflight_aggregations) {
    aggregation->wait(stream);
  }

  auto key_set = make_key_set(new_capacity, stream);
  auto key_loc = std::make_unique<rmm::device_uvector<key_location_t>>(new_capacity, stream, mr);
  auto key_hashes =
    std::make_unique<rmm::device_uvector<hash_value_type>>(new_capacity, stream, mr);

  std::vector<std::unique_ptr<column>> result_columns;
  result_columns.reserve(_agg_results->num_columns());
  for (auto const& old_column : _agg_results->view()) {
    result_columns.push_back(detail::hash::create_result_column(
      old_column.type(),
      new_capacity,
      old_column.nullable() ? mask_state::ALL_NULL : mask_state::UNALLOCATED,
      stream,
      mr));
  }
  auto agg_results = std::make_unique<table>(std::move(result_columns));
  cudf::detail::initialize_with_identity(agg_results->mutable_view(), _agg_kinds, stream);

  if (distinct > 0) {
    thrust::copy_n(
      rmm::exec_policy_nosync(stream, mr), _key_loc->begin(), distinct, key_loc->begin());
    thrust::copy_n(
      rmm::exec_policy_nosync(stream, mr), _key_hashes->begin(), distinct, key_hashes->begin());

    auto const set_ref =
      key_set->ref(cuco::op::insert).rebind_hash_function(dense_id_hasher{key_hashes->data()});
    thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                       cuda::counting_iterator<int64_t>(0),
                       static_cast<int64_t>(_key_set->capacity()),
                       reinsert_fn{_key_set->data(), set_ref});

    for (size_type i = 0; i < agg_results->num_columns(); ++i) {
      auto target = agg_results->get_column(i).mutable_view();
      cudf::copy_range_in_place(_agg_results->view().column(i), target, 0, distinct, 0, stream);
    }
  }

  auto d_agg_results = mutable_table_device_view::create(*agg_results, stream);

  // The old state is released on the host below, and its memory is not freed in the order of
  // the streams whose kernels read it, so everything `stream` waited for has to be done first.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  _key_set       = std::move(key_set);
  _key_loc       = std::move(key_loc);
  _key_hashes    = std::move(key_hashes);
  _agg_results   = std::move(agg_results);
  _d_agg_results = decltype(_d_agg_results){d_agg_results.release(),
                                            +[](mutable_table_device_view* t) { t->destroy(); }};
  _capacity      = new_capacity;
  _inflight_aggregations.clear();
}

}  // namespace cudf::groupby
