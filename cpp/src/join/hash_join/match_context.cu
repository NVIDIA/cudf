/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_match_state.hpp"

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/dispatchers.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace cudf::detail {

template <typename Hasher>
void hash_join<Hasher>::probe(join_kind join,
                              table_view const& left,
                              device_span<size_type> groups,
                              device_span<size_type> counts,
                              cuda::std::uint64_t* total,
                              device_span<cuda::std::uint32_t> matched_groups,
                              cuda::std::uint64_t* matched_build_rows,
                              cuda::stream_ref stream) const
{
  auto const num_rows = static_cast<std::size_t>(left.num_rows());
  CUDF_EXPECTS((groups.empty() || groups.size() == num_rows) &&
                 (counts.empty() || counts.size() == num_rows) &&
                 (matched_groups.empty() ||
                  matched_groups.size() == static_cast<std::size_t>(_right.num_rows())),
               "Probe output sizes do not match the tables");
  auto const data_or_null = [](auto span) { return span.empty() ? nullptr : span.data(); };

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream, temp_mr);
  auto const row_bitmask = cudf::detail::bitmask_and(left, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL ? row_bitmask.data() : nullptr;

  auto probe_rows = [&](auto equality, auto hasher) {
    dispatch_bool(join != join_kind::INNER_JOIN, [&](auto is_outer) {
      launch_hash_csr_probe_count_kernel<decltype(is_outer)::value>(left.num_rows(),
                                                                    valid_rows,
                                                                    data_or_null(groups),
                                                                    data_or_null(counts),
                                                                    total,
                                                                    data_or_null(matched_groups),
                                                                    matched_build_rows,
                                                                    _impl->hash_table(),
                                                                    _impl->csr(),
                                                                    equality,
                                                                    hasher,
                                                                    stream);
    });
  };

  dispatch_join_comparator(
    _right, left, _preprocessed_right, preprocessed_left, _has_nulls, _nulls_equal, probe_rows);
}

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::make_match_context(join_kind join,
                                                               table_view const& left,
                                                               cuda::stream_ref stream,
                                                               cudf::memory_resources mr) const
{
  validate_hash_join_probe(_right, left, _has_nulls);

  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr.get_output_mr());
  cuda::device_buffer<size_type> groups(
    stream, mr.get_output_mr(), _is_empty ? 0 : left.num_rows(), cuda::no_init);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 match_counts->begin(),
                 match_counts->end(),
                 join == join_kind::INNER_JOIN ? 0 : 1);
  } else {
    probe(join, left, groups, *match_counts, nullptr, {}, nullptr, stream);
  }

  return join_match_state::make_context(
    left,
    std::move(match_counts),
    std::make_unique<hash_join_match_state>(this, join, std::move(groups)));
}

template void hash_join<hash_join_hasher>::probe(join_kind,
                                                 cudf::table_view const&,
                                                 device_span<size_type>,
                                                 device_span<size_type>,
                                                 cuda::std::uint64_t*,
                                                 device_span<cuda::std::uint32_t>,
                                                 cuda::std::uint64_t*,
                                                 cuda::stream_ref) const;

template cudf::join_match_context hash_join<hash_join_hasher>::make_match_context(
  join_kind, cudf::table_view const&, cuda::stream_ref, cudf::memory_resources) const;

}  // namespace cudf::detail
