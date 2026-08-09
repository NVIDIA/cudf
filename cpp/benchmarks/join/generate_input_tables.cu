/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_input_tables.hpp"

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <vector>

template <bool Nullable>
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> generate_input_tables(
  std::vector<cudf::type_id> const& key_types,
  cudf::size_type build_table_numrows,
  cudf::size_type probe_table_numrows,
  cudf::size_type num_payload_cols,
  int multiplicity,
  double selectivity)
{
  // Construct build and probe tables
  // Unique table has build_table_numrows / multiplicity numrows
  auto unique_rows_build_table_numrows =
    static_cast<cudf::size_type>(build_table_numrows / multiplicity);
  // Extra rows provide diverse non-matching keys for probe table
  auto const num_extra_nonmatching_rows = unique_rows_build_table_numrows;
  auto const total_unique_rows = unique_rows_build_table_numrows + num_extra_nonmatching_rows;

  double const null_probability = Nullable ? 0.3 : 0;
  auto const profile =
    data_profile{data_profile_builder()
                   .null_probability(null_probability)
                   .cardinality(total_unique_rows)
                   .list_depth(1)
                   .distribution(cudf::type_id::LIST, distribution_id::GEOMETRIC, 0, 8)};
  auto unique_rows_build_table =
    create_random_table(key_types, row_count{total_unique_rows}, profile, 1);

  auto build_table_gather_map =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              build_table_numrows,
                              cudf::mask_state::ALL_VALID);
  thrust::tabulate(thrust::device,
                   build_table_gather_map->mutable_view().begin<cudf::size_type>(),
                   build_table_gather_map->mutable_view().end<cudf::size_type>(),
                   cuda::proclaim_return_type<cudf::size_type>(
                     [unique_rows_build_table_numrows] __device__(cudf::size_type idx) {
                       return idx % unique_rows_build_table_numrows;
                     }));

  auto const num_matching = static_cast<cudf::size_type>(selectivity * probe_table_numrows);
  auto probe_table_gather_map =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              probe_table_numrows,
                              cudf::mask_state::ALL_VALID);
  thrust::uniform_int_distribution<cudf::size_type> non_matching_dist(
    unique_rows_build_table_numrows,
    unique_rows_build_table_numrows + num_extra_nonmatching_rows - 1);
  thrust::tabulate(thrust::device,
                   probe_table_gather_map->mutable_view().begin<cudf::size_type>(),
                   probe_table_gather_map->mutable_view().end<cudf::size_type>(),
                   cuda::proclaim_return_type<cudf::size_type>(
                     [unique_rows_build_table_numrows, num_matching, non_matching_dist] __device__(
                       cudf::size_type idx) mutable {
                       if (idx < num_matching) {
                         // Matching key: cycle through unique build keys
                         return idx % unique_rows_build_table_numrows;
                       } else {
                         // Non-matching key: random index into extra non-matching rows
                         thrust::default_random_engine rng(idx);
                         return non_matching_dist(rng);
                       }
                     }));

  // Shuffle gather maps to avoid cache effects
  thrust::shuffle(thrust::device,
                  build_table_gather_map->mutable_view().begin<cudf::size_type>(),
                  build_table_gather_map->mutable_view().end<cudf::size_type>(),
                  thrust::default_random_engine{12345});
  thrust::shuffle(thrust::device,
                  probe_table_gather_map->mutable_view().begin<cudf::size_type>(),
                  probe_table_gather_map->mutable_view().end<cudf::size_type>(),
                  thrust::default_random_engine{67890});

  auto build_table = cudf::gather(unique_rows_build_table->view(),
                                  build_table_gather_map->view(),
                                  cudf::out_of_bounds_policy::DONT_CHECK);

  auto probe_table = cudf::gather(unique_rows_build_table->view(),
                                  probe_table_gather_map->view(),
                                  cudf::out_of_bounds_policy::DONT_CHECK);

  auto init       = cudf::make_fixed_width_scalar<cudf::size_type>(static_cast<cudf::size_type>(0));
  auto build_cols = build_table->release();
  auto probe_cols = probe_table->release();
  for (auto i = 0; i < num_payload_cols; i++) {
    build_cols.emplace_back(cudf::sequence(build_table_numrows, *init));
    probe_cols.emplace_back(cudf::sequence(probe_table_numrows, *init));
  }

  return std::pair{std::make_unique<cudf::table>(std::move(build_cols)),
                   std::make_unique<cudf::table>(std::move(probe_cols))};
}

// Explicit instantiations
template std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_input_tables<true>(std::vector<cudf::type_id> const& key_types,
                            cudf::size_type build_table_numrows,
                            cudf::size_type probe_table_numrows,
                            cudf::size_type num_payload_cols,
                            int multiplicity,
                            double selectivity);

template std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_input_tables<false>(std::vector<cudf::type_id> const& key_types,
                             cudf::size_type build_table_numrows,
                             cudf::size_type probe_table_numrows,
                             cudf::size_type num_payload_cols,
                             int multiplicity,
                             double selectivity);
