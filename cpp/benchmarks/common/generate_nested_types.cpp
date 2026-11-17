/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_nested_types.hpp"

#include "generate_input.hpp"

#include <vector>

std::unique_ptr<cudf::table> create_lists_data(nvbench::state& state,
                                               cudf::size_type const num_columns,
                                               cudf::size_type const min_val,
                                               cudf::size_type const max_val)
{
  size_t const size_bytes(state.get_int64("size_bytes"));
  cudf::size_type const depth{static_cast<cudf::size_type>(state.get_int64("depth"))};
  auto const null_frequency{state.get_float64("null_frequency")};

  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::LIST, distribution_id::UNIFORM, min_val, max_val);
  table_profile.set_list_depth(depth);
  table_profile.set_null_probability(null_frequency);
  return create_random_table(std::vector<cudf::type_id>(num_columns, cudf::type_id::LIST),
                             table_size_bytes{size_bytes},
                             table_profile);
}

std::unique_ptr<cudf::table> create_structs_data(nvbench::state& state,
                                                 cudf::size_type const n_cols)
{
  auto const n_rows{static_cast<cudf::size_type>(state.get_int64("NumRows"))};
  auto const depth{static_cast<cudf::size_type>(state.get_int64("Depth"))};
  auto const null_frequency = state.get_int64("Nulls") * 0.1;

  data_profile table_profile;
  table_profile.set_distribution_params(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 100);
  table_profile.set_struct_depth(depth);
  table_profile.set_struct_types(std::vector<cudf::type_id>(n_cols, cudf::type_id::INT32));
  table_profile.set_null_probability(null_frequency);
  table_profile.set_cardinality(0);
  return create_random_table({cudf::type_id::STRUCT}, row_count{n_rows}, table_profile);
}
