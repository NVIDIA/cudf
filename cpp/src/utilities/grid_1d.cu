/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::detail {

grid_1d::grid_1d(thread_index_type overall_num_elements,
                 thread_index_type num_threads_per_block,
                 thread_index_type elements_per_thread)
  : num_threads_per_block(num_threads_per_block),
    num_blocks(cudf::detail::div_rounding_up_safe(overall_num_elements,
                                                  elements_per_thread * num_threads_per_block))
{
  CUDF_EXPECTS(num_threads_per_block > 0, "num_threads_per_block must be > 0");
  CUDF_EXPECTS(num_blocks > 0, "num_blocks must be > 0");
}

}  // namespace cudf::detail
