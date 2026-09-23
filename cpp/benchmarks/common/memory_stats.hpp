/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

namespace cudf {

class memory_stats_logger {
 public:
  memory_stats_logger() : statistics_mr(cudf::get_current_device_resource_ref())
  {
    cudf::set_current_device_resource(statistics_mr);
  }

  ~memory_stats_logger()
  {
    cudf::set_current_device_resource(statistics_mr.get_upstream_resource());
  }

  /**
   * @brief Start a fresh measurement interval after all tracked allocations have been released.
   *
   * Callers must release all owners of tracked allocations and synchronize the relevant streams
   * before calling this function. Keep the resource itself unchanged so clients that captured it
   * still contribute statistics.
   *
   * @throws cudf::logic_error if any tracked bytes or allocations remain live.
   */
  void reset_counters()
  {
    CUDF_EXPECTS(statistics_mr.get_bytes_counter().value == 0 &&
                   statistics_mr.get_allocations_counter().value == 0,
                 "Release tracked allocations before resetting memory statistics");
    statistics_mr.push_counters();
  }

  [[nodiscard]] size_t peak_memory_usage() const noexcept
  {
    return statistics_mr.get_bytes_counter().peak;
  }

 private:
  rmm::mr::statistics_resource_adaptor statistics_mr;
};

}  // namespace cudf
