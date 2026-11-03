/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

  [[nodiscard]] size_t peak_memory_usage() const noexcept
  {
    return statistics_mr.get_bytes_counter().peak;
  }

 private:
  rmm::mr::statistics_resource_adaptor statistics_mr;
};

}  // namespace cudf
