/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "lto_benchmark.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>

int main(int argc, char** argv)
{
  try {
    reduction_benchmark::options settings;
    for (int32_t index = 1; index < argc; ++index) {
      std::string_view argument{argv[index]};
      if (argument == "--smoke") {
        settings.smoke = true;
      } else if (argument == "--csv" && index + 1 < argc) {
        settings.csv = argv[++index];
      } else if (argument == "--rows" && index + 1 < argc) {
        auto const rows = std::stoll(argv[++index]);
        if (rows < 0 || rows > std::numeric_limits<cudf::size_type>::max()) {
          throw std::invalid_argument{"Row count is outside the cuDF range"};
        }
        settings.rows = static_cast<cudf::size_type>(rows);
      } else {
        throw std::invalid_argument{"Usage: REDUCE_LTO_BENCH [--smoke] [--rows N] [--csv PATH]"};
      }
    }
    ::setenv("LIBCUDF_JIT_ENABLED", "1", 1);
    size_t free_bytes, total_bytes;
    CUDF_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    auto const budget = (total_bytes / 100 * 90) & ~size_t{255};
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource pool{
      rmm::device_async_resource_ref{upstream}, std::min<size_t>(256 << 20, budget), budget};
    rmm::mr::statistics_resource_adaptor statistics{rmm::device_async_resource_ref{pool}};
    using resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
    auto previous =
      cudf::set_current_device_resource(resource{rmm::device_async_resource_ref{statistics}});
    struct restore_resource {
      resource previous;
      ~restore_resource() { cudf::set_current_device_resource(std::move(previous)); }
    } restore{std::move(previous)};
    std::ofstream output{settings.csv};
    if (!output) { throw std::runtime_error{"Cannot open benchmark CSV output"}; }
    output << std::setprecision(12)
           << "workload,shape,engine,rows,segments,null_percent,hint,output_layout,logical_bytes,"
              "median_ms,min_ms,max_ms,logical_GBps,sample0_ms,sample1_ms,sample2_ms,sample3_ms,"
              "sample4_ms\n";
    reduction_benchmark::run_q13(settings, output);
    reduction_benchmark::run_segmented(settings, output);
    std::cout << "Validated all cases; four warmups and five timed executions per case. CSV: "
              << settings.csv
              << "\nPeak tracked device bytes: " << statistics.get_bytes_counter().peak << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
