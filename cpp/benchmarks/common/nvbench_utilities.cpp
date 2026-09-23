/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvbench_utilities.hpp"

#include <nvbench/nvbench.cuh>

// This function is copied over from
// https://github.com/NVIDIA/nvbench/blob/a171514056e5d6a7f52a035dd6c812fa301d4f4f/nvbench/detail/measure_cold.cu#L190-L224.
void set_throughputs(nvbench::state& state)
{
  double avg_cuda_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");

  if (auto const items = state.get_element_count(); items != 0) {
    auto& summ = state.add_summary("nv/cold/bw/item_rate");
    summ.set_string("name", "Elem/s");
    summ.set_string("hint", "item_rate");
    summ.set_string("description", "Number of input elements processed per second");
    summ.set_float64("value", static_cast<double>(items) / avg_cuda_time);
  }

  if (auto const bytes = state.get_global_memory_rw_bytes(); bytes != 0) {
    auto const avg_used_gmem_bw = static_cast<double>(bytes) / avg_cuda_time;
    {
      auto& summ = state.add_summary("nv/cold/bw/global/bytes_per_second");
      summ.set_string("name", "GlobalMem BW");
      summ.set_string("hint", "byte_rate");
      summ.set_string("description",
                      "Number of bytes read/written per second to the CUDA "
                      "device's global memory");
      summ.set_float64("value", avg_used_gmem_bw);
    }

    {
      auto const peak_gmem_bw =
        static_cast<double>(state.get_device()->get_global_memory_bus_bandwidth());

      auto& summ = state.add_summary("nv/cold/bw/global/utilization");
      summ.set_string("name", "BWUtil");
      summ.set_string("hint", "percentage");
      summ.set_string("description",
                      "Global device memory utilization as a percentage of the "
                      "device's peak bandwidth");
      summ.set_float64("value", avg_used_gmem_bw / peak_gmem_bw);
    }
  }
}
