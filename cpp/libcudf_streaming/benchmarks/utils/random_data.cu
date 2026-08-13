/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "random_data.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

rmm::device_uvector<std::int32_t> random_device_vector(std::size_t nelem,
                                                       std::int32_t min_val,
                                                       std::int32_t max_val,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  // Fill vector with random data.
  using index_t        = std::int64_t;
  auto const end_index = rapidsmpf::safe_cast<index_t>(nelem);
  rmm::device_uvector<std::int32_t> vec(nelem, stream, mr);
  cuda::counting_iterator<index_t> const begin(0);
  cuda::counting_iterator<index_t> const end(end_index);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    begin,
                    end,
                    vec.begin(),
                    [min_val, max_val] __device__(index_t index) {
                      thrust::default_random_engine engine(
                        static_cast<thrust::default_random_engine::result_type>(index));
                      thrust::uniform_int_distribution<std::int32_t> dist(min_val, max_val);
                      return dist(engine);
                    });
  return vec;
}

std::unique_ptr<cudf::column> random_column(cudf::size_type nrows,
                                            std::int32_t min_val,
                                            std::int32_t max_val,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto vec =
    random_device_vector(rapidsmpf::safe_cast<std::size_t>(nrows), min_val, max_val, stream, mr);
  return std::make_unique<cudf::column>(std::move(vec), rmm::device_buffer{0, stream, mr}, 0);
}

cudf::table random_table(cudf::size_type ncolumns,
                         cudf::size_type nrows,
                         std::int32_t min_val,
                         std::int32_t max_val,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (auto i = 0; i < ncolumns; ++i) {
    cols.push_back(random_column(nrows, min_val, max_val, stream, mr));
  }
  return cudf::table(std::move(cols));
}

void random_fill(rapidsmpf::Buffer& buffer, rmm::device_async_resource_ref mr)
{
  switch (buffer.mem_type()) {
    case rapidsmpf::MemoryType::DEVICE: {
      auto const num_elements = std::max<std::size_t>(
        std::size_t{1},
        buffer.size / sizeof(random_data_t) + (buffer.size % sizeof(random_data_t) != 0));
      auto vec = random_device_vector(num_elements,
                                      std::numeric_limits<std::int32_t>::min(),
                                      std::numeric_limits<std::int32_t>::max(),
                                      buffer.stream(),
                                      mr);
      buffer.write_access([&](std::byte* buffer_data, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(
          rapidsmpf::cuda_memcpy_async(buffer_data, vec.data(), buffer.size, stream));
      });
      break;
    }
    default: RAPIDSMPF_FAIL("unsupported memory type", std::invalid_argument);
  }
}
