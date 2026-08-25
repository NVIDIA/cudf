/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "check_nvcomp_output_sizes.hpp"

#include <cudf/utilities/error.hpp>

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>

namespace {

struct java_domain {
  static constexpr char const* name{"Java"};
};

}  // anonymous namespace

namespace cudf {
namespace java {

/**
 * Check that the vector of expected uncompressed sizes matches the vector of actual compressed
 * sizes. Both vectors are assumed to be in device memory and contain num_chunks elements.
 */
bool check_nvcomp_output_sizes(std::size_t const* dev_uncompressed_sizes,
                               std::size_t const* dev_actual_uncompressed_sizes,
                               std::size_t num_chunks,
                               cuda::stream_ref stream)
{
  NVTX3_FUNC_RANGE_IN(java_domain);
  return thrust::equal(rmm::exec_policy_nosync(stream),
                       dev_uncompressed_sizes,
                       dev_uncompressed_sizes + num_chunks,
                       dev_actual_uncompressed_sizes);
}

}  // namespace java
}  // namespace cudf
