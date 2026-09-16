/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cuda/stream>

namespace cudf {
// Create a strings-type column from array of pointer/size pairs
std::unique_ptr<scalar> make_string_scalar(std::string const& string,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto s = new string_scalar(string, true, stream, mr);
  return std::unique_ptr<scalar>(s);
}

}  // namespace cudf
