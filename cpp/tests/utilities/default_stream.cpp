/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/default_stream.hpp>

#include <cudf/utilities/default_stream.hpp>

namespace cudf {
namespace test {

cuda::stream_ref const get_default_stream() { return cudf::get_default_stream(); }

}  // namespace test
}  // namespace cudf
