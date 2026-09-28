/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/stream>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Get the default stream to use for tests.
 *
 * The standard behavior of this function is to return cudf's default stream
 * (cudf::get_default_stream). This function is primarily provided as an
 * overload target for preload libraries (via LD_PRELOAD) so that the default
 * stream used for tests may be modified for tracking purposes. All tests of
 * public APIs that accept streams should pass `cudf::test::get_default_stream`
 * as the stream argument so that a preload library changing the behavior of
 * this function will trigger those tests to run on a different stream than
 * `cudf::get_default_stream`.
 *
 * @return The default stream to use for tests.
 */
cuda::stream_ref const get_default_stream();

}  // namespace test
}  // namespace CUDF_EXPORT cudf
