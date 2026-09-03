/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/stream_ref>

namespace CUDF_EXPORT cudf {

namespace detail {

/**
 * @brief Default stream for cudf
 *
 * Use this value to ensure the correct stream is used when compiled with per
 * thread default stream.
 */
extern cuda::stream_ref const default_stream_value;

}  // namespace detail

}  // namespace CUDF_EXPORT cudf
