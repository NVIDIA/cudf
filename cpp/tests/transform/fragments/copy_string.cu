/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/strings/mutable_string_view.hpp>
#include <cudf/strings/string_view.cuh>

/**
 * @brief Copies a string into a preallocated output buffer, preserving its UTF-8 bytes.
 *
 * The output byte count must equal the input byte count.
 *
 * @param output Preallocated output string
 * @param input String to copy
 * @return 0 on success
 */
extern "C" __device__ int transform(cudf::mutable_string_view* output, cudf::string_view input)
{
  for (cudf::size_type i = 0; i < input.size_bytes(); ++i) {
    output->data()[i] = input.data()[i];
  }
  return 0;
}
