/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>

#include <cstddef>
#include <string>

namespace cudf::io::detail {

/**
 * @brief The size used for padding a data buffer's size to a multiple of the padding.
 *
 * Padding is necessary for input/output buffers of several compression/decompression kernels
 * (inflate_kernel and nvcomp snappy). Such kernels operate on aligned data pointers, which require
 * padding to the buffers so that the pointers can shift along the address space to satisfy their
 * alignment requirement.
 *
 * In the meantime, it is not entirely clear why such padding is needed. We need to further
 * investigate and implement a better fix rather than just padding the buffer.
 * See https://github.com/NVIDIA/cudf/issues/13605.
 */
constexpr std::size_t BUFFER_PADDING_MULTIPLE{8};

[[nodiscard]] std::string compression_type_name(compression_type compression);

}  // namespace cudf::io::detail
