/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace cudf::test::literals {

constexpr std::size_t operator""_MiB(unsigned long long val) { return val << 20; }
constexpr std::size_t operator""_GiB(unsigned long long val) { return val << 30; }

}  // namespace cudf::test::literals
