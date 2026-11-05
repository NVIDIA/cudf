/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cudf::test::literals {

constexpr auto kilo{long{1} << 10};
constexpr auto mega{long{1} << 20};
constexpr auto giga{long{1} << 30};
constexpr auto tera{long{1} << 40};
constexpr auto peta{long{1} << 50};

// user-defined Byte literals
constexpr unsigned long long operator""_B(unsigned long long val) { return val; }
constexpr unsigned long long operator""_KiB(unsigned long long const val) { return kilo * val; }
constexpr unsigned long long operator""_MiB(unsigned long long const val) { return mega * val; }
constexpr unsigned long long operator""_GiB(unsigned long long const val) { return giga * val; }
constexpr unsigned long long operator""_TiB(unsigned long long const val) { return tera * val; }
constexpr unsigned long long operator""_PiB(unsigned long long const val) { return peta * val; }

}  // namespace cudf::test::literals
