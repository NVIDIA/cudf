/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {

/**
 * @brief Returns the NVRTC/nvJitLink architecture name for a compute capability given the
 * comma-separated CMAKE_CUDA_ARCHITECTURES list this build was compiled with
 *
 * Keeps the baseline name (sm_<cc>) whenever the list has a baseline entry for <cc>; otherwise
 * returns the feature-set variant the build has (sm_<cc>a, then sm_<cc>f). Entry suffixes such
 * as -real / -virtual are ignored.
 */
std::string jit_arch_name_for(std::int32_t compute_capability, std::string_view archs);

}  // namespace CUDF_EXPORT cudf
