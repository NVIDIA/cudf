/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief Whether headers are being used for runtime device compilation.
 *
 * NVRTC enables this automatically. Other runtime device compilers can define this macro to 1
 * before including cuDF headers to exclude host-only facilities without changing compiler identity.
 * Precompiled JIT fragments also enable it to match the runtime header interfaces.
 * Use the same value when preparing and consuming precompiled header or module artifacts.
 */
#ifndef CUDF_RUNTIME_COMPILATION
#if defined(__CUDACC_RTC__)
#define CUDF_RUNTIME_COMPILATION 1
#else
#define CUDF_RUNTIME_COMPILATION 0
#endif
#endif
