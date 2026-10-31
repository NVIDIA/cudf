/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file executor.cu
 * @brief NVRTC entry point for compiling a generated regex matcher.
 *
 * Includes the stable device runtime before the precompiled-header boundary and
 * the pattern-specific source afterward.
 */
#include <libregex_ir/executor.cuh>
#pragma nv_hdrstop
#include <libregex_ir/regex_ir_generated.cuh>
