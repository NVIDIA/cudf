/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file
 * @brief Device headers required by CUDA source generated from row IR.
 *
 * The row-IR code generator emits this include to provide device operators and string, timestamp,
 * and duration types. Keeping this common prefix separate from generated expressions allows NVRTC
 * to reuse precompiled headers across row-IR compilations.
 */

#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>
