/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/logger.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <type_traits>

static_assert(std::is_reference_v<decltype(cudf::default_logger())>);

void cudf_export_consumer_compile_test() { (void)cudf::get_default_stream(); }
