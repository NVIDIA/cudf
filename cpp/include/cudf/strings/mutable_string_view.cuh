/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/mutable_string_view.hpp>
#include <cudf/strings/string_view.cuh>

namespace CUDF_EXPORT cudf {

__device__ inline size_type mutable_string_view::length() const
{
  _length = strings::detail::characters_in_string(_data, _bytes);
  return _length;
}

}  // namespace CUDF_EXPORT cudf
