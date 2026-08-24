/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/algorithm>

namespace cudf {
namespace detail {

struct window_wrapper_base {
  cudf::size_type const* group_offsets = nullptr;
  cudf::size_type const* group_labels  = nullptr;
  cudf::size_type window               = 0;
};

struct fixed_window_wrapper : public window_wrapper_base {
  __device__ __host__ fixed_window_wrapper(cudf::size_type window)
    : window_wrapper_base{nullptr, nullptr, window}
  {
  }

  __device__ __host__ fixed_window_wrapper(window_wrapper_base const& base)
    : window_wrapper_base(base)
  {
  }

  __device__ __host__ cudf::size_type operator[](cudf::size_type) const { return window; }
};

struct variable_window_wrapper : public window_wrapper_base {
  __device__ __host__ variable_window_wrapper(cudf::size_type const* group_offsets)
    : window_wrapper_base{group_offsets, nullptr, 0}
  {
  }

  __device__ __host__ variable_window_wrapper(window_wrapper_base const& base)
    : window_wrapper_base(base)
  {
  }

  __device__ __host__ cudf::size_type operator[](cudf::size_type idx) const
  {
    return group_offsets[idx];
  }
};

struct preceding_window_wrapper : public window_wrapper_base {
  __device__ __host__ preceding_window_wrapper(cudf::size_type const* group_offsets,
                                               cudf::size_type const* group_labels,
                                               cudf::size_type window)
    : window_wrapper_base{group_offsets, group_labels, window}
  {
  }

  __device__ __host__ preceding_window_wrapper(window_wrapper_base const& base)
    : window_wrapper_base(base)
  {
  }

  __device__ cudf::size_type operator[](cudf::size_type idx) const
  {
    auto group_label = group_labels[idx];
    auto group_start = group_offsets[group_label];
    return cuda::std::min(window, idx - group_start + 1);  // Preceding includes current row.
  }
};

struct following_window_wrapper : public window_wrapper_base {
  __device__ __host__ following_window_wrapper(cudf::size_type const* group_offsets,
                                               cudf::size_type const* group_labels,
                                               cudf::size_type window)
    : window_wrapper_base{group_offsets, group_labels, window}
  {
  }

  __device__ __host__ following_window_wrapper(window_wrapper_base const& base)
    : window_wrapper_base(base)
  {
  }

  __device__ cudf::size_type operator[](cudf::size_type idx) const
  {
    auto group_label = group_labels[idx];
    auto group_end =
      group_offsets[group_label +
                    1];  // Cannot fall off the end, since offsets is capped with `input.size()`.
    return cuda::std::min(window, (group_end - 1) - idx);
  }
};

static_assert(sizeof(fixed_window_wrapper) == sizeof(variable_window_wrapper));
static_assert(alignof(fixed_window_wrapper) == alignof(variable_window_wrapper));

static_assert(sizeof(variable_window_wrapper) == sizeof(fixed_window_wrapper));
static_assert(alignof(variable_window_wrapper) == alignof(fixed_window_wrapper));

static_assert(sizeof(fixed_window_wrapper) == sizeof(preceding_window_wrapper));
static_assert(alignof(fixed_window_wrapper) == alignof(preceding_window_wrapper));

static_assert(sizeof(fixed_window_wrapper) == sizeof(following_window_wrapper));
static_assert(alignof(fixed_window_wrapper) == alignof(following_window_wrapper));

}  // namespace detail
}  // namespace cudf
