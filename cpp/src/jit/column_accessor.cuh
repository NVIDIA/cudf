/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/type_traits>

namespace cudf {
namespace jit {

template <int32_t Index, typename Column, typename Element, bool AsScalar, int32_t TableIndex>
struct column_accessor {
  static constexpr int32_t index       = Index;
  static constexpr int32_t table_index = TableIndex;
  using column_type                    = Column;
  using element_type                   = Element;
  using optional_element_type          = cuda::std::optional<element_type>;

  static constexpr bool as_scalar = AsScalar;

  static constexpr bool is_mutable_string =
    cuda::std::is_same_v<element_type, cuda::std::span<char>>;

  static __device__ constexpr size_type map_index(size_type row)
  {
    if constexpr (as_scalar) {
      return 0;
    } else {
      return row;
    }
  }

  template <typename T>
  static __device__ auto& column(T const* __restrict__ cols)
    requires(sizeof(T) == sizeof(column_type) && alignof(T) == alignof(column_type))
  { return reinterpret_cast<column_type const&>(cols[index]); }

  static __device__ auto element(auto const* __restrict__ cols, size_type row)
  { return column(cols).template element<element_type>(map_index(row)); }

  static __device__ bool is_null(auto const* __restrict__ cols, size_type row)
  { return column(cols).is_null(map_index(row)); }

  static __device__ bool is_valid(auto const* __restrict__ cols, size_type row)
  { return column(cols).is_valid(map_index(row)); }

  static __device__ auto nullable_element(auto const* __restrict__ cols, size_type row)
  { return column(cols).template nullable_element<element_type>(map_index(row)); }

  static __device__ void set_null_mask_word(auto const* __restrict__ cols,
                                            size_type word_index,
                                            bitmask_type word)
    requires(!as_scalar)
  {
    auto* mask = column(cols).null_mask();

    if (mask == nullptr) { return; }

    mask[word_index] = word;
  }

  static __device__ void assign(auto const* __restrict__ cols, size_type row, element_type value)
    requires(!as_scalar)
  { column(cols).template assign<element_type>(row, value); }

  static __device__ element_type output_arg(auto const* __restrict__ cols, size_type row)
    requires(!as_scalar)
  {
    if constexpr (is_mutable_string) {
      return element(cols, row);
    } else {
      return {};
    }
  }

  static __device__ optional_element_type null_output_arg(auto const* __restrict__ cols,
                                                          size_type row)
    requires(!as_scalar)
  {
    if constexpr (is_mutable_string) {
      return element(cols, row);
    } else {
      return {};
    }
  }
};

}  // namespace jit
}  // namespace cudf
