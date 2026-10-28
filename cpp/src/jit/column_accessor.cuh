/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace cudf {
namespace jit {

template <int32_t Index, typename Element, bool AsScalar, int32_t TableIndex>
struct column_accessor {
  static constexpr int32_t index       = Index;
  using element_type                   = Element;
  static constexpr bool as_scalar      = AsScalar;
  static constexpr int32_t table_index = TableIndex;

  static constexpr bool is_inplace_output =
    cuda::std::is_same_v<element_type, cuda::std::span<char>>;

  static __device__ constexpr size_type map_index(size_type row)
  {
    if constexpr (as_scalar) {
      return 0;
    } else {
      return row;
    }
  }

  static __device__ auto& column(auto const* __restrict__ columns) { return columns[index]; }

  static __device__ decltype(auto) element(auto const* __restrict__ columns, size_type row)
  {
    return columns[index].template element<element_type>(map_index(row));
  }

  static __device__ bool is_null(auto const* __restrict__ columns, size_type row)
  {
    return columns[index].is_null(map_index(row));
  }

  static __device__ bool is_valid(auto const* __restrict__ columns, size_type row)
  {
    return columns[index].is_valid(map_index(row));
  }

  static __device__ decltype(auto) nullable_element(auto const* __restrict__ columns, size_type row)
  {
    return columns[index].template nullable_element<element_type>(map_index(row));
  }

  static __device__ void set_null_mask_word(auto const* __restrict__ columns,
                                            size_type word_index,
                                            bitmask_type word)
    requires(!as_scalar)
  {
    auto* mask = columns[index].null_mask();

    if (mask == nullptr) { return; }

    mask[word_index] = word;
  }

  static __device__ void assign(auto const* __restrict__ columns, size_type row, auto value)
    requires(!as_scalar)
  {
    columns[index].template assign<element_type>(row, value);
  }

  static __device__ decltype(auto) output_arg(auto const* __restrict__ columns, size_type row)
    requires(!as_scalar)
  {
    if constexpr (is_inplace_output) {
      return element(columns, row);
    } else if constexpr (is_repr_element<element_type>) {
      return typename element_type::value_type{};
    } else {
      return element_type{};
    }
  }

  static __device__ decltype(auto) nullable_output_arg(auto const* __restrict__ columns,
                                                       size_type row)
    requires(!as_scalar)
  {
    if constexpr (is_inplace_output) {
      return cuda::std::optional<element_type>(element(columns, row));
    } else if constexpr (is_repr_element<element_type>) {
      return cuda::std::optional<typename element_type::value_type>{};
    } else {
      return cuda::std::optional<element_type>{};
    }
  }
};

}  // namespace jit
}  // namespace cudf
