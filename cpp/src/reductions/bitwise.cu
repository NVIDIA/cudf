/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

namespace cudf::reduction::detail {

namespace {

/**
 * @brief A map from cudf::type_id to cudf type that excludes non-integral types.
 *
 * This is needed because `same_element_type_dispatcher` executes the same code path for all
 * non-dictionary non-nested non-fixed-point types.
 */
template <type_id t>
struct dispatch_void_if_non_integral {
  using type = std::conditional_t<std::is_integral_v<id_to_type<t>>, id_to_type<t>, void>;
};

}  // namespace

std::unique_ptr<scalar> bitwise_reduction(bitwise_op bit_op,
                                          column_view const& col,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const dtype =
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
  auto const do_reduction = [&](auto const op) {
    return cudf::type_dispatcher<dispatch_void_if_non_integral>(
      dtype, op, col, std::nullopt, stream, mr);
  };

  switch (bit_op) {
    case bitwise_op::AND: {
      auto const op = simple::detail::same_element_type_dispatcher<op::bit_and>{};
      return do_reduction(op);
    }
    case bitwise_op::OR: {
      auto const op = simple::detail::same_element_type_dispatcher<op::bit_or>{};
      return do_reduction(op);
    }
    case bitwise_op::XOR: {
      auto const op = simple::detail::same_element_type_dispatcher<op::bit_xor>{};
      return do_reduction(op);
    }
    default: CUDF_UNREACHABLE("Unsupported bitwise operation");
  }
}

}  // namespace cudf::reduction::detail
