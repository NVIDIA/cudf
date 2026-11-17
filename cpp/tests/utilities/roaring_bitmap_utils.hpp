/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/span.hpp>

#include <roaring/roaring.h>
#include <roaring/roaring64.h>

/**
 * @brief Constructs a 32 or 64-bit serialized roaring bitmap containing the given keys
 */
template <typename Key>
inline auto serialize_roaring_bitmap(cudf::host_span<Key const> keys)
{
  static_assert(
    std::is_same_v<Key, cuda::std::uint32_t> or std::is_same_v<Key, cuda::std::uint64_t>,
    "Key must be uint32 or uint64");

  auto serialized = std::vector<cuda::std::byte>();

  if constexpr (std::is_same_v<Key, cuda::std::uint32_t>) {
    auto* bitmap = roaring::api::roaring_bitmap_create();
    auto ctx     = roaring::api::roaring_bulk_context_t{};
    for (auto key : keys) {
      roaring::api::roaring_bitmap_add_bulk(bitmap, &ctx, key);
    }

    auto const num_bytes = roaring::api::roaring_bitmap_portable_size_in_bytes(bitmap);
    serialized.resize(num_bytes);
    roaring::api::roaring_bitmap_portable_serialize(bitmap,
                                                    reinterpret_cast<char*>(serialized.data()));
    roaring::api::roaring_bitmap_free(bitmap);
  } else {
    auto* bitmap = roaring::api::roaring64_bitmap_create();
    auto ctx =
      roaring::api::roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};
    for (auto key : keys) {
      roaring::api::roaring64_bitmap_add_bulk(bitmap, &ctx, key);
    }

    auto const num_bytes = roaring::api::roaring64_bitmap_portable_size_in_bytes(bitmap);
    serialized.resize(num_bytes);
    roaring::api::roaring64_bitmap_portable_serialize(bitmap,
                                                      reinterpret_cast<char*>(serialized.data()));
    roaring::api::roaring64_bitmap_free(bitmap);
  }

  return serialized;
}
